"""
LangGraph 工作流定义

整体流程：

用户输入股票代码
    ↓
fetcher_node：抓取行情、K线图、Prophet预测、新闻、资金流、公司档案
    ↓
并行分析：
    quant_node：量化/资金面分析
    visual_node：K线图视觉分析
    news_node：新闻/舆情分析
    ↓
投研辩论：
    bull_node：构建买入/持有的最强论证
    bear_node：针对牛方报告提出卖出/减仓反驳
    risk_node：风控裁决牛熊分歧并给出置信度上限
    ↓
cio_node：综合三个 Agent 的报告，生成最终投资决策
    ↓
END
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import NotRequired, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph

from src.agents import (
    BEAR_PROMPT,
    BULL_PROMPT,
    CIO_PROMPT,
    NEWS_PROMPT,
    QUANT_PROMPT,
    RISK_PROMPT,
    VisualAgent,
)
from src.models import (
    BearCaseReport,
    BullCaseReport,
    FinalDecision,
    NewsReport,
    QuantReport,
    RiskDebateReport,
    VisualReport,
)
from src.settings import is_english, output_language_instruction
from src.tools import (
    forecast_with_prophet,
    generate_stock_chart,
    get_money_flow,
    get_stock_data,
    get_stock_profile,
    search_news,
)


load_dotenv()


# ============================================================
# 1. LLM 初始化
# ============================================================
base_llm = init_chat_model(
    model=os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo"),
    model_provider="openai",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 视觉模型单独走 VisualAgent，因为它使用 zhipuai SDK 看图
visual_agent = VisualAgent()

# 结构化输出模型
quant_llm = base_llm.with_structured_output(QuantReport)
news_llm = base_llm.with_structured_output(NewsReport)
bull_llm = base_llm.with_structured_output(BullCaseReport)
bear_llm = base_llm.with_structured_output(BearCaseReport)
risk_llm = base_llm.with_structured_output(RiskDebateReport)
cio_llm = base_llm.with_structured_output(FinalDecision)


# ============================================================
# 2. LangGraph State 定义
# ============================================================
class AgentState(TypedDict):
    """
    LangGraph 全局状态。

    注意：
    - 初始输入通常只有 symbol。
    - 后续字段由 fetcher / quant / visual / news / cio 节点逐步写入。
    - 所以除 symbol 外，其余字段使用 NotRequired。
    """

    symbol: str

    price_data: NotRequired[object]  # pandas DataFrame
    chart_path: NotRequired[str]
    prophet_result: NotRequired[str]
    news_data: NotRequired[str]
    fund_flow_data: NotRequired[str]
    profile_data: NotRequired[str]

    quant_report: NotRequired[QuantReport | None]
    visual_report: NotRequired[VisualReport | str | None]
    news_report: NotRequired[NewsReport | None]
    bull_report: NotRequired[BullCaseReport | None]
    bear_report: NotRequired[BearCaseReport | None]
    risk_debate_report: NotRequired[RiskDebateReport | None]
    final_report: NotRequired[FinalDecision | None]


# ============================================================
# 3. 工具安全调用函数
# ============================================================
def safe_tool_invoke(tool, payload: dict, default=None, label: str = "tool"):
    """
    安全调用 LangChain @tool 工具。

    这样做的原因：
    A股数据接口、新闻搜索接口、资金流接口都可能临时失败。
    如果一个工具失败就让整个工作流崩溃，用户体验会很差。

    所以这里统一 catch exception，并返回 default。
    """
    try:
        return tool.invoke(payload)
    except Exception as e:
        print(f"[Warning] {label} failed: {e}")
        return default


def log_text(cn: str, en: str) -> str:
    """
    Return localized log text for console progress messages.
    """
    return en if is_english() else cn


def safe_model_json(obj) -> str:
    """
    将 Pydantic 对象安全转换为 JSON 字符串。
    """
    if obj is None:
        return "N/A"

    if hasattr(obj, "model_dump_json"):
        try:
            return obj.model_dump_json(indent=2)
        except Exception:
            return str(obj)

    return str(obj)


def get_last_price_text(state: AgentState) -> str:
    """
    从 state 中提取最新收盘价文本，供多个 Agent 复用。
    """
    df = state.get("price_data")

    try:
        if df is not None and not getattr(df, "empty", True):
            return f"{float(df.iloc[-1]['close']):.2f}"
    except Exception:
        pass

    return "0.00"


def visual_report_to_text(visual_report) -> str:
    """
    将视觉报告统一转成 prompt 可读文本。
    """
    if isinstance(visual_report, VisualReport):
        return visual_report.model_dump_json(indent=2)

    return str(visual_report or "N/A")


# ============================================================
# 4. 节点定义：Fetcher
# ============================================================
def fetcher_node(state: AgentState):
    """
    数据中心节点。

    负责统一抓取：
    1. 股票行情数据
    2. K线图图片路径
    3. Prophet 时间序列预测结果
    4. 新闻数据
    5. 资金流数据
    6. 公司档案数据
    """
    symbol = state["symbol"]
    print(
        "\n"
        + log_text(
            f"[System] 正在获取 {symbol} 的数据...",
            f"[System] Fetching data for {symbol}...",
        )
    )

    df = safe_tool_invoke(
        get_stock_data,
        {"symbol": symbol},
        default=None,
        label="get_stock_data",
    )

    if df is None or getattr(df, "empty", True):
        print(
            log_text(
                f"[Warning] 未找到 {symbol} 的行情数据。",
                f"[Warning] No price data found for {symbol}.",
            )
        )
        return {
            "price_data": None,
            "chart_path": "",
            "prophet_result": "No price data available.",
            "news_data": "No price data available.",
            "fund_flow_data": "No price data available.",
            "profile_data": f"Symbol: {symbol}\nProfile fetch skipped because price data is empty.",
        }

    chart_path = safe_tool_invoke(
        generate_stock_chart,
        {"df": df, "symbol": symbol},
        default="",
        label="generate_stock_chart",
    )

    prophet_res = safe_tool_invoke(
        forecast_with_prophet,
        {"df": df},
        default="Prophet forecast failed.",
        label="forecast_with_prophet",
    )

    news = safe_tool_invoke(
        search_news,
        {"symbol": symbol},
        default="News fetch failed.",
        label="search_news",
    )

    flow = safe_tool_invoke(
        get_money_flow,
        {"symbol": symbol},
        default="Money flow fetch failed.",
        label="get_money_flow",
    )

    profile = safe_tool_invoke(
        get_stock_profile,
        {"symbol": symbol},
        default=f"Symbol: {symbol}\nCompany profile fetch failed.",
        label="get_stock_profile",
    )

    return {
        "price_data": df,
        "chart_path": chart_path,
        "prophet_result": prophet_res,
        "news_data": news,
        "fund_flow_data": flow,
        "profile_data": profile,
    }


# ============================================================
# 5. 节点定义：Quant Agent
# ============================================================
async def quant_node(state: AgentState):
    """
    量化分析节点。

    输入：
    - price_data
    - prophet_result
    - fund_flow_data

    输出：
    - quant_report: QuantReport
    """
    print(
        log_text(
            "[Agent] Quant Analyst 正在计算...",
            "[Agent] Quant Analyst is calculating...",
        )
    )

    df = state.get("price_data")

    if df is None or getattr(df, "empty", True):
        return {
            "quant_report": QuantReport(
                main_force_intent="washing",
                confidence_score=0.0,
                key_technical_signals=["无行情数据"],
                price_volume_analysis="无法获取行情数据，量价分析不可用。",
                risk_assessment="high",
                detailed_analysis="由于没有有效的行情数据，量化模型无法判断主力资金意图。本次量化报告仅作为失败占位，不应作为真实交易依据。",
            )
        }

    try:
        needed_cols = ["date", "close", "rsi", "macd"]
        available_cols = [col for col in needed_cols if col in df.columns]

        if available_cols:
            recent_str = df.tail(5)[available_cols].to_string()
        else:
            recent_str = df.tail(5).to_string()

        messages = QUANT_PROMPT.invoke(
            {
                "prophet_forecast": state.get("prophet_result", "N/A"),
                "market_data_str": recent_str,
                "fund_flow_data": state.get("fund_flow_data", "N/A"),
                "output_language_instruction": output_language_instruction(),
            }
        )

        result = await quant_llm.ainvoke(messages)
        return {"quant_report": result}

    except Exception as e:
        print(f"[Warning] Quant structured output failed: {e}")

        fallback_report = QuantReport(
            main_force_intent="washing",
            confidence_score=0.3,
            key_technical_signals=["结构化分析失败"],
            price_volume_analysis="量化节点运行失败，可能是模型结构化输出格式不符合 Pydantic 要求。",
            risk_assessment="high",
            detailed_analysis=(
                "量化分析结构化输出失败。本次无法稳定判断主力是在吸筹、洗盘还是出货。"
                "建议检查 OPENAI_MODEL_NAME、OPENAI_BASE_URL、OPENAI_API_KEY 是否正确，"
                "以及 QUANT_PROMPT 是否与 QuantReport 字段保持一致。"
            ),
        )

        return {"quant_report": fallback_report}


# ============================================================
# 6. 节点定义：Visual Agent
# ============================================================
async def visual_node(state: AgentState):
    """
    视觉图表分析节点。

    输入：
    - chart_path

    输出：
    - visual_report: VisualReport | str
    """
    print(
        log_text(
            "[Agent] Visual Analyst 正在查看 K 线图...",
            "[Agent] Visual Analyst is looking at the chart...",
        )
    )

    chart_path = state.get("chart_path", "")

    if not chart_path:
        return {"visual_report": "No Chart Image Available"}

    try:
        report = visual_agent.analyze_chart(chart_path)
        return {"visual_report": report}

    except Exception as e:
        print(f"[Warning] Visual analysis failed: {e}")
        return {"visual_report": f"Visual Analysis Failed: {e}"}


# ============================================================
# 7. 节点定义：News Agent
# ============================================================
async def news_node(state: AgentState):
    """
    新闻舆情分析节点。

    输入：
    - news_data

    输出：
    - news_report: NewsReport
    """
    print(
        log_text(
            "[Agent] News Analyst 正在阅读新闻...",
            "[Agent] News Analyst is reading...",
        )
    )

    try:
        messages = NEWS_PROMPT.invoke(
            {
                "news_data": state.get("news_data", "N/A"),
                "output_language_instruction": output_language_instruction(),
            }
        )

        result = await news_llm.ainvoke(messages)
        return {"news_report": result}

    except Exception as e:
        print(f"[Warning] News structured output failed: {e}")
        print("[Info] Falling back to basic NewsReport...")

        fallback_report = NewsReport(
            company_profile="新闻结构化分析失败，无法提取完整公司档案。",
            sentiment_score=0.0,
            key_news_summary=["新闻结构化分析失败"],
            market_positioning="N/A",
            risk_warnings=["新闻数据解析错误或模型输出格式不符合要求"],
        )

        return {"news_report": fallback_report}


# ============================================================
# 8. 节点定义：Bull / Bear / Risk Debate Agents
# ============================================================
async def bull_node(state: AgentState):
    """
    牛方节点。

    输入：
    - quant_report
    - visual_report
    - news_report
    - profile_data

    输出：
    - bull_report: BullCaseReport
    """
    print(
        log_text(
            "[Agent] Bull Researcher 正在构建最强持有/买入论证...",
            "[Agent] Bull Researcher is building the strongest hold/buy case...",
        )
    )

    try:
        messages = BULL_PROMPT.invoke(
            {
                "last_price": get_last_price_text(state),
                "profile_data": state.get("profile_data", "N/A"),
                "visual_report": visual_report_to_text(state.get("visual_report")),
                "quant_report": safe_model_json(state.get("quant_report")),
                "news_report": safe_model_json(state.get("news_report")),
                "output_language_instruction": output_language_instruction(),
            }
        )

        result = await bull_llm.ainvoke(messages)
        return {"bull_report": result}

    except Exception as e:
        print(f"[Warning] Bull structured output failed: {e}")
        return {
            "bull_report": BullCaseReport(
                stance="HOLD",
                strongest_points=["牛方结构化分析失败，无法形成强看多证据"],
                evidence_quality="weak",
                counter_to_bear_risks="由于牛方节点失败，无法有效反驳潜在看空风险。",
                invalidation_condition="若关键行情、资金流或新闻数据仍无法验证，应视为牛方观点无效。",
            )
        }


async def bear_node(state: AgentState):
    """
    熊方节点。

    输入：
    - bull_report
    - quant_report
    - visual_report
    - news_report
    - profile_data

    输出：
    - bear_report: BearCaseReport
    """
    print(
        log_text(
            "[Agent] Bear Researcher 正在反驳牛方论证...",
            "[Agent] Bear Researcher is challenging the bull case...",
        )
    )

    try:
        messages = BEAR_PROMPT.invoke(
            {
                "last_price": get_last_price_text(state),
                "profile_data": state.get("profile_data", "N/A"),
                "visual_report": visual_report_to_text(state.get("visual_report")),
                "quant_report": safe_model_json(state.get("quant_report")),
                "news_report": safe_model_json(state.get("news_report")),
                "bull_report": safe_model_json(state.get("bull_report")),
                "output_language_instruction": output_language_instruction(),
            }
        )

        result = await bear_llm.ainvoke(messages)
        return {"bear_report": result}

    except Exception as e:
        print(f"[Warning] Bear structured output failed: {e}")
        return {
            "bear_report": BearCaseReport(
                stance="HOLD",
                strongest_points=["熊方结构化分析失败，无法形成可靠卖出证据"],
                evidence_quality="weak",
                rebuttal_to_bull_case="由于熊方节点失败，无法对牛方报告做有效逐点反驳。",
                risk_trigger="若价格放量跌破关键支撑或资金连续流出，应重新评估减仓。",
            )
        }


async def risk_node(state: AgentState):
    """
    风控裁决节点。

    输入：
    - bull_report
    - bear_report
    - quant_report
    - visual_report
    - news_report

    输出：
    - risk_debate_report: RiskDebateReport
    """
    print(
        log_text(
            "[Agent] Risk Manager 正在裁决牛熊辩论...",
            "[Agent] Risk Manager is judging the bull/bear debate...",
        )
    )

    try:
        messages = RISK_PROMPT.invoke(
            {
                "last_price": get_last_price_text(state),
                "bull_report": safe_model_json(state.get("bull_report")),
                "bear_report": safe_model_json(state.get("bear_report")),
                "visual_report": visual_report_to_text(state.get("visual_report")),
                "quant_report": safe_model_json(state.get("quant_report")),
                "news_report": safe_model_json(state.get("news_report")),
                "output_language_instruction": output_language_instruction(),
            }
        )

        result = await risk_llm.ainvoke(messages)
        return {"risk_debate_report": result}

    except Exception as e:
        print(f"[Warning] Risk structured output failed: {e}")
        return {
            "risk_debate_report": RiskDebateReport(
                preferred_action="HOLD",
                confidence_cap=0.5,
                winning_side="balanced",
                key_risk_controls=["风控节点失败，最终决策置信度不得超过 50%"],
                debate_summary=(
                    "风控裁决结构化输出失败，无法可靠比较牛熊双方证据。"
                    "CIO 应优先保持保守，避免高仓位进攻。"
                ),
            )
        }


# ============================================================
# 9. 节点定义：CIO Agent
# ============================================================
async def cio_node(state: AgentState):
    """
    CIO 最终决策节点。

    输入：
    - quant_report
    - visual_report
    - news_report
    - profile_data
    - price_data

    输出：
    - final_report: FinalDecision
    """
    print(
        log_text(
            "[Agent] CIO 正在生成最终决策...",
            "[Agent] CIO is making final decision...",
        )
    )

    last_price = get_last_price_text(state)
    visual_report_str = visual_report_to_text(state.get("visual_report"))
    quant_report_str = safe_model_json(state.get("quant_report"))
    news_report_str = safe_model_json(state.get("news_report"))
    bull_report_str = safe_model_json(state.get("bull_report"))
    bear_report_str = safe_model_json(state.get("bear_report"))
    risk_debate_report_str = safe_model_json(state.get("risk_debate_report"))

    try:
        messages = CIO_PROMPT.invoke(
            {
                "visual_report": visual_report_str,
                "quant_report": quant_report_str,
                "news_report": news_report_str,
                "bull_report": bull_report_str,
                "bear_report": bear_report_str,
                "risk_debate_report": risk_debate_report_str,
                "profile_data": state.get("profile_data", "N/A"),
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "last_price": last_price,
                "output_language_instruction": output_language_instruction(),
            }
        )

        result = await cio_llm.ainvoke(messages)
        return {"final_report": result}

    except Exception as e:
        print(f"[Warning] CIO structured output failed: {e}")

        try:
            current_price = float(last_price)
            if current_price <= 0:
                current_price = 100.0
        except Exception:
            current_price = 100.0

        fallback_decision = FinalDecision(
            action="HOLD",
            confidence=0.3,
            core_logic=(
                "CIO 节点结构化输出失败，系统无法生成可信的完整交易决策。"
                "当前建议保持观望，不进行主动买入或卖出操作。"
                "需要优先检查模型配置、Prompt 字段和 FinalDecision 结构是否一致。"
            ),
            main_force_analysis=(
                "由于最终决策模型输出失败，无法可靠汇总主力资金意图。"
                "如果量化报告中也出现资金流 fetch failed，则说明当前资金面证据不足。"
            ),
            entry_price_range={
                "low": current_price * 0.95,
                "high": current_price * 1.05,
            },
            target_price=current_price * 1.10,
            stop_loss=current_price * 0.90,
            expected_gain_pct=10.0,
            max_loss_pct=-10.0,
            position_size=0.0,
            holding_period="short_term",
            risk_warning="结构化决策生成失败，本次结果不可靠，请检查模型配置和数据源。",
        )

        return {"final_report": fallback_decision}


# ============================================================
# 10. 图构建：Graph Construction
# ============================================================
builder = StateGraph(AgentState)

builder.add_node("fetcher", fetcher_node)
builder.add_node("quant", quant_node)
builder.add_node("visual", visual_node)
builder.add_node("news", news_node)
builder.add_node("bull", bull_node)
builder.add_node("bear", bear_node)
builder.add_node("risk", risk_node)
builder.add_node("cio", cio_node)

builder.set_entry_point("fetcher")

# Fetcher 完成后，进入三个并行分析节点
builder.add_edge("fetcher", "quant")
builder.add_edge("fetcher", "visual")
builder.add_edge("fetcher", "news")

# 三个分析节点全部完成后，进入牛熊风控辩论。
# bull 先提出最强看多/持有理由，bear 再针对牛方反驳，risk 最后裁决。
builder.add_edge(["quant", "visual", "news"], "bull")
builder.add_edge("bull", "bear")
builder.add_edge("bear", "risk")
builder.add_edge("risk", "cio")

# CIO 完成后结束
builder.add_edge("cio", END)

# 导出编译后的应用
app = builder.compile()
