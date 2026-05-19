from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field
from typing import List, Optional
from .tools.push_tool import PushNotificationTool
from .tools.psx_scraper_tool import PSXScraperTool
from .tools.financial_screener_tool import FinancialScreenerTool


# ─── Pydantic Models ────────────────────────────────────────────────────────────

class PSXTrendingCompany(BaseModel):
    """A PSX-listed company that is trending in Pakistan financial news."""
    name: str = Field(description="Full company name")
    ticker: str = Field(description="PSX ticker symbol, e.g. ENGRO, HBL, LUCK")
    sector: str = Field(description="PSX sector classification, e.g. Banking, Cement, Fertiliser")
    catalyst: str = Field(description="Specific reason this stock is trending right now")
    approximate_price_pkr: Optional[float] = Field(
        default=None, description="Current approximate share price in PKR"
    )
    news_source_url: Optional[str] = Field(
        default=None, description="URL of the primary news source"
    )

class PSXTrendingCompanyList(BaseModel):
    """List of 3-5 PSX-listed trending companies."""
    companies: List[PSXTrendingCompany] = Field(
        description="List of PSX-listed trending companies with catalyst context"
    )
    scan_date: str = Field(description="Date when the news scan was performed")


class MacroSectorVerdict(BaseModel):
    """Macro outlook for a single PSX sector."""
    sector: str = Field(description="Sector name")
    verdict: str = Field(description="Tailwind / Neutral / Headwind")
    reasoning: str = Field(description="One-sentence rationale")

class PakistanMacroReport(BaseModel):
    """Comprehensive Pakistan macroeconomic context report."""
    sbp_policy_rate_pct: float = Field(description="Current SBP policy rate in percent")
    sbp_rate_direction: str = Field(description="Cutting / Holding / Hiking")
    pkr_usd_rate: float = Field(description="Current PKR/USD exchange rate")
    pkr_trend_3m: str = Field(description="Appreciation / Depreciation / Stable over 3 months")
    cpi_headline_pct: float = Field(description="Latest CPI headline inflation %")
    fx_reserves_usd_bn: Optional[float] = Field(
        default=None, description="Total foreign exchange reserves in USD billions"
    )
    imf_programme_status: str = Field(description="Current IMF programme status and tranche notes")
    gdp_growth_estimate_pct: Optional[float] = Field(
        default=None, description="Current fiscal year GDP growth estimate %"
    )
    key_macro_themes: List[str] = Field(
        description="2-3 key macro themes the stock picker must weigh"
    )
    sector_verdicts: List[MacroSectorVerdict] = Field(
        description="Tailwind/Neutral/Headwind verdict for each major PSX sector"
    )


class CompanyFundamentals(BaseModel):
    """Detailed fundamental analysis of a single PSX company."""
    ticker: str = Field(description="PSX ticker symbol")
    name: str = Field(description="Company name")
    # Earnings
    latest_eps_pkr: Optional[float] = Field(default=None, description="Latest quarterly EPS in PKR")
    eps_yoy_growth_pct: Optional[float] = Field(default=None, description="EPS YoY growth %")
    annual_revenue_pkr_mn: Optional[float] = Field(
        default=None, description="Latest annual revenue in PKR millions"
    )
    net_profit_margin_pct: Optional[float] = Field(
        default=None, description="Latest net profit margin %"
    )
    # Valuation
    pe_ratio: Optional[float] = Field(default=None, description="Current P/E ratio")
    pe_vs_sector_avg: Optional[str] = Field(
        default=None, description="Premium / Discount / In-Line vs PSX sector P/E average"
    )
    pb_ratio: Optional[float] = Field(default=None, description="Price-to-Book ratio")
    dividend_yield_pct: Optional[float] = Field(
        default=None, description="Current dividend yield %"
    )
    # Balance Sheet
    debt_to_equity: Optional[float] = Field(default=None, description="Debt-to-equity ratio")
    interest_coverage: Optional[float] = Field(
        default=None, description="Interest coverage ratio (EBIT / interest expense)"
    )
    current_ratio: Optional[float] = Field(default=None, description="Current ratio")
    # Quality
    pricing_power: str = Field(description="Strong / Moderate / Weak")
    export_revenue_pct: Optional[float] = Field(
        default=None, description="Export revenue as % of total revenue"
    )
    ownership_type: str = Field(
        description="Family-controlled / Government-owned / Institutional / Mixed"
    )
    broker_consensus: Optional[str] = Field(
        default=None, description="Buy / Hold / Sell from Pakistani brokers"
    )
    broker_price_target_pkr: Optional[float] = Field(
        default=None, description="Broker consensus price target in PKR"
    )
    fundamental_score: float = Field(
        description="Fundamental strength score 1-10 (10 = strongest)"
    )
    investment_thesis: str = Field(
        description="3-sentence investment thesis for this company"
    )

class CompanyFundamentalsList(BaseModel):
    """Fundamental analysis for all shortlisted PSX companies."""
    research_list: List[CompanyFundamentals] = Field(
        description="Fundamental analysis for each company"
    )


class TechnicalAnalysis(BaseModel):
    """Technical analysis for a single PSX company."""
    ticker: str = Field(description="PSX ticker symbol")
    primary_trend: str = Field(description="Uptrend / Downtrend / Sideways on daily chart")
    weekly_trend_confirmation: str = Field(description="Confirms / Contradicts daily trend")
    pct_from_52w_high: Optional[float] = Field(
        default=None, description="% below 52-week high (negative = below)"
    )
    pct_from_52w_low: Optional[float] = Field(
        default=None, description="% above 52-week low"
    )
    above_50d_ma: Optional[bool] = Field(default=None, description="Price above 50-day MA?")
    above_200d_ma: Optional[bool] = Field(default=None, description="Price above 200-day MA?")
    golden_cross_active: Optional[bool] = Field(
        default=None, description="Golden cross active (50 MA > 200 MA)?"
    )
    rsi_14: Optional[float] = Field(default=None, description="14-day RSI value")
    rsi_signal: str = Field(description="Overbought / Oversold / Neutral")
    macd_signal: str = Field(description="Bullish crossover / Bearish crossover / Neutral")
    support_level_pkr: Optional[float] = Field(
        default=None, description="Nearest support level in PKR"
    )
    resistance_level_pkr: Optional[float] = Field(
        default=None, description="Nearest resistance level in PKR"
    )
    volume_trend: str = Field(description="Expanding / Contracting / Neutral")
    kse100_relative_strength: str = Field(
        description="Outperforming / Underperforming / In-line with KSE-100 YTD"
    )
    technical_score: float = Field(
        description="Technical setup score 1-10 (10 = strongest buy signal)"
    )
    suggested_entry_pkr_low: Optional[float] = Field(
        default=None, description="Lower bound of suggested entry zone in PKR"
    )
    suggested_entry_pkr_high: Optional[float] = Field(
        default=None, description="Upper bound of suggested entry zone in PKR"
    )
    stop_loss_pkr: Optional[float] = Field(
        default=None, description="Stop-loss level in PKR"
    )

class TechnicalAnalysisList(BaseModel):
    """Technical analysis for all shortlisted PSX companies."""
    analysis_list: List[TechnicalAnalysis] = Field(
        description="Technical analysis for each company"
    )


class CompanyRiskAssessment(BaseModel):
    """Pakistan-specific risk assessment for a single PSX company."""
    ticker: str = Field(description="PSX ticker symbol")
    currency_risk: str = Field(description="Low / Medium / High / Critical")
    currency_risk_notes: str = Field(description="Brief explanation of currency risk")
    interest_rate_risk: str = Field(description="Low / Medium / High / Critical")
    interest_rate_risk_notes: str = Field(description="Brief explanation of interest rate risk")
    regulatory_governance_risk: str = Field(description="Low / Medium / High / Critical")
    political_risk: str = Field(description="Low / Medium / High / Critical")
    political_risk_notes: str = Field(description="Brief explanation of political risk")
    liquidity_risk: str = Field(description="Low / Medium / High / Critical")
    avg_daily_traded_value_pkr_mn: Optional[float] = Field(
        default=None, description="Average daily traded value in PKR millions"
    )
    sector_specific_risk: str = Field(description="Low / Medium / High / Critical")
    sector_specific_risk_notes: str = Field(description="Key sector risk factors for this company")
    imf_conditionality_risk: str = Field(description="Low / Medium / High / Critical")
    overall_risk_rating: str = Field(description="Low / Medium / High / Critical")
    risk_adjusted_recommendation: str = Field(
        description="Proceed / Proceed with Caution / Avoid"
    )
    risk_score_for_picker: float = Field(
        description="Risk score for the stock picker: 10 = lowest risk, 1 = highest risk"
    )

class RiskAssessmentList(BaseModel):
    """Risk assessments for all shortlisted PSX companies."""
    assessments: List[CompanyRiskAssessment] = Field(
        description="Risk assessment for each company"
    )


class CandidateScore(BaseModel):
    """Weighted score breakdown for a single PSX candidate."""
    ticker: str = Field(description="PSX ticker symbol")
    fundamental_score: float = Field(description="Raw fundamental score (1-10)")
    macro_alignment_score: float = Field(description="Macro tailwind alignment score (1-10)")
    technical_score: float = Field(description="Technical setup score (1-10)")
    risk_score: float = Field(description="Risk score (10=low risk, 1=critical risk)")
    valuation_score: float = Field(description="Valuation vs PSX peers score (1-10)")
    weighted_total: float = Field(description="Weighted total score (out of 10)")

class FinalInvestmentDecision(BaseModel):
    """Final investment decision output."""
    selected_ticker: str = Field(description="Ticker of the selected stock")
    selected_company_name: str = Field(description="Full company name of selected stock")
    current_price_pkr: Optional[float] = Field(
        default=None, description="Current price in PKR"
    )
    price_target_pkr: Optional[float] = Field(
        default=None, description="12-month price target in PKR"
    )
    upside_pct: Optional[float] = Field(
        default=None, description="Upside to price target in %"
    )
    one_line_rationale: str = Field(
        description="One-sentence investment rationale for push notification"
    )
    candidate_scores: List[CandidateScore] = Field(
        description="Scoring breakdown for all candidates"
    )
    rejected_tickers: List[str] = Field(description="Tickers of rejected candidates")
    push_notification_sent: bool = Field(description="Whether push notification was sent")


# ─── Crew Definition ────────────────────────────────────────────────────────────

@CrewBase
class StockPicker():
    """
    Pakistan Stock Exchange (PSX) Stock Picker Crew
    
    A hierarchical multi-agent system that:
    1. Scans Pakistani financial news for KSE-100 trending stocks
    2. Analyzes Pakistan's macroeconomic environment (SBP, PKR, IMF)
    3. Conducts fundamental analysis on shortlisted companies
    4. Performs technical analysis with PSX-specific signals
    5. Assesses Pakistan-specific investment risks
    6. Synthesizes all research into a scored, final investment recommendation
    """

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    # ── Agents ──────────────────────────────────────────────────────────────────

    @agent
    def psx_news_finder(self) -> Agent:
        return Agent(
            config=self.agents_config['psx_news_finder'],
            tools=[SerperDevTool()],
            memory=True,
            verbose=True,
        )

    @agent
    def pakistan_macro_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['pakistan_macro_analyst'],
            tools=[SerperDevTool()],
            verbose=True,
        )

    @agent
    def financial_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['financial_researcher'],
            tools=[
                SerperDevTool(),
                PSXScraperTool(),
                FinancialScreenerTool(),
            ],
            verbose=True,
        )

    @agent
    def technical_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['technical_analyst'],
            tools=[
                SerperDevTool(),
                PSXScraperTool(),
            ],
            verbose=True,
        )

    @agent
    def risk_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['risk_analyst'],
            tools=[SerperDevTool()],
            verbose=True,
        )

    @agent
    def stock_picker(self) -> Agent:
        return Agent(
            config=self.agents_config['stock_picker'],
            tools=[PushNotificationTool()],
            memory=True,
            verbose=True,
        )

    # ── Tasks ───────────────────────────────────────────────────────────────────

    @task
    def find_psx_trending_stocks(self) -> Task:
        return Task(
            config=self.tasks_config['find_psx_trending_stocks'],
            output_pydantic=PSXTrendingCompanyList,
        )

    @task
    def analyze_pakistan_macro(self) -> Task:
        return Task(
            config=self.tasks_config['analyze_pakistan_macro'],
            output_pydantic=PakistanMacroReport,
        )

    @task
    def research_company_financials(self) -> Task:
        return Task(
            config=self.tasks_config['research_company_financials'],
            output_pydantic=CompanyFundamentalsList,
        )

    @task
    def perform_technical_analysis(self) -> Task:
        return Task(
            config=self.tasks_config['perform_technical_analysis'],
            output_pydantic=TechnicalAnalysisList,
        )

    @task
    def assess_investment_risks(self) -> Task:
        return Task(
            config=self.tasks_config['assess_investment_risks'],
            output_pydantic=RiskAssessmentList,
        )

    @task
    def pick_best_psx_stock(self) -> Task:
        return Task(
            config=self.tasks_config['pick_best_psx_stock'],
            # Final task outputs a markdown report; no pydantic to allow rich formatting
        )

    # ── Crew ────────────────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """Creates the PSX StockPicker crew with hierarchical management."""

        manager = Agent(
            config=self.agents_config['manager'],
            allow_delegation=True,
            verbose=True,
        )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=True,
            manager_agent=manager,
            memory=True,
            embedder={
                "provider": "openai",
                "config": {
                    "model": "text-embedding-3-small"
                }
            },
        )