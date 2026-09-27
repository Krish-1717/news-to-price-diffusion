"""
Financial Named Entity Recognition (Rule-Based NER)
Day 15 â news-to-price-diffusion/nlp/named_entity_recognition.py

Extracts financial entities from news text: companies, tickers, currencies,
commodities, macro indicators, and sentiment-bearing phrases.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

ENTITY_TYPES = {
    "TICKER",       # stock tickers (e.g., AAPL, MSFT)
    "COMPANY",      # company names
    "CURRENCY",     # USD, EUR, GBP, etc.
    "COMMODITY",    # gold, oil, wheat, etc.
    "INDEX",        # S&P 500, NASDAQ, etc.
    "MACRO",        # CPI, GDP, unemployment, etc.
    "SENTIMENT",    # bullish, bearish, upgrade, downgrade
    "AMOUNT",       # dollar amounts, percentages
}


@dataclass
class Entity:
    text: str
    entity_type: str
    start: int
    end: int
    normalized: Optional[str] = None   # canonical form


@dataclass
class NERResult:
    text: str
    entities: List[Entity] = field(default_factory=list)

    def by_type(self, entity_type: str) -> List[Entity]:
        return [e for e in self.entities if e.entity_type == entity_type]

    def unique_tickers(self) -> List[str]:
        return list({e.normalized or e.text for e in self.by_type("TICKER")})

    def unique_companies(self) -> List[str]:
        return list({e.text for e in self.by_type("COMPANY")})


# ---------------------------------------------------------------------------
# Entity dictionaries
# ---------------------------------------------------------------------------

COMMON_TICKERS: Set[str] = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "TSLA", "META", "NVDA", "BRK",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "PYPL",
    "JNJ", "PFE", "MRK", "ABBV", "UNH", "CVS",
    "XOM", "CVX", "COP", "SLB",
    "SPY", "QQQ", "DIA", "IWM", "GLD", "SLV", "USO",
    "BTC", "ETH",
}

COMPANY_ALIASES: Dict[str, str] = {
    "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
    "alphabet": "GOOGL", "amazon": "AMZN", "tesla": "TSLA",
    "meta": "META", "facebook": "META", "nvidia": "NVDA",
    "jpmorgan": "JPM", "jp morgan": "JPM", "bank of america": "BAC",
    "wells fargo": "WFC", "citigroup": "C", "goldman sachs": "GS",
    "morgan stanley": "MS", "visa": "V", "mastercard": "MA",
    "johnson & johnson": "JNJ", "pfizer": "PFE", "merck": "MRK",
    "exxon": "XOM", "exxonmobil": "XOM", "chevron": "CVX",
    "berkshire": "BRK", "berkshire hathaway": "BRK",
}

CURRENCIES: Dict[str, str] = {
    "usd": "USD", "dollar": "USD", "dollars": "USD",
    "eur": "EUR", "euro": "EUR", "euros": "EUR",
    "gbp": "GBP", "pound": "GBP", "sterling": "GBP",
    "jpy": "JPY", "yen": "JPY",
    "cny": "CNY", "yuan": "CNY", "renminbi": "CNY",
    "chf": "CHF", "franc": "CHF",
    "cad": "CAD", "aud": "AUD",
    "bitcoin": "BTC", "btc": "BTC",
    "ethereum": "ETH", "eth": "ETH",
}

COMMODITIES: Dict[str, str] = {
    "gold": "GOLD", "silver": "SILVER", "platinum": "PLATINUM",
    "oil": "OIL", "crude": "OIL", "brent": "BRENT_OIL",
    "natural gas": "NAT_GAS", "lng": "NAT_GAS",
    "wheat": "WHEAT", "corn": "CORN", "soybeans": "SOYBEANS",
    "copper": "COPPER", "aluminum": "ALUMINUM",
    "coal": "COAL", "uranium": "URANIUM",
}

INDICES: Dict[str, str] = {
    "s&p 500": "SPX", "s&p500": "SPX", "sp500": "SPX",
    "nasdaq": "NDX", "nasdaq 100": "NDX",
    "dow jones": "DJIA", "dow": "DJIA",
    "russell 2000": "RTY", "ftse": "FTSE", "dax": "DAX",
    "nikkei": "NKY", "hang seng": "HSI",
    "vix": "VIX", "volatility index": "VIX",
}

MACRO_INDICATORS: Dict[str, str] = {
    "cpi": "CPI", "consumer price index": "CPI",
    "pce": "PCE", "core pce": "CORE_PCE",
    "gdp": "GDP", "gross domestic product": "GDP",
    "unemployment": "UNEMPLOYMENT", "jobs": "NONFARM_PAYROLLS",
    "nonfarm payroll": "NONFARM_PAYROLLS", "payrolls": "NONFARM_PAYROLLS",
    "fed funds": "FED_FUNDS_RATE", "federal funds": "FED_FUNDS_RATE",
    "interest rate": "INTEREST_RATE",
    "yield curve": "YIELD_CURVE", "treasury": "TREASURY",
    "pmi": "PMI", "ism": "ISM",
    "retail sales": "RETAIL_SALES",
    "industrial production": "INDUSTRIAL_PRODUCTION",
    "housing starts": "HOUSING_STARTS",
}

SENTIMENT_WORDS: Dict[str, str] = {
    # Positive
    "upgrade": "POSITIVE", "upgraded": "POSITIVE",
    "outperform": "POSITIVE", "buy": "POSITIVE", "strong buy": "POSITIVE",
    "bullish": "POSITIVE", "rally": "POSITIVE", "surge": "POSITIVE",
    "soar": "POSITIVE", "jump": "POSITIVE", "beat": "POSITIVE",
    "exceed": "POSITIVE", "record high": "POSITIVE",
    "growth": "POSITIVE", "expansion": "POSITIVE",
    # Negative
    "downgrade": "NEGATIVE", "downgraded": "NEGATIVE",
    "underperform": "NEGATIVE", "sell": "NEGATIVE",
    "bearish": "NEGATIVE", "crash": "NEGATIVE", "plunge": "NEGATIVE",
    "tumble": "NEGATIVE", "decline": "NEGATIVE", "fall": "NEGATIVE",
    "miss": "NEGATIVE", "disappoint": "NEGATIVE",
    "recession": "NEGATIVE", "contraction": "NEGATIVE",
    # Neutral
    "hold": "NEUTRAL", "neutral": "NEUTRAL", "maintain": "NEUTRAL",
}


# ---------------------------------------------------------------------------
# NER engine
# ---------------------------------------------------------------------------

class FinancialNER:
    """Rule-based financial named entity recogniser."""

    # Ticker: 1-5 uppercase letters optionally preceded by $ or NYSE:/NASDAQ:
    _TICKER_RE = re.compile(
        r'(?:(?:NYSE|NASDAQ|TSX):\s*)?(?:\$)([A-Z]{1,5})\b'
        r'|(?<![A-Za-z])([A-Z]{2,5})(?!\w)'
    )
    _AMOUNT_RE = re.compile(
        r'\$[\d,]+(?:\.\d+)?(?:\s*(?:million|billion|trillion|M|B|T|K))?'
        r'|\b\d+(?:\.\d+)?%'
        r'|\b\d+(?:\.\d+)?\s*(?:million|billion|trillion)\s*(?:dollars?|shares?)?'
    )

    def __init__(self):
        # Build sorted multi-word phrase matchers (longest first)
        self._multi_word_company = sorted(
            COMPANY_ALIASES.keys(), key=len, reverse=True
        )
        self._multi_word_commodity = sorted(
            COMMODITIES.keys(), key=len, reverse=True
        )
        self._multi_word_index = sorted(
            INDICES.keys(), key=len, reverse=True
        )
        self._multi_word_macro = sorted(
            MACRO_INDICATORS.keys(), key=len, reverse=True
        )
        self._multi_word_currency = sorted(
            CURRENCIES.keys(), key=len, reverse=True
        )
        self._multi_word_sentiment = sorted(
            SENTIMENT_WORDS.keys(), key=len, reverse=True
        )

    def _find_phrase(self, text_lower: str, phrases: List[str],
                     entity_type: str, lookup: Dict[str, str],
                     used: Set[Tuple[int, int]]) -> List[Entity]:
        entities = []
        for phrase in phrases:
            start = 0
            while True:
                idx = text_lower.find(phrase, start)
                if idx == -1:
                    break
                end = idx + len(phrase)
                span = (idx, end)
                # Check word boundary
                before_ok = idx == 0 or not text_lower[idx - 1].isalpha()
                after_ok = end == len(text_lower) or not text_lower[end].isalpha()
                if before_ok and after_ok and span not in used:
                    used.add(span)
                    entities.append(Entity(
                        text=text_lower[idx:end],
                        entity_type=entity_type,
                        start=idx,
                        end=end,
                        normalized=lookup.get(phrase),
                    ))
                start = idx + 1
        return entities

    def extract(self, text: str) -> NERResult:
        result = NERResult(text=text)
        text_lower = text.lower()
        used_spans: Set[Tuple[int, int]] = set()

        # 1. Multi-word company names
        result.entities.extend(
            self._find_phrase(text_lower, self._multi_word_company,
                              "COMPANY", COMPANY_ALIASES, used_spans)
        )
        # Also add TICKER entities for matched companies
        for e in list(result.entities):
            if e.entity_type == "COMPANY" and e.normalized:
                result.entities.append(Entity(
                    text=e.normalized, entity_type="TICKER",
                    start=e.start, end=e.end, normalized=e.normalized
                ))

        # 2. Commodities
        result.entities.extend(
            self._find_phrase(text_lower, self._multi_word_commodity,
                              "COMMODITY", COMMODITIES, used_spans)
        )

        # 3. Indices
        result.entities.extend(
            self._find_phrase(text_lower, self._multi_word_index,
                              "INDEX", INDICES, used_spans)
        )

        # 4. Macro
        result.entities.extend(
            self._find_phrase(text_lower, self._multi_word_macro,
                              "MACRO", MACRO_INDICATORS, used_spans)
        )

        # 5. Currencies
        result.entities.extend(
            self._find_phrase(text_lower, self._multi_word_currency,
                              "CURRENCY", CURRENCIES, used_spans)
        )

        # 6. Sentiment
        result.entities.extend(
            self._find_phrase(text_lower, self._multi_word_sentiment,
                              "SENTIMENT", SENTIMENT_WORDS, used_spans)
        )

        # 7. Explicit tickers ($AAPL or standalone AAPL)
        for m in self._TICKER_RE.finditer(text):
            ticker = (m.group(1) or m.group(2) or "").upper()
            span = (m.start(), m.end())
            if ticker in COMMON_TICKERS and span not in used_spans:
                used_spans.add(span)
                result.entities.append(Entity(
                    text=m.group(0), entity_type="TICKER",
                    start=m.start(), end=m.end(), normalized=ticker
                ))

        # 8. Amounts
        for m in self._AMOUNT_RE.finditer(text):
            span = (m.start(), m.end())
            if span not in used_spans:
                used_spans.add(span)
                result.entities.append(Entity(
                    text=m.group(0), entity_type="AMOUNT",
                    start=m.start(), end=m.end()
                ))

        result.entities.sort(key=lambda e: e.start)
        return result


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------

def extract_from_headlines(headlines: List[str],
                            ner: Optional[FinancialNER] = None
                            ) -> List[NERResult]:
    ner = ner or FinancialNER()
    return [ner.extract(h) for h in headlines]


def aggregate_entity_counts(results: List[NERResult],
                             entity_type: str = "TICKER"
                             ) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in results:
        for e ir r.by_type(entity_type):
            key = e.normalized or e.text.upper()
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ner = FinancialNER()

    headlines = [
        "Apple upgraded to Buy at Goldman Sachs; target $220",
        "NVDA surges 8% as AI chip demand beats expectations",
        "Fed signals rate pause; 10-year Treasury yield falls",
        "Gold rallies as USD weakens on soft CPI data",
        "Tesla downgraded to Underperform amid EV price cuts",
        "S&P 500 hits record high; Nasdaq up 1.2%",
        "JPMorgan sees $5 billion in revenue from investment banking",
        "Oil plunges 4% on recession fears and rising US inventories",
    ]

    for headline in headlines:
        result = ner.extract(headline)
        tickers = result.unique_tickers()
        sentiment = [e.text for e in result.by_type("SENTIMENT")]
        amounts = [e.text for e in result.by_type("AMOUNT")]
        print(f"Headline: {headline[:60]}...")
        if tickers:
            print(f"  Tickers: {tickers}")
        if sentiment:
            print(f"  Sentiment: {sentiment}")
        if amounts:
            print(f"  Amounts: {amounts}")
        print()

    print("Ticker mention counts:")
    results = extract_from_headlines(headlines, ner)
    for ticker, count in aggregate_entity_counts(results, "TICKER").items():
        print(f"  {ticker}: {count}")
