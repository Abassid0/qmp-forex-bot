"""
Forex Factory news calendar integration.

Fetches high-impact economic events and creates protection/opportunity
windows around them for the trading pairs we run.

Protection modes:
  BLOCK  — no new trades within the blackout window
  PROTECT — tighten SL to breakeven on open trades
  ALLOW  — window passed, normal trading resumes
"""

import json
import logging
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

FF_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

# Map Forex Factory country codes to the currencies they move
CURRENCY_MAP = {
    "USD": ["GBPUSD", "USDJPY"],
    "GBP": ["GBPUSD", "GBPJPY"],
    "JPY": ["USDJPY", "GBPJPY"],
}

# Minutes before/after a high-impact event to block new trades
PRE_NEWS_MINUTES = 30
POST_NEWS_MINUTES = 30

# Minutes before event to move open-trade SL to breakeven
PROTECT_MINUTES = 15


class NewsAction(Enum):
    CLEAR = "CLEAR"
    BLOCK = "BLOCK"
    PROTECT = "PROTECT"


@dataclass
class NewsEvent:
    title: str
    country: str
    time_utc: datetime
    impact: str
    forecast: str
    previous: str
    affected_pairs: list[str]

    @property
    def block_start(self) -> datetime:
        return self.time_utc - timedelta(minutes=PRE_NEWS_MINUTES)

    @property
    def block_end(self) -> datetime:
        return self.time_utc + timedelta(minutes=POST_NEWS_MINUTES)

    @property
    def protect_start(self) -> datetime:
        return self.time_utc - timedelta(minutes=PROTECT_MINUTES)


class NewsFilter:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.events: list[NewsEvent] = []
        self._last_fetch: Optional[datetime] = None
        self._fetch_interval = timedelta(hours=4)

    def refresh(self) -> bool:
        now = datetime.now(timezone.utc)
        if self._last_fetch and (now - self._last_fetch) < self._fetch_interval:
            return True

        try:
            req = urllib.request.Request(FF_WEEK_URL, headers={
                "User-Agent": "QMP-Forex-Bot/1.0"
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw = json.loads(resp.read().decode())

            self.events = []
            for item in raw:
                if item.get("impact") not in ("High", "Medium"):
                    continue

                country = item.get("country", "")
                affected = []
                for sym in self.symbols:
                    if country in CURRENCY_MAP and sym in CURRENCY_MAP[country]:
                        affected.append(sym)

                if not affected:
                    continue

                dt_str = item.get("date", "")
                if not dt_str:
                    continue

                try:
                    event_dt = datetime.fromisoformat(dt_str)
                    event_utc = event_dt.astimezone(timezone.utc)
                except (ValueError, TypeError):
                    continue

                self.events.append(NewsEvent(
                    title=item["title"],
                    country=country,
                    time_utc=event_utc,
                    impact=item["impact"],
                    forecast=item.get("forecast", ""),
                    previous=item.get("previous", ""),
                    affected_pairs=affected,
                ))

            self.events.sort(key=lambda e: e.time_utc)
            self._last_fetch = now
            logger.info(f"News calendar loaded: {len(self.events)} events for {', '.join(self.symbols)}")
            return True

        except Exception as e:
            logger.warning(f"Failed to fetch news calendar: {e}")
            return False

    def check_symbol(self, symbol: str) -> tuple[NewsAction, Optional[NewsEvent]]:
        """Check if a symbol is in a news blackout or protection window."""
        now = datetime.now(timezone.utc)

        for event in self.events:
            if symbol not in event.affected_pairs:
                continue

            if event.block_end < now:
                continue

            if event.block_start <= now <= event.block_end:
                return NewsAction.BLOCK, event

            if event.protect_start <= now < event.block_start:
                return NewsAction.PROTECT, event

        return NewsAction.CLEAR, None

    def get_upcoming(self, hours: int = 24) -> list[NewsEvent]:
        """Get events in the next N hours."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours)
        return [e for e in self.events if now <= e.time_utc <= cutoff]

    def format_upcoming(self, hours: int = 24) -> str:
        """Human-readable list of upcoming events."""
        upcoming = self.get_upcoming(hours)
        if not upcoming:
            return "No high-impact news in the next 24h"

        lines = []
        for e in upcoming:
            local_time = e.time_utc.strftime("%H:%M UTC")
            pairs = ", ".join(e.affected_pairs)
            impact_marker = "!!!" if e.impact == "High" else "!!"
            forecast_str = f" (F: {e.forecast})" if e.forecast else ""
            lines.append(f"  {impact_marker} {local_time} | {e.country} {e.title}{forecast_str} -> {pairs}")
        return "\n".join(lines)

    def get_breakeven_candidates(self) -> list[str]:
        """Return symbols that need SL moved to breakeven due to imminent news."""
        now = datetime.now(timezone.utc)
        candidates = []
        for event in self.events:
            if event.protect_start <= now < event.block_start:
                candidates.extend(event.affected_pairs)
        return list(set(candidates))
