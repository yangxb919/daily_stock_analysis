#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Select hot A-share sector leaders for GitHub Actions STOCK_LIST.

The script intentionally keeps AkShare imports inside runtime functions so tests can
monkeypatch a fake module and CI can import the file without network access.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd


CN_TZ = timezone(timedelta(hours=8))
DEFAULT_BOARD_TYPES = "industry,concept"
DEFAULT_MIN_AMOUNT_YUAN = 50_000_000.0
STYLE_SECTOR_KEYWORDS = (
    "昨日涨停",
    "昨日连板",
    "昨日触板",
    "近期新高",
    "预盈预增",
    "融资融券",
    "参股",
    "转债",
    "ST板块",
    "ST股",
)


@dataclass
class SectorCandidate:
    name: str
    board_type: str
    board_code: str = ""
    pct_chg: float = 0.0
    amount: float = 0.0
    turnover: float = 0.0
    advancers: float = 0.0
    decliners: float = 0.0
    limit_up_count: float = 0.0
    leading_stock: str = ""
    leading_pct_chg: float = 0.0
    score: float = 0.0


@dataclass
class SelectedLeader:
    code: str
    name: str
    sector: str
    board_type: str
    pct_chg: float = 0.0
    amount: float = 0.0
    turnover: float = 0.0
    volume_ratio: float = 0.0
    score: float = 0.0
    reason: str = ""


@dataclass
class SelectionSummary:
    generated_at: str
    stock_list: List[str] = field(default_factory=list)
    selected: List[SelectedLeader] = field(default_factory=list)
    sectors: List[SectorCandidate] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str = ""


def log(message: str) -> None:
    print(message, file=sys.stderr)


def format_error(exc: Exception, max_len: int = 260) -> str:
    text = str(exc).replace("\n", " ").strip()
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text or exc.__class__.__name__


def parse_number(value: Any, default: float = 0.0) -> float:
    """Parse numbers returned by AkShare, including Chinese units and percent strings."""
    if value is None:
        return default
    try:
        if isinstance(value, float) and math.isnan(value):
            return default
    except TypeError:
        pass
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text or text in {"-", "--", "None", "nan", "NaN", "N/A"}:
        return default
    text = text.replace(",", "").replace("％", "%")
    text = text.replace("元", "").replace("￥", "")
    multiplier = 1.0
    # Prefer longer units first.
    if "万亿" in text:
        multiplier = 1e12
        text = text.replace("万亿", "")
    elif "亿" in text:
        multiplier = 1e8
        text = text.replace("亿", "")
    elif "万" in text:
        multiplier = 1e4
        text = text.replace("万", "")
    text = text.replace("%", "").strip()
    try:
        return float(text) * multiplier
    except ValueError:
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
        if match:
            try:
                return float(match.group(0)) * multiplier
            except ValueError:
                return default
    return default


def first_value(row: pd.Series, candidates: Sequence[str], default: Any = None) -> Any:
    for col in candidates:
        if col in row.index:
            value = row.get(col)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                return value
    return default


def normalize_code(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().upper()
    # AkShare sometimes returns 000001, SZ000001, 000001.SZ, sh600519, etc.
    match = re.search(r"(\d{6})", text)
    if not match:
        return ""
    code = match.group(1)
    if code.startswith(("0", "3", "4", "6", "8")):
        return code
    return ""


def normalize_stock_list(raw: str) -> List[str]:
    result: List[str] = []
    seen = set()
    for item in re.split(r"[,，\s]+", raw or ""):
        code = normalize_code(item)
        if code and code not in seen:
            result.append(code)
            seen.add(code)
    return result


def board_type_label(board_type: str) -> str:
    return "行业" if board_type == "industry" else "概念"


def should_skip_sector(name: str, exclude_style_sectors: bool) -> bool:
    if not name:
        return True
    if not exclude_style_sectors:
        return False
    return any(keyword in name for keyword in STYLE_SECTOR_KEYWORDS)


def retry_call(label: str, func: Callable[[], pd.DataFrame], attempts: int = 2, delay_seconds: float = 1.0) -> pd.DataFrame:
    """Retry transient public-market endpoint failures before falling back.

    Eastmoney/AkShare endpoints occasionally close GitHub Actions connections without
    a response. One attempt made the workflow fall back to 600519, so keep the
    selector resilient without failing the whole report.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return func()
        except Exception as exc:  # pragma: no cover - exercised by live endpoints
            last_exc = exc
            if attempt >= attempts:
                break
            log(f"⚠️ {label} 第 {attempt}/{attempts} 次失败，重试中: {format_error(exc)}")
            time.sleep(delay_seconds * attempt)
    assert last_exc is not None
    raise last_exc


def eastmoney_clist(params: Dict[str, str], hosts: Sequence[str]) -> pd.DataFrame:
    """Fetch Eastmoney clist data directly as a fallback to AkShare wrappers."""
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://quote.eastmoney.com/center/boardlist.html",
    }
    errors: List[str] = []
    for host in hosts:
        url = host.rstrip("/") + "/api/qt/clist/get"
        try:
            response = requests.get(url, params=params, headers=headers, timeout=8)
            response.raise_for_status()
            payload = response.json()
            rows = (payload.get("data") or {}).get("diff") or []
            if rows:
                return pd.DataFrame(rows)
            errors.append(f"{host}: empty diff")
        except Exception as exc:  # pragma: no cover - live fallback path
            errors.append(f"{host}: {format_error(exc, max_len=120)}")
    raise RuntimeError("; ".join(errors) or "Eastmoney clist returned no data")


def normalize_eastmoney_sector_table(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "排名": raw.get("f1"),
            "板块名称": raw.get("f14"),
            "板块代码": raw.get("f12"),
            "最新价": raw.get("f2"),
            "涨跌额": raw.get("f4"),
            "涨跌幅": raw.get("f3"),
            "成交额": raw.get("f6"),
            "总市值": raw.get("f20"),
            "换手率": raw.get("f8"),
            "上涨家数": raw.get("f104"),
            "下跌家数": raw.get("f105"),
            "领涨股票": raw.get("f128"),
            "领涨股票-涨跌幅": raw.get("f136"),
        }
    )
    for col in ("最新价", "涨跌额", "涨跌幅", "成交额", "总市值", "换手率", "上涨家数", "下跌家数", "领涨股票-涨跌幅"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["板块名称", "板块代码"], how="any")


def fetch_sector_table_direct(board_type: str) -> pd.DataFrame:
    fs = "m:90 t:2 f:!50" if board_type == "industry" else "m:90 t:3 f:!50"
    params = {
        "pn": "1",
        "pz": "300",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": fs,
        "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,"
        "f23,f24,f25,f26,f22,f33,f11,f62,f128,f136,f115,f152,f124,f107,f104,f105,"
        "f140,f141,f207,f208,f209,f222",
    }
    raw = eastmoney_clist(
        params,
        hosts=(
            "https://17.push2.eastmoney.com",
            "https://79.push2.eastmoney.com",
            "https://push2.eastmoney.com",
        ),
    )
    return normalize_eastmoney_sector_table(raw)


def fetch_sector_table_sina(ak: Any, board_type: str) -> pd.DataFrame:
    indicator = "行业" if board_type == "industry" else "概念"
    raw = retry_call(f"Sina 获取{board_type_label(board_type)}板块", lambda: ak.stock_sector_spot(indicator=indicator))
    if raw is None or raw.empty:
        return pd.DataFrame()
    labels = raw.get("label")
    if labels is None:
        labels = pd.Series([""] * len(raw), index=raw.index)
    df = pd.DataFrame(
        {
            "排名": range(1, len(raw) + 1),
            "板块名称": raw.get("板块"),
            "板块代码": labels.astype(str).map(lambda value: f"sina:{value}" if value else ""),
            "最新价": raw.get("平均价格"),
            "涨跌额": raw.get("涨跌额"),
            "涨跌幅": raw.get("涨跌幅"),
            "成交额": raw.get("总成交额"),
            "换手率": 0.0,
            "上涨家数": 0.0,
            "下跌家数": 0.0,
            "领涨股票": raw.get("股票名称"),
            "领涨股票-涨跌幅": raw.get("个股-涨跌幅"),
        }
    )
    for col in ("最新价", "涨跌额", "涨跌幅", "成交额", "换手率", "上涨家数", "下跌家数", "领涨股票-涨跌幅"):
        df[col] = pd.Series(pd.to_numeric(df[col], errors="coerce"), index=df.index).fillna(0.0)
    return df.dropna(subset=["板块名称", "板块代码"], how="any")


def fetch_sector_table(ak: Any, board_type: str) -> pd.DataFrame:
    def _ak_fetch() -> pd.DataFrame:
        if board_type == "industry":
            return ak.stock_board_industry_name_em()
        if board_type == "concept":
            return ak.stock_board_concept_name_em()
        raise ValueError(f"unsupported board_type: {board_type}")

    try:
        return retry_call(f"AkShare 获取{board_type_label(board_type)}板块", _ak_fetch)
    except Exception as ak_exc:
        log(f"⚠️ AkShare 获取{board_type_label(board_type)}板块失败，尝试 Sina 备用板块源: {format_error(ak_exc)}")
        try:
            return fetch_sector_table_sina(ak, board_type)
        except Exception as sina_exc:
            log(f"⚠️ Sina 获取{board_type_label(board_type)}板块失败，尝试 Eastmoney 直连: {format_error(sina_exc)}")
            return retry_call(
                f"Eastmoney 直连获取{board_type_label(board_type)}板块",
                lambda: fetch_sector_table_direct(board_type),
            )


def normalize_eastmoney_constituent_table(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "序号": raw.get("f1"),
            "代码": raw.get("f12"),
            "名称": raw.get("f14"),
            "最新价": raw.get("f2"),
            "涨跌幅": raw.get("f3"),
            "涨跌额": raw.get("f4"),
            "成交量": raw.get("f5"),
            "成交额": raw.get("f6"),
            "振幅": raw.get("f7"),
            "换手率": raw.get("f8"),
            "量比": raw.get("f10"),
            "最高": raw.get("f15"),
            "最低": raw.get("f16"),
            "今开": raw.get("f17"),
            "昨收": raw.get("f18"),
            "市盈率-动态": raw.get("f9"),
            "市净率": raw.get("f23"),
        }
    )
    for col in ("最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "振幅", "换手率", "量比", "最高", "最低", "今开", "昨收", "市盈率-动态", "市净率"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["代码", "名称"], how="any")


def normalize_sina_constituent_table(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = pd.DataFrame(
        {
            "序号": range(1, len(raw) + 1),
            "代码": raw.get("symbol"),
            "名称": raw.get("name"),
            "最新价": raw.get("trade"),
            "涨跌幅": raw.get("changepercent"),
            "涨跌额": raw.get("pricechange"),
            "成交量": raw.get("volume"),
            "成交额": raw.get("amount"),
            "换手率": raw.get("turnoverratio"),
            "量比": 0.0,
            "最高": raw.get("high"),
            "最低": raw.get("low"),
            "今开": raw.get("open"),
            "昨收": raw.get("settlement"),
            "市盈率-动态": raw.get("per"),
            "市净率": raw.get("pb"),
        }
    )
    for col in ("最新价", "涨跌幅", "涨跌额", "成交量", "成交额", "换手率", "量比", "最高", "最低", "今开", "昨收", "市盈率-动态", "市净率"):
        df[col] = pd.Series(pd.to_numeric(df[col], errors="coerce"), index=df.index).fillna(0.0)
    return df.dropna(subset=["代码", "名称"], how="any")


def fetch_constituent_table_sina(ak: Any, sector: SectorCandidate) -> pd.DataFrame:
    if not sector.board_code.startswith("sina:"):
        raise ValueError(f"{sector.name} 不是 Sina 板块代码")
    label = sector.board_code.split(":", 1)[1]
    raw = retry_call(f"Sina 获取板块成份股 {sector.name}", lambda: ak.stock_sector_detail(sector=label))
    return normalize_sina_constituent_table(raw)


def fetch_constituent_table_direct(sector: SectorCandidate) -> pd.DataFrame:
    if not sector.board_code:
        raise ValueError(f"{sector.name} 缺少板块代码，无法直连获取成份股")
    params = {
        "pn": "1",
        "pz": "200",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": f"b:{sector.board_code} f:!50",
        "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,"
        "f23,f24,f25,f22,f11,f62,f128,f136,f115,f152,f45",
    }
    raw = eastmoney_clist(
        params,
        hosts=("https://29.push2.eastmoney.com", "https://push2.eastmoney.com"),
    )
    return normalize_eastmoney_constituent_table(raw)


def fetch_constituent_table(ak: Any, sector: SectorCandidate) -> pd.DataFrame:
    if sector.board_code.startswith("sina:"):
        return fetch_constituent_table_sina(ak, sector)

    symbol = sector.board_code or sector.name

    def _ak_fetch() -> pd.DataFrame:
        if sector.board_type == "industry":
            return ak.stock_board_industry_cons_em(symbol=symbol)
        if sector.board_type == "concept":
            return ak.stock_board_concept_cons_em(symbol=symbol)
        raise ValueError(f"unsupported board_type: {sector.board_type}")

    try:
        return retry_call(f"AkShare 获取板块成份股 {sector.name}", _ak_fetch)
    except Exception as ak_exc:
        log(f"⚠️ AkShare 获取板块成份股失败 {sector.name}，尝试 Eastmoney 直连: {format_error(ak_exc)}")
        return retry_call(
            f"Eastmoney 直连获取板块成份股 {sector.name}",
            lambda: fetch_constituent_table_direct(sector),
        )


def score_sector(row: pd.Series, board_type: str, exclude_style_sectors: bool) -> Optional[SectorCandidate]:
    name = str(first_value(row, ("板块名称", "概念名称", "行业名称", "名称"), "")).strip()
    if should_skip_sector(name, exclude_style_sectors):
        return None

    pct = parse_number(first_value(row, ("涨跌幅", "涨幅", "涨跌幅%")))
    amount = parse_number(first_value(row, ("成交额", "板块成交额", "总成交额")))
    turnover = parse_number(first_value(row, ("换手率", "换手", "换手率%")))
    advancers = parse_number(first_value(row, ("上涨家数", "上涨数")))
    decliners = parse_number(first_value(row, ("下跌家数", "下跌数")))
    limit_up = parse_number(first_value(row, ("涨停家数", "涨停数")))
    leading = str(first_value(row, ("领涨股票", "领涨股", "领涨名称"), "")).strip()
    leading_pct = parse_number(first_value(row, ("领涨股票-涨跌幅", "领涨股-涨跌幅", "领涨股涨跌幅")))
    board_code = str(first_value(row, ("板块代码", "概念代码", "行业代码", "代码"), "")).strip()

    # Hot-sector score: price strength first, then liquidity/breadth/continuation proxy.
    amount_yi = amount / 1e8 if amount else 0.0
    breadth = advancers - decliners
    score = (
        pct * 3.0
        + min(amount_yi, 500.0) * 0.06
        + turnover * 0.35
        + breadth * 0.08
        + limit_up * 1.5
        + leading_pct * 0.6
    )
    return SectorCandidate(
        name=name,
        board_type=board_type,
        board_code=board_code,
        pct_chg=round(pct, 3),
        amount=round(amount, 2),
        turnover=round(turnover, 3),
        advancers=advancers,
        decliners=decliners,
        limit_up_count=limit_up,
        leading_stock=leading,
        leading_pct_chg=round(leading_pct, 3),
        score=round(score, 4),
    )


def collect_hot_sectors(
    ak: Any,
    board_types: Sequence[str],
    top_sectors: int,
    max_sector_candidates: int,
    exclude_style_sectors: bool,
    errors: List[str],
) -> List[SectorCandidate]:
    candidates: List[SectorCandidate] = []
    for board_type in board_types:
        try:
            table = fetch_sector_table(ak, board_type)
        except Exception as exc:  # pragma: no cover - real network failure path
            errors.append(f"获取{board_type_label(board_type)}板块失败: {format_error(exc)}")
            continue
        if table is None or table.empty:
            errors.append(f"获取{board_type_label(board_type)}板块为空")
            continue
        for _, row in table.iterrows():
            sector = score_sector(row, board_type, exclude_style_sectors)
            if sector is not None:
                candidates.append(sector)

    positive = [s for s in candidates if s.pct_chg > 0]
    pool = positive if len(positive) >= top_sectors else candidates
    pool.sort(key=lambda s: (s.score, s.pct_chg, s.amount), reverse=True)
    return pool[: max(top_sectors, max_sector_candidates)]


def _rank_series(values: pd.Series) -> pd.Series:
    numeric = pd.Series(pd.to_numeric(values, errors="coerce"), index=values.index).fillna(0.0)
    if len(numeric) == 0:
        return numeric
    # Highest value gets percentile close to 1.0.
    return numeric.rank(method="average", ascending=True, pct=True)


def rank_constituents(
    table: pd.DataFrame,
    sector: SectorCandidate,
    leaders_per_sector: int,
    min_amount_yuan: float,
) -> List[SelectedLeader]:
    if table is None or table.empty:
        return []
    rows: List[Dict[str, Any]] = []
    for _, row in table.iterrows():
        code = normalize_code(first_value(row, ("代码", "股票代码", "证券代码", "symbol")))
        name = str(first_value(row, ("名称", "股票名称", "证券简称", "name"), "")).strip()
        if not code or not name:
            continue
        if "ST" in name.upper() or "退" in name:
            continue
        pct = parse_number(first_value(row, ("涨跌幅", "涨幅", "涨跌幅%")))
        amount = parse_number(first_value(row, ("成交额", "成交金额", "amount")))
        turnover = parse_number(first_value(row, ("换手率", "换手", "turnover_rate")))
        volume_ratio = parse_number(first_value(row, ("量比", "volume_ratio")))
        latest_price = parse_number(first_value(row, ("最新价", "最新", "现价", "price")))
        if latest_price <= 0:
            continue
        if pct <= 0:
            continue
        if min_amount_yuan > 0 and amount < min_amount_yuan:
            continue
        rows.append(
            {
                "code": code,
                "name": name,
                "pct_chg": pct,
                "amount": amount,
                "turnover": turnover,
                "volume_ratio": volume_ratio,
            }
        )

    if not rows:
        return []

    df = pd.DataFrame(rows)
    df["pct_rank"] = _rank_series(df["pct_chg"])
    df["amount_rank"] = _rank_series(df["amount"])
    df["turnover_rank"] = _rank_series(df["turnover"])
    df["volume_ratio_rank"] = _rank_series(df["volume_ratio"])
    df["leader_score"] = (
        df["pct_rank"] * 0.38
        + df["amount_rank"] * 0.34
        + df["turnover_rank"] * 0.16
        + df["volume_ratio_rank"] * 0.12
    ) * 100
    df = df.sort_values(
        ["leader_score", "pct_chg", "amount"],
        ascending=[False, False, False],
    )

    selected: List[SelectedLeader] = []
    for _, row in df.head(leaders_per_sector).iterrows():
        reason = (
            f"{board_type_label(sector.board_type)}板块{sector.name}走强，"
            f"个股涨幅{row['pct_chg']:.2f}%、成交额{row['amount'] / 1e8:.2f}亿元"
        )
        selected.append(
            SelectedLeader(
                code=str(row["code"]),
                name=str(row["name"]),
                sector=sector.name,
                board_type=sector.board_type,
                pct_chg=round(float(row["pct_chg"]), 3),
                amount=round(float(row["amount"]), 2),
                turnover=round(float(row["turnover"]), 3),
                volume_ratio=round(float(row["volume_ratio"]), 3),
                score=round(float(row["leader_score"]), 3),
                reason=reason,
            )
        )
    return selected


def select_hot_sector_leaders(
    top_sectors: int = 3,
    leaders_per_sector: int = 2,
    board_types: Sequence[str] = ("industry", "concept"),
    min_amount_yuan: float = DEFAULT_MIN_AMOUNT_YUAN,
    fallback_stocks: Sequence[str] = (),
    max_sector_candidates: Optional[int] = None,
    exclude_style_sectors: bool = True,
) -> SelectionSummary:
    generated_at = datetime.now(CN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    summary = SelectionSummary(generated_at=generated_at)
    max_sector_candidates = max_sector_candidates or max(top_sectors * 4, top_sectors)

    try:
        import akshare as ak  # type: ignore
    except Exception as exc:
        summary.errors.append(f"导入 AkShare 失败: {exc}")
        return apply_fallback(summary, fallback_stocks, "AkShare 不可用")

    sectors = collect_hot_sectors(
        ak=ak,
        board_types=board_types,
        top_sectors=top_sectors,
        max_sector_candidates=max_sector_candidates,
        exclude_style_sectors=exclude_style_sectors,
        errors=summary.errors,
    )
    summary.sectors = sectors[:top_sectors]

    selected: List[SelectedLeader] = []
    seen_codes = set()
    used_sector_count = 0
    for sector in sectors:
        if used_sector_count >= top_sectors:
            break
        try:
            table = fetch_constituent_table(ak, sector)
            leaders = rank_constituents(
                table,
                sector,
                leaders_per_sector=leaders_per_sector,
                min_amount_yuan=min_amount_yuan,
            )
        except Exception as exc:  # pragma: no cover - real network failure path
            summary.errors.append(f"获取板块成份股失败 {sector.name}: {format_error(exc)}")
            continue
        unique_leaders = []
        for leader in leaders:
            if leader.code in seen_codes:
                continue
            unique_leaders.append(leader)
            seen_codes.add(leader.code)
        if not unique_leaders:
            continue
        selected.extend(unique_leaders)
        used_sector_count += 1

    target_count = max(1, top_sectors * leaders_per_sector)
    summary.selected = selected[:target_count]
    summary.stock_list = [item.code for item in summary.selected]
    if not summary.stock_list:
        return apply_fallback(summary, fallback_stocks, "未选出满足流动性/涨幅条件的板块龙头")
    return summary


def apply_fallback(summary: SelectionSummary, fallback_stocks: Sequence[str], reason: str) -> SelectionSummary:
    stocks = [code for code in fallback_stocks if normalize_code(code)]
    if stocks:
        summary.stock_list = stocks
        summary.fallback_used = True
        summary.fallback_reason = reason
        log(f"⚠️ 动态选股失败，使用 fallback STOCK_LIST: {','.join(stocks)} ({reason})")
    else:
        summary.fallback_reason = reason
        log(f"❌ 动态选股失败且没有 fallback: {reason}")
    return summary


def render_markdown(summary: SelectionSummary) -> str:
    lines = [
        "# 热门板块龙头动态选股",
        "",
        f"生成时间：{summary.generated_at}",
        f"最终 STOCK_LIST：{','.join(summary.stock_list) if summary.stock_list else '无'}",
        f"是否使用 fallback：{'是' if summary.fallback_used else '否'}",
    ]
    if summary.fallback_reason:
        lines.append(f"fallback 原因：{summary.fallback_reason}")
    if summary.errors:
        lines.extend(["", "## 运行提示 / 错误", ""])
        lines.extend(f"- {err}" for err in summary.errors)

    lines.extend(["", "## 入选标的", ""])
    if summary.selected:
        lines.append("| 股票 | 板块 | 类型 | 涨跌幅 | 成交额 | 换手率 | 量比 | 评分 |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|")
        for item in summary.selected:
            lines.append(
                f"| {item.name}({item.code}) | {item.sector} | {board_type_label(item.board_type)} | "
                f"{item.pct_chg:.2f}% | {item.amount / 1e8:.2f}亿 | {item.turnover:.2f}% | "
                f"{item.volume_ratio:.2f} | {item.score:.1f} |"
            )
    elif summary.stock_list:
        for code in summary.stock_list:
            lines.append(f"- {code}")
    else:
        lines.append("无")

    lines.extend(["", "## 候选热门板块", ""])
    if summary.sectors:
        lines.append("| 板块 | 类型 | 涨跌幅 | 成交额 | 换手率 | 领涨股 | 板块分 |")
        lines.append("|---|---|---:|---:|---:|---|---:|")
        for sector in summary.sectors:
            lines.append(
                f"| {sector.name} | {board_type_label(sector.board_type)} | {sector.pct_chg:.2f}% | "
                f"{sector.amount / 1e8:.2f}亿 | {sector.turnover:.2f}% | "
                f"{sector.leading_stock or '-'} | {sector.score:.1f} |"
            )
    else:
        lines.append("无")
    lines.extend(
        [
            "",
            "说明：该结果只用于盘后/定时复盘选股，不构成投资建议；个股仍需结合后续报告里的支撑、压力、量能和风险事件判断。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(summary: SelectionSummary, output: Optional[str], json_output: Optional[str]) -> None:
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(summary), encoding="utf-8")
        log(f"✅ 动态选股报告已保存: {path}")
    if json_output:
        path = Path(json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(summary)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"✅ 动态选股 JSON 已保存: {path}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    env = os.environ
    parser = argparse.ArgumentParser(description="Select hot A-share sector leaders for STOCK_LIST")
    parser.add_argument("--top-sectors", type=int, default=int(env.get("HOT_SECTOR_TOP_N", "3") or 3))
    parser.add_argument("--leaders-per-sector", type=int, default=int(env.get("HOT_SECTOR_LEADERS_PER_SECTOR", "2") or 2))
    parser.add_argument("--board-types", default=env.get("HOT_SECTOR_BOARD_TYPES", DEFAULT_BOARD_TYPES))
    parser.add_argument("--min-amount-yuan", type=float, default=float(env.get("HOT_SECTOR_MIN_AMOUNT_YUAN", DEFAULT_MIN_AMOUNT_YUAN) or DEFAULT_MIN_AMOUNT_YUAN))
    parser.add_argument("--fallback-stocks", default=env.get("HOT_SECTOR_FALLBACK_STOCKS") or env.get("STOCK_LIST", ""))
    parser.add_argument("--max-sector-candidates", type=int, default=int(env.get("HOT_SECTOR_MAX_SECTOR_CANDIDATES", "0") or 0))
    parser.add_argument("--no-exclude-style-sectors", action="store_true", help="Do not filter style/statistic pseudo sectors")
    parser.add_argument("--output", default=env.get("HOT_SECTOR_SELECTION_REPORT", ""))
    parser.add_argument("--json-output", default=env.get("HOT_SECTOR_SELECTION_JSON", ""))
    parser.add_argument("--format", choices=("csv", "json", "markdown"), default="csv")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    board_types = [item.strip() for item in args.board_types.split(",") if item.strip()]
    board_types = [item for item in board_types if item in {"industry", "concept"}]
    if not board_types:
        board_types = ["industry", "concept"]
    fallback = normalize_stock_list(args.fallback_stocks)
    summary = select_hot_sector_leaders(
        top_sectors=max(1, args.top_sectors),
        leaders_per_sector=max(1, args.leaders_per_sector),
        board_types=board_types,
        min_amount_yuan=max(0.0, args.min_amount_yuan),
        fallback_stocks=fallback,
        max_sector_candidates=args.max_sector_candidates or None,
        exclude_style_sectors=not args.no_exclude_style_sectors,
    )
    write_outputs(summary, args.output or None, args.json_output or None)

    if args.format == "csv":
        print(",".join(summary.stock_list))
    elif args.format == "json":
        print(json.dumps(asdict(summary), ensure_ascii=False))
    else:
        print(render_markdown(summary))

    # Do not fail the whole workflow when fallback is available.
    return 0 if summary.stock_list else 1


if __name__ == "__main__":
    raise SystemExit(main())
