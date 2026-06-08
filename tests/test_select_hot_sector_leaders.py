# -*- coding: utf-8 -*-
"""Tests for hot-sector leader selection helper used by GitHub Actions."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from scripts.select_hot_sector_leaders import (
    normalize_stock_list,
    parse_number,
    render_markdown,
    select_hot_sector_leaders,
)


class HotSectorLeaderSelectionTestCase(unittest.TestCase):
    def test_parse_number_handles_chinese_units_and_percent(self) -> None:
        self.assertEqual(parse_number("12.5%"), 12.5)
        self.assertEqual(parse_number("3.2亿"), 320000000.0)
        self.assertEqual(parse_number("1.5万"), 15000.0)
        self.assertEqual(parse_number("--", default=-1), -1)

    def test_normalize_stock_list_accepts_common_code_formats(self) -> None:
        self.assertEqual(
            normalize_stock_list("SZ000001, 600519.SH,sh688001, invalid, 300750"),
            ["000001", "600519", "688001", "300750"],
        )

    def test_selects_top_leaders_from_fake_akshare_tables(self) -> None:
        fake_ak = SimpleNamespace()
        fake_ak.stock_board_industry_name_em = lambda: pd.DataFrame(
            [
                {
                    "板块名称": "半导体",
                    "涨跌幅": 3.2,
                    "成交额": "220亿",
                    "换手率": 4.5,
                    "上涨家数": 80,
                    "下跌家数": 10,
                    "领涨股票": "芯片龙头",
                    "领涨股票-涨跌幅": 9.1,
                },
                {
                    "板块名称": "银行",
                    "涨跌幅": 0.5,
                    "成交额": "500亿",
                    "换手率": 1.0,
                    "上涨家数": 20,
                    "下跌家数": 18,
                    "领涨股票": "银行A",
                    "领涨股票-涨跌幅": 1.2,
                },
            ]
        )
        fake_ak.stock_board_concept_name_em = lambda: pd.DataFrame(
            [
                {
                    "板块名称": "机器人概念",
                    "涨跌幅": 5.6,
                    "成交额": "180亿",
                    "换手率": 8.0,
                    "上涨家数": 120,
                    "下跌家数": 8,
                    "领涨股票": "机器人龙头",
                    "领涨股票-涨跌幅": 12.0,
                }
            ]
        )

        def industry_cons(symbol: str) -> pd.DataFrame:
            self.assertEqual(symbol, "半导体")
            return pd.DataFrame(
                [
                    {"代码": "688001", "名称": "芯片龙头", "最新价": 50, "涨跌幅": 7.0, "成交额": "12亿", "换手率": 9.0, "量比": 2.4},
                    {"代码": "688002", "名称": "芯片跟随", "最新价": 30, "涨跌幅": 3.0, "成交额": "2亿", "换手率": 3.0, "量比": 1.2},
                ]
            )

        def concept_cons(symbol: str) -> pd.DataFrame:
            self.assertEqual(symbol, "机器人概念")
            return pd.DataFrame(
                [
                    {"代码": "300001", "名称": "机器人龙头", "最新价": 25, "涨跌幅": 10.0, "成交额": "15亿", "换手率": 12.0, "量比": 3.0},
                    {"代码": "300002", "名称": "机器人跟随", "最新价": 18, "涨跌幅": 4.0, "成交额": "3亿", "换手率": 5.0, "量比": 1.5},
                ]
            )

        fake_ak.stock_board_industry_cons_em = industry_cons
        fake_ak.stock_board_concept_cons_em = concept_cons

        with patch.dict(sys.modules, {"akshare": fake_ak}):
            summary = select_hot_sector_leaders(
                top_sectors=2,
                leaders_per_sector=1,
                min_amount_yuan=50_000_000,
                fallback_stocks=["600519"],
            )

        self.assertFalse(summary.fallback_used)
        self.assertEqual(summary.stock_list, ["300001", "688001"])
        self.assertEqual(summary.selected[0].sector, "机器人概念")
        self.assertIn("机器人龙头(300001)", render_markdown(summary))

    def test_fallback_when_no_constituent_passes_filters(self) -> None:
        fake_ak = SimpleNamespace()
        fake_ak.stock_board_industry_name_em = lambda: pd.DataFrame(
            [{"板块名称": "半导体", "涨跌幅": 2.0, "成交额": "100亿"}]
        )
        fake_ak.stock_board_concept_name_em = lambda: pd.DataFrame([])
        fake_ak.stock_board_industry_cons_em = lambda symbol: pd.DataFrame(
            [{"代码": "688001", "名称": "芯片龙头", "最新价": 50, "涨跌幅": -1.0, "成交额": "10亿"}]
        )
        fake_ak.stock_board_concept_cons_em = lambda symbol: pd.DataFrame([])

        with patch.dict(sys.modules, {"akshare": fake_ak}):
            summary = select_hot_sector_leaders(
                top_sectors=1,
                leaders_per_sector=1,
                min_amount_yuan=50_000_000,
                fallback_stocks=["600519", "000333"],
            )

        self.assertTrue(summary.fallback_used)
        self.assertEqual(summary.stock_list, ["600519", "000333"])
        self.assertIn("未选出", summary.fallback_reason)


if __name__ == "__main__":
    unittest.main()
