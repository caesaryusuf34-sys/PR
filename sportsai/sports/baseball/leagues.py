"""League configurations: MLB, KBO, NPB, CPBL.

Each league = a data source + time zone + rating constants + learning policy. The constants are
starting values only: the walk-forward learner re-validates the model configuration as games accrue.
"""
from __future__ import annotations

from .adapter import BaseballAdapter
from .features import BaseballFeatureBuilder, BASE_FEATURES, LINE_FEATURES, TOTAL_FEATURES, TOTAL_LINE_FEATURES, EXTRA_FEATURES
from .sources import MLBSource, KBOSource, NPBSource, CPBLSource


# ------------------------------------------------------------------------------------------- MLB
class MLBFeatures(BaseballFeatureBuilder):
    feature_version = "mlb-f1"
    SPORT, TZ, FIRST_SEASON = "mlb", "America/New_York", 2021
    ELO = dict(K=4.0, HFA=24.0, revert=1 / 3)
    LAM_T, LAM_P = 30.0, 12.0


class MLBAdapter(BaseballAdapter):
    sport, league_name, source_cls, builder_cls = "mlb", "Major League Baseball (USA)", MLBSource, MLBFeatures
    local_tz = "America/New_York"
    first_train_season = 2022
    void_ties = True
    policy_overrides = {"min_new_games": 150, "eval_window_games": 3000, "min_confirm_games": 300, "min_subgroup_n": 60, "high_conf": 0.65}
    team_aliases = {"yanks": "Yankees", "nyy": "Yankees", "nym": "Mets", "bosox": "Red Sox", "sox": "Red Sox", "chisox": "White Sox",
                    "cws": "White Sox", "dodgers": "Los Angeles Dodgers", "lad": "Los Angeles Dodgers", "laa": "Angels",
                    "halos": "Angels", "sf": "Giants", "sd": "Padres", "stl": "Cardinals", "cards": "Cardinals", "kc": "Royals",
                    "tb": "Rays", "a's": "Athletics", "as": "Athletics", "oakland": "Athletics", "phx": "Diamondbacks",
                    "dbacks": "Diamondbacks", "d-backs": "Diamondbacks", "jays": "Blue Jays", "tor": "Blue Jays",
                    "phils": "Phillies", "nats": "Nationals", "cle": "Guardians", "det": "Tigers", "atl": "Braves",
                    "mil": "Brewers", "brew crew": "Brewers", "chc": "Cubs", "sea": "Mariners", "m's": "Mariners"}


# ------------------------------------------------------------------------------------------- KBO
class KBOFeatures(BaseballFeatureBuilder):
    """KBO: starters (with ids) and scores from the official game list; no box-score pitching lines,
    so starter quality comes from the opponent-adjusted run model only."""
    feature_version = "kbo-f1"
    SPORT, TZ, FIRST_SEASON = "kbo", "Asia/Seoul", 2021
    HAS_LINES = False
    ELO = dict(K=5.0, HFA=20.0, revert=1 / 3)
    LAM_T, LAM_P = 20.0, 8.0
    model_features = list(BASE_FEATURES)
    total_features = list(TOTAL_FEATURES)
    candidate_features = model_features + [f for f in EXTRA_FEATURES if not f.startswith(("sp_fip", "sp_kbb", "sp_ra9", "sp_depth", "sp_starts", "pen_"))]


class KBOAdapter(BaseballAdapter):
    sport, league_name, source_cls, builder_cls = "kbo", "KBO League (Korea)", KBOSource, KBOFeatures
    local_tz = "Asia/Seoul"
    has_ties = True
    tie_note = "KBO regular-season games can end tied at the extra-inning limit; P(win) is for a decided game and a tie voids the pick"
    policy_overrides = {"min_new_games": 60, "eval_window_games": 1500, "min_confirm_games": 250, "min_subgroup_n": 40, "high_conf": 0.65}
    team_aliases = {"doosan": "Doosan Bears", "bears": "Doosan Bears", "두산": "Doosan Bears", "lg": "LG Twins", "twins": "LG Twins",
                    "kiwoom": "Kiwoom Heroes", "heroes": "Kiwoom Heroes", "키움": "Kiwoom Heroes", "samsung": "Samsung Lions",
                    "삼성": "Samsung Lions", "kia": "KIA Tigers", "tigers": "KIA Tigers", "kt": "KT Wiz", "wiz": "KT Wiz",
                    "ssg": "SSG Landers", "landers": "SSG Landers", "sk": "SSG Landers", "nc": "NC Dinos", "dinos": "NC Dinos",
                    "lotte": "Lotte Giants", "롯데": "Lotte Giants", "giants": "Lotte Giants", "hanwha": "Hanwha Eagles",
                    "한화": "Hanwha Eagles", "eagles": "Hanwha Eagles", "lions": "Samsung Lions"}


# ------------------------------------------------------------------------------------------- NPB
class NPBFeatures(BaseballFeatureBuilder):
    feature_version = "npb-f1"
    SPORT, TZ, FIRST_SEASON = "npb", "Asia/Tokyo", 2021
    ELO = dict(K=5.0, HFA=18.0, revert=1 / 3)
    LAM_T, LAM_P = 22.0, 10.0


class NPBAdapter(BaseballAdapter):
    sport, league_name, source_cls, builder_cls = "npb", "Nippon Professional Baseball (Japan)", NPBSource, NPBFeatures
    local_tz = "Asia/Tokyo"
    has_ties = True
    tie_note = "NPB games can end tied after 12 innings; P(win) is for a decided game and a tie voids the pick"
    policy_overrides = {"min_new_games": 60, "eval_window_games": 1500, "min_confirm_games": 250, "min_subgroup_n": 40, "high_conf": 0.65}
    team_aliases = {"giants": "Yomiuri Giants", "yomiuri": "Yomiuri Giants", "巨人": "Yomiuri Giants", "hanshin": "Hanshin Tigers",
                    "tigers": "Hanshin Tigers", "阪神": "Hanshin Tigers", "dena": "Yokohama DeNA BayStars", "baystars": "Yokohama DeNA BayStars",
                    "yokohama": "Yokohama DeNA BayStars", "carp": "Hiroshima Toyo Carp", "hiroshima": "Hiroshima Toyo Carp",
                    "広島": "Hiroshima Toyo Carp", "yakult": "Tokyo Yakult Swallows", "swallows": "Tokyo Yakult Swallows",
                    "ヤクルト": "Tokyo Yakult Swallows", "chunichi": "Chunichi Dragons", "dragons": "Chunichi Dragons", "中日": "Chunichi Dragons",
                    "softbank": "Fukuoka SoftBank Hawks", "hawks": "Fukuoka SoftBank Hawks", "ソフトバンク": "Fukuoka SoftBank Hawks",
                    "nippon-ham": "Hokkaido Nippon-Ham Fighters", "nippon ham": "Hokkaido Nippon-Ham Fighters", "ham": "Hokkaido Nippon-Ham Fighters",
                    "fighters": "Hokkaido Nippon-Ham Fighters", "日本ハム": "Hokkaido Nippon-Ham Fighters", "lotte": "Chiba Lotte Marines",
                    "marines": "Chiba Lotte Marines", "ロッテ": "Chiba Lotte Marines", "rakuten": "Tohoku Rakuten Golden Eagles",
                    "eagles": "Tohoku Rakuten Golden Eagles", "楽天": "Tohoku Rakuten Golden Eagles", "orix": "ORIX Buffaloes",
                    "buffaloes": "ORIX Buffaloes", "オリックス": "ORIX Buffaloes", "seibu": "Saitama Seibu Lions", "lions": "Saitama Seibu Lions",
                    "西武": "Saitama Seibu Lions"}


# ------------------------------------------------------------------------------------------- CPBL
class CPBLFeatures(BaseballFeatureBuilder):
    feature_version = "cpbl-f1"
    SPORT, TZ, FIRST_SEASON = "cpbl", "Asia/Taipei", 2021
    ELO = dict(K=6.0, HFA=15.0, revert=1 / 3)
    LAM_T, LAM_P = 15.0, 8.0


class CPBLAdapter(BaseballAdapter):
    sport, league_name, source_cls, builder_cls = "cpbl", "CPBL (Taiwan)", CPBLSource, CPBLFeatures
    local_tz = "Asia/Taipei"
    has_ties = True
    tie_note = "CPBL games can end tied after 12 innings; P(win) is for a decided game and a tie voids the pick"
    policy_overrides = {"min_new_games": 40, "eval_window_games": 900, "min_confirm_games": 150, "min_subgroup_n": 30, "high_conf": 0.65}
    team_aliases = {"brothers": "CTBC Brothers", "ctbc": "CTBC Brothers", "中信": "CTBC Brothers", "中信兄弟": "CTBC Brothers",
                    "lions": "Uni-President Lions", "uni-lions": "Uni-President Lions", "uni lions": "Uni-President Lions",
                    "統一": "Uni-President Lions", "統一獅": "Uni-President Lions", "fubon": "Fubon Guardians", "guardians": "Fubon Guardians",
                    "富邦": "Fubon Guardians", "rakuten": "Rakuten Monkeys", "monkeys": "Rakuten Monkeys", "樂天": "Rakuten Monkeys",
                    "wei chuan": "Wei Chuan Dragons", "weichuan": "Wei Chuan Dragons", "dragons": "Wei Chuan Dragons", "味全": "Wei Chuan Dragons",
                    "tsg": "TSG Hawks", "hawks": "TSG Hawks", "台鋼": "TSG Hawks"}


LEAGUES = {"mlb": MLBAdapter, "kbo": KBOAdapter, "npb": NPBAdapter, "cpbl": CPBLAdapter}
