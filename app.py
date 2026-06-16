import streamlit as st
import re

st.set_page_config(page_title="AI 글로벌 퀀트 터미널", layout="wide", page_icon="📈")

import yfinance as yf
import feedparser
import urllib.parse
import google.generativeai as genai
import json
import requests
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# ==========================================
# 세션 초기화
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "안녕하세요! 30년 경력의 수석 애널리스트입니다. 방금 분석하신 종목이나 현재 시장 상황에 대해 무엇이든 물어보세요."}]
if "current_stock" not in st.session_state:
    st.session_state.current_stock = "없음"
if "kis_token" not in st.session_state:
    st.session_state.kis_token = None
if "kis_token_expire" not in st.session_state:
    st.session_state.kis_token_expire = None

US_STOCKS    = {"애플": "AAPL", "엔비디아": "NVDA", "테슬라": "TSLA", "마이크로소프트": "MSFT"}
YF_HEADERS   = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
NAVER_MOBILE = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15"}

# ==========================================
# 1. 한국투자증권 Open API
# ==========================================

def get_kis_base_url():
    is_real = st.secrets.get("KIS_IS_REAL", False)
    return "https://openapi.koreainvestment.com:9443" if is_real else "https://openapivts.koreainvestment.com:29443"

def get_kis_token():
    now = datetime.now()
    if (st.session_state.kis_token and
        st.session_state.kis_token_expire and
        now < st.session_state.kis_token_expire):
        return st.session_state.kis_token
    try:
        app_key    = st.secrets.get("KIS_APP_KEY", "")
        app_secret = st.secrets.get("KIS_APP_SECRET", "")
        if not app_key or not app_secret:
            return None
        res = requests.post(
            f"{get_kis_base_url()}/oauth2/tokenP",
            json={"grant_type": "client_credentials",
                  "appkey": app_key, "appsecret": app_secret},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        token = res.json().get("access_token")
        if token:
            st.session_state.kis_token        = token
            st.session_state.kis_token_expire = now + timedelta(hours=11)
            return token
    except:
        pass
    return None

def kis_headers(tr_id, token):
    return {
        "Content-Type":  "application/json",
        "authorization": f"Bearer {token}",
        "appkey":         st.secrets.get("KIS_APP_KEY", ""),
        "appsecret":      st.secrets.get("KIS_APP_SECRET", ""),
        "tr_id":          tr_id,
        "custtype":       "P",
    }

@st.cache_data(ttl=60)
def get_kis_price(code):
    token = get_kis_token()
    if not token:
        return None
    try:
        res = requests.get(
            f"{get_kis_base_url()}/uapi/domestic-stock/v1/quotations/inquire-price",
            headers=kis_headers("FHKST01010100", token),
            params={"fid_cond_mrkt_div_code": "J", "fid_input_iscd": code},
            timeout=5
        )
        d = res.json().get("output", {})
        if not d:
            return None
        def sf(v):
            try: return float(str(v).replace(",","").strip())
            except: return None
        return {
            "price":    sf(d.get("stck_prpr")),
            "chg_rate": sf(d.get("prdy_ctrt")),
            "volume":   sf(d.get("acml_vol")),
            "high":     sf(d.get("stck_hgpr")),
            "low":      sf(d.get("stck_lwpr")),
            "open":     sf(d.get("stck_oprc")),
            "per":      sf(d.get("per")),
            "pbr":      sf(d.get("pbr")),
            "eps":      sf(d.get("eps")),
            "bps":      sf(d.get("bps")),
            "name":     d.get("hts_kor_isnm", code),
        }
    except:
        return None

@st.cache_data(ttl=60)
def get_kis_volume_rank(market="J"):
    token = get_kis_token()
    if not token:
        return []
    try:
        res = requests.get(
            f"{get_kis_base_url()}/uapi/domestic-stock/v1/quotations/volume-rank",
            headers=kis_headers("FHPST01710000", token),
            params={
                "fid_cond_mrkt_div_code": market,
                "fid_cond_scr_div_code":  "20171",
                "fid_input_iscd":         "0000",
                "fid_div_cls_code":       "0",
                "fid_blng_cls_code":      "0",
                "fid_trgt_cls_code":      "111111111",
                "fid_trgt_exls_cls_code": "000000",
                "fid_input_price_1":      "",
                "fid_input_price_2":      "",
                "fid_vol_cnt":            "",
                "fid_input_date_1":       "",
            },
            timeout=10
        )
        rows = []
        for item in res.json().get("output", []):
            code   = item.get("mksc_shrn_iscd", "")
            suffix = ".KQ" if market == "Q" else ".KS"
            rows.append({
                "Name":    item.get("hts_kor_isnm", ""),
                "Code":    code,
                "Ticker":  f"{code}{suffix}",
                "Market":  "KOSDAQ" if market == "Q" else "KOSPI",
                "Close":   int(str(item.get("stck_prpr",  "0")).replace(",", "")),
                "ChgRate": float(str(item.get("prdy_ctrt","0")).replace(",", "")),
                "Volume":  int(str(item.get("acml_vol",   "0")).replace(",", "")),
                "Marcap":  0,
            })
        return rows
    except:
        return []

# ==========================================
# 2. 네이버 모바일 API
# ==========================================

def search_naver_stock(name):
    """네이버 자동완성 - df_krx에 없는 종목도 검색 가능"""
    try:
        res = requests.get(
            "https://ac.stock.naver.com/ac",
            params={"q": name, "target": "stock,index,fund,futures,option"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=5
        )
        data       = res.json()
        items_list = data.get("items", [])
        if items_list and items_list[0]:
            item       = items_list[0][0]
            code       = item.get("code", "")
            stock_name = item.get("name", name)
            type_code  = str(item.get("typeCode", ""))
            suffix     = ".KQ" if type_code == "12" else ".KS"
            if code:
                return f"{code}{suffix}", stock_name
    except:
        pass
    return None, None

@st.cache_data(ttl=60)
def get_naver_marcap_rank(market="KOSPI", n=100):
    rows     = []
    page     = 1
    per_page = 60
    while len(rows) < n:
        try:
            res = requests.get(
                f"https://m.stock.naver.com/api/stocks/marketValue/{market}",
                params={"page": page, "pageSize": per_page},
                headers=NAVER_MOBILE, timeout=5
            )
            data   = res.json()
            stocks = data.get("stocks", [])
            if not stocks:
                break
            for s in stocks:
                code       = s.get("itemCode", "")
                suffix     = ".KQ" if market == "KOSDAQ" else ".KS"
                marcap_raw = str(s.get("marketValue", "0")).replace(",", "")
                rows.append({
                    "Name":    s.get("stockName", ""),
                    "Code":    code,
                    "Ticker":  f"{code}{suffix}",
                    "Market":  market,
                    "Close":   float(str(s.get("closePrice",               "0")).replace(",", "")),
                    "ChgRate": float(str(s.get("fluctuationsRatio",        "0")).replace(",", "")),
                    "Volume":  float(str(s.get("accumulatedTradingVolume", "0")).replace(",", "")),
                    "Marcap":  float(marcap_raw) * 100_000_000 if marcap_raw else 0,
                })
            total = data.get("totalCount", 0)
            if page * per_page >= total or len(rows) >= n:
                break
            page += 1
        except:
            break
    return rows[:n]

@st.cache_data(ttl=60)
def get_naver_volume_rank(market="KOSPI", n=30):
    rows = []
    try:
        res = requests.get(
            f"https://m.stock.naver.com/api/stocks/quant/{market}",
            params={"page": 1, "pageSize": n},
            headers=NAVER_MOBILE, timeout=5
        )
        for s in res.json().get("stocks", []):
            code   = s.get("itemCode", "")
            suffix = ".KQ" if market == "KOSDAQ" else ".KS"
            rows.append({
                "Name":    s.get("stockName", ""),
                "Code":    code,
                "Ticker":  f"{code}{suffix}",
                "Market":  market,
                "Close":   float(str(s.get("closePrice",               "0")).replace(",", "")),
                "ChgRate": float(str(s.get("fluctuationsRatio",        "0")).replace(",", "")),
                "Volume":  float(str(s.get("accumulatedTradingVolume", "0")).replace(",", "")),
                "Marcap":  float(str(s.get("marketValue",              "0")).replace(",", "")) * 100_000_000,
            })
    except:
        pass
    return rows

@st.cache_data(ttl=60)
def get_naver_change_rank(market="KOSPI", direction="rise", n=30):
    rows = []
    try:
        res = requests.get(
            f"https://m.stock.naver.com/api/stocks/{direction}/{market}",
            params={"page": 1, "pageSize": n},
            headers=NAVER_MOBILE, timeout=5
        )
        for s in res.json().get("stocks", []):
            code   = s.get("itemCode", "")
            suffix = ".KQ" if market == "KOSDAQ" else ".KS"
            rows.append({
                "Name":    s.get("stockName", ""),
                "Code":    code,
                "Ticker":  f"{code}{suffix}",
                "Market":  market,
                "Close":   float(str(s.get("closePrice",               "0")).replace(",", "")),
                "ChgRate": float(str(s.get("fluctuationsRatio",        "0")).replace(",", "")),
                "Volume":  float(str(s.get("accumulatedTradingVolume", "0")).replace(",", "")),
                "Marcap":  float(str(s.get("marketValue",              "0")).replace(",", "")) * 100_000_000,
            })
    except:
        pass
    return rows

# ==========================================
# 3. 전체 KRX 데이터 통합
# ==========================================

@st.cache_data(ttl=60)
def get_krx_full_data():
    rows  = []
    token = get_kis_token()

    rows += get_naver_marcap_rank("KOSPI",  100)
    rows += get_naver_marcap_rank("KOSDAQ", 100)

    if token:
        kis_kospi  = get_kis_volume_rank("J")
        kis_kosdaq = get_kis_volume_rank("Q")
        rows += kis_kospi  if kis_kospi  else get_naver_volume_rank("KOSPI",  30)
        rows += kis_kosdaq if kis_kosdaq else get_naver_volume_rank("KOSDAQ", 30)
    else:
        rows += get_naver_volume_rank("KOSPI",  30)
        rows += get_naver_volume_rank("KOSDAQ", 30)

    rows += get_naver_change_rank("KOSPI",  "rise", 30)
    rows += get_naver_change_rank("KOSPI",  "fall", 30)
    rows += get_naver_change_rank("KOSDAQ", "rise", 30)
    rows += get_naver_change_rank("KOSDAQ", "fall", 30)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df["Close"] > 0].drop_duplicates(subset=["Code"]).reset_index(drop=True)
    return df

def get_ranking_tables(df_krx):
    empty = {k: pd.DataFrame() for k in
             ["kospi_cap","kosdaq_cap","kospi_vol","kosdaq_vol","price_up","price_down"]}
    if df_krx.empty:
        return empty

    kospi  = df_krx[df_krx["Market"] == "KOSPI"].copy()
    kosdaq = df_krx[df_krx["Market"] == "KOSDAQ"].copy()

    def make_table(df, sort_col, ascending=False, n=30):
        if sort_col not in df.columns or df.empty:
            return pd.DataFrame()
        df = df[df[sort_col].notna() & (df[sort_col] > 0)]
        df = df.sort_values(sort_col, ascending=ascending).head(n)
        cols = [c for c in ["Name","Close","ChgRate","Volume","Marcap"] if c in df.columns]
        return df[cols].rename(columns={
            "Name":"종목명","Close":"현재가","ChgRate":"등락률(%)","Volume":"거래량","Marcap":"시가총액"
        }).reset_index(drop=True)

    return {
        "kospi_cap":  make_table(kospi,  "Marcap",  False),
        "kosdaq_cap": make_table(kosdaq, "Marcap",  False),
        "kospi_vol":  make_table(kospi,  "Volume",  False),
        "kosdaq_vol": make_table(kosdaq, "Volume",  False),
        "price_up":   make_table(df_krx, "ChgRate", False),
        "price_down": make_table(df_krx, "ChgRate", True),
    }

def get_ticker_from_name(name, df_krx=None):
    name = name.strip()
    if name in US_STOCKS:
        return US_STOCKS[name], True, name
    if re.match(r'^[A-Za-z]+$', name):
        return name.upper(), True, name.upper()

    # 1순위: df_krx 내 검색
    if df_krx is not None and not df_krx.empty and "Name" in df_krx.columns:
        clean   = name.replace(" ", "").upper()
        exact   = df_krx[df_krx["Name"].str.replace(" ", "").str.upper() == clean]
        if not exact.empty:
            row = exact.iloc[0]
            return row["Ticker"], False, row["Name"]
        partial = df_krx[df_krx["Name"].str.replace(" ", "").str.upper().str.contains(clean, na=False)]
        if not partial.empty:
            row = partial.iloc[0]
            return row["Ticker"], False, row["Name"]

    # 2순위: 네이버 자동완성
    ticker, found_name = search_naver_stock(name)
    if ticker:
        return ticker, False, found_name

    # 3순위: Yahoo Finance
    try:
        res = requests.get(
            "https://query1.finance.yahoo.com/v1/finance/search",
            params={"q": name, "lang": "en-US", "region": "KR", "quotesCount": 5},
            headers=YF_HEADERS, timeout=5
        )
        for q in res.json().get("quotes", []):
            symbol = q.get("symbol", "")
            if symbol.endswith(".KS") or symbol.endswith(".KQ"):
                return symbol, False, q.get("longname") or q.get("shortname") or name
    except:
        pass

    return None, False, name

# ==========================================
# 4. 주가 히스토리
# ==========================================

@st.cache_data(ttl=300)
def get_stock_history(ticker, is_us, period_days=180):
    if is_us:
        try:
            hist = yf.Ticker(ticker).history(period="6mo", auto_adjust=False)
            return hist.dropna(subset=["Close"])
        except:
            return pd.DataFrame()

    raw_code = ticker.split(".")[0]
    token    = get_kis_token()

    if token:
        try:
            end_date   = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=period_days)).strftime("%Y%m%d")
            res = requests.get(
                f"{get_kis_base_url()}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
                headers=kis_headers("FHKST03010100", token),
                params={
                    "fid_cond_mrkt_div_code": "J",
                    "fid_input_iscd":          raw_code,
                    "fid_input_date_1":        start_date,
                    "fid_input_date_2":        end_date,
                    "fid_period_div_code":     "D",
                    "fid_org_adj_prc":         "0",
                },
                timeout=10
            )
            output = res.json().get("output2", [])
            if output:
                df = pd.DataFrame(output)
                df = df.rename(columns={
                    "stck_bsop_date": "Date",
                    "stck_clpr":      "Close",
                    "stck_oprc":      "Open",
                    "stck_hgpr":      "High",
                    "stck_lwpr":      "Low",
                    "acml_vol":       "Volume",
                })
                df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
                df = df.set_index("Date").sort_index()
                for col in ["Close","Open","High","Low","Volume"]:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.replace(",", "")
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                return df.dropna(subset=["Close"])
        except:
            pass

    for suffix in [".KS", ".KQ"]:
        try:
            hist = yf.Ticker(f"{raw_code}{suffix}").history(period="6mo", auto_adjust=False)
            if not hist.empty:
                return hist.dropna(subset=["Close"])
        except:
            pass
    return pd.DataFrame()

# ==========================================
# 5. PER/PBR - KIS 직접 사용
# ==========================================

@st.cache_data(ttl=60)
def get_fundamentals(ticker, is_us_stock, df_krx=None):
    per, pbr = "N/A", "N/A"

    if is_us_stock:
        try:
            info = yf.Ticker(ticker).info
            if info.get("trailingPE"):  per = f"{info['trailingPE']:.2f}배"
            if info.get("priceToBook"): pbr = f"{info['priceToBook']:.2f}배"
        except:
            pass
        return per, pbr

    raw_code   = ticker.split(".")[0]
    price_data = get_kis_price(raw_code)

    if price_data:
        p = price_data.get("per")
        b = price_data.get("pbr")
        try:
            if p and float(p) > 0: per = f"{float(p):.2f}배"
        except: pass
        try:
            if b and float(b) > 0: pbr = f"{float(b):.2f}배"
        except: pass

    if per == "N/A" or pbr == "N/A":
        try:
            res = requests.get(
                f"https://finance.daum.net/api/quotes/A{raw_code}",
                headers={"User-Agent": "Mozilla/5.0",
                         "Referer":    f"https://finance.daum.net/quotes/A{raw_code}",
                         "Accept":     "application/json"},
                timeout=5
            )
            if res.status_code == 200:
                d = res.json()
                if per == "N/A" and d.get("per") and float(d["per"]) > 0:
                    per = f"{float(d['per']):.2f}배"
                if pbr == "N/A" and d.get("pbr") and float(d["pbr"]) > 0:
                    pbr = f"{float(d['pbr']):.2f}배"
        except:
            pass

    return per, pbr

# ==========================================
# 6. 퀀트 팩터 스코어링
# ==========================================

@st.cache_data(ttl=3600)
def get_factor_data(code):
    """KIS API 2번 호출로 4개 팩터 계산 - 순수 데이터 함수 (UI 없음)"""
    token = get_kis_token()
    if not token:
        return None

    result = {"code": code}

    try:
        res = requests.get(
            f"{get_kis_base_url()}/uapi/domestic-stock/v1/quotations/inquire-price",
            headers=kis_headers("FHKST01010100", token),
            params={"fid_cond_mrkt_div_code": "J", "fid_input_iscd": code},
            timeout=5
        )
        d = res.json().get("output", {})
        if not d:
            return None

        def sf(v):
            try: return float(str(v).replace(",","").strip())
            except: return None

        result["price"] = sf(d.get("stck_prpr"))
        result["per"]   = sf(d.get("per"))
        result["pbr"]   = sf(d.get("pbr"))
        result["eps"]   = sf(d.get("eps"))
        result["bps"]   = sf(d.get("bps"))
        # KIS 종목명은 보조용으로만 저장 (나중에 네이버로 덮어씀)
        result["name"]  = d.get("hts_kor_isnm", "")

        if result["eps"] and result["bps"] and result["bps"] > 0:
            result["roe"] = result["eps"] / result["bps"] * 100
        else:
            result["roe"] = None

    except:
        return None

    try:
        end_date   = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
        res2 = requests.get(
            f"{get_kis_base_url()}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice",
            headers=kis_headers("FHKST03010100", token),
            params={
                "fid_cond_mrkt_div_code": "J",
                "fid_input_iscd":          code,
                "fid_input_date_1":        start_date,
                "fid_input_date_2":        end_date,
                "fid_period_div_code":     "D",
                "fid_org_adj_prc":         "0",
            },
            timeout=10
        )
        output = res2.json().get("output2", [])
        if output and len(output) >= 20:
            prices = []
            for row in output:
                p = str(row.get("stck_clpr","0")).replace(",","")
                try: prices.append(float(p))
                except: continue
            prices = list(reversed(prices))
            curr   = prices[-1]

            r1m = (curr / prices[-20]  - 1) * 100 if len(prices) >= 20  else None
            r3m = (curr / prices[-60]  - 1) * 100 if len(prices) >= 60  else None
            r6m = (curr / prices[-120] - 1) * 100 if len(prices) >= 120 else None

            mom_list = [x for x in [r1m, r3m, r6m] if x is not None]
            result["momentum"] = sum(mom_list) / len(mom_list) if mom_list else None

            if len(prices) >= 21:
                daily_rets = [(prices[i]/prices[i-1]-1) for i in range(len(prices)-20, len(prices))]
                result["volatility"] = float(np.std(daily_rets) * 100)
            else:
                result["volatility"] = None
    except:
        result["momentum"]   = None
        result["volatility"] = None

    return result


def score_factors(factor_rows):
    """팩터 점수화 - 순수 계산 함수 (UI 없음, 캐시 없음)"""
    if len(factor_rows) < 10:
        return pd.DataFrame()

    df = pd.DataFrame(factor_rows)

    def percentile_rank(series, ascending=True):
        return series.rank(pct=True, ascending=ascending).fillna(0.5) * 100

    scores = pd.DataFrame(index=df.index)

    per_valid = df["per"].notna() & (df["per"] > 0) & (df["per"] < 500)
    pbr_valid = df["pbr"].notna() & (df["pbr"] > 0) & (df["pbr"] < 50)
    per_score = pd.Series(50.0, index=df.index)
    pbr_score = pd.Series(50.0, index=df.index)
    if per_valid.sum() > 5:
        per_score[per_valid] = percentile_rank(df.loc[per_valid, "per"], ascending=False)
    if pbr_valid.sum() > 5:
        pbr_score[pbr_valid] = percentile_rank(df.loc[pbr_valid, "pbr"], ascending=False)
    scores["value"] = (per_score + pbr_score) / 2

    mom_valid = df["momentum"].notna()
    scores["momentum"] = 50.0
    if mom_valid.sum() > 5:
        scores.loc[mom_valid, "momentum"] = percentile_rank(df.loc[mom_valid, "momentum"], ascending=True)

    roe_valid = df["roe"].notna() & (df["roe"] > -100) & (df["roe"] < 200)
    scores["quality"] = 50.0
    if roe_valid.sum() > 5:
        scores.loc[roe_valid, "quality"] = percentile_rank(df.loc[roe_valid, "roe"], ascending=True)

    vol_valid = df["volatility"].notna() & (df["volatility"] > 0)
    scores["volatility"] = 50.0
    if vol_valid.sum() > 5:
        scores.loc[vol_valid, "volatility"] = percentile_rank(df.loc[vol_valid, "volatility"], ascending=False)

    df["종합점수"]   = (scores["value"]*0.30 + scores["momentum"]*0.30 +
                        scores["quality"]*0.25 + scores["volatility"]*0.15).round(1)
    df["밸류점수"]   = scores["value"].round(1)
    df["모멘텀점수"] = scores["momentum"].round(1)
    df["퀄리티점수"] = scores["quality"].round(1)
    df["저변동점수"] = scores["volatility"].round(1)

    result = df[[
        "name","code","price","per","pbr","roe","momentum",
        "종합점수","밸류점수","모멘텀점수","퀄리티점수","저변동점수"
    ]].rename(columns={
        "name":     "종목명",
        "code":     "코드",
        "price":    "현재가",
        "per":      "PER",
        "pbr":      "PBR",
        "roe":      "ROE(%)",
        "momentum": "6M수익률(%)",
    })

    return result.sort_values("종합점수", ascending=False).reset_index(drop=True)

# ==========================================
# 7. 유틸리티
# ==========================================

@st.cache_data(ttl=60)
def get_major_indices():
    indices = {"KOSPI":"^KS11","KOSDAQ":"^KQ11","NASDAQ":"^IXIC","S&P 500":"^GSPC"}
    res = {}
    for name, ticker in indices.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d").dropna(subset=["Close"])
            if len(hist) >= 2:
                curr = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
                res[name] = {"price":curr,"diff":curr-prev,"pct":((curr-prev)/prev)*100}
            elif len(hist) == 1:
                curr = float(hist["Close"].iloc[-1])
                res[name] = {"price":curr,"diff":0.0,"pct":0.0}
            else:
                res[name] = {"price":0.0,"diff":0.0,"pct":0.0}
        except:
            res[name] = {"price":0.0,"diff":0.0,"pct":0.0}
    return res

@st.cache_data(ttl=3600)
def get_macro_data():
    data = {}
    for key, ticker in {"USD_KRW":"KRW=X","US_10Y":"^TNX","VIX":"^VIX"}.items():
        try:
            hist    = yf.Ticker(ticker).history(period="5d").dropna(subset=["Close"])
            data[key] = round(float(hist["Close"].iloc[-1]),2) if not hist.empty else "N/A"
        except:
            data[key] = "N/A"
    return data

def fetch_google_news(keyword, limit=5):
    try:
        url  = f"https://news.google.com/rss/search?q={urllib.parse.quote(keyword)}&hl=ko&gl=KR&ceid=KR:ko"
        feed = feedparser.parse(url)
        return [e.title for e in feed.entries[:limit]]
    except:
        return []

def get_peer_group(company_name):
    peers_map = {
        "삼성전자":    ["SK하이닉스","삼성전기","DB하이텍"],
        "SK하이닉스":  ["삼성전자","한미반도체","DB하이텍"],
        "현대차":      ["기아","현대모비스","현대로템"],
        "기아":        ["현대차","현대모비스","현대로템"],
        "카카오":      ["NAVER","카카오게임즈","크래프톤"],
        "NAVER":       ["카카오","카카오게임즈","크래프톤"],
        "셀트리온":    ["삼성바이오로직스","한미약품","유한양행"],
        "에코프로비엠": ["포스코퓨처엠","코스모신소재","엘앤에프"],
        "AAPL":        ["MSFT","GOOGL","AMZN"],
        "NVDA":        ["AMD","INTC","AVGO"],
        "TSLA":        ["RIVN","LCID","F"],
    }
    name = company_name.replace(" ","")
    for key, peers in peers_map.items():
        if key in name or name in key:
            return peers[:3]
    return []

@st.cache_data(ttl=3600)
def run_backtest(ticker, is_us, start_years=3):
    try:
        df = get_stock_history(ticker, is_us, 365*start_years)
        if df is None or df.empty: return None
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA60"] = df["Close"].rolling(60).mean()
        delta = df["Close"].diff()
        gain  = delta.where(delta>0,0).rolling(14).mean()
        loss  = (-delta.where(delta<0,0)).rolling(14).mean()
        df["RSI"] = 100-(100/(1+(gain/loss)))
        df["Signal"] = 0
        df.loc[(df["MA20"]>df["MA60"])&(df["RSI"]<50),"Signal"] = 1
        df.loc[(df["MA20"]<df["MA60"])|(df["RSI"]>70),"Signal"] = -1
        df["Position"]            = df["Signal"].replace(-1,0).shift()
        df["Market_Return"]       = df["Close"].pct_change()
        df["Strategy_Return"]     = df["Position"]*df["Market_Return"]
        df["Cumulative_Market"]   = (1+df["Market_Return"]).cumprod()
        df["Cumulative_Strategy"] = (1+df["Strategy_Return"]).cumprod()
        return df
    except:
        return None

@st.cache_data(ttl=600, show_spinner=False)
def get_ai_response(prompt_text, api_key, model_choice, instruction_text, is_json=True):
    try:
        genai.configure(api_key=api_key)
        config   = {"temperature":0.2}
        if is_json: config["response_mime_type"] = "application/json"
        model    = genai.GenerativeModel(model_name=model_choice,
                       system_instruction=instruction_text, generation_config=config)
        response = model.generate_content(prompt_text)
        clean    = response.text.strip()
        ts       = time.strftime("%H:%M:%S")
        if is_json:
            if clean.startswith("```"):
                clean = clean.replace("```json","").replace("```","").strip()
            return json.loads(clean), ts
        return clean, ts
    except Exception as e:
        st.error(f"⚠️ AI 오류: {e}")
        return None, None

def run_quick_analysis(company_name, api_key, model_choice, df_krx):
    ticker, is_us, display_name = get_ticker_from_name(company_name, df_krx)
    if not ticker:
        st.error(f"'{company_name}' 종목을 찾지 못했습니다.")
        return
    st.session_state.current_stock = display_name
    with st.spinner(f"'{display_name}' 퀵 스캔 중..."):
        try:
            hist = get_stock_history(ticker, is_us, 180)
            if hist.empty:
                st.error("주가 데이터를 불러오지 못했습니다.")
                return
            curr  = hist["Close"].iloc[-1]
            hist["MA20"] = hist["Close"].rolling(20).mean()
            hist["MA60"] = hist["Close"].rolling(60).mean()
            ma20  = hist["MA20"].iloc[-1] if len(hist)>=20 else curr
            ma60  = hist["MA60"].iloc[-1] if len(hist)>=60 else curr
            delta = hist["Close"].diff()
            gain  = delta.where(delta>0,0).rolling(14).mean()
            loss  = (-delta.where(delta<0,0)).rolling(14).mean()
            rsi   = (100-(100/(1+(gain/loss)))).iloc[-1] if not gain.isna().all() else 50
            per, pbr  = get_fundamentals(ticker, is_us, df_krx)
            news      = fetch_google_news(display_name, 3)
            news_text = "\n".join([f"- {n}" for n in news]) if news else "최근 뉴스 없음"
            sym       = "$" if is_us else "₩"
            price_fmt = f"{sym}{curr:,.2f}" if is_us else f"{sym}{int(curr):,}"
            result, called_at = get_ai_response(
                f"[{display_name}] 현재가:{price_fmt}, RSI:{rsi:.2f}, MA20:{ma20:.0f}, MA60:{ma60:.0f}, PER:{per}, PBR:{pbr}, 뉴스:{news_text}. 트레이딩 투자 의견.",
                api_key, model_choice,
                "당신은 월스트리트 퀀트 애널리스트. JSON({'action':'BUY/SELL/HOLD','reason':'코멘트'})으로만 응답.",
                is_json=True
            )
            if result is None: return
            col1, col2 = st.columns([2,1])
            with col1:
                st.subheader(f"📈 {display_name} 6개월 추이")
                st.line_chart(hist[["Close","MA20","MA60"]])
            with col2:
                st.subheader("⚡ 퀵 스캔 리포트")
                action = result.get("action","HOLD")
                if action=="BUY":    st.success(f"### 의견: {action} 🟢")
                elif action=="SELL": st.error(f"### 의견: {action} 🔴")
                else:                st.warning(f"### 의견: {action} 🟡")
                st.markdown(f"**RSI(14):** `{rsi:.2f}` | **PER:** `{per}` | **PBR:** `{pbr}`")
                st.info(f"**AI 코멘트:**\n{result.get('reason','')}")
                st.caption(f"기준 시각: {called_at}")
        except Exception as e:
            st.error(f"오류: {e}")

# ==========================================
# 8. 메인 UI
# ==========================================

with st.sidebar:
    st.header("⚙️ 터미널 설정")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini API 연동")
    except:
        st.error("⚠️ GEMINI_API_KEY 설정 필요")
        api_key = ""

    token = get_kis_token()
    if token:
        is_real = st.secrets.get("KIS_IS_REAL", False)
        st.success(f"✅ KIS API ({'실전' if is_real else '모의'}) 실시간 연동")
    else:
        st.error("⚠️ KIS API 연결 실패")

    # 🔥 실제 최신 Gemini 모델명
    model_choice = st.selectbox("AI 모델", ("gemini-3.5-flash", "gemini-3.1-pro"))

    st.divider()
    webhook_url = st.text_input("Webhook URL", placeholder="https://...")
    if st.button("테스트 알림"):
        if webhook_url:
            try:
                requests.post(webhook_url, json={"text":"📈 AI 퀀트 터미널 테스트"}, timeout=3)
                st.success("전송 완료!")
            except:
                st.error("전송 실패")

st.title("🤖 AI 글로벌 퀀트 터미널")

indices_data = get_major_indices()
if indices_data:
    cols = st.columns(len(indices_data))
    for i, (name, data) in enumerate(indices_data.items()):
        cols[i].metric(
            name,
            f"{data['price']:,.2f}" if data["price"] > 0 else "장 마감",
            f"{data['diff']:,.2f} ({data['pct']:+.2f}%)" if data["price"] > 0 else None
        )
st.divider()

try:
    with st.spinner("📡 실시간 시장 데이터 수집 중..."):
        df_krx = get_krx_full_data()
except Exception as e:
    st.warning(f"시장 데이터 수집 오류: {e}")
    df_krx = pd.DataFrame()

tab_main, tab_search, tab_recommend, tab_backtest, tab_chat = st.tabs([
    "🌐 실시간 시장 랭킹","🔍 딥다이브 분석 & Peer 비교",
    "🏆 퀀트 팩터 스크리닝","⏳ 알고리즘 백테스팅","💬 전담 AI 애널리스트 챗봇"
])

# ── 탭1: 실시간 시장 랭킹 ──
with tab_main:
    if df_krx.empty:
        st.warning("⚠️ 실시간 랭킹 데이터를 가져오지 못했습니다.")
    else:
        st.caption(f"✅ 총 {len(df_krx)}개 종목 수집 완료 | 🔄 1분마다 자동 갱신")
        m_data = get_ranking_tables(df_krx)
        t1,t2,t3,t4,t5,t6 = st.tabs([
            "👑 KOSPI 시총","👑 KOSDAQ 시총",
            "🌊 KOSPI 거래량","🌊 KOSDAQ 거래량",
            "🚀 상승률","📉 하락률"
        ])
        events = {}
        with t1: events["kospi_cap"]  = st.dataframe(m_data["kospi_cap"],  use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t1")
        with t2: events["kosdaq_cap"] = st.dataframe(m_data["kosdaq_cap"], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t2")
        with t3: events["kospi_vol"]  = st.dataframe(m_data["kospi_vol"],  use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t3")
        with t4: events["kosdaq_vol"] = st.dataframe(m_data["kosdaq_vol"], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t4")
        with t5: events["price_up"]   = st.dataframe(m_data["price_up"],   use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t5")
        with t6: events["price_down"] = st.dataframe(m_data["price_down"], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t6")

        selected_stock = None
        for cat, event in events.items():
            if len(event.selection.rows) > 0:
                df_ref = m_data[cat]
                if not df_ref.empty and "종목명" in df_ref.columns:
                    selected_stock = df_ref.iloc[event.selection.rows[0]]["종목명"]
                break
        if selected_stock:
            st.divider()
            st.markdown(f"### ⚡ **{selected_stock}** 실시간 퀵 스캔")
            run_quick_analysis(selected_stock, api_key, model_choice, df_krx)

# ── 탭2: 딥다이브 분석 ──
with tab_search:
    st.header("기관 투자자용 심층 종목 분석")
    company_name = st.text_input("검색할 종목명 또는 티커", value="삼성전자")
    if st.button("심층 펀더멘털 분석 시작"):
        ticker, is_us, display_name = get_ticker_from_name(company_name, df_krx)
        if not ticker:
            st.error("종목을 찾을 수 없습니다.")
        else:
            st.session_state.current_stock = display_name
            with st.spinner(f"'{display_name}' 데이터 수집 중..."):
                try:
                    hist = get_stock_history(ticker, is_us, 180)
                    if not hist.empty:
                        curr      = hist["Close"].iloc[-1]
                        per, pbr  = get_fundamentals(ticker, is_us, df_krx)
                        macro     = get_macro_data()
                        news_list = fetch_google_news(display_name, 5)
                        news_text = "\n".join([f"- {n}" for n in news_list]) if news_list else "없음"
                        peers     = get_peer_group(display_name)
                        peer_data = []
                        for p in peers:
                            pt, pu, pn = get_ticker_from_name(p, df_krx)
                            if pt:
                                pp, pb = get_fundamentals(pt, pu, df_krx)
                                peer_data.append({"기업명":pn,"PER":pp,"PBR":pb})
                        sym       = "$" if is_us else "₩"
                        price_fmt = f"{sym}{curr:,.2f}" if is_us else f"{sym}{int(curr):,}"
                        report_text, _ = get_ai_response(
                            f"**[타깃]** {display_name} (현재가:{price_fmt}, PER:{per}, PBR:{pbr})\n**[Peer]** {json.dumps(peer_data,ensure_ascii=False)}\n**[매크로]** 환율:{macro.get('USD_KRW')}, 미10년물:{macro.get('US_10Y')}%, VIX:{macro.get('VIX')}\n**[뉴스]** {news_text}\n\n1.비즈니스모델 2.밸류에이션 3.투자의견 및 목표가를 마크다운으로 작성하라.",
                            api_key, model_choice,
                            "당신은 월스트리트 수석 애널리스트입니다. 마크다운으로 작성하세요.",
                            is_json=False
                        )
                        if report_text:
                            st.success(f"✅ {display_name} 리포트 완료")
                            col1, col2 = st.columns([1,2])
                            with col1:
                                st.metric("PER", per)
                                st.metric("PBR", pbr)
                                if peer_data: st.dataframe(pd.DataFrame(peer_data), hide_index=True)
                                st.line_chart(hist["Close"])
                            with col2:
                                st.subheader("🧠 딥다이브 퀀트 리포트")
                                st.markdown(report_text)
                except Exception as e:
                    st.error(f"분석 오류: {e}")

# ── 탭3: 퀀트 팩터 스크리닝 ──
with tab_recommend:
    st.header("🏆 퀀트 팩터 스크리닝")
    st.markdown("""
    **4개 팩터 가중 합산으로 종목 점수화** (KIS 실시간 데이터 기반)

    | 팩터 | 지표 | 가중치 |
    |---|---|---|
    | 밸류 | PER, PBR | 30% |
    | 모멘텀 | 1M/3M/6M 수익률 | 30% |
    | 퀄리티 | ROE (EPS/BPS) | 25% |
    | 저변동성 | 20일 수익률 표준편차 | 15% |
    """)

    col_a, col_b, col_c = st.columns(3)
    with col_a: market_filter = st.selectbox("시장", ["전체","KOSPI","KOSDAQ"])
    with col_b: top_n         = st.slider("상위 종목 수", 5, 20, 10)
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        run_screen = st.button("🔬 퀀트 스크리닝 시작", use_container_width=True)

    if run_screen:
        if df_krx.empty:
            st.error("시장 데이터가 없습니다. 새로고침 해주세요.")
        elif not get_kis_token():
            st.error("KIS API 연결이 필요합니다.")
        else:
            pool = df_krx.copy()
            if market_filter != "전체":
                pool = pool[pool["Market"] == market_filter]
            pool = pool[pool["Marcap"] > 0].sort_values("Marcap", ascending=False).head(200)

            st.info(f"📊 분석 대상: {len(pool)}개 종목 (시총 상위 기준)")

            codes       = pool["Code"].tolist()
            factor_rows = []
            progress    = st.progress(0, text="팩터 데이터 수집 중...")

            for i, code in enumerate(codes):
                progress.progress(
                    (i+1)/len(codes),
                    text=f"팩터 계산 중... ({i+1}/{len(codes)}) - {code}"
                )
                data = get_factor_data(code)
                if data and data.get("price") and data["price"] > 0:
                    # 🔥 네이버 pool 종목명 항상 우선 적용
                    naver_row = pool[pool["Code"] == code]
                    if not naver_row.empty:
                        data["name"] = naver_row.iloc[0]["Name"]
                    elif not data.get("name") or data["name"] == code:
                        _, found_name = search_naver_stock(code)
                        if found_name:
                            data["name"] = found_name
                    factor_rows.append(data)
                time.sleep(0.05)

            progress.empty()

            result_df = score_factors(factor_rows)

            if result_df.empty:
                st.error("팩터 데이터 수집 실패. 잠시 후 다시 시도해주세요.")
            else:
                top_df = result_df.head(top_n)
                st.success(f"✅ 스크리닝 완료 | 분석 종목: {len(result_df)}개 | 상위 {top_n}종목")

                cols = st.columns(min(5, top_n))
                for idx, row in top_df.head(5).iterrows():
                    with cols[idx]:
                        score = row["종합점수"]
                        color = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
                        st.metric(
                            label=f"{color} {row['종목명']}",
                            value=f"₩{int(row['현재가']):,}" if pd.notna(row["현재가"]) else "N/A",
                            delta=f"종합 {score}점"
                        )
                        if pd.notna(row.get("PER")) and pd.notna(row.get("PBR")):
                            st.caption(
                                f"PER: {row['PER']:.1f} | PBR: {row['PBR']:.1f} | "
                                f"ROE: {row['ROE(%)']:.1f}%"
                            )

                st.divider()

                st.subheader(f"📊 팩터 스코어 상위 {top_n}종목")
                st.dataframe(
                    top_df.style.background_gradient(
                        subset=["종합점수","밸류점수","모멘텀점수","퀄리티점수","저변동점수"],
                        cmap="RdYlGn"
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader("🧠 AI 팩터 분석 해설")
                top5_summary = top_df.head(5)[
                    ["종목명","PER","PBR","ROE(%)","6M수익률(%)","종합점수"]
                ].to_dict("records")
                ai_comment, _ = get_ai_response(
                    f"퀀트 팩터 스크리닝 결과 상위 5종목:\n{json.dumps(top5_summary,ensure_ascii=False)}\n\n각 종목의 팩터 특징과 투자 포인트를 간결하게 설명하라. 투자 권유가 아닌 데이터 기반 팩터 분석임을 명시하라.",
                    api_key, model_choice,
                    "당신은 퀀트 애널리스트입니다. 마크다운으로 작성하세요.",
                    is_json=False
                )
                if ai_comment:
                    st.markdown(ai_comment)

                st.caption("⚠️ 본 결과는 투자 권유가 아니며 과거 팩터 데이터 기반의 참고 정보입니다.")

# ── 탭4: 백테스팅 ──
with tab_backtest:
    st.header("⏳ 퀀트 알고리즘 백테스팅")
    col1, col2 = st.columns([1,3])
    with col1:
        test_input = st.text_input("종목명", value="삼성전자")
        test_years = st.slider("기간(년)", 1, 5, 3)
        run_bt     = st.button("백테스트 시작")
    with col2:
        if run_bt and test_input:
            t_ticker, t_is_us, t_name = get_ticker_from_name(test_input, df_krx)
            if t_ticker:
                st.session_state.current_stock = t_name
                with st.spinner("시뮬레이션 중..."):
                    bt_df = run_backtest(t_ticker, t_is_us, test_years)
                    if bt_df is not None and not bt_df.empty:
                        mr = (bt_df["Cumulative_Market"].iloc[-1]-1)*100
                        sr = (bt_df["Cumulative_Strategy"].iloc[-1]-1)*100
                        st.subheader(f"📊 {t_name} {test_years}년 시뮬레이션")
                        c1, c2 = st.columns(2)
                        c1.metric("Buy & Hold",  f"{mr:.2f}%")
                        c2.metric("전략 수익률", f"{sr:.2f}%", delta=f"{sr-mr:.2f}% 초과수익")
                        st.line_chart(bt_df[["Cumulative_Market","Cumulative_Strategy"]])
                    else:
                        st.error("백테스트 데이터를 불러오지 못했습니다.")
            else:
                st.error("정확한 종목명을 입력하세요.")

# ── 탭5: AI 챗봇 ──
with tab_chat:
    st.header("💬 전담 AI 애널리스트 채팅")
    current_focus = st.session_state.get("current_stock","없음")
    if current_focus != "없음":
        st.info(f"💡 현재 분석 종목: **'{current_focus}'**")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    if prompt := st.chat_input("질문을 입력하세요..."):
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.chat_history.append({"role":"user","content":prompt})
        history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:]])
        resp, _ = get_ai_response(
            f"분석 중 종목: {current_focus}\n[대화]\n{history}\n[질문] {prompt}\n전문 애널리스트 톤으로 답변해.",
            api_key, model_choice, "당신은 퀀트 애널리스트입니다.", is_json=False
        )
        if resp:
            with st.chat_message("assistant"): st.markdown(resp)
            st.session_state.chat_history.append({"role":"assistant","content":resp})
