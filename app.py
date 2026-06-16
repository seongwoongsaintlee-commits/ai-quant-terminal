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
import FinanceDataReader as fdr
from pykrx_openapi import KRXOpenAPI

# ==========================================
# 세션 초기화
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "안녕하세요! 30년 경력의 수석 애널리스트입니다. 방금 분석하신 종목이나 현재 시장 상황에 대해 무엇이든 물어보세요."}]
if "current_stock" not in st.session_state:
    st.session_state.current_stock = "없음"

US_STOCKS = {"애플": "AAPL", "엔비디아": "NVDA", "테슬라": "TSLA", "마이크로소프트": "MSFT"}

# ==========================================
# KRX OpenAPI 클라이언트 초기화
# ==========================================
@st.cache_resource
def get_krx_client():
    try:
        api_key = st.secrets["KRX_API_KEY"]
        return KRXOpenAPI(api_key=api_key)
    except Exception as e:
        st.error(f"KRX OpenAPI 키 로딩 실패: {e}")
        return None

# ==========================================
# 1. 핵심 데이터 수집 함수 (KRX OpenAPI 기반)
# ==========================================

def get_today_str():
    """오늘 날짜 문자열 반환 (YYYYMMDD)"""
    return datetime.now().strftime('%Y%m%d')

def get_recent_bizday_str():
    """가장 최근 영업일 (토/일이면 전주 금요일)"""
    d = datetime.now()
    while d.weekday() >= 5:  # 토=5, 일=6
        d -= timedelta(days=1)
    return d.strftime('%Y%m%d')

@st.cache_data(ttl=3600)
def get_krx_full_data():
    """
    KRX OpenAPI를 라이브러리 없이 직접 HTTP 호출
    pykrx-openapi 라이브러리 불필요
    """
    try:
        krx_api_key = st.secrets.get("KRX_API_KEY", "")
        today = datetime.now().strftime('%Y%m%d')

        headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json',
        }

        all_dfs = []

        for market_nm in ["KOSPI", "KOSDAQ"]:
            # KRX OpenAPI: 주식 시세 (종가, 거래량, 등락률, 시가총액)
            url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
            params = {
                "bld":        "dbms/MDC/STAT/standard/MDCSTAT01501",
                "mktId":      "STK" if market_nm == "KOSPI" else "KSQ",
                "trdDd":      today,
                "share":      "1",
                "money":      "1",
                "csvxls_isNo":"false",
            }

            try:
                res = requests.post(url, data=params, headers=headers, timeout=10)
                data = res.json()
                rows = data.get("OutBlock_1", [])

                if rows:
                    df = pd.DataFrame(rows)
                    # KRX 응답 컬럼명 매핑
                    col_map = {
                        'ISU_ABBRV':  'Name',     # 종목명
                        'ISU_CD':     'Code',     # 종목코드
                        'TDD_CLSPRC': 'Close',    # 종가
                        'ACC_TRDVOL': 'Volume',   # 거래량
                        'FLUC_RT':    'ChgRate',  # 등락률
                        'MKTCAP':     'Marcap',   # 시가총액
                    }
                    df.rename(columns=col_map, inplace=True)
                    df['Market'] = market_nm

                    # 숫자 변환 (KRX는 콤마 포함 문자열)
                    for col in ['Close', 'Volume', 'ChgRate', 'Marcap']:
                        if col in df.columns:
                            df[col] = df[col].astype(str).str.replace(',','').str.replace('%','')
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    all_dfs.append(df)
            except Exception as e:
                st.warning(f"{market_nm} 시세 로딩 실패: {e}")
                continue

        if not all_dfs:
            raise ValueError("KRX 시세 데이터 없음")

        df_all = pd.concat(all_dfs, ignore_index=True)

        # PER/PBR: KRX OpenAPI 직접 호출
        per_dfs = []
        for market_nm in ["KOSPI", "KOSDAQ"]:
            try:
                params_fund = {
                    "bld":        "dbms/MDC/STAT/standard/MDCSTAT03501",
                    "mktId":      "STK" if market_nm == "KOSPI" else "KSQ",
                    "trdDd":      today,
                    "csvxls_isNo":"false",
                }
                res_fund = requests.post(url, data=params_fund, headers=headers, timeout=10)
                data_fund = res_fund.json()
                rows_fund = data_fund.get("OutBlock_1", [])
                if rows_fund:
                    df_fund = pd.DataFrame(rows_fund)
                    fund_col_map = {
                        'ISU_CD': 'Code',
                        'PER':    'PER',
                        'PBR':    'PBR',
                    }
                    df_fund.rename(columns=fund_col_map, inplace=True)
                    for col in ['PER', 'PBR']:
                        if col in df_fund.columns:
                            df_fund[col] = df_fund[col].astype(str).str.replace(',','')
                            df_fund[col] = pd.to_numeric(df_fund[col], errors='coerce')
                    per_dfs.append(df_fund[['Code','PER','PBR']])
            except:
                continue

        if per_dfs:
            df_per = pd.concat(per_dfs, ignore_index=True)
            df_all = df_all.merge(df_per, on='Code', how='left')
        else:
            df_all['PER'] = np.nan
            df_all['PBR'] = np.nan

        keep = [c for c in ['Name','Code','Market','Close','ChgRate','Volume','Marcap','PER','PBR'] if c in df_all.columns]
        return df_all[keep].dropna(subset=['Name','Code']).reset_index(drop=True)

    except Exception as e:
        st.warning(f"KRX 직접 호출 실패 ({e}), FDR로 전환합니다.")
        return get_krx_data_fdr_fallback()


@st.cache_data(ttl=3600)
def get_krx_data_fdr_fallback():
    """KRX OpenAPI 실패 시 FDR로 대체"""
    try:
        df_kospi  = fdr.StockListing('KOSPI')
        df_kosdaq = fdr.StockListing('KOSDAQ')
        df_kospi['Market']  = 'KOSPI'
        df_kosdaq['Market'] = 'KOSDAQ'
        df = pd.concat([df_kospi, df_kosdaq], ignore_index=True)
        df.rename(columns={'ChagesRatio': 'ChgRate'}, inplace=True)
        df['PER'] = np.nan
        df['PBR'] = np.nan
        keep = [c for c in ['Name','Code','Market','Close','ChgRate','Volume','Marcap','PER','PBR'] if c in df.columns]
        df = df[keep].dropna(subset=['Name','Code'])
        for col in ['Close','ChgRate','Volume','Marcap']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except:
        return pd.DataFrame()

# 🔥 PER/PBR 개별 종목 조회 (KRX OpenAPI → Daum API 순서로 폴백)
@st.cache_data(ttl=3600)
def get_per_pbr(raw_code, _krx_client):
    per, pbr = "N/A", "N/A"

    # 플랜 A: KRX OpenAPI 개별 조회
    if _krx_client is not None:
        try:
            today = get_recent_bizday_str()
            df = _krx_client.get_stock_per_pbr(bas_dd=today, isu_cd=raw_code)
            if not df.empty:
                p = pd.to_numeric(df.iloc[0].get('PER',''), errors='coerce')
                b = pd.to_numeric(df.iloc[0].get('PBR',''), errors='coerce')
                if pd.notna(p) and p > 0: per = f"{p:.2f}배"
                if pd.notna(b) and b > 0: pbr = f"{b:.2f}배"
        except: pass

    # 플랜 B: Daum Finance API (네이버와 동일한 TTM 기준)
    if per == "N/A" or pbr == "N/A":
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0',
                'Referer': f'https://finance.daum.net/quotes/A{raw_code}',
                'Accept': 'application/json',
            }
            res = requests.get(
                f"https://finance.daum.net/api/quotes/A{raw_code}",
                headers=headers, timeout=5
            )
            if res.status_code == 200:
                data = res.json()
                p = data.get('per')
                b = data.get('pbr')
                if p and float(p) > 0: per = f"{float(p):.2f}배"
                if b and float(b) > 0: pbr = f"{float(b):.2f}배"
        except: pass

    return per, pbr

def get_fundamentals(ticker, is_us_stock, krx_client):
    if is_us_stock:
        try:
            info = yf.Ticker(ticker).info
            per = f"{info['trailingPE']:.2f}배" if info.get('trailingPE') else "N/A"
            pbr = f"{info['priceToBook']:.2f}배" if info.get('priceToBook') else "N/A"
            return per, pbr
        except:
            return "N/A", "N/A"
    else:
        raw_code = ticker.split('.')[0]
        return get_per_pbr(raw_code, krx_client)

def get_ticker_from_name(name, df_krx):
    name = name.strip()
    if name in US_STOCKS: return US_STOCKS[name], True, name
    if re.match(r'^[A-Za-z]+$', name): return name.upper(), True, name.upper()

    if not df_krx.empty and 'Name' in df_krx.columns:
        clean_input = name.replace(" ","").upper()
        exact = df_krx[df_krx['Name'].str.replace(" ","").str.upper() == clean_input]
        if not exact.empty:
            row = exact.iloc[0]
            suffix = '.KQ' if row.get('Market') == 'KOSDAQ' else '.KS'
            return f"{row['Code']}{suffix}", False, row['Name']
        partial = df_krx[df_krx['Name'].str.replace(" ","").str.upper().str.contains(clean_input, na=False)]
        if not partial.empty:
            row = partial.iloc[0]
            suffix = '.KQ' if row.get('Market') == 'KOSDAQ' else '.KS'
            return f"{row['Code']}{suffix}", False, row['Name']

    try:
        encoded = urllib.parse.quote(name.encode('euc-kr'))
        url = f"https://ac.finance.naver.com/ac?q={encoded}&q_enc=euc-kr&st=111&r_format=json&r_enc=utf-8"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=3)
        items = res.json().get('items', [[]])[0]
        if items:
            code, market = items[0][1], items[0][2]
            suffix = '.KQ' if "KOSDAQ" in market or "코스닥" in market else '.KS'
            return f"{code}{suffix}", False, items[0][0]
    except: pass

    return None, False, name

def get_ranking_tables(df_krx):
    if df_krx.empty:
        return {k: pd.DataFrame() for k in ['kospi_cap','kosdaq_cap','kospi_vol','kosdaq_vol','price_up','price_down']}
    kospi  = df_krx[df_krx['Market'] == 'KOSPI'].copy()
    kosdaq = df_krx[df_krx['Market'] == 'KOSDAQ'].copy()

    def make_table(df, sort_col, ascending=False, n=30):
        if sort_col not in df.columns or df.empty: return pd.DataFrame()
        df = df[df[sort_col].notna()].sort_values(sort_col, ascending=ascending).head(n)
        cols = [c for c in ['Name','Close','ChgRate','Volume','Marcap','PER','PBR'] if c in df.columns]
        rename = {'Name':'종목명','Close':'현재가','ChgRate':'등락률(%)','Volume':'거래량','Marcap':'시가총액','PER':'PER','PBR':'PBR'}
        return df[cols].rename(columns=rename).reset_index(drop=True)

    return {
        'kospi_cap':  make_table(kospi,  'Marcap',  ascending=False),
        'kosdaq_cap': make_table(kosdaq, 'Marcap',  ascending=False),
        'kospi_vol':  make_table(kospi,  'Volume',  ascending=False),
        'kosdaq_vol': make_table(kosdaq, 'Volume',  ascending=False),
        'price_up':   make_table(df_krx, 'ChgRate', ascending=False),
        'price_down': make_table(df_krx, 'ChgRate', ascending=True),
    }

@st.cache_data(ttl=300)
def get_major_indices():
    indices = {"KOSPI": "^KS11", "KOSDAQ": "^KQ11", "NASDAQ": "^IXIC", "S&P 500": "^GSPC"}
    res = {}
    for name, ticker in indices.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) >= 2:
                curr = float(hist['Close'].iloc[-1])
                prev = float(hist['Close'].iloc[-2])
                res[name] = {"price": curr, "diff": curr-prev, "pct": ((curr-prev)/prev)*100}
            else: res[name] = {"price": 0.0, "diff": 0.0, "pct": 0.0}
        except: res[name] = {"price": 0.0, "diff": 0.0, "pct": 0.0}
    return res

@st.cache_data(ttl=3600)
def get_macro_data():
    macros = {"USD_KRW": "KRW=X", "US_10Y": "^TNX", "VIX": "^VIX"}
    data = {}
    for key, ticker in macros.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            data[key] = round(hist['Close'].iloc[-1], 2) if not hist.empty else "N/A"
        except: data[key] = "N/A"
    return data

def fetch_google_news(keyword, limit=5):
    try:
        encoded = urllib.parse.quote(keyword)
        url = f"https://news.google.com/rss/search?q={encoded}&hl=ko&gl=KR&ceid=KR:ko"
        feed = feedparser.parse(url)
        return [entry.title for entry in feed.entries[:limit]]
    except: return []

def get_stock_history_kr(raw_code, period_days=180):
    """FDR로 한국 주식 실제 거래가 (수정주가 X)"""
    try:
        end   = datetime.now().strftime('%Y-%m-%d')
        start = (datetime.now() - timedelta(days=period_days)).strftime('%Y-%m-%d')
        df = fdr.DataReader(raw_code, start, end)
        if not df.empty and 'Close' in df.columns:
            return df
    except: pass
    # 폴백: Yahoo
    for suffix in ['.KS', '.KQ']:
        try:
            hist = yf.Ticker(f"{raw_code}{suffix}").history(period="6mo", auto_adjust=False)
            if not hist.empty: return hist
        except: pass
    return pd.DataFrame()

@st.cache_data(ttl=1800)
def get_trending_stocks_with_news(df_krx):
    if df_krx.empty or 'Volume' not in df_krx.columns: return []
    pool = df_krx[df_krx['Volume'].notna()].sort_values('Volume', ascending=False).head(20)
    candidates = []
    for _, row in pool.iterrows():
        name = str(row.get('Name',''))
        code = str(row.get('Code',''))
        try:
            hist = get_stock_history_kr(code, 5)
            if not hist.empty:
                candidates.append({"종목명": name, "현재가": int(hist['Close'].iloc[-1]), "최신뉴스": fetch_google_news(name, limit=3)})
        except: pass
    return candidates

@st.cache_data(ttl=1800)
def get_hidden_gem_stocks(df_krx):
    if df_krx.empty or 'Marcap' not in df_krx.columns: return []
    pool = df_krx[df_krx['Marcap'].notna()].sort_values('Marcap', ascending=False).head(200)
    if 'Volume' in pool.columns:
        vol_median = pool['Volume'].median()
        filtered = pool[pool['Volume'] < vol_median]
        if not filtered.empty: pool = filtered.head(60)
    sampled = pool.sample(n=min(15, len(pool))) if not pool.empty else pool
    candidates = []
    for _, row in sampled.iterrows():
        name = str(row.get('Name',''))
        code = str(row.get('Code',''))
        try:
            hist = get_stock_history_kr(code, 5)
            if not hist.empty:
                candidates.append({"종목명": name, "현재가": int(hist['Close'].iloc[-1]), "최신뉴스": fetch_google_news(name, limit=3)})
        except: pass
    return candidates

@st.cache_data(ttl=1800)
def get_us_trending_stocks_with_news():
    pool = ['AAPL','NVDA','MSFT','TSLA','AMZN','GOOGL','META','AMD','NFLX','TSM','SMCI','PLTR','AVGO']
    candidates = []
    for ticker in pool:
        try:
            hist = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
            if not hist.empty:
                candidates.append({"종목명": ticker, "현재가": round(float(hist['Close'].iloc[-1]),2), "최신뉴스": fetch_google_news(f"{ticker} 주식", limit=3)})
        except: pass
    return candidates

@st.cache_data(ttl=1800)
def get_us_hidden_gems_with_news():
    pool = ['PYPL','DIS','PFE','NKE','SBUX','BA','INTC','C','WFC','T','VZ','O','QCOM','TXN','CVX']
    candidates = []
    for ticker in pool:
        try:
            hist = yf.Ticker(ticker).history(period="5d", auto_adjust=False)
            if not hist.empty:
                candidates.append({"종목명": ticker, "현재가": round(float(hist['Close'].iloc[-1]),2), "최신뉴스": fetch_google_news(f"{ticker} stock", limit=3)})
        except: pass
    return candidates

def get_peer_group(company_name):
    name = company_name.replace(" ","")
    if "엔비티" in name or "노바렉스" in name or "서흥" in name: peers = ["노바렉스","서흥","콜마비앤에이치"]
    elif "콜마" in name or "코스맥스" in name or "코스메카" in name: peers = ["한국콜마","코스메카코리아","씨앤씨인터내셔널"]
    elif "삼성전자" in name or "SK하이닉스" in name: peers = ["SK하이닉스","한미반도체","DB하이텍"]
    elif "AAPL" in name or "MSFT" in name or "GOOGL" in name: peers = ["MSFT","GOOGL","AMZN"]
    elif "TSLA" in name: peers = ["RIVN","LCID","F"]
    else: peers = []
    return [p for p in peers if p != company_name][:3]

@st.cache_data(ttl=3600)
def run_backtest(raw_code, is_us_stock, start_years=3):
    try:
        df = yf.Ticker(raw_code).history(period=f"{start_years}y", auto_adjust=False) if is_us_stock else get_stock_history_kr(raw_code, 365*start_years)
        if df is None or df.empty: return None
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        delta = df['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['Signal'] = 0
        df.loc[(df['MA20'] > df['MA60']) & (df['RSI'] < 50), 'Signal'] = 1
        df.loc[(df['MA20'] < df['MA60']) | (df['RSI'] > 70), 'Signal'] = -1
        df['Position'] = df['Signal'].replace(-1, 0).shift()
        df['Market_Return']    = df['Close'].pct_change()
        df['Strategy_Return']  = df['Position'] * df['Market_Return']
        df['Cumulative_Market']   = (1 + df['Market_Return']).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
        return df
    except: return None

@st.cache_data(ttl=600, show_spinner=False)
def get_ai_response(prompt_text, api_key, model_choice, instruction_text, is_json=True):
    try:
        genai.configure(api_key=api_key)
        config = {"temperature": 0.2}
        if is_json: config["response_mime_type"] = "application/json"
        model = genai.GenerativeModel(model_name=model_choice, system_instruction=instruction_text, generation_config=config)
        response   = model.generate_content(prompt_text)
        clean_text = response.text.strip()
        called_at  = time.strftime("%H:%M:%S")
        if is_json:
            ticks = "`" * 3
            if clean_text.startswith(ticks):
                clean_text = clean_text.replace(ticks+"json","").replace(ticks,"").strip()
            return json.loads(clean_text), called_at
        else: return clean_text, called_at
    except Exception as e:
        st.error(f"⚠️ AI 연산 오류: {e}")
        return None, None

def run_quick_analysis(company_name, api_key, model_choice, df_krx, krx_client):
    ticker, is_us_stock, display_name = get_ticker_from_name(company_name, df_krx)
    if not ticker:
        st.error(f"'{company_name}' 종목을 찾지 못했습니다.")
        return
    st.session_state.current_stock = display_name
    with st.spinner(f"'{display_name}' 실시간 퀵 스캔 중..."):
        try:
            raw_code = ticker.split('.')[0]
            hist = get_stock_history_kr(raw_code, 180) if not is_us_stock else yf.Ticker(ticker).history(period="6mo", auto_adjust=False)
            if hist.empty:
                st.error("주가 데이터를 불러오지 못했습니다.")
                return

            current_price = hist['Close'].iloc[-1]
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA60'] = hist['Close'].rolling(window=60).mean()
            ma20 = hist['MA20'].iloc[-1] if len(hist) >= 20 else current_price
            ma60 = hist['MA60'].iloc[-1] if len(hist) >= 60 else current_price
            delta = hist['Close'].diff()
            gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            current_rsi = (100 - (100/(1+(gain/loss)))).iloc[-1] if not gain.isna().all() else 50

            per, pbr    = get_fundamentals(ticker, is_us_stock, krx_client)
            news_list   = fetch_google_news(display_name, limit=3)
            news_text   = "\n".join([f"- {n}" for n in news_list]) if news_list else "최근 주요 뉴스 없음"
            currency_sym = "$" if is_us_stock else "₩"
            price_fmt   = f"{currency_sym}{current_price:,.2f}" if is_us_stock else f"{currency_sym}{int(current_price):,}"

            instruction = "당신은 월스트리트 퀀트 애널리스트입니다. JSON 형식({'action':'BUY/SELL/HOLD','reason':'분석 코멘트'})으로만 응답하세요."
            prompt      = f"[{display_name}] 현재가:{price_fmt}, RSI:{current_rsi:.2f}, MA20:{ma20:.0f}, MA60:{ma60:.0f}, PER:{per}, PBR:{pbr}, 뉴스:{news_text}. 트레이딩 관점 투자 의견을 제시하세요."
            result, called_at = get_ai_response(prompt, api_key, model_choice, instruction, is_json=True)
            if result is None: return

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"📈 {display_name} 6개월 추이")
                st.line_chart(hist[['Close','MA20','MA60']])
            with col2:
                st.subheader("⚡ 퀵 스캔 리포트")
                action = result.get('action','HOLD')
                if action == "BUY":   st.success(f"### 의견: {action} 🟢")
                elif action == "SELL": st.error(f"### 의견: {action} 🔴")
                else:                  st.warning(f"### 의견: {action} 🟡")
                st.markdown(f"**RSI(14):** `{current_rsi:.2f}` | **PER:** `{per}` | **PBR:** `{pbr}`")
                st.info(f"**AI 코멘트:**\n{result.get('reason','')}")
                st.caption(f"기준 시각: {called_at}")
        except Exception as e:
            st.error(f"오류: {e}")

# ==========================================
# 2. 메인 UI
# ==========================================

krx_client = get_krx_client()
df_krx     = get_krx_full_data(krx_client)

with st.sidebar:
    st.header("⚙️ 터미널 설정")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini API 연동 완료")
    except:
        st.error("⚠️ GEMINI_API_KEY Secrets 설정 필요")
        api_key = ""

    if krx_client:
        st.success("✅ KRX OpenAPI 연동 완료")
    else:
        st.warning("⚠️ KRX_API_KEY Secrets 설정 필요")

    model_choice = st.selectbox("사용할 AI 모델", ("gemini-3.5-flash","gemini-3.5-pro"))
    st.divider()
    st.header("🔗 자동화 워크플로우")
    webhook_url = st.text_input("Webhook URL", placeholder="https://...")
    if st.button("테스트 알림 발송"):
        if webhook_url:
            try:
                requests.post(webhook_url, json={"text":"📈 AI 퀀트 터미널: 테스트 완료."}, timeout=3)
                st.success("웹훅 전송 완료!")
            except: st.error("전송 실패")

st.title("🤖 AI 글로벌 퀀트 터미널")

indices_data = get_major_indices()
if indices_data:
    cols = st.columns(len(indices_data))
    for i, (name, data) in enumerate(indices_data.items()):
        cols[i].metric(label=name, value=f"{data['price']:,.2f}", delta=f"{data['diff']:,.2f} ({data['pct']:+.2f}%)")
st.divider()

tab_main, tab_search, tab_recommend, tab_backtest, tab_chat = st.tabs([
    "🌐 실시간 시장 랭킹","🔍 딥다이브 분석 & Peer 비교",
    "🏆 AI 주도주 & 숨은 보석 추천","⏳ 알고리즘 백테스팅","💬 전담 AI 애널리스트 챗봇"
])

with tab_main:
    if df_krx.empty:
        st.error("🚨 KRX 데이터를 불러오지 못했습니다. 잠시 후 새로고침 해주세요.")
    else:
        m_data = get_ranking_tables(df_krx)
        t1,t2,t3,t4,t5,t6 = st.tabs(["👑 KOSPI 시총상위","👑 KOSDAQ 시총상위","🌊 KOSPI 거래량","🌊 KOSDAQ 거래량","🚀 상승률 상위","📉 하락률 상위"])
        events = {}
        with t1: events['kospi_cap']  = st.dataframe(m_data['kospi_cap'],  use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t1")
        with t2: events['kosdaq_cap'] = st.dataframe(m_data['kosdaq_cap'], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t2")
        with t3: events['kospi_vol']  = st.dataframe(m_data['kospi_vol'],  use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t3")
        with t4: events['kosdaq_vol'] = st.dataframe(m_data['kosdaq_vol'], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t4")
        with t5: events['price_up']   = st.dataframe(m_data['price_up'],   use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t5")
        with t6: events['price_down'] = st.dataframe(m_data['price_down'], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t6")

        selected_stock = None
        for category_key, event in events.items():
            if len(event.selection.rows) > 0:
                df_ref = m_data[category_key]
                if not df_ref.empty and '종목명' in df_ref.columns:
                    selected_stock = df_ref.iloc[event.selection.rows[0]]['종목명']
                break

        if selected_stock:
            st.divider()
            st.markdown(f"### ⚡ **{selected_stock}** 실시간 퀵 스캔")
            run_quick_analysis(selected_stock, api_key, model_choice, df_krx, krx_client)

with tab_search:
    st.header("기관 투자자용 심층 종목 분석")
    company_name = st.text_input("검색할 종목명 또는 티커", value="삼성전자")
    analyze_btn  = st.button("심층 펀더멘털 분석 시작")

    if analyze_btn and company_name:
        ticker, is_us_stock, display_name = get_ticker_from_name(company_name, df_krx)
        if not ticker: st.error("종목을 찾을 수 없습니다.")
        else:
            st.session_state.current_stock = display_name
            with st.spinner(f"'{display_name}' 데이터 수집 중..."):
                raw_code = ticker.split('.')[0]
                hist = get_stock_history_kr(raw_code, 180) if not is_us_stock else yf.Ticker(ticker).history(period="6mo", auto_adjust=False)
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    per, pbr      = get_fundamentals(ticker, is_us_stock, krx_client)
                    macro         = get_macro_data()
                    news_list     = fetch_google_news(display_name, limit=5)
                    news_text     = "\n".join([f"- {n}" for n in news_list]) if news_list else "최근 주요 뉴스 없음"

                    peers     = get_peer_group(display_name)
                    peer_data = []
                    for p in peers:
                        p_ticker, p_is_us, p_name = get_ticker_from_name(p, df_krx)
                        if p_ticker:
                            p_per, p_pbr = get_fundamentals(p_ticker, p_is_us, krx_client)
                            peer_data.append({"기업명": p_name, "PER": p_per, "PBR": p_pbr})

                    currency_sym = "$" if is_us_stock else "₩"
                    price_fmt    = f"{currency_sym}{current_price:,.2f}" if is_us_stock else f"{currency_sym}{int(current_price):,}"

                    instruction = "당신은 월스트리트 수석 애널리스트입니다. 마크다운 포맷으로 작성하세요."
                    prompt = f"""
**[타깃 기업]** {display_name} (현재가: {price_fmt}, PER: {per}, PBR: {pbr})
**[Peer 데이터]** {json.dumps(peer_data, ensure_ascii=False)}
**[매크로]** 환율: {macro.get('USD_KRW')}, 미10년물: {macro.get('US_10Y')}%, VIX: {macro.get('VIX')}
**[뉴스]** {news_text}

1. 비즈니스 모델 분석 2. 밸류에이션 매력도 3. 투자의견(Buy/Hold/Sell) 및 목표가를 작성하라.
"""
                    report_text, _ = get_ai_response(prompt, api_key, model_choice, instruction, is_json=False)
                    if report_text:
                        st.success(f"✅ {display_name} 심층 리포트 완료")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.metric("PER", per)
                            st.metric("PBR", pbr)
                            if peer_data: st.dataframe(pd.DataFrame(peer_data), hide_index=True)
                            st.line_chart(hist['Close'])
                        with col2:
                            st.subheader("🧠 딥다이브 퀀트 리포트")
                            st.markdown(report_text)

with tab_recommend:
    st.header("🏆 듀얼 엔진 주식 스크리너")
    col_a, col_b = st.columns(2)
    with col_a: market_choice   = st.radio("분석할 시장:", ["🇰🇷 국내 증시","🇺🇸 미국 증시"], horizontal=True)
    with col_b: strategy_choice = st.radio("투자 전략:", ["🔥 시장 주도주","💎 숨은 보석 발굴"], horizontal=True)

    if st.button("🚀 실시간 추천받기", use_container_width=True):
        with st.spinner("전체 상장사 실시간 퀀트 스크리닝 중..."):
            is_us_recom   = "미국" in market_choice
            is_hidden_gem = "숨은 보석" in strategy_choice
            if is_us_recom:
                trending_data = get_us_hidden_gems_with_news() if is_hidden_gem else get_us_trending_stocks_with_news()
                currency = "$"
            else:
                trending_data = get_hidden_gem_stocks(df_krx) if is_hidden_gem else get_trending_stocks_with_news(df_krx)
                currency = "₩"

            if not trending_data: st.error("❌ 시장 데이터 수집 실패.")
            else:
                instruction     = "당신은 퀀트 매니저입니다. [{'stock':'종목명','current_price':'숫자만','sentiment':'호재 요약','reason':'추천 사유'}] JSON 배열로 응답하세요."
                strategy_prompt = "저평가 턴어라운드 가치주 상위 5개 추천" if is_hidden_gem else "모멘텀 강한 주도주 상위 5개 추천"
                prompt = f"후보: {json.dumps(trending_data, ensure_ascii=False)}\n\n찌라시 제외, 진짜 펀더멘털 개선 종목으로 {strategy_prompt}."
                try:
                    recommendations, _ = get_ai_response(prompt, api_key, model_choice, instruction, is_json=True)
                    if recommendations:
                        st.success("🎉 퀀트 스크리닝 완료!")
                        cols = st.columns(len(recommendations))
                        for idx, rec in enumerate(recommendations):
                            with cols[idx]:
                                st.info(f"### 🥇 TOP {idx+1}\n**{rec.get('stock')}**")
                                raw_price = str(rec.get('current_price','0')).replace(',','').replace('원','').replace('$','').replace(' ','')
                                try: fmt_price = f"${float(raw_price):,.2f}" if is_us_recom else f"₩{int(float(raw_price)):,}"
                                except: fmt_price = f"{currency}{raw_price}"
                                st.metric("현재가", fmt_price)
                                st.markdown(f"**📰 요약:** {rec.get('sentiment')}")
                                st.write(f"**💡 사유:** {rec.get('reason')}")
                except Exception as e: st.error(f"오류: {e}")

with tab_backtest:
    st.header("⏳ 퀀트 알고리즘 백테스팅")
    col1, col2 = st.columns([1, 3])
    with col1:
        test_input = st.text_input("백테스트 종목명", value="삼성전자")
        test_years = st.slider("기간 (년)", 1, 5, 3)
        run_bt     = st.button("백테스트 시작")
    with col2:
        if run_bt and test_input:
            t_ticker, t_is_us, t_name = get_ticker_from_name(test_input, df_krx)
            if t_ticker:
                t_raw = t_ticker.split('.')[0]
                st.session_state.current_stock = t_name
                with st.spinner("시뮬레이션 중..."):
                    bt_df = run_backtest(t_raw if not t_is_us else t_ticker, t_is_us, test_years)
                    if bt_df is not None and not bt_df.empty:
                        mr = (bt_df['Cumulative_Market'].iloc[-1]-1)*100
                        sr = (bt_df['Cumulative_Strategy'].iloc[-1]-1)*100
                        st.subheader(f"📊 {t_name} {test_years}년 시뮬레이션")
                        c1, c2 = st.columns(2)
                        c1.metric("Buy & Hold", f"{mr:.2f}%")
                        c2.metric("전략 수익률", f"{sr:.2f}%", delta=f"{sr-mr:.2f}% 초과수익")
                        st.line_chart(bt_df[['Cumulative_Market','Cumulative_Strategy']])
            else: st.error("정확한 종목을 입력하세요.")

with tab_chat:
    st.header("💬 전담 AI 애널리스트 채팅")
    current_focus = st.session_state.get('current_stock','없음')
    if current_focus != "없음": st.info(f"💡 현재 분석 종목: **'{current_focus}'**")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요..."):
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.chat_history.append({"role":"user","content":prompt})
        history_text  = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:]])
        context_prompt = f"분석 중 종목: {current_focus}\n[대화]\n{history_text}\n[질문] {prompt}\n\n전문 애널리스트 톤으로 답변해."
        with st.spinner("답변 작성 중..."):
            response_text, _ = get_ai_response(context_prompt, api_key, model_choice, "당신은 퀀트 애널리스트입니다.", is_json=False)
            if response_text:
                with st.chat_message("assistant"): st.markdown(response_text)
                st.session_state.chat_history.append({"role":"assistant","content":response_text})
