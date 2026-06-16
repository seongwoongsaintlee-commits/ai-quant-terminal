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

US_STOCKS = {"애플": "AAPL", "엔비디아": "NVDA", "테슬라": "TSLA", "마이크로소프트": "MSFT"}

KRX_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'http://data.krx.co.kr/',
    'Accept': 'application/json, text/plain, */*',
}

def get_recent_bizday():
    d = datetime.now()
    # 장 마감 전(오전)이면 전날 기준
    if d.hour < 16:
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime('%Y%m%d')

# ==========================================
# 1. KRX 직접 HTTP 호출 (라이브러리 불필요)
# ==========================================

@st.cache_data(ttl=3600)
def get_krx_full_data():
    """
    KRX 공식 서버(data.krx.co.kr) 직접 HTTP 호출
    PER/PBR 포함 전 종목 시세 수집
    Python 버전 무관하게 작동
    """
    today = get_recent_bizday()
    base_url = "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd"
    all_price = []
    all_fund  = []

    for mkt_id, mkt_nm in [("STK", "KOSPI"), ("KSQ", "KOSDAQ")]:
        # 시세 (종가, 거래량, 등락률, 시가총액)
        try:
            r = requests.post(base_url, data={
                "bld":         "dbms/MDC/STAT/standard/MDCSTAT01501",
                "mktId":       mkt_id,
                "trdDd":       today,
                "share":       "1",
                "money":       "1",
                "csvxls_isNo": "false",
            }, headers=KRX_HEADERS, timeout=10)
            rows = r.json().get("OutBlock_1", [])
            if rows:
                df = pd.DataFrame(rows)
                df["Market"] = mkt_nm
                all_price.append(df)
        except Exception as e:
            st.warning(f"{mkt_nm} 시세 오류: {e}")

        # PER/PBR
        try:
            r2 = requests.post(base_url, data={
                "bld":         "dbms/MDC/STAT/standard/MDCSTAT03501",
                "mktId":       mkt_id,
                "trdDd":       today,
                "csvxls_isNo": "false",
            }, headers=KRX_HEADERS, timeout=10)
            rows2 = r2.json().get("OutBlock_1", [])
            if rows2:
                all_fund.append(pd.DataFrame(rows2))
        except:
            pass

    if not all_price:
        return pd.DataFrame()

    df_price = pd.concat(all_price, ignore_index=True)

    # 컬럼 매핑
    price_col_map = {
        'ISU_ABBRV':  'Name',
        'ISU_CD':     'Code',
        'TDD_CLSPRC': 'Close',
        'ACC_TRDVOL': 'Volume',
        'FLUC_RT':    'ChgRate',
        'MKTCAP':     'Marcap',
    }
    df_price.rename(columns=price_col_map, inplace=True)

    for col in ['Close', 'Volume', 'ChgRate', 'Marcap']:
        if col in df_price.columns:
            df_price[col] = df_price[col].astype(str).str.replace(',', '').str.replace('%', '')
            df_price[col] = pd.to_numeric(df_price[col], errors='coerce')

    # PER/PBR 병합
    if all_fund:
        df_fund = pd.concat(all_fund, ignore_index=True)
        fund_col_map = {'ISU_CD': 'Code', 'PER': 'PER', 'PBR': 'PBR'}
        df_fund.rename(columns=fund_col_map, inplace=True)
        for col in ['PER', 'PBR']:
            if col in df_fund.columns:
                df_fund[col] = df_fund[col].astype(str).str.replace(',', '')
                df_fund[col] = pd.to_numeric(df_fund[col], errors='coerce')
        if 'Code' in df_fund.columns:
            df_price = df_price.merge(df_fund[['Code', 'PER', 'PBR']], on='Code', how='left')

    if 'PER' not in df_price.columns: df_price['PER'] = np.nan
    if 'PBR' not in df_price.columns: df_price['PBR'] = np.nan

    keep = [c for c in ['Name','Code','Market','Close','ChgRate','Volume','Marcap','PER','PBR'] if c in df_price.columns]
    return df_price[keep].dropna(subset=['Name','Code']).reset_index(drop=True)

def get_ticker_from_name(name, df_krx):
    name = name.strip()
    if name in US_STOCKS: return US_STOCKS[name], True, name
    if re.match(r'^[A-Za-z]+$', name): return name.upper(), True, name.upper()

    if not df_krx.empty and 'Name' in df_krx.columns:
        clean = name.replace(" ", "").upper()
        exact = df_krx[df_krx['Name'].str.replace(" ", "").str.upper() == clean]
        if not exact.empty:
            row = exact.iloc[0]
            return f"{row['Code']}{'.KQ' if row.get('Market')=='KOSDAQ' else '.KS'}", False, row['Name']
        partial = df_krx[df_krx['Name'].str.replace(" ", "").str.upper().str.contains(clean, na=False)]
        if not partial.empty:
            row = partial.iloc[0]
            return f"{row['Code']}{'.KQ' if row.get('Market')=='KOSDAQ' else '.KS'}", False, row['Name']

    # 네이버 자동완성 폴백
    try:
        encoded = urllib.parse.quote(name.encode('euc-kr'))
        res = requests.get(
            f"https://ac.finance.naver.com/ac?q={encoded}&q_enc=euc-kr&st=111&r_format=json&r_enc=utf-8",
            headers={'User-Agent': 'Mozilla/5.0'}, timeout=3
        )
        items = res.json().get('items', [[]])[0]
        if items:
            code, market = items[0][1], items[0][2]
            return f"{code}{'.KQ' if '코스닥' in market or 'KOSDAQ' in market else '.KS'}", False, items[0][0]
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
        return df[cols].rename(columns={
            'Name':'종목명','Close':'현재가','ChgRate':'등락률(%)','Volume':'거래량','Marcap':'시가총액'
        }).reset_index(drop=True)

    return {
        'kospi_cap':  make_table(kospi,  'Marcap',  False),
        'kosdaq_cap': make_table(kosdaq, 'Marcap',  False),
        'kospi_vol':  make_table(kospi,  'Volume',  False),
        'kosdaq_vol': make_table(kosdaq, 'Volume',  False),
        'price_up':   make_table(df_krx, 'ChgRate', False),
        'price_down': make_table(df_krx, 'ChgRate', True),
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
            else:
                res[name] = {"price": 0.0, "diff": 0.0, "pct": 0.0}
        except:
            res[name] = {"price": 0.0, "diff": 0.0, "pct": 0.0}
    return res

@st.cache_data(ttl=3600)
def get_macro_data():
    macros = {"USD_KRW": "KRW=X", "US_10Y": "^TNX", "VIX": "^VIX"}
    data = {}
    for key, ticker in macros.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            data[key] = round(hist['Close'].iloc[-1], 2) if not hist.empty else "N/A"
        except:
            data[key] = "N/A"
    return data

def fetch_google_news(keyword, limit=5):
    try:
        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(keyword)}&hl=ko&gl=KR&ceid=KR:ko"
        feed = feedparser.parse(url)
        return [e.title for e in feed.entries[:limit]]
    except:
        return []

# 🔥 PER/PBR: KRX 데이터 우선, 없으면 Daum API
def get_fundamentals(ticker, is_us_stock, df_krx):
    per, pbr = "N/A", "N/A"

    if is_us_stock:
        try:
            info = yf.Ticker(ticker).info
            if info.get('trailingPE'): per = f"{info['trailingPE']:.2f}배"
            if info.get('priceToBook'): pbr = f"{info['priceToBook']:.2f}배"
        except: pass
        return per, pbr

    raw_code = ticker.split('.')[0]

    # 플랜 A: KRX 전체 데이터에서 직접 읽기
    if not df_krx.empty and 'PER' in df_krx.columns:
        match = df_krx[df_krx['Code'] == raw_code]
        if not match.empty:
            p = match.iloc[0].get('PER')
            b = match.iloc[0].get('PBR')
            if pd.notna(p) and float(p) > 0: per = f"{float(p):.2f}배"
            if pd.notna(b) and float(b) > 0: pbr = f"{float(b):.2f}배"

    # 플랜 B: Daum Finance API
    if per == "N/A" or pbr == "N/A":
        try:
            res = requests.get(
                f"https://finance.daum.net/api/quotes/A{raw_code}",
                headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Referer': f'https://finance.daum.net/quotes/A{raw_code}',
                    'Accept': 'application/json',
                }, timeout=5
            )
            if res.status_code == 200:
                d = res.json()
                if per == "N/A" and d.get('per') and float(d['per']) > 0:
                    per = f"{float(d['per']):.2f}배"
                if pbr == "N/A" and d.get('pbr') and float(d['pbr']) > 0:
                    pbr = f"{float(d['pbr']):.2f}배"
        except: pass

    return per, pbr

# 🔥 한국 주가: KRX OHLCV 직접 호출 (수정주가 X, 실제 거래가)
@st.cache_data(ttl=600)
def get_stock_history_kr(raw_code, period_days=180):
    try:
        end   = datetime.now()
        start = end - timedelta(days=period_days)
        r = requests.post(
            "http://data.krx.co.kr/comm/bldAttendant/getJsonData.cmd",
            data={
                "bld":         "dbms/MDC/STAT/standard/MDCSTAT01701",
                "isuCd":       raw_code,
                "strtDd":      start.strftime('%Y%m%d'),
                "endDd":       end.strftime('%Y%m%d'),
                "csvxls_isNo": "false",
            },
            headers=KRX_HEADERS, timeout=10
        )
        rows = r.json().get("output", [])
        if rows:
            df = pd.DataFrame(rows)
            col_map = {
                'TRD_DD':     'Date',
                'TDD_CLSPRC': 'Close',
                'TDD_OPNPRC': 'Open',
                'TDD_HGPRC':  'High',
                'TDD_LWPRC':  'Low',
                'ACC_TRDVOL': 'Volume',
            }
            df.rename(columns=col_map, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'], format='%Y/%m/%d', errors='coerce')
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            for col in ['Close','Open','High','Low','Volume']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(',','')
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
    except: pass

    # 폴백: Yahoo Finance
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
        name, code = str(row.get('Name','')), str(row.get('Code',''))
        try:
            hist = get_stock_history_kr(code, 5)
            if not hist.empty:
                candidates.append({"종목명": name, "현재가": int(hist['Close'].iloc[-1]), "최신뉴스": fetch_google_news(name, 3)})
        except: pass
    return candidates

@st.cache_data(ttl=1800)
def get_hidden_gem_stocks(df_krx):
    if df_krx.empty or 'Marcap' not in df_krx.columns: return []
    pool = df_krx[df_krx['Marcap'].notna()].sort_values('Marcap', ascending=False).head(200)
    if 'Volume' in pool.columns:
        vol_med = pool['Volume'].median()
        filtered = pool[pool['Volume'] < vol_med]
        if not filtered.empty: pool = filtered.head(60)
    sampled = pool.sample(n=min(15, len(pool))) if not pool.empty else pool
    candidates = []
    for _, row in sampled.iterrows():
        name, code = str(row.get('Name','')), str(row.get('Code',''))
        try:
            hist = get_stock_history_kr(code, 5)
            if not hist.empty:
                candidates.append({"종목명": name, "현재가": int(hist['Close'].iloc[-1]), "최신뉴스": fetch_google_news(name, 3)})
        except: pass
    return candidates

@st.cache_data(ttl=1800)
def get_us_trending_stocks_with_news():
    pool = ['AAPL','NVDA','MSFT','TSLA','AMZN','GOOGL','META','AMD','NFLX','TSM','SMCI','PLTR','AVGO']
    candidates = []
    for t in pool:
        try:
            hist = yf.Ticker(t).history(period="5d", auto_adjust=False)
            if not hist.empty:
                candidates.append({"종목명": t, "현재가": round(float(hist['Close'].iloc[-1]),2), "최신뉴스": fetch_google_news(f"{t} 주식", 3)})
        except: pass
    return candidates

@st.cache_data(ttl=1800)
def get_us_hidden_gems_with_news():
    pool = ['PYPL','DIS','PFE','NKE','SBUX','BA','INTC','C','WFC','T','VZ','O','QCOM','TXN','CVX']
    candidates = []
    for t in pool:
        try:
            hist = yf.Ticker(t).history(period="5d", auto_adjust=False)
            if not hist.empty:
                candidates.append({"종목명": t, "현재가": round(float(hist['Close'].iloc[-1]),2), "최신뉴스": fetch_google_news(f"{t} stock", 3)})
        except: pass
    return candidates

def get_peer_group(company_name):
    name = company_name.replace(" ","")
    if "삼성전자" in name or "SK하이닉스" in name: peers = ["SK하이닉스","한미반도체","DB하이텍"]
    elif "콜마" in name or "코스맥스" in name: peers = ["한국콜마","코스메카코리아","씨앤씨인터내셔널"]
    elif "AAPL" in name or "MSFT" in name: peers = ["MSFT","GOOGL","AMZN"]
    elif "TSLA" in name: peers = ["RIVN","LCID","F"]
    else: peers = []
    return [p for p in peers if p != company_name][:3]

@st.cache_data(ttl=3600)
def run_backtest(raw_code, is_us_stock, start_years=3):
    try:
        if is_us_stock:
            df = yf.Ticker(raw_code).history(period=f"{start_years}y", auto_adjust=False)
        else:
            df = get_stock_history_kr(raw_code, 365*start_years)
        if df is None or df.empty: return None
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        delta = df['Close'].diff()
        gain  = delta.where(delta>0, 0).rolling(14).mean()
        loss  = (-delta.where(delta<0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100/(1+(gain/loss)))
        df['Signal'] = 0
        df.loc[(df['MA20']>df['MA60'])&(df['RSI']<50), 'Signal'] = 1
        df.loc[(df['MA20']<df['MA60'])|(df['RSI']>70), 'Signal'] = -1
        df['Position']           = df['Signal'].replace(-1,0).shift()
        df['Market_Return']      = df['Close'].pct_change()
        df['Strategy_Return']    = df['Position'] * df['Market_Return']
        df['Cumulative_Market']  = (1+df['Market_Return']).cumprod()
        df['Cumulative_Strategy']= (1+df['Strategy_Return']).cumprod()
        return df
    except: return None

@st.cache_data(ttl=600, show_spinner=False)
def get_ai_response(prompt_text, api_key, model_choice, instruction_text, is_json=True):
    try:
        genai.configure(api_key=api_key)
        config = {"temperature": 0.2}
        if is_json: config["response_mime_type"] = "application/json"
        model    = genai.GenerativeModel(model_name=model_choice, system_instruction=instruction_text, generation_config=config)
        response = model.generate_content(prompt_text)
        clean    = response.text.strip()
        ts       = time.strftime("%H:%M:%S")
        if is_json:
            if clean.startswith("```"):
                clean = clean.replace("```json","").replace("```","").strip()
            return json.loads(clean), ts
        return clean, ts
    except Exception as e:
        st.error(f"⚠️ AI 연산 오류: {e}")
        return None, None

def run_quick_analysis(company_name, api_key, model_choice, df_krx):
    ticker, is_us, display_name = get_ticker_from_name(company_name, df_krx)
    if not ticker:
        st.error(f"'{company_name}' 종목을 찾지 못했습니다.")
        return
    st.session_state.current_stock = display_name
    with st.spinner(f"'{display_name}' 실시간 퀵 스캔 중..."):
        try:
            raw_code = ticker.split('.')[0]
            hist = get_stock_history_kr(raw_code, 180) if not is_us else yf.Ticker(ticker).history(period="6mo", auto_adjust=False)
            if hist.empty:
                st.error("주가 데이터를 불러오지 못했습니다.")
                return

            curr  = hist['Close'].iloc[-1]
            hist['MA20'] = hist['Close'].rolling(20).mean()
            hist['MA60'] = hist['Close'].rolling(60).mean()
            ma20  = hist['MA20'].iloc[-1] if len(hist)>=20 else curr
            ma60  = hist['MA60'].iloc[-1] if len(hist)>=60 else curr
            delta = hist['Close'].diff()
            gain  = delta.where(delta>0,0).rolling(14).mean()
            loss  = (-delta.where(delta<0,0)).rolling(14).mean()
            rsi   = (100-(100/(1+(gain/loss)))).iloc[-1] if not gain.isna().all() else 50

            per, pbr  = get_fundamentals(ticker, is_us, df_krx)
            news      = fetch_google_news(display_name, 3)
            news_text = "\n".join([f"- {n}" for n in news]) if news else "최근 주요 뉴스 없음"
            sym       = "$" if is_us else "₩"
            price_fmt = f"{sym}{curr:,.2f}" if is_us else f"{sym}{int(curr):,}"

            result, called_at = get_ai_response(
                f"[{display_name}] 현재가:{price_fmt}, RSI:{rsi:.2f}, MA20:{ma20:.0f}, MA60:{ma60:.0f}, PER:{per}, PBR:{pbr}, 뉴스:{news_text}. 트레이딩 투자 의견을 제시하세요.",
                api_key, model_choice,
                "당신은 월스트리트 퀀트 애널리스트입니다. JSON({'action':'BUY/SELL/HOLD','reason':'코멘트'})으로만 응답하세요.",
                is_json=True
            )
            if result is None: return

            col1, col2 = st.columns([2,1])
            with col1:
                st.subheader(f"📈 {display_name} 6개월 추이")
                st.line_chart(hist[['Close','MA20','MA60']])
            with col2:
                st.subheader("⚡ 퀵 스캔 리포트")
                action = result.get('action','HOLD')
                if action=="BUY":   st.success(f"### 의견: {action} 🟢")
                elif action=="SELL": st.error(f"### 의견: {action} 🔴")
                else:                st.warning(f"### 의견: {action} 🟡")
                st.markdown(f"**RSI(14):** `{rsi:.2f}` | **PER:** `{per}` | **PBR:** `{pbr}`")
                st.info(f"**AI 코멘트:**\n{result.get('reason','')}")
                st.caption(f"기준 시각: {called_at}")
        except Exception as e:
            st.error(f"오류: {e}")

# ==========================================
# 2. 메인 UI
# ==========================================

df_krx = get_krx_full_data()

with st.sidebar:
    st.header("⚙️ 터미널 설정")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini API 연동 완료")
    except:
        st.error("⚠️ GEMINI_API_KEY 설정 필요")
        api_key = ""
    model_choice = st.selectbox("AI 모델", ("gemini-3.5-flash","gemini-3.5-pro"))
    st.divider()
    webhook_url = st.text_input("Webhook URL", placeholder="https://...")
    if st.button("테스트 알림"):
        if webhook_url:
            try:
                requests.post(webhook_url, json={"text":"📈 AI 퀀트 터미널 테스트"}, timeout=3)
                st.success("전송 완료!")
            except: st.error("전송 실패")

st.title("🤖 AI 글로벌 퀀트 터미널")

indices_data = get_major_indices()
if indices_data:
    cols = st.columns(len(indices_data))
    for i, (name, data) in enumerate(indices_data.items()):
        cols[i].metric(name, f"{data['price']:,.2f}", f"{data['diff']:,.2f} ({data['pct']:+.2f}%)")
st.divider()

tab_main, tab_search, tab_recommend, tab_backtest, tab_chat = st.tabs([
    "🌐 실시간 시장 랭킹","🔍 딥다이브 분석 & Peer 비교",
    "🏆 AI 주도주 & 숨은 보석 추천","⏳ 알고리즘 백테스팅","💬 전담 AI 애널리스트 챗봇"
])

with tab_main:
    if df_krx.empty:
        st.error("🚨 KRX 데이터를 불러오지 못했습니다. 새로고침 해주세요.")
    else:
        m_data = get_ranking_tables(df_krx)
        t1,t2,t3,t4,t5,t6 = st.tabs(["👑 KOSPI 시총","👑 KOSDAQ 시총","🌊 KOSPI 거래량","🌊 KOSDAQ 거래량","🚀 상승률","📉 하락률"])
        events = {}
        with t1: events['kospi_cap']  = st.dataframe(m_data['kospi_cap'],  use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t1")
        with t2: events['kosdaq_cap'] = st.dataframe(m_data['kosdaq_cap'], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t2")
        with t3: events['kospi_vol']  = st.dataframe(m_data['kospi_vol'],  use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t3")
        with t4: events['kosdaq_vol'] = st.dataframe(m_data['kosdaq_vol'], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t4")
        with t5: events['price_up']   = st.dataframe(m_data['price_up'],   use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t5")
        with t6: events['price_down'] = st.dataframe(m_data['price_down'], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t6")

        selected_stock = None
        for cat, event in events.items():
            if len(event.selection.rows) > 0:
                df_ref = m_data[cat]
                if not df_ref.empty and '종목명' in df_ref.columns:
                    selected_stock = df_ref.iloc[event.selection.rows[0]]['종목명']
                break
        if selected_stock:
            st.divider()
            st.markdown(f"### ⚡ **{selected_stock}** 실시간 퀵 스캔")
            run_quick_analysis(selected_stock, api_key, model_choice, df_krx)

with tab_search:
    st.header("기관 투자자용 심층 종목 분석")
    company_name = st.text_input("검색할 종목명 또는 티커", value="삼성전자")
    if st.button("심층 펀더멘털 분석 시작"):
        ticker, is_us, display_name = get_ticker_from_name(company_name, df_krx)
        if not ticker: st.error("종목을 찾을 수 없습니다.")
        else:
            st.session_state.current_stock = display_name
            with st.spinner(f"'{display_name}' 데이터 수집 중..."):
                raw_code = ticker.split('.')[0]
                hist = get_stock_history_kr(raw_code, 180) if not is_us else yf.Ticker(ticker).history(period="6mo", auto_adjust=False)
                if not hist.empty:
                    curr          = hist['Close'].iloc[-1]
                    per, pbr      = get_fundamentals(ticker, is_us, df_krx)
                    macro         = get_macro_data()
                    news_list     = fetch_google_news(display_name, 5)
                    news_text     = "\n".join([f"- {n}" for n in news_list]) if news_list else "없음"
                    peers         = get_peer_group(display_name)
                    peer_data     = []
                    for p in peers:
                        pt, pu, pn = get_ticker_from_name(p, df_krx)
                        if pt:
                            pp, pb = get_fundamentals(pt, pu, df_krx)
                            peer_data.append({"기업명": pn, "PER": pp, "PBR": pb})
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
                            st.line_chart(hist['Close'])
                        with col2:
                            st.subheader("🧠 딥다이브 퀀트 리포트")
                            st.markdown(report_text)

with tab_recommend:
    st.header("🏆 듀얼 엔진 주식 스크리너")
    col_a, col_b = st.columns(2)
    with col_a: market_choice   = st.radio("시장:", ["🇰🇷 국내 증시","🇺🇸 미국 증시"], horizontal=True)
    with col_b: strategy_choice = st.radio("전략:", ["🔥 시장 주도주","💎 숨은 보석 발굴"], horizontal=True)

    if st.button("🚀 실시간 추천받기", use_container_width=True):
        with st.spinner("전체 상장사 실시간 퀀트 스크리닝 중..."):
            is_us_r = "미국" in market_choice
            is_gem  = "숨은 보석" in strategy_choice
            if is_us_r:
                data = get_us_hidden_gems_with_news() if is_gem else get_us_trending_stocks_with_news()
                currency = "$"
            else:
                data = get_hidden_gem_stocks(df_krx) if is_gem else get_trending_stocks_with_news(df_krx)
                currency = "₩"

            if not data: st.error("❌ 시장 데이터 수집 실패.")
            else:
                recs, _ = get_ai_response(
                    f"후보: {json.dumps(data,ensure_ascii=False)}\n\n{'저평가 가치주 상위 5개' if is_gem else '모멘텀 강한 주도주 상위 5개'} 추천. JSON 배열로.",
                    api_key, model_choice,
                    "당신은 퀀트 매니저입니다. [{'stock':'종목명','current_price':'숫자만','sentiment':'호재요약','reason':'사유'}] JSON으로만 응답.",
                    is_json=True
                )
                if recs:
                    st.success("🎉 퀀트 스크리닝 완료!")
                    cols = st.columns(len(recs))
                    for idx, rec in enumerate(recs):
                        with cols[idx]:
                            st.info(f"### 🥇 TOP {idx+1}\n**{rec.get('stock')}**")
                            raw_p = str(rec.get('current_price','0')).replace(',','').replace('원','').replace('$','').replace(' ','')
                            try: fmt_p = f"${float(raw_p):,.2f}" if is_us_r else f"₩{int(float(raw_p)):,}"
                            except: fmt_p = f"{currency}{raw_p}"
                            st.metric("현재가", fmt_p)
                            st.markdown(f"**📰 요약:** {rec.get('sentiment')}")
                            st.write(f"**💡 사유:** {rec.get('reason')}")

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
            else: st.error("정확한 종목명을 입력하세요.")

with tab_chat:
    st.header("💬 전담 AI 애널리스트 채팅")
    current_focus = st.session_state.get('current_stock','없음')
    if current_focus != "없음": st.info(f"💡 현재 분석 종목: **'{current_focus}'**")

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
