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

# ==========================================
# 한국 주요 종목 코드 사전 (KOSPI + KOSDAQ 300선)
# 실시간 스크리닝 대상 풀 - 종목명:야후티커 매핑
# ==========================================
KR_STOCK_MAP = {
    # KOSPI 대형주
    "삼성전자": "005930.KS", "SK하이닉스": "000660.KS", "삼성바이오로직스": "207940.KS",
    "LG에너지솔루션": "373220.KS", "현대차": "005380.KS", "삼성SDI": "006400.KS",
    "기아": "000270.KS", "POSCO홀딩스": "005490.KS", "NAVER": "035420.KS",
    "셀트리온": "068270.KS", "KB금융": "105560.KS", "신한지주": "055550.KS",
    "LG화학": "051910.KS", "하나금융지주": "086790.KS", "현대모비스": "012330.KS",
    "삼성물산": "028260.KS", "우리금융지주": "316140.KS", "카카오": "035720.KS",
    "두산에너빌리티": "034020.KS", "HD현대": "267250.KS", "LG전자": "066570.KS",
    "삼성생명": "032830.KS", "KT&G": "033780.KS", "고려아연": "010130.KS",
    "한국전력": "015760.KS", "SK": "034730.KS", "SK이노베이션": "096770.KS",
    "롯데케미칼": "011170.KS", "대한항공": "003490.KS", "아모레퍼시픽": "090430.KS",
    "삼성전기": "009150.KS", "한화에어로스페이스": "012450.KS", "LG": "003550.KS",
    "SK텔레콤": "017670.KS", "한화솔루션": "009830.KS", "현대건설": "000720.KS",
    "CJ제일제당": "097950.KS", "GS": "078930.KS", "포스코인터내셔널": "047050.KS",
    "S-Oil": "010950.KS", "HMM": "011200.KS", "강원랜드": "035250.KS",
    "기업은행": "024110.KS", "미래에셋증권": "006800.KS", "현대제철": "004020.KS",
    "코웨이": "021240.KS", "한국타이어앤테크놀로지": "161390.KS", "BGF리테일": "282330.KS",
    # KOSDAQ 대형주
    "에코프로비엠": "247540.KQ", "에코프로": "086520.KQ", "HLB": "028300.KQ",
    "알테오젠": "196170.KQ", "리가켐바이오": "141080.KQ", "삼천당제약": "000250.KQ",
    "한미반도체": "042700.KQ", "레인보우로보틱스": "277810.KQ", "클래시스": "214150.KQ",
    "파마리서치": "214450.KQ", "셀트리온헬스케어": "091990.KQ", "카카오게임즈": "293490.KQ",
    "펄어비스": "263750.KQ", "크래프톤": "259960.KS", "엔씨소프트": "036570.KS",
    "위메이드": "112040.KQ", "넷마블": "251270.KS", "더블유게임즈": "192080.KQ",
    "케이카": "381970.KS", "오스템임플란트": "048260.KQ", "덴티움": "145720.KQ",
    "동국제강": "460860.KS", "고려제강": "004560.KS", "세아베스틸지주": "001430.KS",
    "DB하이텍": "000990.KS", "하이비젼시스템": "126700.KQ", "이오테크닉스": "039030.KQ",
    "원익IPS": "240810.KQ", "주성엔지니어링": "036930.KQ", "HPSP": "403870.KQ",
    "피에스케이": "319660.KQ", "디이엔티": "079810.KQ", "코미코": "183300.KQ",
    "솔브레인": "357780.KQ", "동진쎄미켐": "005290.KS", "후성": "093370.KS",
    "SK바이오팜": "326030.KS", "셀트리온제약": "068760.KQ", "유한양행": "000100.KS",
    "한미약품": "128940.KS", "종근당": "185750.KS", "대웅제약": "069620.KS",
    "녹십자": "006280.KS", "동아에스티": "170900.KS", "보령": "003850.KS",
    "일동제약": "249420.KS", "광동제약": "009290.KS", "JW중외제약": "001060.KS",
    "메디톡스": "086900.KQ", "휴젤": "145020.KQ", "대봉엘에스": "078140.KQ",
    "삼양식품": "003230.KS", "오리온": "271560.KS", "농심": "004370.KS",
    "롯데웰푸드": "280360.KS", "CJ": "001040.KS", "하이트진로": "000080.KS",
    "이마트": "139480.KS", "현대백화점": "069960.KS", "롯데쇼핑": "023530.KS",
    "GS리테일": "007070.KS", "BGF": "027410.KS", "한화": "000880.KS",
    "두산": "000150.KS", "효성": "004800.KS", "LS": "006260.KS",
    "코오롱인더": "120110.KS", "태광산업": "003240.KS", "일진머티리얼즈": "020150.KS",
    "엔켐": "348370.KQ", "코스모신소재": "005070.KS", "에코프로에이치엔": "383310.KQ",
    "포스코퓨처엠": "003670.KS", "LG에너지솔루션": "373220.KS",
    "한화오션": "042660.KS", "HD현대중공업": "329180.KS", "삼성중공업": "010140.KS",
    "현대로템": "064350.KS", "LIG넥스원": "079550.KS", "빅텍": "065450.KQ",
}

# 종목명 → 티커 역방향 매핑
TICKER_TO_NAME = {v: k for k, v in KR_STOCK_MAP.items()}

def get_ticker_from_name(name, _=None):
    name = name.strip()
    if name in US_STOCKS: return US_STOCKS[name], True, name
    if re.match(r'^[A-Za-z]+$', name): return name.upper(), True, name.upper()

    # 정확 일치
    if name in KR_STOCK_MAP: return KR_STOCK_MAP[name], False, name

    # 부분 일치
    clean = name.replace(" ", "").upper()
    for k, v in KR_STOCK_MAP.items():
        if clean in k.replace(" ", "").upper():
            return v, False, k

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
            suffix = '.KQ' if '코스닥' in market or 'KOSDAQ' in market else '.KS'
            return f"{code}{suffix}", False, items[0][0]
    except: pass

    return None, False, name

# ==========================================
# 1. 핵심 데이터 수집 (Yahoo Finance 100% 활용)
# ==========================================

@st.cache_data(ttl=1800)
def get_krx_full_data():
    """
    Yahoo Finance로 한국 주요 종목 300선 실시간 스크리닝
    - 현재가, 거래량, 등락률, 시가총액 실시간 수집
    - 어떤 서버도 차단 불가
    """
    tickers = list(KR_STOCK_MAP.values())
    rows = []

    try:
        # yf.download로 오늘 + 어제 데이터 한 번에 받기
        data = yf.download(
            tickers, period="5d", auto_adjust=False,
            group_by="ticker", progress=False, threads=True
        )

        for name, ticker in KR_STOCK_MAP.items():
            try:
                if len(tickers) == 1:
                    df = data
                else:
                    df = data[ticker] if ticker in data.columns.get_level_values(0) else pd.DataFrame()

                if df is None or df.empty: continue
                df = df.dropna(subset=['Close'])
                if len(df) < 2: continue

                close_today = float(df['Close'].iloc[-1])
                close_prev  = float(df['Close'].iloc[-2])
                volume      = float(df['Volume'].iloc[-1])
                chg_rate    = ((close_today - close_prev) / close_prev) * 100

                # 시가총액: 야후 info에서 (느리므로 캐시 활용)
                rows.append({
                    'Name':    name,
                    'Code':    ticker.split('.')[0],
                    'Ticker':  ticker,
                    'Market':  'KOSDAQ' if ticker.endswith('.KQ') else 'KOSPI',
                    'Close':   close_today,
                    'ChgRate': round(chg_rate, 2),
                    'Volume':  volume,
                    'Marcap':  np.nan,  # 별도 수집 생략 (속도 우선)
                    'PER':     np.nan,
                    'PBR':     np.nan,
                })
            except: continue

    except Exception as e:
        st.warning(f"야후 파이낸스 일괄 수집 오류: {e}")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df

def get_ranking_tables(df_krx):
    if df_krx.empty:
        return {k: pd.DataFrame() for k in ['kospi_cap','kosdaq_cap','kospi_vol','kosdaq_vol','price_up','price_down']}

    kospi  = df_krx[df_krx['Market'] == 'KOSPI'].copy()
    kosdaq = df_krx[df_krx['Market'] == 'KOSDAQ'].copy()

    def make_table(df, sort_col, ascending=False, n=30):
        if sort_col not in df.columns or df.empty: return pd.DataFrame()
        df = df[df[sort_col].notna()].sort_values(sort_col, ascending=ascending).head(n)
        cols = [c for c in ['Name','Close','ChgRate','Volume'] if c in df.columns]
        return df[cols].rename(columns={
            'Name':'종목명','Close':'현재가','ChgRate':'등락률(%)','Volume':'거래량'
        }).reset_index(drop=True)

    return {
        'kospi_cap':  make_table(kospi,  'Volume',  False),  # 시총 대신 거래량 기준
        'kosdaq_cap': make_table(kosdaq, 'Volume',  False),
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

# 🔥 PER/PBR: Daum Finance API (TTM 기준, 네이버와 동일)
@st.cache_data(ttl=3600)
def get_per_pbr_daum(raw_code):
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
            per = f"{float(d['per']):.2f}배" if d.get('per') and float(d['per']) > 0 else "N/A"
            pbr = f"{float(d['pbr']):.2f}배" if d.get('pbr') and float(d['pbr']) > 0 else "N/A"
            return per, pbr
    except: pass
    return "N/A", "N/A"

def get_fundamentals(ticker, is_us_stock, _=None):
    if is_us_stock:
        try:
            info = yf.Ticker(ticker).info
            per = f"{info['trailingPE']:.2f}배" if info.get('trailingPE') else "N/A"
            pbr = f"{info['priceToBook']:.2f}배" if info.get('priceToBook') else "N/A"
            return per, pbr
        except: return "N/A", "N/A"
    raw_code = ticker.split('.')[0]
    return get_per_pbr_daum(raw_code)

@st.cache_data(ttl=600)
def get_stock_history_kr(ticker, period_days=180):
    try:
        hist = yf.Ticker(ticker).history(period="6mo", auto_adjust=False)
        if not hist.empty: return hist
    except: pass
    return pd.DataFrame()

# 🔥 실시간 스크리닝: 전체 풀에서 실제 거래량/등락 기준 상위 추출
@st.cache_data(ttl=1800)
def get_trending_stocks_with_news(df_krx):
    if df_krx.empty: return []
    pool = df_krx.sort_values('Volume', ascending=False).head(20)
    candidates = []
    for _, row in pool.iterrows():
        name, ticker = str(row['Name']), str(row['Ticker'])
        try:
            candidates.append({
                "종목명": name,
                "현재가": int(row['Close']),
                "등락률": round(row['ChgRate'], 2),
                "최신뉴스": fetch_google_news(name, 3)
            })
        except: pass
    return candidates

@st.cache_data(ttl=1800)
def get_hidden_gem_stocks(df_krx):
    if df_krx.empty: return []
    # 거래량 하위 50% 이면서 하락률이 낮은(소외된) 종목
    vol_median = df_krx['Volume'].median()
    pool = df_krx[df_krx['Volume'] <= vol_median].copy()
    pool = pool[pool['ChgRate'].between(-3, 3)]  # 급등/급락 제외한 안정 종목
    if pool.empty: pool = df_krx
    sampled = pool.sample(n=min(15, len(pool)))
    candidates = []
    for _, row in sampled.iterrows():
        name, ticker = str(row['Name']), str(row['Ticker'])
        try:
            per, pbr = get_per_pbr_daum(row['Code'])
            candidates.append({
                "종목명": name,
                "현재가": int(row['Close']),
                "등락률": round(row['ChgRate'], 2),
                "PER": per,
                "PBR": pbr,
                "최신뉴스": fetch_google_news(name, 3)
            })
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
    if "삼성전자" in name: peers = ["SK하이닉스","삼성전기","DB하이텍"]
    elif "SK하이닉스" in name: peers = ["삼성전자","한미반도체","DB하이텍"]
    elif "현대차" in name or "기아" in name: peers = ["기아","현대모비스","현대로템"]
    elif "카카오" in name: peers = ["NAVER","카카오게임즈","크래프톤"]
    elif "셀트리온" in name: peers = ["삼성바이오로직스","한미약품","유한양행"]
    elif "에코프로" in name: peers = ["에코프로비엠","포스코퓨처엠","코스모신소재"]
    elif "AAPL" in name or "MSFT" in name: peers = ["MSFT","GOOGL","AMZN"]
    elif "TSLA" in name: peers = ["RIVN","LCID","F"]
    elif "NVDA" in name: peers = ["AMD","INTC","AVGO"]
    else: peers = []
    return [p for p in peers if p not in company_name][:3]

@st.cache_data(ttl=3600)
def run_backtest(ticker, is_us_stock, start_years=3):
    try:
        hist = yf.Ticker(ticker).history(period=f"{start_years}y", auto_adjust=False)
        if hist is None or hist.empty: return None
        df = hist.copy()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        delta = df['Close'].diff()
        gain  = delta.where(delta>0,0).rolling(14).mean()
        loss  = (-delta.where(delta<0,0)).rolling(14).mean()
        df['RSI'] = 100-(100/(1+(gain/loss)))
        df['Signal'] = 0
        df.loc[(df['MA20']>df['MA60'])&(df['RSI']<50),'Signal'] = 1
        df.loc[(df['MA20']<df['MA60'])|(df['RSI']>70),'Signal'] = -1
        df['Position']            = df['Signal'].replace(-1,0).shift()
        df['Market_Return']       = df['Close'].pct_change()
        df['Strategy_Return']     = df['Position']*df['Market_Return']
        df['Cumulative_Market']   = (1+df['Market_Return']).cumprod()
        df['Cumulative_Strategy'] = (1+df['Strategy_Return']).cumprod()
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
        st.error(f"⚠️ AI 오류: {e}")
        return None, None

def run_quick_analysis(company_name, api_key, model_choice, df_krx):
    ticker, is_us, display_name = get_ticker_from_name(company_name)
    if not ticker:
        st.error(f"'{company_name}' 종목을 찾지 못했습니다.")
        return
    st.session_state.current_stock = display_name
    with st.spinner(f"'{display_name}' 실시간 퀵 스캔 중..."):
        try:
            hist = yf.Ticker(ticker).history(period="6mo", auto_adjust=False)
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

            per, pbr  = get_fundamentals(ticker, is_us)
            news      = fetch_google_news(display_name, 3)
            news_text = "\n".join([f"- {n}" for n in news]) if news else "최근 주요 뉴스 없음"
            sym       = "$" if is_us else "₩"
            price_fmt = f"{sym}{curr:,.2f}" if is_us else f"{sym}{int(curr):,}"

            result, called_at = get_ai_response(
                f"[{display_name}] 현재가:{price_fmt}, RSI:{rsi:.2f}, MA20:{ma20:.0f}, MA60:{ma60:.0f}, PER:{per}, PBR:{pbr}, 뉴스:{news_text}. 트레이딩 투자 의견.",
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
                if action=="BUY":    st.success(f"### 의견: {action} 🟢")
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

# 앱 시작 시 전체 데이터 로드
with st.spinner("📡 한국 주요 종목 실시간 데이터 수집 중..."):
    df_krx = get_krx_full_data()

tab_main, tab_search, tab_recommend, tab_backtest, tab_chat = st.tabs([
    "🌐 실시간 시장 랭킹","🔍 딥다이브 분석 & Peer 비교",
    "🏆 AI 주도주 & 숨은 보석 추천","⏳ 알고리즘 백테스팅","💬 전담 AI 애널리스트 챗봇"
])

with tab_main:
    if df_krx.empty:
        st.error("🚨 데이터를 불러오지 못했습니다. 새로고침 해주세요.")
    else:
        st.caption(f"✅ 총 {len(df_krx)}개 종목 실시간 데이터 수집 완료")
        m_data = get_ranking_tables(df_krx)
        t1,t2,t3,t4,t5,t6 = st.tabs(["👑 KOSPI 거래량","👑 KOSDAQ 거래량","🌊 KOSPI 거래량상위","🌊 KOSDAQ 거래량상위","🚀 상승률 상위","📉 하락률 상위"])
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
        ticker, is_us, display_name = get_ticker_from_name(company_name)
        if not ticker: st.error("종목을 찾을 수 없습니다.")
        else:
            st.session_state.current_stock = display_name
            with st.spinner(f"'{display_name}' 데이터 수집 중..."):
                hist = yf.Ticker(ticker).history(period="6mo", auto_adjust=False)
                if not hist.empty:
                    curr          = hist['Close'].iloc[-1]
                    per, pbr      = get_fundamentals(ticker, is_us)
                    macro         = get_macro_data()
                    news_list     = fetch_google_news(display_name, 5)
                    news_text     = "\n".join([f"- {n}" for n in news_list]) if news_list else "없음"
                    peers         = get_peer_group(display_name)
                    peer_data     = []
                    for p in peers:
                        pt, pu, pn = get_ticker_from_name(p)
                        if pt:
                            pp, pb = get_fundamentals(pt, pu)
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
        with st.spinner("실시간 퀀트 스크리닝 중..."):
            is_us_r = "미국" in market_choice
            is_gem  = "숨은 보석" in strategy_choice
            if is_us_r:
                data     = get_us_hidden_gems_with_news() if is_gem else get_us_trending_stocks_with_news()
                currency = "$"
            else:
                data     = get_hidden_gem_stocks(df_krx) if is_gem else get_trending_stocks_with_news(df_krx)
                currency = "₩"

            if not data: st.error("❌ 데이터 수집 실패.")
            else:
                recs, _ = get_ai_response(
                    f"후보 종목 실시간 데이터: {json.dumps(data,ensure_ascii=False)}\n\n{'저평가 가치주 상위 5개' if is_gem else '모멘텀 강한 주도주 상위 5개'} 추천.",
                    api_key, model_choice,
                    "당신은 퀀트 매니저입니다. [{'stock':'종목명','current_price':'숫자만','sentiment':'호재요약','reason':'추천사유'}] JSON 배열로만 응답.",
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
            t_ticker, t_is_us, t_name = get_ticker_from_name(test_input)
            if t_ticker:
                st.session_state.current_stock = t_name
                with st.spinner("시뮬레이션 중..."):
                    bt_df = run_backtest(t_ticker, t_is_us, test_years)
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
