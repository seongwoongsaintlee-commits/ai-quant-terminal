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

YF_HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

# ==========================================
# 1. Yahoo Finance Screener - 실시간 동적 스크리닝
# ==========================================

@st.cache_data(ttl=1800)
def get_krx_full_data():
    """
    Yahoo Finance Screener API로 KOSPI/KOSDAQ 전 종목 실시간 스크리닝
    하드코딩 리스트 없음 - 오늘 실제 시장 데이터만 사용
    """
    rows = []
    screener_url = "https://query1.finance.yahoo.com/v1/finance/screener"

    for exchange, market_label in [("KSC", "KOSPI"), ("KOE", "KOSDAQ")]:
        offset = 0
        while offset < 600:
            try:
                payload = {
                    "offset": offset,
                    "size": 100,
                    "sortField": "intradayvolume",
                    "sortType": "DESC",
                    "quoteType": "EQUITY",
                    "query": {
                        "operator": "AND",
                        "operands": [
                            {"operator": "EQ", "operands": ["exchange", exchange]}
                        ]
                    },
                    "userId": "",
                    "userIdType": "guid"
                }
                res = requests.post(
                    screener_url,
                    headers=YF_HEADERS,
                    params={"formatted": "false", "lang": "en-US", "region": "KR"},
                    json=payload,
                    timeout=10
                )
                quotes = res.json().get("finance", {}).get("result", [{}])[0].get("quotes", [])
                if not quotes:
                    break

                for q in quotes:
                    try:
                        symbol = q.get('symbol', '')
                        if not symbol:
                            continue
                        rows.append({
                            'Name':    q.get('longName') or q.get('shortName', symbol),
                            'Code':    symbol.replace('.KS', '').replace('.KQ', ''),
                            'Ticker':  symbol,
                            'Market':  market_label,
                            'Close':   float(q.get('regularMarketPrice', 0) or 0),
                            'ChgRate': round(float(q.get('regularMarketChangePercent', 0) or 0), 2),
                            'Volume':  float(q.get('regularMarketVolume', 0) or 0),
                            'Marcap':  float(q.get('marketCap', 0) or 0),
                        })
                    except:
                        continue

                if len(quotes) < 100:
                    break
                offset += 100

            except Exception as e:
                st.warning(f"{market_label} 스크리닝 오류: {e}")
                break

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df[df['Close'] > 0].drop_duplicates(subset=['Ticker']).reset_index(drop=True)
    return df

def get_ticker_from_name(name, df_krx=None):
    name = name.strip()
    if name in US_STOCKS:
        return US_STOCKS[name], True, name
    if re.match(r'^[A-Za-z]+$', name):
        return name.upper(), True, name.upper()

    # 실시간 수집 데이터에서 검색
    if df_krx is not None and not df_krx.empty and 'Name' in df_krx.columns:
        clean = name.replace(" ", "").upper()
        exact = df_krx[df_krx['Name'].str.replace(" ", "").str.upper() == clean]
        if not exact.empty:
            row = exact.iloc[0]
            return row['Ticker'], False, row['Name']
        partial = df_krx[df_krx['Name'].str.replace(" ", "").str.upper().str.contains(clean, na=False)]
        if not partial.empty:
            row = partial.iloc[0]
            return row['Ticker'], False, row['Name']

    # Yahoo Finance 검색 API
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

def get_ranking_tables(df_krx):
    if df_krx.empty:
        return {k: pd.DataFrame() for k in ['kospi_cap','kosdaq_cap','kospi_vol','kosdaq_vol','price_up','price_down']}

    kospi  = df_krx[df_krx['Market'] == 'KOSPI'].copy()
    kosdaq = df_krx[df_krx['Market'] == 'KOSDAQ'].copy()

    def make_table(df, sort_col, ascending=False, n=30):
        if sort_col not in df.columns or df.empty:
            return pd.DataFrame()
        df = df[df[sort_col].notna() & (df[sort_col] > 0)].sort_values(sort_col, ascending=ascending).head(n)
        cols = [c for c in ['Name','Close','ChgRate','Volume','Marcap'] if c in df.columns]
        return df[cols].rename(columns={
            'Name':'종목명', 'Close':'현재가', 'ChgRate':'등락률(%)',
            'Volume':'거래량', 'Marcap':'시가총액'
        }).reset_index(drop=True)

    return {
        'kospi_cap':  make_table(kospi,  'Marcap',  False),
        'kosdaq_cap': make_table(kosdaq, 'Marcap',  False),
        'kospi_vol':  make_table(kospi,  'Volume',  False),
        'kosdaq_vol': make_table(kosdaq, 'Volume',  False),
        'price_up':   make_table(df_krx, 'ChgRate', False),
        'price_down': make_table(df_krx, 'ChgRate', True),
    }

# ==========================================
# 2. DART API - PER/PBR 근본 해결
#    네이버 증권과 동일한 원천 데이터
# ==========================================

@st.cache_data(ttl=86400)
def get_dart_corp_code(stock_code):
    """DART 고유번호 조회 (종목코드 → DART corp_code 변환)"""
    try:
        dart_key = st.secrets.get("DART_API_KEY", "")
        if not dart_key:
            return None

        # DART 전체 기업 목록 (zip 파일)
        res = requests.get(
            "https://opendart.fss.or.kr/api/corpCode.xml",
            params={"crtfc_key": dart_key},
            timeout=10
        )
        if res.status_code != 200:
            return None

        import zipfile
        import io
        import xml.etree.ElementTree as ET

        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            with z.open('CORPCODE.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                for corp in root.findall('list'):
                    code = corp.findtext('stock_code', '').strip()
                    if code == stock_code:
                        return corp.findtext('corp_code', '').strip()
    except:
        pass
    return None

@st.cache_data(ttl=86400)
def get_dart_financials(corp_code, year=None):
    """
    DART API로 최근 재무제표 수집
    - 당기순이익 (Net Income)
    - 자본총계 (Total Equity)
    - 발행주식수 (Shares Outstanding)
    """
    try:
        dart_key = st.secrets.get("DART_API_KEY", "")
        if not dart_key or not corp_code:
            return None

        if year is None:
            year = str(datetime.now().year - 1)

        # 단일회사 주요계정 (재무상태표 + 손익계산서)
        res = requests.get(
            "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json",
            params={
                "crtfc_key": dart_key,
                "corp_code":  corp_code,
                "bsns_year":  year,
                "reprt_code": "11011",  # 사업보고서
                "fs_div":     "CFS",    # 연결재무제표
            },
            timeout=10
        )
        data = res.json()
        if data.get('status') != '000':
            # 연결 없으면 별도재무제표
            res = requests.get(
                "https://opendart.fss.or.kr/api/fnlttSinglAcnt.json",
                params={
                    "crtfc_key": dart_key,
                    "corp_code":  corp_code,
                    "bsns_year":  year,
                    "reprt_code": "11011",
                    "fs_div":     "OFS",
                },
                timeout=10
            )
            data = res.json()

        if data.get('status') != '000':
            return None

        items = data.get('list', [])
        result = {}
        for item in items:
            acnt = item.get('account_nm', '')
            amt  = item.get('thstrm_amount', '0') or '0'
            amt  = float(str(amt).replace(',', '').replace(' ', '') or '0')

            if '당기순이익' in acnt and '지배' not in acnt and 'net_income' not in result:
                result['net_income'] = amt
            if '자본총계' in acnt and '지배' not in acnt and 'total_equity' not in result:
                result['total_equity'] = amt

        return result if result else None
    except:
        return None

@st.cache_data(ttl=86400)
def get_shares_outstanding(ticker):
    """발행주식수: Yahoo Finance info에서 추출"""
    try:
        info = yf.Ticker(ticker).fast_info
        shares = getattr(info, 'shares', None)
        if shares and shares > 0:
            return shares
        info2 = yf.Ticker(ticker).info
        return info2.get('sharesOutstanding') or info2.get('impliedSharesOutstanding')
    except:
        return None

@st.cache_data(ttl=3600)
def get_fundamentals(ticker, is_us_stock, df_krx=None):
    """
    PER/PBR 계산 우선순위:
    1. DART 재무제표 기반 직접 계산 (네이버와 동일한 원천)
    2. Yahoo Finance (미국 주식, 또는 DART 실패 시)
    """
    per, pbr = "N/A", "N/A"

    # ── 미국 주식: Yahoo Finance 그대로 사용 ──
    if is_us_stock:
        try:
            info = yf.Ticker(ticker).info
            if info.get('trailingPE'): per = f"{info['trailingPE']:.2f}배"
            if info.get('priceToBook'): pbr = f"{info['priceToBook']:.2f}배"
        except:
            pass
        return per, pbr

    # ── 한국 주식: DART 재무제표로 직접 계산 ──
    raw_code = ticker.split('.')[0]

    dart_key = st.secrets.get("DART_API_KEY", "")
    if dart_key:
        try:
            # 1. DART corp_code 변환
            corp_code = get_dart_corp_code(raw_code)

            if corp_code:
                # 2. 최근 2개년 시도 (전년도 → 전전년도)
                financials = None
                for y in [str(datetime.now().year - 1), str(datetime.now().year - 2)]:
                    financials = get_dart_financials(corp_code, y)
                    if financials:
                        break

                if financials:
                    # 3. 현재 주가 + 발행주식수
                    hist = yf.Ticker(ticker).history(period="1d", auto_adjust=False)
                    price = float(hist['Close'].iloc[-1]) if not hist.empty else 0
                    shares = get_shares_outstanding(ticker)

                    if price > 0 and shares and shares > 0:
                        net_income   = financials.get('net_income', 0)
                        total_equity = financials.get('total_equity', 0)

                        # PER = 시가총액 / 당기순이익
                        if net_income and net_income > 0:
                            market_cap = price * shares
                            per_val    = market_cap / net_income
                            per = f"{per_val:.2f}배"

                        # PBR = 시가총액 / 자본총계
                        if total_equity and total_equity > 0:
                            market_cap = price * shares
                            pbr_val    = market_cap / total_equity
                            pbr = f"{pbr_val:.2f}배"

        except Exception as e:
            pass  # DART 실패 시 아래 폴백으로

    # ── 폴백: Daum Finance API ──
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
        except:
            pass

    return per, pbr

# ==========================================
# 3. 나머지 유틸리티 함수들
# ==========================================

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
                res[name] = {"price": curr, "diff": curr - prev, "pct": ((curr - prev) / prev) * 100}
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
        return [e.title for e in feedparser.parse(url).entries[:limit]]
    except:
        return []

@st.cache_data(ttl=1800)
def get_trending_stocks_with_news(df_krx):
    if df_krx.empty: return []
    pool = df_krx[df_krx['Volume'] > 0].sort_values('Volume', ascending=False).head(20)
    candidates = []
    for _, row in pool.iterrows():
        name = str(row['Name'])
        try:
            candidates.append({
                "종목명": name,
                "현재가": int(row['Close']),
                "등락률": round(row['ChgRate'], 2),
                "최신뉴스": fetch_google_news(name, 3)
            })
        except:
            pass
    return candidates

@st.cache_data(ttl=1800)
def get_hidden_gem_stocks(df_krx):
    if df_krx.empty: return []
    pool = df_krx[df_krx['Marcap'] > 0].sort_values('Marcap', ascending=False).head(200)
    vol_median = pool['Volume'].median()
    filtered = pool[(pool['Volume'] <= vol_median) & (pool['ChgRate'].between(-5, 5))]
    if filtered.empty: filtered = pool
    sampled = filtered.sample(n=min(15, len(filtered)))
    candidates = []
    for _, row in sampled.iterrows():
        name = str(row['Name'])
        try:
            candidates.append({
                "종목명": name,
                "현재가": int(row['Close']),
                "등락률": round(row['ChgRate'], 2),
                "최신뉴스": fetch_google_news(name, 3)
            })
        except:
            pass
    return candidates

@st.cache_data(ttl=1800)
def get_us_trending_stocks_with_news():
    pool = ['AAPL','NVDA','MSFT','TSLA','AMZN','GOOGL','META','AMD','NFLX','TSM','SMCI','PLTR','AVGO']
    candidates = []
    for t in pool:
        try:
            hist = yf.Ticker(t).history(period="5d", auto_adjust=False)
            if not hist.empty:
                candidates.append({"종목명": t, "현재가": round(float(hist['Close'].iloc[-1]), 2), "최신뉴스": fetch_google_news(f"{t} stock news", 3)})
        except:
            pass
    return candidates

@st.cache_data(ttl=1800)
def get_us_hidden_gems_with_news():
    pool = ['PYPL','DIS','PFE','NKE','SBUX','BA','INTC','C','WFC','T','VZ','O','QCOM','TXN','CVX']
    candidates = []
    for t in pool:
        try:
            hist = yf.Ticker(t).history(period="5d", auto_adjust=False)
            if not hist.empty:
                candidates.append({"종목명": t, "현재가": round(float(hist['Close'].iloc[-1]), 2), "최신뉴스": fetch_google_news(f"{t} stock news", 3)})
        except:
            pass
    return candidates

def get_peer_group(company_name, df_krx=None):
    name = company_name.replace(" ", "")
    peers_map = {
        "삼성전자": ["SK하이닉스","삼성전기","DB하이텍"],
        "SK하이닉스": ["삼성전자","한미반도체","DB하이텍"],
        "현대차": ["기아","현대모비스","현대로템"],
        "기아": ["현대차","현대모비스","쌍용자동차"],
        "카카오": ["NAVER","카카오게임즈","크래프톤"],
        "NAVER": ["카카오","카카오게임즈","크래프톤"],
        "셀트리온": ["삼성바이오로직스","한미약품","유한양행"],
        "에코프로비엠": ["포스코퓨처엠","코스모신소재","엘앤에프"],
        "AAPL": ["MSFT","GOOGL","AMZN"],
        "NVDA": ["AMD","INTC","AVGO"],
        "TSLA": ["RIVN","LCID","F"],
    }
    for key, peers in peers_map.items():
        if key in name or name in key:
            return peers[:3]
    return []

@st.cache_data(ttl=3600)
def run_backtest(ticker, start_years=3):
    try:
        hist = yf.Ticker(ticker).history(period=f"{start_years}y", auto_adjust=False)
        if hist is None or hist.empty: return None
        df = hist.copy()
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA60'] = df['Close'].rolling(60).mean()
        delta = df['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        df['Signal'] = 0
        df.loc[(df['MA20'] > df['MA60']) & (df['RSI'] < 50), 'Signal'] = 1
        df.loc[(df['MA20'] < df['MA60']) | (df['RSI'] > 70), 'Signal'] = -1
        df['Position']            = df['Signal'].replace(-1, 0).shift()
        df['Market_Return']       = df['Close'].pct_change()
        df['Strategy_Return']     = df['Position'] * df['Market_Return']
        df['Cumulative_Market']   = (1 + df['Market_Return']).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
        return df
    except:
        return None

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
                clean = clean.replace("```json", "").replace("```", "").strip()
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
    with st.spinner(f"'{display_name}' 실시간 퀵 스캔 중..."):
        try:
            hist = yf.Ticker(ticker).history(period="6mo", auto_adjust=False)
            if hist.empty:
                st.error("주가 데이터를 불러오지 못했습니다.")
                return

            curr  = hist['Close'].iloc[-1]
            hist['MA20'] = hist['Close'].rolling(20).mean()
            hist['MA60'] = hist['Close'].rolling(60).mean()
            ma20  = hist['MA20'].iloc[-1] if len(hist) >= 20 else curr
            ma60  = hist['MA60'].iloc[-1] if len(hist) >= 60 else curr
            delta = hist['Close'].diff()
            gain  = delta.where(delta > 0, 0).rolling(14).mean()
            loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rsi   = (100 - (100 / (1 + (gain / loss)))).iloc[-1] if not gain.isna().all() else 50

            per, pbr  = get_fundamentals(ticker, is_us, df_krx)
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

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"📈 {display_name} 6개월 추이")
                st.line_chart(hist[['Close', 'MA20', 'MA60']])
            with col2:
                st.subheader("⚡ 퀵 스캔 리포트")
                action = result.get('action', 'HOLD')
                if action == "BUY":    st.success(f"### 의견: {action} 🟢")
                elif action == "SELL": st.error(f"### 의견: {action} 🔴")
                else:                  st.warning(f"### 의견: {action} 🟡")
                st.markdown(f"**RSI(14):** `{rsi:.2f}` | **PER:** `{per}` | **PBR:** `{pbr}`")
                st.info(f"**AI 코멘트:**\n{result.get('reason', '')}")
                st.caption(f"기준 시각: {called_at}")
        except Exception as e:
            st.error(f"오류: {e}")

# ==========================================
# 4. 메인 UI
# ==========================================

with st.sidebar:
    st.header("⚙️ 터미널 설정")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini API 연동")
    except:
        st.error("⚠️ GEMINI_API_KEY 설정 필요")
        api_key = ""

    dart_key = st.secrets.get("DART_API_KEY", "")
    if dart_key:
        st.success("✅ DART API 연동 (PER/PBR 정확)")
    else:
        st.warning("⚠️ DART_API_KEY 미설정\nopendart.fss.or.kr에서 무료 발급 후 Secrets에 추가하면 PER/PBR이 네이버와 동일해집니다.")

    model_choice = st.selectbox("AI 모델", ("gemini-2.5-flash", "gemini-2.5-pro"))
    st.divider()
    webhook_url = st.text_input("Webhook URL", placeholder="https://...")
    if st.button("테스트 알림"):
        if webhook_url:
            try:
                requests.post(webhook_url, json={"text": "📈 AI 퀀트 터미널 테스트"}, timeout=3)
                st.success("전송 완료!")
            except:
                st.error("전송 실패")

st.title("🤖 AI 글로벌 퀀트 터미널")

indices_data = get_major_indices()
if indices_data:
    cols = st.columns(len(indices_data))
    for i, (name, data) in enumerate(indices_data.items()):
        cols[i].metric(name, f"{data['price']:,.2f}", f"{data['diff']:,.2f} ({data['pct']:+.2f}%)")
st.divider()

with st.spinner("📡 실시간 시장 데이터 수집 중..."):
    df_krx = get_krx_full_data()

if df_krx.empty:
    st.warning("⚠️ 실시간 스크리닝 데이터 수집 중 문제가 발생했습니다. 잠시 후 새로고침 해주세요.")

tab_main, tab_search, tab_recommend, tab_backtest, tab_chat = st.tabs([
    "🌐 실시간 시장 랭킹", "🔍 딥다이브 분석 & Peer 비교",
    "🏆 AI 주도주 & 숨은 보석 추천", "⏳ 알고리즘 백테스팅", "💬 전담 AI 애널리스트 챗봇"
])

with tab_main:
    if df_krx.empty:
        st.error("🚨 데이터를 불러오지 못했습니다. 새로고침 해주세요.")
    else:
        st.caption(f"✅ 총 {len(df_krx)}개 종목 실시간 수집 완료")
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
        if not ticker:
            st.error("종목을 찾을 수 없습니다.")
        else:
            st.session_state.current_stock = display_name
            with st.spinner(f"'{display_name}' 데이터 수집 중..."):
                hist = yf.Ticker(ticker).history(period="6mo", auto_adjust=False)
                if not hist.empty:
                    curr      = hist['Close'].iloc[-1]
                    per, pbr  = get_fundamentals(ticker, is_us, df_krx)
                    macro     = get_macro_data()
                    news_list = fetch_google_news(display_name, 5)
                    news_text = "\n".join([f"- {n}" for n in news_list]) if news_list else "없음"
                    peers     = get_peer_group(display_name, df_krx)
                    peer_data = []
                    for p in peers:
                        pt, pu, pn = get_ticker_from_name(p, df_krx)
                        if pt:
                            pp, pb = get_fundamentals(pt, pu, df_krx)
                            peer_data.append({"기업명": pn, "PER": pp, "PBR": pb})
                    sym       = "$" if is_us else "₩"
                    price_fmt = f"{sym}{curr:,.2f}" if is_us else f"{sym}{int(curr):,}"

                    report_text, _ = get_ai_response(
                        f"**[타깃]** {display_name} (현재가:{price_fmt}, PER:{per}, PBR:{pbr})\n**[Peer]** {json.dumps(peer_data, ensure_ascii=False)}\n**[매크로]** 환율:{macro.get('USD_KRW')}, 미10년물:{macro.get('US_10Y')}%, VIX:{macro.get('VIX')}\n**[뉴스]** {news_text}\n\n1.비즈니스모델 2.밸류에이션 3.투자의견 및 목표가를 마크다운으로 작성하라.",
                        api_key, model_choice,
                        "당신은 월스트리트 수석 애널리스트입니다. 마크다운으로 작성하세요.",
                        is_json=False
                    )
                    if report_text:
                        st.success(f"✅ {display_name} 리포트 완료")
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
    with col_a: market_choice   = st.radio("시장:", ["🇰🇷 국내 증시", "🇺🇸 미국 증시"], horizontal=True)
    with col_b: strategy_choice = st.radio("전략:", ["🔥 시장 주도주", "💎 숨은 보석 발굴"], horizontal=True)

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

            if not data:
                st.error("❌ 데이터 수집 실패.")
            else:
                recs, _ = get_ai_response(
                    f"후보 종목 실시간 데이터: {json.dumps(data, ensure_ascii=False)}\n\n{'저평가 가치주 상위 5개' if is_gem else '모멘텀 강한 주도주 상위 5개'} 추천.",
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
                            raw_p = str(rec.get('current_price', '0')).replace(',', '').replace('원', '').replace('$', '').replace(' ', '')
                            try:    fmt_p = f"${float(raw_p):,.2f}" if is_us_r else f"₩{int(float(raw_p)):,}"
                            except: fmt_p = f"{currency}{raw_p}"
                            st.metric("현재가", fmt_p)
                            st.markdown(f"**📰 요약:** {rec.get('sentiment')}")
                            st.write(f"**💡 사유:** {rec.get('reason')}")

with tab_backtest:
    st.header("⏳ 퀀트 알고리즘 백테스팅")
    col1, col2 = st.columns([1, 3])
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
                    bt_df = run_backtest(t_ticker, test_years)
                    if bt_df is not None and not bt_df.empty:
                        mr = (bt_df['Cumulative_Market'].iloc[-1] - 1) * 100
                        sr = (bt_df['Cumulative_Strategy'].iloc[-1] - 1) * 100
                        st.subheader(f"📊 {t_name} {test_years}년 시뮬레이션")
                        c1, c2 = st.columns(2)
                        c1.metric("Buy & Hold", f"{mr:.2f}%")
                        c2.metric("전략 수익률", f"{sr:.2f}%", delta=f"{sr-mr:.2f}% 초과수익")
                        st.line_chart(bt_df[['Cumulative_Market', 'Cumulative_Strategy']])
            else:
                st.error("정확한 종목명을 입력하세요.")

with tab_chat:
    st.header("💬 전담 AI 애널리스트 채팅")
    current_focus = st.session_state.get('current_stock', '없음')
    if current_focus != "없음": st.info(f"💡 현재 분석 종목: **'{current_focus}'**")

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])

    if prompt := st.chat_input("질문을 입력하세요..."):
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        history = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:]])
        resp, _ = get_ai_response(
            f"분석 중 종목: {current_focus}\n[대화]\n{history}\n[질문] {prompt}\n전문 애널리스트 톤으로 답변해.",
            api_key, model_choice, "당신은 퀀트 애널리스트입니다.", is_json=False
        )
        if resp:
            with st.chat_message("assistant"): st.markdown(resp)
            st.session_state.chat_history.append({"role": "assistant", "content": resp})
