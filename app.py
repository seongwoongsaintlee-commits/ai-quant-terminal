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
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

# ==========================================
# 공통 요청 헤더 및 변수 세팅
# ==========================================
NAVER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    'Connection': 'keep-alive',
    'Referer': 'https://finance.naver.com/',
}

US_STOCKS = {"애플": "AAPL", "엔비디아": "NVDA", "테슬라": "TSLA", "마이크로소프트": "MSFT"}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "안녕하세요! 30년 경력의 수석 애널리스트입니다. 방금 분석하신 종목이나 현재 시장 상황에 대해 무엇이든 물어보세요."}]
if "current_stock" not in st.session_state:
    st.session_state.current_stock = "없음"

# ==========================================
# 1. 크롤링 및 유틸리티 함수 (KRX 우회)
# ==========================================

# 🔥 핵심 로직: KRX 차단을 무시하는 네이버 자동완성 API 기반 실시간 종목 매핑
@st.cache_data(ttl=86400)
def get_ticker_from_naver(keyword):
    keyword = keyword.strip()
    if keyword in US_STOCKS: 
        return US_STOCKS[keyword], True, keyword
    # 영어로만 이루어진 입력은 미국 티커로 간주
    if re.match(r'^[A-Za-z]+$', keyword): 
        return keyword.upper(), True, keyword.upper()

    # 네이버 금융 자동완성 API를 호출하여 즉시 코드 변환
    url = f"https://ac.finance.naver.com/ac?q={urllib.parse.quote(keyword)}&q_enc=euc-kr&st=111&r_format=json&r_enc=euc-kr"
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        items = res.json().get('items', [[]])[0]
        if items:
            for item in items:
                name, code, market = item[0], item[1], item[2]
                if keyword.replace(" ", "").upper() in name.replace(" ", "").upper():
                    ticker = f"{code}.KQ" if "KOSDAQ" in market or "코스닥" in market else f"{code}.KS"
                    return ticker, False, name
            
            # 완벽히 일치하는 이름이 없으면 최상단 첫 번째 검색 결과 반환
            name, code, market = items[0][0], items[0][1], items[0][2]
            ticker = f"{code}.KQ" if "KOSDAQ" in market or "코스닥" in market else f"{code}.KS"
            return ticker, False, name
    except Exception as e:
        print(f"네이버 맵핑 오류: {e}")
        pass
    
    return None, False, keyword

@st.cache_data(ttl=60)
def get_naver_list(url, price_idx, rate_idx):
    try:
        res = requests.get(url, headers=NAVER_HEADERS, timeout=5)
        res.encoding = 'euc-kr'
        soup = BeautifulSoup(res.text, 'lxml') 
        data = []
        for a_tag in soup.find_all('a', {'class': 'tltle'}):
            row = a_tag.find_parent('tr')
            if row:
                cols = row.find_all('td')
                if len(cols) > max(price_idx, rate_idx):
                    name  = a_tag.text.strip()
                    price = cols[price_idx].text.strip()
                    rate  = cols[rate_idx].text.strip()
                    if name and price:
                        data.append({"종목명": name, "현재가": price, "등락률": rate})
                        if len(data) == 30:
                            break
        return pd.DataFrame(data)
    except Exception as e:
        return pd.DataFrame()

def load_market_data():
    return {
        "kospi_cap":  get_naver_list("https://finance.naver.com/sise/sise_market_sum.naver?sosok=0", 2, 4),
        "kosdaq_cap": get_naver_list("https://finance.naver.com/sise/sise_market_sum.naver?sosok=1", 2, 4),
        "search_top": get_naver_list("https://finance.naver.com/sise/lastsearch2.naver", 3, 5),
        "volume_top": get_naver_list("https://finance.naver.com/sise/sise_quant.naver", 2, 4),
        "price_up":   get_naver_list("https://finance.naver.com/sise/sise_rise.naver", 2, 4),
        "price_down": get_naver_list("https://finance.naver.com/sise/sise_fall.naver", 2, 4),
    }

@st.cache_data(ttl=300)
def get_major_indices():
    res = {}
    try:
        r = requests.get("https://finance.naver.com/", headers=NAVER_HEADERS, timeout=5)
        r.encoding = 'euc-kr'
        soup = BeautifulSoup(r.text, 'lxml')
        for name, css_class in [("KOSPI", ".kospi_area"), ("KOSDAQ", ".kosdaq_area")]:
            box = soup.select_one(css_class)
            if box:
                num_tag = box.select_one('.num')
                num2_tag = box.select_one('.num2')
                num3_tag = box.select_one('.num3')
                price = float(num_tag.text.strip().replace(',', '')) if num_tag else 0.0
                diff = float(num2_tag.text.strip().replace(',', '')) if num2_tag else 0.0
                pct_text = num3_tag.text.strip().replace('%', '') if num3_tag else "0"
                
                state_tag = box.select_one('.blind')
                is_down = False
                if state_tag and "하락" in state_tag.text: is_down = True
                if num2_tag and "txt_down" in "".join(num2_tag.get('class', [])): is_down = True
                    
                if is_down:
                    diff = -abs(diff)
                    if not pct_text.startswith("-"): pct_text = "-" + pct_text
                try: pct = float(pct_text.replace('+', ''))
                except: pct = 0.0
                res[name] = {"price": price, "diff": diff, "pct": pct}
    except Exception:
        res["KOSPI"] = {"price": 0.0, "diff": 0.0, "pct": 0.0}
        res["KOSDAQ"] = {"price": 0.0, "diff": 0.0, "pct": 0.0}
        
    us_indices = {"NASDAQ": "^IXIC", "S&P 500": "^GSPC"}
    for name, ticker in us_indices.items():
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if len(hist) >= 2:
                curr = hist['Close'].iloc[-1]
                prev = hist['Close'].iloc[-2]
                res[name] = {"price": curr, "diff": curr - prev, "pct": ((curr - prev) / prev) * 100}
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
        except: data[key] = "N/A"
    return data

def fetch_google_news(keyword, limit=5):
    try:
        encoded_keyword = urllib.parse.quote(keyword)
        url = f"https://news.google.com/rss/search?q={encoded_keyword}&hl=ko&gl=KR&ceid=KR:ko"
        feed = feedparser.parse(url)
        return [entry.title for entry in feed.entries[:limit]]
    except: return []

@st.cache_data(ttl=1800)
def get_trending_stocks_with_news():
    m_data = load_market_data()
    search_top = m_data.get('search_top', pd.DataFrame())
    volume_top = m_data.get('volume_top', pd.DataFrame())
    
    if search_top.empty and volume_top.empty: return []
    pool = pd.concat([search_top, volume_top]).drop_duplicates(subset=['종목명']).head(15)
    candidates = []
    
    for _, row in pool.iterrows():
        name = row['종목명']
        ticker, is_us, mapped_name = get_ticker_from_naver(name)
        if not ticker: continue
        
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                curr_price = int(hist['Close'].iloc[-1])
                candidates.append({"종목명": name, "현재가": curr_price, "최신뉴스": fetch_google_news(name, limit=3)})
        except: pass
    return candidates

@st.cache_data(ttl=1800)
def get_us_trending_stocks_with_news():
    us_pool = ['AAPL', 'NVDA', 'MSFT', 'TSLA', 'AMZN', 'GOOGL', 'META', 'AMD', 'NFLX', 'TSM', 'SMCI', 'PLTR', 'AVGO']
    candidates = []
    for ticker in us_pool:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                curr_price = float(hist['Close'].iloc[-1])
                candidates.append({"종목명": ticker, "현재가": round(curr_price, 2), "최신뉴스": fetch_google_news(f"{ticker} 주식", limit=3)})
        except: pass
    return candidates

@st.cache_data(ttl=1800)
def get_hidden_gem_stocks():
    m_data = load_market_data()
    noisy_stocks = set()
    if not m_data.get('search_top', pd.DataFrame()).empty:
        noisy_stocks.update(m_data['search_top']['종목명'].tolist())
    if not m_data.get('volume_top', pd.DataFrame()).empty:
        noisy_stocks.update(m_data['volume_top']['종목명'].tolist())
        
    kospi_cap = m_data.get('kospi_cap', pd.DataFrame())
    kosdaq_cap = m_data.get('kosdaq_cap', pd.DataFrame())
    pool = pd.concat([kospi_cap, kosdaq_cap]).drop_duplicates(subset=['종목명'])
    
    quiet_pool = pool[~pool['종목명'].isin(noisy_stocks)].head(40)
    sampled_pool = quiet_pool.sample(n=min(15, len(quiet_pool))) if not quiet_pool.empty else quiet_pool
    
    candidates = []
    for _, row in sampled_pool.iterrows():
        name = row['종목명']
        ticker, is_us, mapped_name = get_ticker_from_naver(name)
        if not ticker: continue
        
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                curr_price = int(hist['Close'].iloc[-1])
                candidates.append({"종목명": name, "현재가": curr_price, "최신뉴스": fetch_google_news(name, limit=3)})
        except: pass
    return candidates

@st.cache_data(ttl=1800)
def get_us_hidden_gems_with_news():
    us_value_pool = ['PYPL', 'DIS', 'PFE', 'NKE', 'SBUX', 'BA', 'INTC', 'C', 'WFC', 'T', 'VZ', 'O', 'QCOM', 'TXN', 'CVX']
    candidates = []
    for ticker in us_value_pool:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                curr_price = float(hist['Close'].iloc[-1])
                candidates.append({"종목명": ticker, "현재가": round(curr_price, 2), "최신뉴스": fetch_google_news(f"{ticker} stock", limit=3)})
        except: pass
    return candidates

def get_peer_group(company_name):
    name = company_name.replace(" ", "")
    if "엔비티" in name or "노바렉스" in name or "서흥" in name or "콜마비앤에이치" in name or "에이치엘사이언스" in name: peers = ["노바렉스", "서흥", "콜마비앤에이치"]
    elif name == "코스맥스" or "한국콜마" in name or "씨앤씨인터내셔널" in name or "코스메카코리아" in name: peers = ["한국콜마", "씨앤씨인터내셔널", "코스메카코리아"]
    elif "삼성전자" in name or "SK하이닉스" in name: peers = ["SK하이닉스", "한미반도체", "DB하이텍"]
    elif "AAPL" in name or "MSFT" in name or "GOOGL" in name: peers = ["MSFT", "GOOGL", "AMZN"]
    elif "TSLA" in name: peers = ["RIVN", "LCID", "F"]
    else: peers = []
    return [p for p in peers if p != company_name][:3]

def get_fundamentals(ticker, is_us_stock):
    per, pbr = "N/A", "N/A"
    if is_us_stock:
        try:
            info = yf.Ticker(ticker).info
            if info.get('trailingPE'): per = f"{info['trailingPE']:.2f}배"
            if info.get('priceToBook'): pbr = f"{info['priceToBook']:.2f}배"
        except: pass
    else:
        try:
            raw_code = ticker.split('.')[0]
            url = f"https://finance.naver.com/item/main.naver?code={raw_code}"
            res = requests.get(url, headers=NAVER_HEADERS, timeout=5)
            soup = BeautifulSoup(res.text, 'lxml')
            per_tag = soup.select_one('#_per')
            pbr_tag = soup.select_one('#_pbr')
            if per_tag and per_tag.text.strip(): per = per_tag.text.strip() + "배"
            if pbr_tag and pbr_tag.text.strip(): pbr = pbr_tag.text.strip() + "배"
        except: pass
    return per, pbr

@st.cache_data(ttl=3600)
def run_backtest(ticker, start_years=3):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * start_years)
        df = yf.Ticker(ticker).history(start=start_date, end=end_date)
        if df.empty: return None
        
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA60'] = df['Close'].rolling(window=60).mean()
        
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df['RSI'] = 100 - (100 / (1 + (gain / loss)))
        
        df['Signal'] = 0
        df.loc[(df['MA20'] > df['MA60']) & (df['RSI'] < 50), 'Signal'] = 1
        df.loc[(df['MA20'] < df['MA60']) | (df['RSI'] > 70), 'Signal'] = -1
        
        df['Position'] = df['Signal'].replace(-1, 0).shift()
        df['Market_Return'] = df['Close'].pct_change()
        df['Strategy_Return'] = df['Position'] * df['Market_Return']
        
        df['Cumulative_Market'] = (1 + df['Market_Return']).cumprod()
        df['Cumulative_Strategy'] = (1 + df['Strategy_Return']).cumprod()
        return df
    except: return None

@st.cache_data(ttl=600, show_spinner=False)
def get_ai_response(prompt_text, api_key, model_choice, instruction_text, is_json=True):
    try:
        genai.configure(api_key=api_key)
        config = {"temperature": 0.2}
        if is_json: config["response_mime_type"] = "application/json"
            
        model = genai.GenerativeModel(
            model_name=model_choice,
            system_instruction=instruction_text,
            generation_config=config,
        )
        response = model.generate_content(prompt_text)
        clean_text = response.text.strip()
        called_at = time.strftime("%H:%M:%S")
        
        if is_json:
            ticks = "`" * 3
            if clean_text.startswith(ticks):
                clean_text = clean_text.replace(ticks + "json", "").replace(ticks, "").strip()
            return json.loads(clean_text), called_at
        else:
            return clean_text, called_at
    except Exception as e:
        st.error(f"⚠️ AI 연산 오류: {e}")
        return None, None

def run_quick_analysis(company_name, api_key, model_choice):
    ticker, is_us_stock, display_name = get_ticker_from_naver(company_name)
    if not ticker:
        st.error(f"'{company_name}' 종목의 코드를 찾지 못했습니다. 네이버 금융에서 검색되는 정확한 이름을 입력해 주세요.")
        return

    st.session_state.current_stock = display_name

    with st.spinner(f"'{display_name}' 실시간 퀵 스캔 중..."):
        try:
            stock_data = yf.Ticker(ticker)
            hist = stock_data.history(period="6mo")
            if hist.empty:
                st.error("주가 데이터를 불러오지 못했습니다.")
                return

            current_price = hist['Close'].iloc[-1]
            hist['MA20'] = hist['Close'].rolling(window=20).mean()
            hist['MA60'] = hist['Close'].rolling(window=60).mean()
            ma20 = hist['MA20'].iloc[-1] if len(hist) >= 20 else current_price
            ma60 = hist['MA60'].iloc[-1] if len(hist) >= 60 else current_price

            delta = hist['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            current_rsi = (100 - (100 / (1 + (gain / loss)))).iloc[-1] if not gain.isna().all() else 50

            per, pbr = get_fundamentals(ticker, is_us_stock)
            news_list = fetch_google_news(display_name, limit=3)
            news_text = "\n".join([f"- {news}" for news in news_list]) if news_list else "최근 주요 뉴스 없음"

            currency_sym = "$" if is_us_stock else "₩"
            price_fmt = f"{currency_sym}{current_price:,.2f}" if is_us_stock else f"{currency_sym}{int(current_price):,}"

            instruction = "당신은 월스트리트 퀀트 애널리스트입니다. JSON 형식({'action':'BUY/SELL/HOLD', 'reason':'직관적인 분석 코멘트'})으로만 응답하세요."
            prompt = f"[{display_name}] 현재가:{price_fmt}, RSI:{current_rsi:.2f}, 20일선:{ma20:.2f}, 60일선:{ma60:.2f}, PER:{per}, PBR:{pbr}, 뉴스:{news_text}. 위 데이터를 바탕으로 트레이딩 관점의 투자 의견과 짧은 이유를 제시하세요."
            
            result, called_at = get_ai_response(prompt, api_key, model_choice, instruction, is_json=True)
            if result is None: return

            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader(f"📈 {display_name} 6개월 추이")
                st.line_chart(hist[['Close', 'MA20', 'MA60']])
            with col2:
                st.subheader("⚡ 퀵 스캔 리포트")
                action = result.get('action', 'HOLD')
                if action == "BUY": st.success(f"### 의견: {action} 🟢")
                elif action == "SELL": st.error(f"### 의견: {action} 🔴")
                else: st.warning(f"### 의견: {action} 🟡")
                
                st.markdown(f"**RSI(14):** `{current_rsi:.2f}` | **PER:** `{per}` | **PBR:** `{pbr}`")
                st.info(f"**AI 분석 코멘트:**\n{result.get('reason', '')}")
                st.caption(f"기준 시각: {called_at}")
        except Exception as e:
            st.error(f"데이터 연동 중 오류가 발생했습니다.\n에러: {e}")

# ==========================================
# 2. 메인 UI 및 설정
# ==========================================

with st.sidebar:
    st.header("⚙️ 터미널 설정")
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ 클라우드 보안 키 자동 연동 완료")
    except:
        st.error("⚠️ Secrets 설정이 필요합니다.")
        api_key = "" 
        
    model_choice = st.selectbox("사용할 AI 모델", ("gemini-3.5-flash", "gemini-3.5-pro"))
    
    st.divider()
    st.header("🔗 자동화 워크플로우")
    webhook_url = st.text_input("Webhook URL (Apps Script / Slack)", placeholder="https://script.google.com/macros/s/...")
    if st.button("테스트 알림 발송"):
        if webhook_url:
            try:
                requests.post(webhook_url, json={"text": "📈 AI 퀀트 터미널: 실시간 자동화 연동 테스트 완료."}, timeout=3)
                st.success("웹훅 전송 완료!")
            except: st.error("웹훅 전송 실패. URL을 확인하세요.")
        else: st.warning("URL을 입력하세요.")

st.title("🤖 AI 글로벌 퀀트 터미널")

indices_data = get_major_indices()
if indices_data:
    cols = st.columns(len(indices_data))
    for i, (name, data) in enumerate(indices_data.items()):
        cols[i].metric(label=name, value=f"{data['price']:,.2f}", delta=f"{data['diff']:,.2f} ({data['pct']:+.2f}%)")
st.divider()

tab_main, tab_search, tab_recommend, tab_backtest, tab_chat = st.tabs([
    "🌐 실시간 시장 랭킹", 
    "🔍 딥다이브 분석 & Peer 비교", 
    "🏆 AI 주도주 & 숨은 보석 추천", 
    "⏳ 알고리즘 백테스팅",
    "💬 전담 AI 애널리스트 챗봇"
])

with tab_main:
    m_data = load_market_data()
    if all(df.empty for df in m_data.values()):
        st.error("🚨 네이버 금융 서버 접속이 지연되고 있습니다. 잠시 후 새로고침 해주세요.")
    else:
        t1, t2, t3, t4, t5, t6 = st.tabs(["👑 KOSPI 시총상위", "👑 KOSDAQ 시총상위", "🔥 검색량 상위", "🌊 거래량 상위", "🚀 상승률 상위", "📉 하락률 상위"])
        events = {}
        with t1: events['kospi_cap']  = st.dataframe(m_data['kospi_cap'],  use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t1")
        with t2: events['kosdaq_cap'] = st.dataframe(m_data['kosdaq_cap'], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t2")
        with t3: events['search_top'] = st.dataframe(m_data['search_top'], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t3")
        with t4: events['volume_top'] = st.dataframe(m_data['volume_top'], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t4")
        with t5: events['price_up']   = st.dataframe(m_data['price_up'],   use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t5")
        with t6: events['price_down'] = st.dataframe(m_data['price_down'], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row", key="t6")

        selected_stock = None
        for category_key, event in events.items():
            if len(event.selection.rows) > 0:
                selected_stock = m_data[category_key].iloc[event.selection.rows[0]]['종목명']
                break

        if selected_stock:
            st.divider()
            st.markdown(f"### ⚡ **{selected_stock}** 실시간 퀵 스캔")
            run_quick_analysis(selected_stock, api_key, model_choice)

with tab_search:
    st.header("기관 투자자용 심층 종목 분석")
    company_name = st.text_input("검색할 종목명 또는 티커", value="삼성전자") 
    analyze_btn  = st.button("심층 펀더멘털 분석 시작")
    
    if analyze_btn and company_name:
        ticker, is_us_stock, display_name = get_ticker_from_naver(company_name)
        if not ticker: st.error("종목을 찾을 수 없습니다. 정확한 종목명을 입력하세요.")
        else:
            st.session_state.current_stock = display_name
            with st.spinner(f"'{display_name}' 및 경쟁사(Peer) 데이터를 수집 중입니다..."):
                stock_data = yf.Ticker(ticker)
                hist = stock_data.history(period="6mo")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    per, pbr = get_fundamentals(ticker, is_us_stock)
                    macro = get_macro_data()
                    news_list = fetch_google_news(display_name, limit=5)
                    news_text = "\n".join([f"- {news}" for news in news_list]) if news_list else "최근 주요 뉴스 없음"

                    peers = get_peer_group(display_name)
                    peer_data = []
                    for p in peers:
                        p_ticker, p_is_us, p_name = get_ticker_from_naver(p)
                        if p_ticker:
                            p_per, p_pbr = get_fundamentals(p_ticker, p_is_us)
                            peer_data.append({"기업명": p_name, "PER": p_per, "PBR": p_pbr})

                    currency_sym = "$" if is_us_stock else "₩"
                    price_fmt = f"{currency_sym}{current_price:,.2f}" if is_us_stock else f"{currency_sym}{int(current_price):,}"

                    instruction = "당신은 월스트리트 수석 애널리스트입니다. 응답은 마크다운 포맷으로 작성하세요."
                    prompt = f"""
**[타깃 기업]** {display_name} (현재가: {price_fmt}, PER: {per}, PBR: {pbr})
**[Peer 데이터]** {json.dumps(peer_data, ensure_ascii=False)}
**[실시간 최신 뉴스]** {news_text}

위 데이터를 바탕으로: 1. 비즈니스 모델 분석 2. 밸류에이션 매력도 3. 투자 의견(Buy/Hold/Sell) 및 목표가, 근거를 3줄 요약하여 작성하라.
"""
                    report_text, called_at = get_ai_response(prompt, api_key, model_choice, instruction, is_json=False)
                    if report_text:
                        st.success(f"✅ {display_name} 심층 리포트 발간 완료")
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.subheader("📈 기본 가치 및 피어(Peer) 비교")
                            st.metric(f"{display_name} PER", per)
                            if peer_data: st.dataframe(pd.DataFrame(peer_data), hide_index=True)
                            st.line_chart(hist['Close'])
                        with col2:
                            st.subheader("🧠 딥다이브 퀀트 리포트")
                            st.markdown(report_text)

with tab_recommend:
    st.header("🏆 듀얼 엔진 주식 스크리너")
    col_a, col_b = st.columns(2)
    with col_a: market_choice = st.radio("분석할 시장:", ["🇰🇷 국내 증시", "🇺🇸 미국 증시"], horizontal=True)
    with col_b: strategy_choice = st.radio("투자 전략:", ["🔥 시장 주도주", "💎 숨은 보석 발굴"], horizontal=True)

    if st.button("🚀 실시간 추천받기", use_container_width=True):
        with st.spinner("시장 데이터 스크리닝 및 AI 필터 가동 중..."):
            is_us_recom = "미국" in market_choice
            is_hidden_gem = "숨은 보석" in strategy_choice
            
            if is_us_recom:
                trending_data = get_us_hidden_gems_with_news() if is_hidden_gem else get_us_trending_stocks_with_news()
                currency = "$"
            else:
                trending_data = get_hidden_gem_stocks() if is_hidden_gem else get_trending_stocks_with_news()
                currency = "₩"

            if not trending_data: st.error("❌ 시장 데이터를 가져오지 못했습니다.")
            else:
                instruction = "당신은 탁월한 퀀트 매니저입니다. [{'stock':'종목명','current_price':'숫자만','sentiment':'진짜 호재 요약','reason':'추천 사유'}] 형식의 JSON 배열로 응답하세요."
                strategy_prompt = "저평가 턴어라운드 가치주(Hidden Gem) 상위 5개를 추천해" if is_hidden_gem else "모멘텀이 강력한 유망 주도주 상위 5개를 추천해"
                prompt = f"후보 리스트 및 뉴스: {json.dumps(trending_data, ensure_ascii=False)}\n\n단순 찌라시를 거르고 진짜 펀더멘털 개선이 기대되는 종목으로 {strategy_prompt}."

                try:
                    recommendations, _ = get_ai_response(prompt, api_key, model_choice, instruction, is_json=True)
                    if recommendations:
                        st.success("🎉 AI 데스킹 완료! 최종 추천 리스트입니다.")
                        cols = st.columns(len(recommendations))
                        for idx, rec in enumerate(recommendations):
                            with cols[idx]:
                                st.info(f"### 🥇 TOP {idx + 1}\n**{rec.get('stock')}**")
                                raw_price = str(rec.get('current_price', '0')).replace(',', '').replace('원', '').replace('$', '').replace(' ', '')
                                try:
                                    fmt_price = f"${float(raw_price):,.2f}" if is_us_recom else f"₩{int(float(raw_price)):,}"
                                except: fmt_price = f"{currency}{raw_price}"
                                st.metric("현재가", fmt_price)
                                st.markdown(f"**📰 요약:** {rec.get('sentiment')}")
                                st.write(f"**💡 사유:** {rec.get('reason')}")
                except Exception as e: st.error(f"오류: {e}")

with tab_backtest:
    st.header("⏳ 퀀트 알고리즘 과거 수익률 백테스팅")
    col1, col2 = st.columns([1, 3])
    with col1:
        test_ticker_input = st.text_input("백테스트 종목명", value="삼성전자")
        test_years = st.slider("테스트 기간 (년)", 1, 5, 3)
        run_bt = st.button("백테스트 시작")
    with col2:
        if run_bt and test_ticker_input:
            t_code, _, t_name = get_ticker_from_naver(test_ticker_input)
            if t_code:
                st.session_state.current_stock = t_name 
                with st.spinner("시뮬레이션 중..."):
                    bt_df = run_backtest(t_code, test_years)
                    if bt_df is not None and not bt_df.empty:
                        market_return = (bt_df['Cumulative_Market'].iloc[-1] - 1) * 100
                        strat_return = (bt_df['Cumulative_Strategy'].iloc[-1] - 1) * 100
                        st.subheader(f"📊 {t_name} 과거 {test_years}년 시뮬레이션 결과")
                        c1, c2 = st.columns(2)
                        c1.metric("단순 보유 (Buy & Hold)", f"{market_return:.2f}%")
                        c2.metric("전략 매매 수익률", f"{strat_return:.2f}%", delta=f"{strat_return - market_return:.2f}% (초과수익)")
                        st.line_chart(bt_df[['Cumulative_Market', 'Cumulative_Strategy']])
            else: st.error("정확한 종목을 입력하세요.")

with tab_chat:
    st.header("💬 전담 AI 애널리스트 실시간 채팅")
    current_focus = st.session_state.get('current_stock', '없음')
    if current_focus != "없음": st.info(f"💡 AI가 분석 중인 종목: **'{current_focus}'**")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("질문을 입력하세요..."):
        with st.chat_message("user"): st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:]])
        context_prompt = f"현재 분석 중 종목: {current_focus}\n[대화 기록]\n{history_text}\n[새 질문] {prompt}\n\n전문적인 애널리스트 톤으로 대답해."
        
        with st.spinner("답변 작성 중..."):
            response_text, _ = get_ai_response(context_prompt, api_key, model_choice, "당신은 퀀트 애널리스트입니다.", is_json=False)
            if response_text:
                with st.chat_message("assistant"): st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
