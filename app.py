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
import FinanceDataReader as fdr

# ==========================================
# 공통 요청 헤더 및 세션 초기화
# ==========================================
NAVER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7',
    'Connection': 'keep-alive',
    'Referer': 'https://finance.naver.com/',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'same-origin',
}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "assistant", "content": "안녕하세요! 30년 경력의 수석 애널리스트입니다. 방금 분석하신 종목이나 현재 시장 상황에 대해 무엇이든 물어보세요."}]
if "current_stock" not in st.session_state:
    st.session_state.current_stock = "없음"

# ==========================================
# 1. 크롤링 및 유틸리티 함수
# ==========================================

# 💡 함수 이름을 완전히 바꿔서 예전에 꼬여있던 캐시(저장소)를 강제로 무효화합니다!
@st.cache_data(ttl=86400)
def fetch_korean_tickers():
    try:
        # 1차 시도: FDR을 사용하여 KOSPI, KOSDAQ 개별 호출 (단일 호출보다 에러 확률이 낮음)
        df_kospi = fdr.StockListing('KOSPI')
        df_kosdaq = fdr.StockListing('KOSDAQ')
        df = pd.concat([df_kospi, df_kosdaq])
        
        if df.empty:
            raise ValueError("데이터가 비어있습니다.")
            
        code_dict = dict(zip(df['Name'], df['Code']))
        market_dict = dict(zip(df['Name'], df['Market']))
        return code_dict, market_dict
    except Exception as e1:
        # 2차 시도: 1차가 실패하면 한국거래소(KIND) 웹사이트 우회 크롤링으로 백업
        try:
            url = 'http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13'
            res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
            res.encoding = 'cp949'
            df_kind = pd.read_html(res.text, header=0)[0]
            df_kind['종목코드'] = df_kind['종목코드'].astype(str).str.zfill(6)
            
            code_dict = dict(zip(df_kind['회사명'], df_kind['종목코드']))
            market_col = '시장구분' if '시장구분' in df_kind.columns else '시장'
            market_dict = dict(zip(df_kind['회사명'], df_kind.get(market_col, '유가증권시장')))
            return code_dict, market_dict
        except Exception as e2:
            # 💡 2차까지 다 실패하면 빈 데이터를 캐시하지 않도록 아예 None을 반환! (다음번 새로고침 시 다시 시도하게 함)
            return None, None

def get_ticker_from_name(name, stock_codes, market_dict, us_stocks):
    name = name.strip()
    if name in us_stocks: return us_stocks[name], True, name
    if re.match(r'^[A-Za-z]+$', name): return name.upper(), True, name.upper()
    
    matched_name = name
    if name not in stock_codes:
        clean_input = name.replace(" ", "").upper()
        for k in stock_codes.keys():
            if clean_input in k.replace(" ", "").upper():
                matched_name = k
                break
                
    if matched_name in stock_codes:
        raw_code = stock_codes[matched_name]
        market = str(market_dict.get(matched_name, "유가증권시장")).upper()
        ticker = f"{raw_code}.KQ" if ("KOSDAQ" in market or "코스닥" in market) else f"{raw_code}.KS"
        return ticker, False, matched_name
        
    return None, False, name

@st.cache_data(ttl=60)
def get_naver_list(url, price_idx, rate_idx):
    try:
        res = requests.get(url, headers=NAVER_HEADERS, timeout=5)
        res.raise_for_status() 
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
        r.raise_for_status()
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
                if state_tag and "하락" in state_tag.text:
                    is_down = True
                if num2_tag and "txt_down" in "".join(num2_tag.get('class', [])):
                    is_down = True
                    
                if is_down:
                    diff = -abs(diff)
                    if not pct_text.startswith("-"):
                        pct_text = "-" + pct_text
                
                try: pct = float(pct_text.replace('+', ''))
                except: pct = 0.0
                
                res[name] = {"price": price, "diff": diff, "pct": pct}
    except Exception as e:
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
    except:
        return []

@st.cache_data(ttl=1800)
def get_trending_stocks_with_news(stock_codes_dict, market_dict):
    m_data = load_market_data()
    search_top = m_data.get('search_top', pd.DataFrame())
    volume_top = m_data.get('volume_top', pd.DataFrame())
    
    if search_top.empty and volume_top.empty: return []
    pool = pd.concat([search_top, volume_top]).drop_duplicates(subset=['종목명']).head(15)
    candidates = []
    
    for _, row in pool.iterrows():
        name = row['종목명']
        raw_code = stock_codes_dict.get(name)
        if not raw_code: continue
        market = str(market_dict.get(name, "유가증권시장")).upper()
        ticker = f"{raw_code}.KQ" if ("KOSDAQ" in market or "코스닥" in market) else f"{raw_code}.KS"
        
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                curr_price = int(hist['Close'].iloc[-1])
                recent_news = fetch_google_news(name, limit=3)
                candidates.append({"종목명": name, "현재가": curr_price, "최신뉴스": recent_news})
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
                recent_news = fetch_google_news(f"{ticker} 주식", limit=3)
                candidates.append({"종목명": ticker, "현재가": round(curr_price, 2), "최신뉴스": recent_news})
        except: pass
    return candidates

@st.cache_data(ttl=1800)
def get_hidden_gem_stocks(stock_codes_dict, market_dict):
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
        raw_code = stock_codes_dict.get(name)
        if not raw_code: continue
        market = str(market_dict.get(name, "유가증권시장")).upper()
        ticker = f"{raw_code}.KQ" if ("KOSDAQ" in market or "코스닥" in market) else f"{raw_code}.KS"
        
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                curr_price = int(hist['Close'].iloc[-1])
                recent_news = fetch_google_news(name, limit=3)
                candidates.append({"종목명": name, "현재가": curr_price, "최신뉴스": recent_news})
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
                recent_news = fetch_google_news(f"{ticker} stock", limit=3)
                candidates.append({"종목명": ticker, "현재가": round(curr_price, 2), "최신뉴스": recent_news})
        except: pass
    return candidates

def get_peer_group(company_name):
    name = company_name.replace(" ", "")
    if "엔비티" in name or "노바렉스" in name or "서흥" in name or "콜마비앤에이치" in name or "에이치엘사이언스" in name:
        peers = ["노바렉스", "서흥", "콜마비앤에이치"]
    elif name == "코스맥스" or "한국콜마" in name or "씨앤씨인터내셔널" in name or "코스메카코리아" in name:
        peers = ["한국콜마", "씨앤씨인터내셔널", "코스메카코리아"]
    elif "삼성전자" in name or "SK하이닉스" in name:
        peers = ["SK하이닉스", "한미반도체", "DB하이텍"]
    elif "AAPL" in name or "MSFT" in name or "GOOGL" in name:
        peers = ["MSFT", "GOOGL", "AMZN"]
    elif "TSLA" in name:
        peers = ["RIVN", "LCID", "F"]
    else:
        peers = []
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
            res.raise_for_status()
            soup = BeautifulSoup(res.text, 'lxml')
            per_tag = soup.select_one('#_per')
            pbr_tag = soup.select_one('#_pbr')
            if per_tag and per_tag.text.strip(): per = per_tag.text.strip() + "배"
            if pbr_tag and pbr_tag.text.strip(): pbr = pbr_tag.text.strip() + "배"
        except Exception: 
            pass
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
    except:
        return None

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

def run_quick_analysis(company_name, api_key, model_choice, stock_codes, market_dict, us_stocks):
    ticker, is_us_stock, display_name = get_ticker_from_name(company_name, stock_codes, market_dict, us_stocks)
    if not ticker:
        st.error(f"'{company_name}' 종목을 찾을 수 없습니다. (종목코드 맵핑 실패)")
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
            if result is None:
                return

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

# 💡 캐시가 꼬이는 걸 방지하는 로직 추가
stock_codes, market_dict = fetch_korean_tickers()
if stock_codes is None:
    st.error("🚨 서버에서 종목 코드를 일시적으로 불러오지 못했습니다. 잠시 후 새로고침(F5)을 눌러주세요.")
    stock_codes, market_dict = {}, {}

us_stocks = {"애플": "AAPL", "엔비디아": "NVDA", "테슬라": "TSLA", "마이크로소프트": "MSFT"}

with st.sidebar:
    st.header("⚙️ 터미널 설정")
    
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ 클라우드 보안 키 자동 연동 완료")
    except:
        st.error("⚠️ Secrets 설정이 필요합니다. (또는 테스트용으로 하드코딩 필요)")
        api_key = "" 
        
    model_choice = st.selectbox("사용할 AI 모델", ("gemini-3.5-flash", "gemini-3.5-pro"))
    
    st.divider()
    st.header("🔗 자동화 워크플로우")
    webhook_url = st.text_input("Webhook URL (Apps Script / Slack)", placeholder="https://script.google.com/macros/s/...")
    if st.button("테스트 알림 발송"):
        if webhook_url:
            try:
                payload = {"text": "📈 AI 퀀트 터미널: 실시간 자동화 연동이 정상적으로 테스트 되었습니다."}
                requests.post(webhook_url, json=payload, timeout=3)
                st.success("웹훅 전송 완료!")
            except:
                st.error("웹훅 전송 실패. URL을 확인하세요.")
        else:
            st.warning("URL을 입력하세요.")

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

# ---------------------------------------------------------
# [탭 1] 실시간 시장 랭킹
# ---------------------------------------------------------
with tab_main:
    m_data = load_market_data()
    
    if all(df.empty for df in m_data.values()):
        st.error("🚨 네이버 금융 등에서 클라우드 서버 접속을 차단(403 Error)하여 실시간 데이터를 불러올 수 없습니다.")
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
            run_quick_analysis(selected_stock, api_key, model_choice, stock_codes, market_dict, us_stocks)

# ---------------------------------------------------------
# [탭 2] 딥다이브 & Peer 비교 분석
# ---------------------------------------------------------
with tab_search:
    st.header("기관 투자자용 심층 종목 분석")
    company_name = st.text_input("검색할 종목명 또는 티커", value="삼성전자") 
    analyze_btn  = st.button("심층 펀더멘털 분석 시작")
    
    if analyze_btn and company_name:
        ticker, is_us_stock, display_name = get_ticker_from_name(company_name, stock_codes, market_dict, us_stocks)
        
        if not ticker:
            st.error(f"'{company_name}' 종목을 찾을 수 없습니다. (종목 리스트 로딩 지연 또는 오타)")
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
                        p_ticker, p_is_us, p_name = get_ticker_from_name(p, stock_codes, market_dict, us_stocks)
                        if p_ticker:
                            p_per, p_pbr = get_fundamentals(p_ticker, p_is_us)
                            peer_data.append({"기업명": p_name, "PER": p_per, "PBR": p_pbr})

                    currency_sym = "$" if is_us_stock else "₩"
                    price_fmt = f"{currency_sym}{current_price:,.2f}" if is_us_stock else f"{currency_sym}{int(current_price):,}"

                    instruction = "당신은 월스트리트 수석 애널리스트입니다. 응답은 마크다운(Markdown) 포맷으로 작성하세요."
                    prompt = f"""
**[역할 부여]**
당신은 연평균 50% 이상의 수익률을 기록 중인 30년 경력의 수석 애널리스트입니다. 

**[분석 대상 기업 및 실시간 데이터]**
*   **타깃 기업:** {display_name} (티커: {ticker})
*   **현재 주가:** {price_fmt}
*   **가치 지표:** PER {per}, PBR {pbr}
*   **동종 업계(Peer) 데이터:** {json.dumps(peer_data, ensure_ascii=False)}
*   **매크로 환경:** 환율 {macro['USD_KRW']}, 미10년물 {macro['US_10Y']}%
*   **실시간 최신 뉴스:** 
{news_text}

**[분석 모듈 지침]**
1. **비즈니스 모델 및 경제적 해자:** 기업의 독점적 지위와 협상력을 분석하라.
2. **정량적 밸류에이션 및 Peer 비교:** 동종 업계 경쟁사(위 Peer 데이터 참조) 대비 현재 {display_name}의 밸류에이션 매력도를 상대평가하라. 
3. **노이즈 필터링 된 촉매제 및 리스크:** 단순 홍보성 기사는 배제하고 실적과 펀더멘털에 직결되는 팩트만 판독하여 분석하라.

**[최종 출력 형식]**
*   **Investment Rating:** [Strong Buy / Buy / Hold / Sell]
*   **Target Price:** 구체적 목표가 및 근거
*   **1-Minute Pitch:** 핵심 투자 아이디어 3줄 요약
"""
                    report_text, called_at = get_ai_response(prompt, api_key, model_choice, instruction, is_json=False)
                    
                    if report_text:
                        st.success(f"✅ {display_name} 기관 투자자용 심층 리포트 발간 완료")
                        
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            st.subheader("📈 기본 가치 및 피어(Peer) 비교")
                            st.metric(f"{display_name} PER", per)
                            if peer_data:
                                st.dataframe(pd.DataFrame(peer_data), hide_index=True)
                            st.line_chart(hist['Close'])
                            with st.expander("📰 필터링 대상 원본 뉴스"):
                                st.write(news_text)

                        with col2:
                            st.subheader("🧠 딥다이브 퀀트 리포트")
                            st.markdown(report_text)
                            st.caption(f"AI 응답 기준 시각: {called_at}")

# ---------------------------------------------------------
# [탭 3] AI 주도주 & 숨은 보석 추천
# ---------------------------------------------------------
with tab_recommend:
    st.header("🏆 듀얼 엔진 주식 스크리너")
    st.write("알맹이 없는 홍보성 기사는 거르고, 실제 실적과 펀더멘털을 움직이는 '진짜 호재'만 골라내는 고도화된 센티먼트 분석을 수행합니다.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        market_choice = st.radio("분석할 시장 선택:", ["🇰🇷 국내 증시", "🇺🇸 미국 증시"], horizontal=True)
    with col_b:
        strategy_choice = st.radio("투자 전략 선택:", ["🔥 시장 주도주 (모멘텀 트레이딩)", "💎 숨은 보석 발굴 (역발상 가치투자)"], horizontal=True)

    if st.button(f"🚀 실시간 {strategy_choice.split(' ')[1]} 추천받기", use_container_width=True):
        with st.spinner("시장 데이터 스크리닝 및 데스킹(Desking) 필터 가동 중..."):
            
            is_us_recom = "미국" in market_choice
            is_hidden_gem = "숨은 보석" in strategy_choice
            
            if is_us_recom:
                trending_data = get_us_hidden_gems_with_news() if is_hidden_gem else get_us_trending_stocks_with_news()
                currency = "$"
            else:
                trending_data = get_hidden_gem_stocks(stock_codes, market_dict) if is_hidden_gem else get_trending_stocks_with_news(stock_codes, market_dict)
                currency = "₩"

            if not trending_data:
                st.error("❌ 데이터 수집에 실패했습니다. 스트림릿 서버 IP가 네이버 금융 등에서 차단되었을 수 있습니다.")
            else:
                instruction = "당신은 시장을 움직이는 기사가 무엇인지 본능적으로 꿰뚫어 보는 탁월한 취재 감각을 지닌 금융 데스크이자 퀀트 매니저입니다. [{\"stock\":\"종목명\",\"current_price\":\"숫자만\",\"sentiment\":\"노이즈를 제거한 진짜 호재 요약\",\"reason\":\"선정 이유\"}] 형식의 JSON 배열로 응답하세요."
                strategy_prompt = "소외되어 있지만 턴어라운드가 기대되는 저평가 가치주(Hidden Gem) 상위 5개를 발굴하세요." if is_hidden_gem else "모멘텀 속에서 가장 강력한 호재를 품은 유망 종목 상위 5개를 추천하세요."
                prompt = f"후보 리스트 및 뉴스:\n{json.dumps(trending_data, ensure_ascii=False)}\n\n[조건]\n단순 MOU, 기부 행사 등 뜬구름 잡는 홍보성 찌라시는 무시하십시오. 실적 증명, 수주, M&A 등 펀더멘털을 바꾸는 진짜 팩트만 호재로 인정하여, {strategy_prompt}"

                try:
                    recommendations, called_at = get_ai_response(prompt, api_key, model_choice, instruction, is_json=True)
                    if recommendations:
                        st.success(f"🎉 데스킹 완료! 노이즈가 제거된 찐(Real) 추천 종목입니다.")
                        cols = st.columns(len(recommendations))
                        for idx, rec in enumerate(recommendations):
                            with cols[idx]:
                                st.info(f"### 🥇 TOP {idx + 1}\n**{rec.get('stock')}**")
                                try:
                                    raw_price = str(rec.get('current_price', '0')).replace(',', '').replace('원', '').replace('$', '').replace(' ', '')
                                    if is_us_recom: fmt_price = f"${float(raw_price):,.2f}"
                                    else: fmt_price = f"₩{int(float(raw_price)):,}"
                                except:
                                    fmt_price = f"{currency}{rec.get('current_price', 0)}"
                                        
                                st.metric("현재가", fmt_price)
                                st.markdown(f"**📰 데스크 판독 결과:**\n{rec.get('sentiment')}")
                                st.write(f"**💡 추천 사유:**\n{rec.get('reason')}")
                except Exception as e:
                    st.error(f"오류: {e}")

# ---------------------------------------------------------
# [탭 4] 알고리즘 백테스팅
# ---------------------------------------------------------
with tab_backtest:
    st.header("⏳ 퀀트 알고리즘 과거 수익률 백테스팅")
    st.write("AI가 추천한 종목이 실제로 기술적 매수/매도 로직(MA Crossover + RSI)에서 과거 3년간 어느 정도의 승률을 기록했는지 시뮬레이션합니다.")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        test_ticker_input = st.text_input("백테스트 종목명 또는 티커", value="AAPL")
        test_years = st.slider("테스트 기간 (년)", min_value=1, max_value=5, value=3)
        run_bt = st.button("백테스트 시뮬레이션 가동")
        
    with col2:
        if run_bt and test_ticker_input:
            t_code, _, t_name = get_ticker_from_name(test_ticker_input, stock_codes, market_dict, us_stocks)
            if t_code:
                st.session_state.current_stock = t_name 
                with st.spinner("과거 주가 데이터 연산 및 포지션 매매 시뮬레이션 중..."):
                    bt_df = run_backtest(t_code, test_years)
                    if bt_df is not None and not bt_df.empty:
                        market_return = (bt_df['Cumulative_Market'].iloc[-1] - 1) * 100
                        strat_return = (bt_df['Cumulative_Strategy'].iloc[-1] - 1) * 100
                        
                        st.subheader(f"📊 {t_name} 과거 {test_years}년 시뮬레이션 결과")
                        c1, c2 = st.columns(2)
                        c1.metric("그냥 들고 있었을 때 (Buy & Hold)", f"{market_return:.2f}%")
                        c2.metric("전략 매매 수익률 (알고리즘)", f"{strat_return:.2f}%", delta=f"{strat_return - market_return:.2f}% (초과수익)")
                        
                        st.line_chart(bt_df[['Cumulative_Market', 'Cumulative_Strategy']])
                        st.caption("※ 전략 로직: 20일선이 60일선을 돌파(골든크로스)하고 RSI가 50 이하일 때 매수 / 데드크로스 또는 RSI 70 이상일 때 매도")
                    else:
                        st.error("백테스트 데이터를 가져오지 못했습니다.")
            else:
                st.error("정확한 종목을 입력하세요.")

# ---------------------------------------------------------
# [탭 5] 전담 AI 애널리스트 챗봇
# ---------------------------------------------------------
with tab_chat:
    st.header(f"💬 전담 AI 애널리스트 실시간 채팅")
    
    current_focus = st.session_state.get('current_stock', '없음')
    if current_focus != "없음":
        st.info(f"💡 AI가 현재 사용자님이 **'{current_focus}'** 종목을 분석하고 있다는 것을 인지하고 있습니다. 관련된 추가 질문을 자유롭게 남겨주세요!")
    else:
        st.write("시장 상황이나 종목에 대해 자유롭게 질문해 보세요.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("예: 코스맥스엔비티 단기 전망은 어떨까? / 애플 지금 매수해도 돼?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-5:]])
        context_prompt = f"""
당신은 방금 사용자와 함께 종목을 분석한 30년 경력의 월스트리트 수석 퀀트 애널리스트입니다.
현재 사용자가 대시보드에서 분석 중이거나 주목하고 있는 종목: {current_focus}

[이전 대화 기록]
{history_text}

[사용자의 새로운 질문]
{prompt}

위 정보와 대화 기록을 바탕으로, 애널리스트로서 전문적이고 논리적인 답변을 작성하세요. 너무 길지 않게 핵심 위주로 대화체로 답변하세요.
"""
        with st.spinner("애널리스트가 시장 데이터를 검토하며 답변을 작성 중입니다..."):
            bot_instruction = "당신은 월스트리트 수석 퀀트 애널리스트로서 사용자의 주식 관련 질문에 전문적이고 친절하게 답하는 챗봇입니다."
            response_text, _ = get_ai_response(context_prompt, api_key, model_choice, bot_instruction, is_json=False)
            
            if response_text:
                with st.chat_message("assistant"):
                    st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
