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
        
