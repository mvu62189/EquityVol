import requests
import pandas as pd

def get_ticker_suggestions(query: str):
    """
    Queries Yahoo Finance Typeahead API for top 5 ticker suggestions.
    """
    if not query or len(query) < 1:
        return pd.DataFrame()

    url = f"https://query2.finance.yahoo.com/v1/finance/search"
    params = {
        'q': query,
        'quotesCount': 5,
        'newsCount': 0,
        'enableFuzzyQuery': 'false',
        'quotesQueryId': 'tss_match_phrase_query'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=5)
        data = r.json()
        
        if 'quotes' in data and len(data['quotes']) > 0:
            # Extract relevant fields
            results = []
            for q in data['quotes']:
                # Filter for Equity or ETF only to avoid noise
                if q.get('quoteType') in ['EQUITY', 'ETF', 'INDEX']:
                    results.append({
                        'Ticker': q.get('symbol'),
                        'Name': q.get('longname') or q.get('shortname'),
                        'Type': q.get('quoteType'),
                        'Exchange': q.get('exchange')
                    })
            return pd.DataFrame(results)
            
    except Exception as e:
        print(f"Search API Error: {e}")
        
    return pd.DataFrame()