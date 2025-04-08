"""
Quantum RSI - DEX Data Module

This module handles retrieving market data from decentralized exchanges (Uniswap, dYdX, 1inch)
and calculating technical indicators needed for the quantum RSI model.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional, Union
import datetime
import time
import logging
import requests
import json
import asyncio
from web3 import Web3, AsyncWeb3
from web3.providers.legacy_websocket import LegacyWebSocketProvider
import os
from dotenv import load_dotenv
import base64
import websocket

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('quantum_rsi.data')

# Load environment variables
load_dotenv()

# Common token addresses for Ethereum mainnet
TOKEN_ADDRESSES = {
    # Ethereum and wrapped versions
    "ETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH address
    "WETH": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    
    # Stablecoins
    "USDT": "0xdAC17F958D2ee523a2206206994597C13D831ec7",
    "USDC": "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "DAI": "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "TUSD": "0x0000000000085d4780B73119b644AE5ecd22b376",
    "BUSD": "0x4Fabb145d64652a948d72533023f6E7A623C7C53",
    "FRAX": "0x853d955aCEf822Db058eb8505911ED77F175b99e",
    
    # Major cryptocurrencies
    "WBTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "BTC": "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",  # Wrapped BTC
    "LINK": "0x514910771AF9Ca656af840dff83E8264EcF986CA",
    "UNI": "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
    "AAVE": "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
    "SNX": "0xC011a73ee8576Fb46F5E1c5751cA3B9Fe0af2a6F",
    "MKR": "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2",
    "CRV": "0xD533a949740bb3306d119CC777fa900bA034cd52",
    "COMP": "0xc00e94Cb662C3520282E6f5717214004A7f26888",
    "SUSHI": "0x6B3595068778DD592e39A122f4f5a5cF09C90fE2",
    
    # L2 and scaling
    "MATIC": "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",
    "LRC": "0xBBbbCA6A901c926F240b89EacB641d8Aec7AEafD",
    "OP": "0x4200000000000000000000000000000000000042",  # Optimism
    "ARB": "0xB50721BCf8d664c30412Cfbc6cf7a15145234ad1"   # Arbitrum
}

# Common timeframes
TIMEFRAME_1M = '1m'
TIMEFRAME_5M = '5m'
TIMEFRAME_15M = '15m'
TIMEFRAME_1H = '1h'
TIMEFRAME_4H = '4h'
TIMEFRAME_1D = '1d'

# Function to get token address with more detailed error handling
def get_token_address(token_symbol: str) -> str:
    """
    Get the contract address for a token symbol with better error handling.
    
    Args:
        token_symbol: Token symbol (e.g., "ETH", "USDT")
        
    Returns:
        Token contract address or the input if it's already an address
    """
    # If it's already an address, return it
    if token_symbol.startswith('0x') and len(token_symbol) == 42:
        return token_symbol
        
    # Try to find the token in our dictionary (case insensitive)
    token_upper = token_symbol.upper()
    if token_upper in TOKEN_ADDRESSES:
        return TOKEN_ADDRESSES[token_upper]
        
    # Check for common aliases and normalize them
    if token_upper == "BTC":
        return TOKEN_ADDRESSES["WBTC"]
    if token_upper == "ETH":
        return TOKEN_ADDRESSES["WETH"]
        
    # If we couldn't find it, log a warning
    logger.warning(f"Unknown token symbol: {token_symbol}. Using as-is, but this may cause errors.")
    return token_symbol

class DEXDataProvider:
    """
    Provides data from decentralized exchanges for the quantum RSI model.
    """
    
    def __init__(self, pairs: List[str] = None, timeframe: str = TIMEFRAME_1H, provider: str = "uniswap"):
        """
        Initialize the DEX data provider.
        
        Args:
            pairs: List of trading pairs to track (e.g., ["ETH/USDT", "ETH/USDC"])
            timeframe: Timeframe for data (default: 1 hour)
            provider: DEX provider to use for on-chain data ("uniswap", "dydx", or "1inch")
                      Note: CryptoCompare is used as the primary data source for historical data
        """
        self.pairs = pairs or ["ETH/USDT", "ETH/USDC"]
        self.timeframe = timeframe
        self.provider = provider.lower()
        self.initialized = False
        self.web3 = None
        self.ws_provider = None
        
        # Dictionary to store calculated indicators
        self.indicators = {}
        
        # API endpoints
        if self.provider == "uniswap":
            # Use the official Uniswap V3 subgraph
            self.api_endpoint = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
            
            # Check for environment override
            custom_endpoint = os.getenv('UNISWAP_SUBGRAPH_URL')
            if custom_endpoint:
                logger.info(f"Using custom Uniswap subgraph endpoint from environment: {custom_endpoint}")
                self.api_endpoint = custom_endpoint
                
            logger.warning("Uniswap subgraph API usage is deprecated. CryptoCompare will be used for historical data.")
        elif self.provider == "dydx":
            self.api_endpoint = "https://api.dydx.exchange"
        elif self.provider == "1inch":
            self.api_endpoint = "https://api.1inch.io/v5.0"
        else:
            logger.warning(f"Unsupported provider: {provider}. Defaulting to Uniswap.")
            self.provider = "uniswap"
            self.api_endpoint = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
            
        # Initialize CryptoCompare API settings
        self.cryptocompare_api_key = os.getenv('CRYPTOCOMPARE_API_KEY')
        if self.cryptocompare_api_key:
            logger.info("CryptoCompare API key found. Using for historical data.")
        else:
            logger.info("No CryptoCompare API key found. Using public API limits.")
            
        # Print data source information
        logger.info("Data source priority: 1. CryptoCompare, 2. DEX (for on-chain data)")
        logger.info(f"Selected DEX for on-chain data: {self.provider}")
        
    def initialize(self) -> bool:
        """
        Initialize connection to Web3 and check provider availability.
        
        Returns:
            True if connection was successful, False otherwise
        """
        if self.initialized:
            return True
            
        # Get Web3 provider from environment
        provider_uri = os.getenv('WEB3_PROVIDER_URI')
        if not provider_uri:
            logger.error("No WEB3_PROVIDER_URI found in environment variables")
            return False
        
        # Initialize Web3 based on provider type
        try:
            logger.info(f"Initializing connection to Web3 provider...")
            
            # Create provider based on URL scheme
            if provider_uri.startswith('wss://') or provider_uri.startswith('ws://'):
                logger.info("Using WebSocket provider")
                # In web3.py v7, we need to use LegacyWebSocketProvider for synchronous code
                self.web3 = Web3(LegacyWebSocketProvider(provider_uri))
            else:
                logger.info("Using HTTP provider")
                self.web3 = Web3(Web3.HTTPProvider(provider_uri))
                
            # Check if Web3 connection is working
            if not self.web3.is_connected():
                logger.error(f"Failed to connect to Web3 provider")
                return False
                
            logger.info(f"Successfully connected to Web3 provider")
            
            # Check if wallet is configured if needed
            if os.getenv('ETH_PRIVATE_KEY'):
                try:
                    account = self.web3.eth.account.from_key(os.getenv('ETH_PRIVATE_KEY'))
                    logger.info(f"Wallet configured: {account.address}")
                except Exception as e:
                    logger.warning(f"Wallet configuration issue: {str(e)}")
            else:
                logger.info("No wallet configured. Read-only mode enabled.")
                
            # Check if API endpoint is reachable
            try:
                if self.provider == "uniswap":
                    # Simple query to test connection
                    query = """
                    {
                      pools(first: 1, orderBy: volumeUSD, orderDirection: desc) {
                        id
                      }
                    }
                    """
                    response = requests.post(self.api_endpoint, json={'query': query})
                    if response.status_code != 200:
                        logger.error(f"Failed to connect to {self.provider} API: {response.text}")
                        return False
                elif self.provider == "dydx":
                    response = requests.get(f"{self.api_endpoint}/v3/markets")
                    if response.status_code != 200:
                        logger.error(f"Failed to connect to {self.provider} API: {response.text}")
                        return False
                elif self.provider == "1inch":
                    response = requests.get(f"{self.api_endpoint}/1/healthcheck")
                    if response.status_code != 200:
                        logger.error(f"Failed to connect to {self.provider} API: {response.text}")
                        return False
            except Exception as e:
                logger.error(f"Error connecting to {self.provider} API: {str(e)}")
                return False
                    
            self.initialized = True
            logger.info(f"DEX data provider initialized successfully using {self.provider}")
            return True
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            return False
        
    def get_account_info(self) -> dict:
        """
        Get information about the connected wallet.
        
        Returns:
            Dictionary with account information
        """
        if not self.initialized and not self.initialize():
            return {}
            
        if not os.getenv('ETH_PRIVATE_KEY'):
            return {
                "wallet": None,
                "balance": 0,
                "connected": False,
                "read_only": True
            }
            
        try:
            account = self.web3.eth.account.from_key(os.getenv('ETH_PRIVATE_KEY'))
            address = account.address
            balance = self.web3.eth.get_balance(address)
            balance_eth = self.web3.from_wei(balance, 'ether')
            
            return {
                "wallet": address,
                "balance": float(balance_eth),
                "connected": True,
                "read_only": False
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {str(e)}")
            return {}
        
    def get_historical_data(self, pair: str, 
                          bars: int = 500,
                          start_time: Optional[datetime.datetime] = None,
                          data_source: str = "cryptocompare") -> pd.DataFrame:
        """
        Get historical OHLCV data for a trading pair.
        
        Args:
            pair: Trading pair to get data for (e.g., "ETH/USDT")
            bars: Number of bars to retrieve
            start_time: Optional start time (if None, gets the most recent bars)
            data_source: Data source to use ("cryptocompare", "dex", or "auto")
            
        Returns:
            DataFrame with OHLCV data
        """
        if not self.initialized and not self.initialize():
            return pd.DataFrame()
            
        # Convert pair format if needed
        base, quote = self._parse_pair(pair)
        
        # Get data based on the specified source
        df = pd.DataFrame()
        
        # Use CryptoCompare as the primary data source
        if data_source == "cryptocompare" or data_source == "auto":
            logger.info(f"Fetching {pair} data from CryptoCompare")
            df = self._get_cryptocompare_data(base, quote, bars, start_time)
            
            # If using "auto" mode and CryptoCompare failed, try DEX as fallback
            if df.empty and data_source == "auto":
                logger.info(f"CryptoCompare data unavailable for {pair}, trying DEX data")
                df = self._get_dex_data(base, quote, pair, bars, start_time)
        
        # Only use DEX data if explicitly requested
        elif data_source == "dex":
            logger.info(f"Fetching {pair} data from {self.provider}")
            df = self._get_dex_data(base, quote, pair, bars, start_time)
            
        # Log result
        if not df.empty:
            logger.info(f"Successfully retrieved {len(df)} bars of data for {pair}")
        else:
            logger.error(f"Failed to retrieve data for {pair} from any source")
            
        return df
        
    def _parse_pair(self, pair: str) -> Tuple[str, str]:
        """Parse trading pair into base and quote currencies"""
        if '/' in pair:
            base, quote = pair.split('/')
        else:
            # Assume format like ETHUSDT
            # This is simplified and would need to be improved for production
            for token in TOKEN_ADDRESSES.keys():
                if pair.startswith(token):
                    base = token
                    quote = pair[len(token):]
                    break
            else:
                base, quote = pair, 'USDT'  # Default assumption
                
        return base, quote
        
    def _get_uniswap_data(self, base: str, quote: str, 
                         bars: int, 
                         start_time: Optional[datetime.datetime]) -> pd.DataFrame:
        """Get historical data from Uniswap"""
        logger.info(f"Fetching Uniswap data for {base}/{quote}")
        
        # Find pool ID for the pair
        base_address = get_token_address(base)
        quote_address = get_token_address(quote)
        
        # Log the addresses being used
        logger.info(f"Using addresses: {base}={base_address}, {quote}={quote_address}")
        
        # Check if addresses are properly formatted
        if not base_address.startswith('0x'):
            logger.error(f"Invalid base token address: {base_address}")
            return pd.DataFrame()
            
        if not quote_address.startswith('0x'):
            logger.error(f"Invalid quote token address: {quote_address}")
            return pd.DataFrame()
        
        # First get the pool ID - use a simpler query approach
        query = """
        {
          pools(where: {token0_in: ["TOKEN0", "TOKEN1"], token1_in: ["TOKEN0", "TOKEN1"]}, 
                orderBy: volumeUSD, orderDirection: desc, first: 5) {
            id
            token0 {
              id
              symbol
            }
            token1 {
              id
              symbol
            }
            volumeUSD
            token0Price
            token1Price
          }
        }
        """.replace("TOKEN0", base_address.lower()).replace("TOKEN1", quote_address.lower())
        
        # Log the actual query for debugging
        logger.info(f"GraphQL Query: {query}")
        
        try:
            logger.info(f"Querying Uniswap subgraph: {self.api_endpoint}")
            
            # Try alternative endpoint for Uniswap v3
            alternative_endpoints = [
                "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3",
                "https://gateway.thegraph.com/api/[api-key]/subgraphs/id/ELUcwgpm14LKPLrBRuVvPvNKHQ9HvwmtKgKSH6123cr7" # Replace [api-key] with actual key if available
            ]
            
            # First try with original endpoint
            response = requests.post(
                self.api_endpoint, 
                json={'query': query}
            )
            
            # Log full response for debugging
            if response.status_code != 200:
                logger.error(f"Failed to get Uniswap pool: {response.text}")
                
                # Try alternative endpoints
                success = False
                for endpoint in alternative_endpoints:
                    if endpoint == self.api_endpoint:
                        continue
                        
                    logger.info(f"Trying alternative endpoint: {endpoint}")
                    try:
                        response = requests.post(endpoint, json={'query': query})
                        if response.status_code == 200:
                            self.api_endpoint = endpoint
                            logger.info(f"Successfully connected to alternative endpoint: {endpoint}")
                            success = True
                            break
                    except Exception as e:
                        logger.error(f"Error with alternative endpoint: {str(e)}")
                
                if not success:
                    # Try with a very simple query to check if any GraphQL endpoint works
                    test_query = "{ _meta { block { number } } }"
                    logger.info(f"Testing with simple query: {test_query}")
                    
                    for endpoint in alternative_endpoints:
                        try:
                            test_response = requests.post(endpoint, json={'query': test_query})
                            logger.info(f"Test response from {endpoint}: {test_response.status_code} - {test_response.text[:100]}")
                            
                            if test_response.status_code == 200:
                                self.api_endpoint = endpoint
                                response = requests.post(endpoint, json={'query': query})
                                if response.status_code == 200:
                                    logger.info(f"Successfully connected to working endpoint: {endpoint}")
                                    break
                        except Exception as e:
                            logger.error(f"Test query error: {str(e)}")
                            
                if response.status_code != 200:
                    logger.error("All Uniswap endpoints failed")
                    return pd.DataFrame()
                
            # Print full response for debugging
            logger.info(f"Response status: {response.status_code}")
            response_text = response.text[:500] + "..." if len(response.text) > 500 else response.text
            logger.info(f"Response text: {response_text}")
            
            data = response.json()
            if 'errors' in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                return pd.DataFrame()
                
            pools = data.get('data', {}).get('pools', [])
            
            if not pools:
                logger.error(f"No Uniswap pool found for {base}/{quote}")
                logger.info("Let's try querying for top pools to see what's available")
                
                # Query for top pools to see what's available
                top_pools_query = """
                {
                  pools(orderBy: volumeUSD, orderDirection: desc, first: 10) {
                    id
                    token0 {
                      id
                      symbol
                    }
                    token1 {
                      id
                      symbol
                    }
                    volumeUSD
                  }
                }
                """
                
                try:
                    top_response = requests.post(self.api_endpoint, json={'query': top_pools_query})
                    if top_response.status_code == 200:
                        top_data = top_response.json()
                        top_pools = top_data.get('data', {}).get('pools', [])
                        logger.info(f"Available top pools:")
                        for idx, pool in enumerate(top_pools):
                            token0 = pool.get('token0', {})
                            token1 = pool.get('token1', {})
                            logger.info(f"  {idx+1}. {token0.get('symbol', 'Unknown')}/{token1.get('symbol', 'Unknown')} - Volume: ${float(pool.get('volumeUSD', 0)):.2f}")
                except Exception as e:
                    logger.error(f"Error getting top pools: {str(e)}")
                    
                return pd.DataFrame()
            
            # Find the best pool matching our tokens
            selected_pool = None
            for pool in pools:
                token0_symbol = pool.get('token0', {}).get('symbol', '').upper()
                token1_symbol = pool.get('token1', {}).get('symbol', '').upper()
                
                logger.info(f"Found pool: {token0_symbol}/{token1_symbol} (ID: {pool['id']})")
                
                if (token0_symbol == base.upper() and token1_symbol == quote.upper()) or \
                   (token0_symbol == quote.upper() and token1_symbol == base.upper()):
                    selected_pool = pool
                    logger.info(f"Selected pool {token0_symbol}/{token1_symbol}")
                    break
            
            if not selected_pool and pools:
                # Just use the highest volume pool
                selected_pool = pools[0]
                logger.info(f"No exact match found. Using highest volume pool: {selected_pool['id']}")
            
            if not selected_pool:
                logger.error(f"No suitable pool found for {base}/{quote}")
                return pd.DataFrame()
                
            pool_id = selected_pool['id']
            logger.info(f"Using pool ID: {pool_id}")
            
            # Determine token ordering in the pool
            token0_address = selected_pool.get('token0', {}).get('id', '').lower()
            token0_is_base = token0_address == base_address.lower()
            
            # Now get pool day data for this pool
            time_filter = ""
            if start_time:
                unix_time = int(start_time.timestamp())
                time_filter = f", where: {{ date_gt: {unix_time} }}"
                
            # Get poolDayData instead of swaps for better OHLCV representation
            query = f"""
            query {{
              poolDayDatas(
                first: {bars}, 
                orderBy: date, 
                orderDirection: desc,
                where: {{ pool: "{pool_id}" {time_filter} }}
              ) {{
                date
                volumeUSD
                tvlUSD
                token0Price
                token1Price
                high
                low
                open
                close
              }}
            }}
            """
            
            response = requests.post(self.api_endpoint, json={'query': query})
            
            if response.status_code != 200:
                logger.error(f"Failed to get pool data: {response.text}")
                return pd.DataFrame()
                
            data = response.json()
            pool_days = data.get('data', {}).get('poolDayDatas', [])
            
            if not pool_days:
                logger.error(f"No historical data found for pool {pool_id}")
                return pd.DataFrame()
                
            # Process poolDayData into OHLCV
            records = []
            for day in pool_days:
                # Convert to timestamp
                timestamp = datetime.datetime.fromtimestamp(int(day['date']))
                
                # Get the correct price based on token ordering
                if token0_is_base:
                    price = float(day['token0Price'])
                    # Price is inverted in this case
                    price = 1.0 / price if price > 0 else 0
                else:
                    price = float(day['token1Price'])
                
                try:
                    # Use existing OHLC if available or derive from price
                    open_price = float(day.get('open', price))
                    high_price = float(day.get('high', price * 1.02))  # Estimate
                    low_price = float(day.get('low', price * 0.98))    # Estimate
                    close_price = float(day.get('close', price))
                    volume = float(day.get('volumeUSD', 0))
                    
                    records.append({
                        'timestamp': timestamp,
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': close_price,
                        'volume': volume
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing data point: {str(e)}")
                    continue
            
            if not records:
                logger.error("Failed to process any valid data points")
                return pd.DataFrame()
                
            # Create DataFrame
            df = pd.DataFrame(records)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Resample data if timeframe is not daily
            if self.timeframe != '1d':
                # Map to pandas frequency
                tf_map = {
                    '1m': '1min', '5m': '5min', '15m': '15min',
                    '1h': '1H', '4h': '4H', '1d': '1D'
                }
                pandas_tf = tf_map.get(self.timeframe, '1H')
                
                # Resample
                df = df.resample(pandas_tf).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            
            logger.info(f"Successfully retrieved {len(df)} bars of data for {base}/{quote}")
            return df
            
        except Exception as e:
            logger.error(f"Error getting Uniswap data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
            
    def _get_dydx_data(self, pair: str, bars: int, 
                      start_time: Optional[datetime.datetime]) -> pd.DataFrame:
        """Get historical data from dYdX"""
        # dYdX has a specific format for market pairs
        market = pair.replace('/', '-')
        
        # Map timeframe to dYdX format
        tf_map = {
            '1m': '1MIN', '5m': '5MINS', '15m': '15MINS',
            '1h': '1HOUR', '4h': '4HOURS', '1d': '1DAY'
        }
        dydx_tf = tf_map.get(self.timeframe, '1HOUR')
        
        # Calculate start time if not provided
        if not start_time:
            # Get latest data and go back by bars
            time_filter = ""
        else:
            # Format timestamp for API
            from_iso = start_time.strftime('%Y-%m-%dT%H:%M:%S.000Z')
            time_filter = f"&fromISO={from_iso}"
            
        try:
            url = f"{self.api_endpoint}/v3/candles?market={market}&resolution={dydx_tf}&limit={bars}{time_filter}"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"Failed to get dYdX data: {response.text}")
                return pd.DataFrame()
                
            data = response.json()
            candles = data.get('candles', [])
            
            if not candles:
                logger.error(f"No candles found for {market} on dYdX")
                return pd.DataFrame()
                
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            df['startedAt'] = pd.to_datetime(df['startedAt'])
            df.rename(columns={
                'startedAt': 'time',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'baseTokenVolume': 'volume'
            }, inplace=True)
            
            # Set index and select relevant columns
            df.set_index('time', inplace=True)
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Convert to numeric
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting dYdX data: {str(e)}")
            return pd.DataFrame()
            
    def _get_1inch_data(self, base: str, quote: str, 
                       bars: int, 
                       start_time: Optional[datetime.datetime]) -> pd.DataFrame:
        """
        Get historical data approximated from 1inch
        
        Note: 1inch doesn't provide OHLCV data directly.
        We'll use external price feeds like CoinGecko for this implementation.
        """
        # Since 1inch doesn't have historical candles API, we use CoinGecko as a proxy
        # A real implementation would use a proper data provider or aggregate from events
        base_id = base.lower()
        vs_currency = quote.lower()
        
        # Map timeframe to days
        days = 1
        if self.timeframe in ['1d', '4h']:
            days = 30
        elif self.timeframe in ['1h', '15m']:
            days = 7
        else:
            days = 1
            
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{base_id}/market_chart"
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': self.timeframe if self.timeframe in ['daily', 'hourly'] else 'hourly'
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to get price data: {response.text}")
                return pd.DataFrame()
                
            data = response.json()
            
            # Process price data
            prices = data.get('prices', [])
            volumes = data.get('total_volumes', [])
            
            if not prices:
                logger.error(f"No price data found for {base}/{quote}")
                return pd.DataFrame()
                
            # Create price DataFrame
            price_df = pd.DataFrame(prices, columns=['timestamp', 'price'])
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='ms')
            
            # Create volume DataFrame
            volume_df = pd.DataFrame(volumes, columns=['timestamp', 'volume'])
            volume_df['timestamp'] = pd.to_datetime(volume_df['timestamp'], unit='ms')
            
            # Merge DataFrames
            df = pd.merge(price_df, volume_df, on='timestamp', how='outer')
            
            # Set index
            df.set_index('timestamp', inplace=True)
            
            # Resample to desired timeframe
            tf_map = {
                '1m': '1min', '5m': '5min', '15m': '15min',
                '1h': '1H', '4h': '4H', '1d': '1D'
            }
            pandas_tf = tf_map.get(self.timeframe, '1H')
            
            # Resample and create OHLCV
            resampled = df.resample(pandas_tf)
            
            ohlcv = pd.DataFrame()
            ohlcv['open'] = resampled['price'].first()
            ohlcv['high'] = resampled['price'].max()
            ohlcv['low'] = resampled['price'].min()
            ohlcv['close'] = resampled['price'].last()
            ohlcv['volume'] = resampled['volume'].sum()
            
            # Clean up the data
            ohlcv = ohlcv.dropna()
            
            return ohlcv
            
        except Exception as e:
            logger.error(f"Error getting 1inch/CoinGecko data: {str(e)}")
            return pd.DataFrame()
            
    def calculate_indicators(self, df: pd.DataFrame, 
                           rsi_period: int = 14,
                           ma_period: int = 20,
                           vwap_period: int = 20,
                           volume_lookback: int = 20) -> pd.DataFrame:
        """
        Calculate technical indicators needed for the quantum RSI model.
        
        Args:
            df: DataFrame with OHLCV data
            rsi_period: Period for RSI calculation
            ma_period: Period for moving average calculation
            vwap_period: Period for VWAP calculation
            volume_lookback: Lookback period for volume analysis
            
        Returns:
            DataFrame with added indicators
        """
        if df.empty:
            return df
            
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate RSI
        delta = result['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=rsi_period).mean()
        
        # Handle zero division
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = gain / loss
            result['rsi'] = 100 - (100 / (1 + rs))
        
        # Fill NaN values
        result['rsi'].fillna(50, inplace=True)
        
        # Calculate moving average
        result['ma'] = result['close'].rolling(window=ma_period).mean()
        
        # Calculate VWAP approximation
        result['vwap'] = self._calculate_rolling_vwap(result, period=vwap_period)
        
        # Normalize volume (current vol / average of last N periods)
        result['volume_ratio'] = result['volume'] / result['volume'].rolling(window=volume_lookback).mean()
        result['volume_ratio'].fillna(1.0, inplace=True)
        
        # Calculate price to VWAP ratio
        result['price_to_vwap'] = result['close'] / result['vwap']
        result['price_to_vwap'].fillna(1.0, inplace=True)
        
        # Calculate up/down volume as proxy
        result['up_volume'] = result['volume'] * (delta > 0).astype(float)
        result['down_volume'] = result['volume'] * (delta < 0).astype(float)
        result['delta_volume'] = result['up_volume'] - result['down_volume']
        
        # Calculate normalized delta (for estimating buying/selling pressure)
        vol_sum = result['up_volume'] + result['down_volume']
        # Avoid division by zero
        vol_sum = vol_sum.replace(0, 1)
        result['delta_ratio'] = result['delta_volume'] / vol_sum
        
        return result
    
    def _calculate_rolling_vwap(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate a rolling VWAP approximation.
        
        Args:
            df: DataFrame with OHLCV data
            period: VWAP calculation period
            
        Returns:
            Series with VWAP values
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        return vwap
        
    def get_latest_data(self, pair: str, lookback_bars: int = 100) -> Dict:
        """
        Get the latest market data with calculated indicators.
        
        Args:
            pair: Trading pair to get data for
            lookback_bars: Number of historical bars to retrieve
            
        Returns:
            Dictionary with latest market data and indicators
        """
        if not self.initialized and not self.initialize():
            return {}
            
        # Get historical data
        df = self.get_historical_data(pair, lookback_bars)
        if df.empty:
            return {}
            
        # Calculate indicators
        df_with_indicators = self.calculate_indicators(df)
        
        # Store in cache
        self.indicators[pair] = df_with_indicators
        
        # Get latest values (last row)
        latest = df_with_indicators.iloc[-1].to_dict()
        
        # Add some derived values that might be useful
        latest['normalized_rsi'] = (latest['rsi'] - 50) / 50  # Map RSI to [-1, 1]
        latest['rsi_momentum'] = df_with_indicators['rsi'].diff(3).iloc[-1]  # RSI momentum (3-period change)
        
        # Get current prices from DEX
        latest_price = self._get_current_price(pair)
        if latest_price:
            latest.update(latest_price)
            
        return latest
    
    def _get_current_price(self, pair: str) -> Dict:
        """Get current price from DEX"""
        base, quote = self._parse_pair(pair)
        
        if self.provider == "uniswap":
            return self._get_uniswap_price(base, quote)
        elif self.provider == "dydx":
            return self._get_dydx_price(pair)
        elif self.provider == "1inch":
            return self._get_1inch_price(base, quote)
        else:
            return {}
            
    def _get_uniswap_price(self, base: str, quote: str) -> Dict:
        """Get current price from Uniswap"""
        base_address = get_token_address(base)
        quote_address = get_token_address(quote)
        
        query = """
        query ($token0: String!, $token1: String!) {
          pools(where: {
            token0: $token0,
            token1: $token1
          }, orderBy: volumeUSD, orderDirection: desc, first: 1) {
            token0Price
            token1Price
          }
        }
        """
        
        variables = {
            "token0": base_address,
            "token1": quote_address
        }
        
        try:
            response = requests.post(
                self.api_endpoint, 
                json={'query': query, 'variables': variables}
            )
            
            if response.status_code != 200:
                return {}
                
            data = response.json()
            pools = data.get('data', {}).get('pools', [])
            
            if not pools:
                # Try with tokens reversed
                variables = {
                    "token0": quote_address,
                    "token1": base_address
                }
                response = requests.post(
                    self.api_endpoint, 
                    json={'query': query, 'variables': variables}
                )
                data = response.json()
                pools = data.get('data', {}).get('pools', [])
                
                if not pools:
                    return {}
                    
                # If we had to reverse, use token0Price
                return {
                    'current_price': float(pools[0]['token0Price']),
                    'bid': float(pools[0]['token0Price']) * 0.998,  # Approximate
                    'ask': float(pools[0]['token0Price']) * 1.002   # Approximate
                }
                
            return {
                'current_price': float(pools[0]['token1Price']),
                'bid': float(pools[0]['token1Price']) * 0.998,  # Approximate
                'ask': float(pools[0]['token1Price']) * 1.002   # Approximate
            }
            
        except Exception as e:
            logger.error(f"Error getting Uniswap price: {str(e)}")
            return {}
            
    def _get_dydx_price(self, pair: str) -> Dict:
        """Get current price from dYdX"""
        market = pair.replace('/', '-')
        
        try:
            url = f"{self.api_endpoint}/v3/markets/{market}"
            response = requests.get(url)
            
            if response.status_code != 200:
                return {}
                
            data = response.json()
            market_data = data.get('market', {})
            
            if not market_data:
                return {}
                
            return {
                'current_price': float(market_data.get('oraclePrice', 0)),
                'bid': float(market_data.get('indexPrice', 0)),
                'ask': float(market_data.get('indexPrice', 0))
            }
            
        except Exception as e:
            logger.error(f"Error getting dYdX price: {str(e)}")
            return {}
            
    def _get_1inch_price(self, base: str, quote: str) -> Dict:
        """Get current price from 1inch"""
        from_token = get_token_address(base)
        to_token = get_token_address(quote)
        
        try:
            url = f"{self.api_endpoint}/1/quote"
            params = {
                'fromTokenAddress': from_token,
                'toTokenAddress': to_token,
                'amount': '1000000000000000000'  # 1 ETH in wei
            }
            
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                return {}
                
            data = response.json()
            
            if 'toTokenAmount' not in data:
                return {}
                
            # Calculate price
            from_decimals = int(data.get('fromToken', {}).get('decimals', 18))
            to_decimals = int(data.get('toToken', {}).get('decimals', 18))
            
            from_amount = float(data.get('fromTokenAmount')) / (10 ** from_decimals)
            to_amount = float(data.get('toTokenAmount')) / (10 ** to_decimals)
            
            price = to_amount / from_amount
            
            return {
                'current_price': price,
                'bid': price * 0.995,  # Approximate with slippage
                'ask': price * 1.005   # Approximate with slippage
            }
            
        except Exception as e:
            logger.error(f"Error getting 1inch price: {str(e)}")
            return {}
    
    def shutdown(self) -> None:
        """Clean up resources."""
        if hasattr(self, 'web3') and self.web3 and isinstance(self.web3.provider, LegacyWebSocketProvider):
            try:
                self.web3.provider.disconnect()
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {str(e)}")
            
        self.initialized = False
        logger.info("DEX data provider shut down")

    async def subscribe_to_new_blocks(self, callback):
        """
        Subscribe to new block headers and call the callback function for each new block.
        This is an example of using WebSockets for real-time data.
        
        Args:
            callback: A function that takes a block header as argument
        """
        if not self.initialized and not self.initialize():
            logger.error("Failed to initialize provider for subscription")
            return
        
        provider_uri = os.getenv('WEB3_PROVIDER_URI')
        if not provider_uri.startswith('wss://') and not provider_uri.startswith('ws://'):
            logger.error("WebSocket subscription requires a WebSocket endpoint (wss:// or ws://)")
            return
        
        logger.info("Setting up WebSocket subscription for new blocks...")
        
        # We need to use AsyncWeb3 for subscriptions
        async with AsyncWeb3.persistent_websocket(LegacyWebSocketProvider(provider_uri)) as w3:
            # Subscribe to new block headers
            subscription_id = await w3.eth.subscribe("newHeads")
            logger.info(f"Subscribed to new block headers (ID: {subscription_id})")
            
            try:
                # Process subscription messages
                async for message in w3.ws.listen_to_websocket():
                    if "result" in message:
                        block_header = message["result"]
                        # Log block information
                        logger.info(f"New block: {block_header['number']} - Gas used: {int(block_header['gasUsed'], 16)}")
                        
                        # Call the user-provided callback
                        await callback(block_header)
            except Exception as e:
                logger.error(f"Error in WebSocket subscription: {str(e)}")
            finally:
                # Unsubscribe when done
                try:
                    await w3.eth.unsubscribe(subscription_id)
                    logger.info("Unsubscribed from new block headers")
                except Exception as e:
                    logger.error(f"Error unsubscribing: {str(e)}")
                
    def _get_cryptocompare_data(self, base: str, quote: str, 
                              bars: int,
                              start_time: Optional[datetime.datetime]) -> pd.DataFrame:
        """
        Get historical data from CryptoCompare API
        
        Args:
            base: Base currency (e.g., "ETH")
            quote: Quote currency (e.g., "USDT")
            bars: Number of bars to retrieve
            start_time: Optional start time (if None, gets the most recent bars)
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"\n>>> Fetching CryptoCompare data for {base}/{quote}")
        logger.info(f"Fetching CryptoCompare data for {base}/{quote}")
        
        # Map timeframe to CryptoCompare format
        tf_map = {
            '1m': 'minute', '5m': 'minute', '15m': 'minute', '30m': 'minute',
            '1h': 'hour', '4h': 'hour', 
            '1d': 'day', '1w': 'day'
        }
        
        # Get timeframe values
        if self.timeframe in ['1m', '5m', '15m', '30m']:
            # Extract the number from the timeframe
            aggregate = int(self.timeframe[:-1])
            cc_tf = 'minute'
        elif self.timeframe in ['1h', '2h', '4h', '6h', '12h']:
            aggregate = int(self.timeframe[:-1])
            cc_tf = 'hour'
        elif self.timeframe == '1d':
            aggregate = 1
            cc_tf = 'day'
        elif self.timeframe == '1w':
            aggregate = 7
            cc_tf = 'day'
        else:
            # Default to hourly
            aggregate = 1
            cc_tf = 'hour'
        
        # CryptoCompare API endpoint
        base_url = "https://min-api.cryptocompare.com/data/v2"
        
        # Calculate parameters
        limit = min(bars, 2000)  # CryptoCompare has a limit of 2000 points
        
        # Convert base and quote to uppercase
        base = base.upper()
        quote = quote.upper()
        
        # Determine endpoint based on timeframe
        if cc_tf == 'minute':
            endpoint = f"/histominute"
        elif cc_tf == 'hour':
            endpoint = f"/histohour"
        else:
            endpoint = f"/histoday"
        
        # Build query parameters
        params = {
            'fsym': base,
            'tsym': quote,
            'limit': limit,
            'aggregate': aggregate,
            'e': 'CCCAGG'  # Use the CryptoCompare aggregate index
        }
        
        # CryptoCompare API needs a timestamp to get historical data
        # It returns data going backward from the specified timestamp
        if start_time:
            # To get data for a specific date range, we need to:
            # 1. Set toTs to a date AFTER our desired end date
            # 2. Get enough data to cover our desired range
            # 3. Filter the results after

            # Calculate time per bar in seconds
            time_per_bar = 0
            if cc_tf == 'minute':
                time_per_bar = 60 * aggregate  # seconds per minute * aggregate
            elif cc_tf == 'hour':
                time_per_bar = 3600 * aggregate  # seconds per hour * aggregate
            else:  # day
                time_per_bar = 86400 * aggregate  # seconds per day * aggregate
            
            # Calculate how many days worth of data we need
            # For backtesting, we typically need data from start_time to end_time
            # Let's add 30 days (or equivalent in our timeframe) to the start_time
            # to ensure we have enough data
            end_time = start_time + datetime.timedelta(days=30)
            
            # Set toTs to end_time
            params['toTs'] = int(end_time.timestamp())
            logger.info(f"Requesting data with end timestamp: {end_time}")
        
        # Check for API key and add if available
        api_key = os.getenv('CRYPTOCOMPARE_API_KEY')
        headers = {}
        if api_key:
            headers['authorization'] = f"Apikey {api_key}"
        
        try:
            url = f"{base_url}{endpoint}"
            logger.info(f"CryptoCompare URL: {url}, Params: {params}")
            
            # Make request to CryptoCompare
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code != 200:
                logger.error(f"CryptoCompare API error: {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            if data.get('Response') == 'Error':
                logger.error(f"CryptoCompare error: {data.get('Message')}")
                return pd.DataFrame()
            
            # Parse response
            ohlcv_data = data.get('Data', {}).get('Data', [])
            
            if not ohlcv_data:
                logger.error(f"No data returned from CryptoCompare for {base}/{quote}")
                return pd.DataFrame()
            
            # Create DataFrame
            df = pd.DataFrame(ohlcv_data)
            
            # Rename columns to standard format
            df = df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volumefrom': 'volume'
            })
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Set index
            df = df.set_index('timestamp')
            
            # Select only the columns we need
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Log date range for debugging
            logger.info(f"Data retrieved from {df.index.min()} to {df.index.max()}")
            
            # Filter by date range if start_time provided
            if start_time:
                logger.info(f"Filtering data to start from {start_time}")
                # The API often returns data from earlier than the requested time
                # We need to filter to only include data after start_time
                df = df[df.index >= start_time]
                
            # Check if we got any data after filtering
            if df.empty:
                logger.warning(f"No data available for {base}/{quote} after date filtering")
                return pd.DataFrame()
                
            logger.info(f"Successfully fetched {len(df)} bars from CryptoCompare for {base}/{quote}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching CryptoCompare data: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    def _get_dex_data(self, base: str, quote: str, pair: str, 
                    bars: int, start_time: Optional[datetime.datetime]) -> pd.DataFrame:
        """Get historical data from the configured DEX provider"""
        # Get data based on provider
        if self.provider == "uniswap":
            df = self._get_uniswap_data(base, quote, bars, start_time)
        elif self.provider == "dydx":
            df = self._get_dydx_data(pair, bars, start_time)
        elif self.provider == "1inch":
            df = self._get_1inch_data(base, quote, bars, start_time)
        else:
            logger.error(f"Unsupported provider: {self.provider}")
            df = pd.DataFrame()
            
        return df

async def run_websocket_example():
    """
    Example of using WebSocket subscription to monitor new blocks.
    This function demonstrates how to set up an async environment and
    listen for blockchain events in real-time.
    """
    provider = DEXDataProvider()
    
    async def process_block(block_header):
        """Process a new block header"""
        block_number = int(block_header['number'], 16)
        gas_used = int(block_header['gasUsed'], 16)
        gas_limit = int(block_header['gasLimit'], 16)
        usage_percent = gas_used / gas_limit * 100
        
        print(f"Block #{block_number}: Gas usage {usage_percent:.2f}% ({gas_used}/{gas_limit})")
        
        # You could analyze trading data, update indicators, or execute trades here
        
    try:
        await provider.subscribe_to_new_blocks(process_block)
    except KeyboardInterrupt:
        print("Subscription stopped by user")
    finally:
        provider.shutdown()


def check_web3_availability() -> bool:
    """
    Check if Web3 provider is available.
    
    Returns:
        True if Web3 can be initialized, False otherwise
    """
    try:
        # Get provider URI from environment
        provider_uri = os.getenv('WEB3_PROVIDER_URI')
        if not provider_uri:
            logger.error("No WEB3_PROVIDER_URI found in environment variables")
            return False
        
        # Print provider details (without credentials)
        if '@' in provider_uri:
            provider_host = provider_uri.split('@')[1].split('/')[0]
        else:
            provider_host = provider_uri.split('/')[2]
        
        protocol = "WSS" if provider_uri.startswith('wss://') else "HTTP"
        logger.info(f"Connecting to Web3 provider: {protocol}://{provider_host}")
            
        # Create Web3 instance with appropriate provider
        try:
            if provider_uri.startswith('wss://') or provider_uri.startswith('ws://'):
                logger.info("Using WebSocket provider")
                web3 = Web3(LegacyWebSocketProvider(provider_uri))
            else:
                logger.info("Using HTTP provider")
                web3 = Web3(Web3.HTTPProvider(provider_uri))
                
        except Exception as e:
            logger.error(f"Error initializing Web3 provider: {str(e)}")
            return False
            
        # Check connection
        if web3.is_connected():
            logger.info(f"Successfully connected to Web3 provider")
            
            # Get current block number as test
            try:
                block_number = web3.eth.block_number
                logger.info(f"Current block number: {block_number}")
            except Exception as e:
                logger.error(f"Error getting block number: {str(e)}")
                
            # Close WebSocket connection properly if needed
            if isinstance(web3.provider, LegacyWebSocketProvider):
                try:
                    web3.provider.disconnect()
                    logger.info("WebSocket connection closed")
                except Exception as e:
                    logger.error(f"Error closing WebSocket connection: {str(e)}")
            
            return True
        else:
            logger.error(f"Could not connect to Web3 provider")
            return False
    except Exception as e:
        logger.error(f"Web3 initialization error: {str(e)}")
        return False


# Test the data provider
if __name__ == "__main__":
    # Check if Web3 is available
    if not check_web3_availability():
        print("Web3 is not available. Please install and set up Web3 with a demo account.")
        exit(1)
        
    # Create data provider
    data_provider = DEXDataProvider(pairs=["ETH/USDT", "ETH/USDC"])
    
    # Initialize connection
    if not data_provider.initialize():
        print("Failed to initialize Web3 connection.")
        exit(1)
        
    # Print account info
    account_info = data_provider.get_account_info()
    print("Account Information:")
    for key, value in account_info.items():
        print(f"  {key}: {value}")
    
    # Get and print the latest data for ETH/USDT
    latest_eth_usdt = data_provider.get_latest_data("ETH/USDT")
    print("\nLatest ETH/USDT Data:")
    print(f"  Close: {latest_eth_usdt.get('close', 'N/A')}")
    print(f"  RSI: {latest_eth_usdt.get('rsi', 'N/A'):.2f}")
    print(f"  Volume Ratio: {latest_eth_usdt.get('volume_ratio', 'N/A'):.2f}")
    print(f"  Price to VWAP: {latest_eth_usdt.get('price_to_vwap', 'N/A'):.4f}")
    print(f"  Delta Volume Ratio: {latest_eth_usdt.get('delta_ratio', 'N/A'):.4f}")
    
    # Shutdown
    data_provider.shutdown() 