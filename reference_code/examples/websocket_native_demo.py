"""
Native WebSocket Demo for Quantum RSI

This script demonstrates how to use WebSockets directly with the websockets library
to get real-time data from the Ethereum blockchain using Chainstack.

It follows Chainstack's best practices for WebSocket connections with reconnection logic.
"""

import asyncio
import json
import websockets
import logging
import sys
import os
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('websocket_native_demo')

# Load environment variables
load_dotenv()

# Maximum reconnection attempts
MAX_RECONNECT_ATTEMPTS = 10
# Time to wait between reconnection attempts (in seconds)
RECONNECT_DELAY = 5

async def subscribe_to_blocks(ws_url):
    """
    Subscribe to new block headers with automatic reconnection logic.
    Uses the native websockets library to connect directly to Ethereum JSON-RPC.
    
    Args:
        ws_url: WebSocket URL for the Ethereum node
    """
    reconnect_attempts = 0
    last_block_time = time.time()
    
    # Extract host for logging (without credentials)
    if '@' in ws_url:
        host = ws_url.split('@')[1].split('/')[0]
    else:
        host = ws_url.split('/')[2]
    
    logger.info(f"Connecting to WebSocket provider: wss://{host}")
    
    while reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
        try:
            logger.info(f"Attempt #{reconnect_attempts + 1} to connect to WebSocket...")
            
            # Establish a WebSocket connection to the Ethereum node
            async with websockets.connect(ws_url) as websocket:
                logger.info("WebSocket connection established successfully")
                
                # Send a subscription request for new block headers
                await websocket.send(json.dumps({
                    "id": 1,
                    "method": "eth_subscribe",
                    "params": ["newHeads"],
                    "jsonrpc": "2.0"
                }))
                
                # Wait for the subscription response
                subscription_response = await websocket.recv()
                subscription_data = json.loads(subscription_response)
                
                if "result" in subscription_data:
                    subscription_id = subscription_data["result"]
                    logger.info(f"Subscribed to new block headers. Subscription ID: {subscription_id}")
                    
                    # Reset reconnection attempts on successful connection
                    reconnect_attempts = 0
                    
                    # Continuously process incoming blocks
                    while True:
                        try:
                            # Receive block data
                            block_data_raw = await websocket.recv()
                            block_data = json.loads(block_data_raw)
                            
                            if "params" in block_data and "result" in block_data["params"]:
                                # Extract block header information
                                block_header = block_data["params"]["result"]
                                
                                # Parse important block information
                                block_number = int(block_header["number"], 16)
                                gas_used = int(block_header["gasUsed"], 16)
                                gas_limit = int(block_header["gasLimit"], 16)
                                usage_percent = (gas_used / gas_limit) * 100
                                
                                # Log block information
                                logger.info(f"Block #{block_number}: Gas usage {usage_percent:.2f}% ({gas_used}/{gas_limit})")
                                
                                # Calculate time since last block
                                current_time = time.time()
                                time_diff = current_time - last_block_time
                                logger.info(f"Time since last block: {time_diff:.2f} seconds")
                                last_block_time = current_time
                                
                                # Example: You could perform trading analysis here
                                # - Check price changes
                                # - Calculate indicators
                                # - Execute trades
                                
                        except websockets.exceptions.ConnectionClosed:
                            logger.error("WebSocket connection closed unexpectedly")
                            break
                else:
                    logger.error(f"Failed to subscribe: {subscription_data}")
                    
        except Exception as e:
            logger.error(f"WebSocket connection error: {str(e)}")
            
            # Increment reconnection attempts with exponential backoff
            reconnect_attempts += 1
            adjusted_delay = RECONNECT_DELAY * reconnect_attempts
            
            if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                logger.error(f"Maximum reconnection attempts reached ({MAX_RECONNECT_ATTEMPTS}). Stopping.")
                break
                
            logger.info(f"Reconnect attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS} in {adjusted_delay} seconds...")
            await asyncio.sleep(adjusted_delay)
            
    logger.error("WebSocket connection failed after multiple attempts")

async def main():
    """Main function to run the WebSocket subscription."""
    # Get WebSocket URL from environment
    ws_url = os.getenv('WEB3_PROVIDER_URI')
    
    if not ws_url:
        logger.error("No WEB3_PROVIDER_URI found in environment variables")
        return
        
    if not ws_url.startswith('wss://') and not ws_url.startswith('ws://'):
        logger.error("WebSocket subscription requires a WebSocket endpoint (wss:// or ws://)")
        return
        
    try:
        await subscribe_to_blocks(ws_url)
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting Native WebSocket Demo for Quantum RSI with Chainstack")
    
    try:
        # Run the async function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.error(f"Error in WebSocket demo: {e}")
        
    logger.info("WebSocket demo completed") 