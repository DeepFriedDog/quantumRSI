"""
WebSocket Demo for Quantum RSI

This script demonstrates how to use WebSockets with the Quantum RSI system
to get real-time data from the Ethereum blockchain using Chainstack.

It includes robust reconnection logic for maintaining a stable connection.
"""

import asyncio
import sys
import os
import logging
import json
import time
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the quantum_rsi module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import web3 components
from web3 import Web3, AsyncWeb3
from web3.providers.legacy_websocket import LegacyWebSocketProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('websocket_demo')

# Load environment variables
load_dotenv()

# Maximum reconnection attempts
MAX_RECONNECT_ATTEMPTS = 10
# Time to wait between reconnection attempts (in seconds)
RECONNECT_DELAY = 5

async def subscribe_to_new_blocks():
    """
    Subscribe to new block headers with automatic reconnection logic.
    """
    # Get the WebSocket URL from environment variables
    ws_url = os.getenv('WEB3_PROVIDER_URI')
    if not ws_url:
        logger.error("No WEB3_PROVIDER_URI found in environment variables")
        return
        
    if not ws_url.startswith('wss://') and not ws_url.startswith('ws://'):
        logger.error("WebSocket subscription requires a WebSocket endpoint (wss:// or ws://)")
        return

    # Extract provider details (without credentials) for logging
    if '@' in ws_url:
        provider_host = ws_url.split('@')[1].split('/')[0]
    else:
        provider_host = ws_url.split('/')[2]
    
    logger.info(f"Connecting to WebSocket provider: wss://{provider_host}")
    
    reconnect_attempts = 0
    while reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
        try:
            # Create a WebSocket connection with the provider
            logger.info(f"Attempt #{reconnect_attempts + 1} to connect to WebSocket...")
            
            # Create a WebSocket provider
            ws_provider = LegacyWebSocketProvider(ws_url)
            w3 = AsyncWeb3(ws_provider)
            
            try:
                # Check if the connection is working
                is_connected = await w3.is_connected()
                if not is_connected:
                    logger.error("Failed to connect to Web3 provider")
                    raise Exception("WebSocket connection failed")
                
                logger.info("WebSocket connection established successfully")
                
                # Subscribe to new block headers
                subscription_id = await w3.eth.subscribe("newHeads")
                logger.info(f"Subscribed to new block headers (ID: {subscription_id})")
                
                # Reset reconnection attempts on successful connection
                reconnect_attempts = 0
                
                # Process incoming messages
                try:
                    # Create a message handler for new blocks
                    async def handle_message(message):
                        if "result" in message and "params" in message:
                            # Process block header
                            block_header = message["params"]["result"]
                            # Parse block information
                            block_number = int(block_header['number'], 16)
                            gas_used = int(block_header['gasUsed'], 16)
                            gas_limit = int(block_header['gasLimit'], 16)
                            usage_percent = gas_used / gas_limit * 100
                            
                            # Log block information
                            logger.info(f"Block #{block_number}: Gas usage {usage_percent:.2f}% ({gas_used}/{gas_limit})")
                            
                            # Calculate time since last block (if available)
                            current_time = time.time()
                            if hasattr(subscribe_to_new_blocks, 'last_block_time'):
                                time_diff = current_time - subscribe_to_new_blocks.last_block_time
                                logger.info(f"Time since last block: {time_diff:.2f} seconds")
                            subscribe_to_new_blocks.last_block_time = current_time
                    
                    # Register the message handler
                    ws_provider.on_message = handle_message
                    
                    # Keep the connection alive
                    while True:
                        await asyncio.sleep(30)  # Check every 30 seconds
                        
                        # Verify the connection is still active
                        if not await w3.is_connected():
                            logger.error("WebSocket connection lost")
                            raise Exception("WebSocket connection lost")
                            
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {str(e)}")
                    # Try to unsubscribe before reconnecting
                    try:
                        await w3.eth.unsubscribe(subscription_id)
                        logger.info("Unsubscribed from new block headers")
                    except Exception as unsubscribe_error:
                        logger.error(f"Error unsubscribing: {str(unsubscribe_error)}")
                
                # Close connection
                try:
                    if ws_provider:
                        ws_provider.disconnect()
                        logger.info("WebSocket connection closed")
                except Exception as disconnect_error:
                    logger.error(f"Error disconnecting: {str(disconnect_error)}")
                    
            except Exception as processing_error:
                logger.error(f"Error during WebSocket processing: {str(processing_error)}")
                # Close connection on error
                try:
                    if ws_provider:
                        ws_provider.disconnect()
                except Exception:
                    pass
                
                # Increment reconnect attempts and wait before retrying
                reconnect_attempts += 1
                adjusted_delay = RECONNECT_DELAY * reconnect_attempts
                logger.info(f"Waiting {adjusted_delay} seconds before reconnecting...")
                await asyncio.sleep(adjusted_delay)
                continue
                
        except Exception as connection_error:
            logger.error(f"Connection error: {str(connection_error)}")
            # Check if we need to increase the delay for exponential backoff
            adjusted_delay = RECONNECT_DELAY * (reconnect_attempts + 1)
            reconnect_attempts += 1
            
            if reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                logger.error(f"Maximum reconnection attempts reached ({MAX_RECONNECT_ATTEMPTS}). Stopping.")
                break
                
            logger.info(f"Reconnect attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS} in {adjusted_delay} seconds...")
            await asyncio.sleep(adjusted_delay)

    logger.error("WebSocket connection failed after multiple attempts")

if __name__ == "__main__":
    logger.info("Starting WebSocket demo for Quantum RSI with Chainstack")
    
    # Initialize the timestamp for the first block
    subscribe_to_new_blocks.last_block_time = time.time()
    
    try:
        # Run the async function
        asyncio.run(subscribe_to_new_blocks())
    except KeyboardInterrupt:
        logger.info("Demo stopped by user")
    except Exception as e:
        logger.error(f"Error in WebSocket demo: {e}")
        
    logger.info("WebSocket demo completed") 