/**
 * WebSocket Demo for Quantum RSI - JavaScript Version
 * 
 * This script demonstrates how to use WebSockets with web3.js to get
 * real-time data from the Ethereum blockchain using Chainstack.
 * 
 * It includes robust reconnection logic for maintaining a stable connection.
 */

// Import required packages
const { Web3 } = require('web3');
const dotenv = require('dotenv');
const fs = require('fs');
const path = require('path');

// Load environment variables
dotenv.config({ path: path.resolve(__dirname, '../.env') });

// Get WebSocket URL from environment
const wsUrl = process.env.WEB3_PROVIDER_URI;

// Check if WebSocket URL is available
if (!wsUrl) {
  console.error('No WEB3_PROVIDER_URI found in environment variables');
  process.exit(1);
}

// Check if it's a WebSocket URL
if (!wsUrl.startsWith('wss://') && !wsUrl.startsWith('ws://')) {
  console.error('WebSocket subscription requires a WebSocket endpoint (wss:// or ws://)');
  process.exit(1);
}

// Extract provider details for logging (without credentials)
let providerHost = '';
if (wsUrl.includes('@')) {
  providerHost = wsUrl.split('@')[1].split('/')[0];
} else {
  providerHost = wsUrl.split('/')[2];
}

console.log(`Connecting to WebSocket provider: wss://${providerHost}`);

// Reconnection settings
const maxReconnectAttempts = 10;
const reconnectDelay = 5000; // in milliseconds

// WebSocket provider options with reconnection logic
const wsOptions = {
  autoReconnect: true,
  reconnectDelay: reconnectDelay,
  maxReconnectAttempts: maxReconnectAttempts,
  clientConfig: {
    keepalive: true,
    keepaliveInterval: 30000, // 30 seconds
  }
};

// Create web3 instance with WebSocket provider
let web3;
let subscription;
let reconnectAttempts = 0;
let lastBlockTime = Date.now();

// Initialize WebSocket connection
function initializeWebSocket() {
  console.log(`Attempt #${reconnectAttempts + 1} to connect to WebSocket...`);
  
  // Create Web3 instance with WebSocket provider
  const provider = new Web3.providers.WebsocketProvider(wsUrl, wsOptions);
  web3 = new Web3(provider);
  
  // Set up event handlers for the provider
  setupProviderListeners(provider);
  
  // Subscribe to new block headers
  subscribeToNewBlocks();
}

// Set up event listeners for the provider
function setupProviderListeners(provider) {
  // Handle provider connection
  provider.on('connect', () => {
    console.log('WebSocket connection established successfully');
    reconnectAttempts = 0;
  });
  
  // Handle provider error
  provider.on('error', (error) => {
    console.error('WebSocket provider error:', error);
  });
  
  // Handle provider end/disconnect
  provider.on('end', (code, reason) => {
    console.log(`WebSocket connection closed: ${code} - ${reason}`);
    handleReconnect();
  });
}

// Subscribe to new block headers
async function subscribeToNewBlocks() {
  try {
    // Create a new subscription to the 'newBlockHeaders' event
    subscription = await web3.eth.subscribe('newBlockHeaders');
    console.log(`Subscribed to new block headers`);
    
    // Handle new block headers
    subscription.on('data', handleNewBlock);
    subscription.on('error', handleSubscriptionError);
  } catch (error) {
    console.error(`Error subscribing to new blocks: ${error}`);
    handleReconnect();
  }
}

// Handle new block header
function handleNewBlock(blockHeader) {
  // Calculate block information
  const blockNumber = blockHeader.number;
  const gasUsed = blockHeader.gasUsed;
  const gasLimit = blockHeader.gasLimit;
  const usagePercent = (gasUsed / gasLimit) * 100;
  
  // Log block information
  console.log(`Block #${blockNumber}: Gas usage ${usagePercent.toFixed(2)}% (${gasUsed}/${gasLimit})`);
  
  // Calculate time since last block
  const currentTime = Date.now();
  const timeDiff = (currentTime - lastBlockTime) / 1000;
  console.log(`Time since last block: ${timeDiff.toFixed(2)} seconds`);
  lastBlockTime = currentTime;
  
  // Example: Trade signal analysis could be done here
}

// Handle subscription error
function handleSubscriptionError(error) {
  console.error('Subscription error:', error);
  
  // Unsubscribe and reconnect
  if (subscription) {
    try {
      subscription.unsubscribe((error, success) => {
        if (success) {
          console.log('Successfully unsubscribed');
        }
        handleReconnect();
      });
    } catch (unsubError) {
      console.error('Error unsubscribing:', unsubError);
      handleReconnect();
    }
  } else {
    handleReconnect();
  }
}

// Handle reconnection
function handleReconnect() {
  reconnectAttempts++;
  
  if (reconnectAttempts >= maxReconnectAttempts) {
    console.error(`Maximum reconnection attempts reached (${maxReconnectAttempts}). Stopping.`);
    process.exit(1);
  }
  
  // Use exponential backoff for reconnection
  const adjustedDelay = reconnectDelay * reconnectAttempts;
  console.log(`Reconnect attempt ${reconnectAttempts}/${maxReconnectAttempts} in ${adjustedDelay/1000} seconds...`);
  
  setTimeout(() => {
    initializeWebSocket();
  }, adjustedDelay);
}

// Handle script termination
process.on('SIGINT', async () => {
  console.log('Terminating WebSocket demo...');
  
  // Clean up subscription if it exists
  if (subscription) {
    try {
      await subscription.unsubscribe();
      console.log('Successfully unsubscribed');
    } catch (error) {
      console.error('Error unsubscribing:', error);
    }
  }
  
  process.exit(0);
});

// Start the WebSocket demo
console.log('Starting WebSocket demo for Quantum RSI with Chainstack');
initializeWebSocket(); 