# Quantum RSI Examples

This directory contains example scripts that demonstrate various features of the Quantum RSI trading system, with a focus on real-time blockchain data access using WebSockets.

## WebSocket Demos

WebSockets are strongly recommended over HTTP for real-time blockchain data because they:

- Provide a persistent, bidirectional connection (no need to constantly establish new connections)
- Allow for real-time data streaming via subscriptions (e.g., new blocks, logs, pending transactions)
- Have a longer idle timeout (3600 seconds vs 60 seconds for HTTP)
- Are more efficient for applications requiring continuous data flow

We provide multiple WebSocket implementations to demonstrate best practices.

### Native Python WebSocket Demo (Recommended)

The `websocket_native_demo.py` script demonstrates how to use the `websockets` library to establish a direct WebSocket connection to Chainstack's Ethereum node. This is the most efficient approach for Python applications.

To run the native Python WebSocket demo:

```bash
pip install websockets
python examples/websocket_native_demo.py
```

### JavaScript WebSocket Demo with web3.js

The `websocket_demo.js` script demonstrates how to use WebSockets with web3.js to get real-time data from the Ethereum blockchain using Chainstack.

To run the JavaScript WebSocket demo:

1. First, install the dependencies:
```bash
cd examples
npm install
```

2. Then run the demo:
```bash
node websocket_demo.js
```

### Python WebSocket Demo with web3.py

The `websocket_demo.py` script demonstrates how to use WebSockets with the Quantum RSI system using web3.py's legacy WebSocket provider.

To run the Python WebSocket demo:

```bash
python websocket_demo.py
```

## Features of the WebSocket Demos

All WebSocket demos showcase:

1. **Real-time Block Monitoring**: Subscribe to new blocks as they're mined
2. **Robust Reconnection Logic**: Automatically reconnect if the connection drops
3. **Exponential Backoff**: Increase wait time between reconnection attempts
4. **Error Handling**: Proper handling of WebSocket errors
5. **Clean Shutdown**: Properly unsubscribe and close connections on exit

## Connection Limitations

When working with WebSockets, be aware of these limitations from Chainstack:

- **Idle Timeout**: 3,600 seconds (1 hour) - connection will be terminated if idle for this duration
- **Maximum Connections**: 500 concurrent WebSocket connections maximum
- **Common Errors**: Code 1006 (abnormal closure) and ECONNRESET indicate disconnection

## WebSocket vs HTTP for EVM Node Connections

The general recommendation is to use WebSocket rather than HTTP when connecting to an EVM node. WebSocket has the advantage of connection reusability and the keep-alive mechanism.

| Feature | HTTP | WebSocket |
|---------|------|-----------|
| Idle Timeout | 60 seconds | 3600 seconds (1 hour) |
| Connection Type | New connection per request | Persistent connection |
| Recommended For | Short requests (eth_call, eth_sendRawTransaction) | Long requests (eth_getLogs), subscriptions |
| Maximum Requests | 1,000 per connection | Not applicable (stream-based) |

## Requirements

- A Chainstack WebSocket endpoint (set in your `.env` file as `WEB3_PROVIDER_URI`)
- For Native Python: `websockets` and `dotenv` libraries
- For web3.py: `web3`, `asyncio`, and other dependencies from `requirements.txt`
- For JavaScript: `web3.js` and `dotenv` (installed via npm) 