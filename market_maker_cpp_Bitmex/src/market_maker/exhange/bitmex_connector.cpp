#include "bitmex_connector.h"
#include <pybind11/embed.h>

BitMEXConnector::BitMEXConnector(const Config& config) : config_(config) {
    init_python();
    
    // Import BitMEX module and create instance
    py::module bitmex = py::module::import("market_maker.auth.bitmex");
    bitmex_instance_ = bitmex.attr("BitMEX")(
        py::arg("base_url") = config.base_url,
        py::arg("symbol") = config.symbol,
        py::arg("apiKey") = config.api_key,
        py::arg("apiSecret") = config.api_secret,
        py::arg("orderIDPrefix") = config.order_id_prefix,
        py::arg("shouldWSAuth") = config.should_ws_auth,
        py::arg("postOnly") = config.post_only,
        py::arg("timeout") = config.timeout
    );
}

void BitMEXConnector::init_python() {
    static pybind11::scoped_interpreter guard{};
    
    // Import required Python modules
    py::module::import("market_maker.auth.APIKeyAuth");
    py::module::import("market_maker.auth.APIKeyAuthWithExpires");
}

bool BitMEXConnector::place_order(const Order& order) {
    try {
        py::dict order_dict;
        convert_order_to_dict(order, order_dict);
        
        py::object result = bitmex_instance_.attr("place_order")(order_dict);
        return !result.is_none();
    }
    catch (const py::error_already_set& e) {
        // Handle Python exceptions
        return false;
    }
}

MarketDepth BitMEXConnector::convert_orderbook_to_depth(const py::dict& orderbook) {
    MarketDepth depth;
    
    // Convert asks
    py::list asks = orderbook["asks"].cast<py::list>();
    for (size_t i = 0; i < std::min(asks.size(), MarketDepth::MAX_LEVELS); ++i) {
        py::dict level = asks[i].cast<py::dict>();
        depth.update_ask(i,
            level["price"].cast<double>(),
            level["size"].cast<double>()
        );
    }
    
    // Convert bids
    py::list bids = orderbook["bids"].cast<py::list>();
    for (size_t i = 0; i < std::min(bids.size(), MarketDepth::MAX_LEVELS); ++i) {
        py::dict level = bids[i].cast<py::dict>();
        depth.update_bid(i,
            level["price"].cast<double>(),
            level["size"].cast<double>()
        );
    }
    
    return depth;
}

void BitMEXConnector::convert_order_to_dict(const Order& order, py::dict& order_dict) {
    order_dict["symbol"] = config_.symbol;
    order_dict["side"] = order.side == OrderSide::BUY ? "Buy" : "Sell";
    order_dict["orderQty"] = order.quantity;
    order_dict["price"] = order.price;
    order_dict["ordType"] = "Limit";
    
    if (config_.post_only) {
        order_dict["execInst"] = "ParticipateDoNotInitiate";
    }
}

void BitMEXConnector::subscribe_market_data(
    const std::function<void(const MarketDepth&)>& callback) {
    
    // Import WebSocket module
    py::module ws = py::module::import("market_maker.ws.ws_thread");
    
    // Create WebSocket instance
    ws_thread_ = ws.attr("BitMEXWebsocket")(
        py::arg("endpoint") = config_.base_url,
        py::arg("symbol") = config_.symbol,
        py::arg("api_key") = config_.api_key,
        py::arg("api_secret") = config_.api_secret
    );
    
    // Subscribe to orderBook10 topic
    ws_thread_.attr("subscribe")(py::str("orderBook10"));
    
    // Start WebSocket thread
    ws_thread_.attr("connect")();
    
    // Create callback wrapper
    auto py_callback = [callback](const py::dict& data) {
        MarketDepth depth = convert_orderbook_to_depth(data);
        callback(depth);
    };
    
    // Register callback
    ws_thread_.attr("on_message")(py::cpp_function(py_callback));
} 

bool BitMEXConnector::ensure_connection() {
    if (connection_state_.is_connected) {
        return true;
    }
    
    if (connection_state_.retry_count >= ConnectionState::MAX_RETRIES) {
        throw std::runtime_error("Max connection retries exceeded");
    }
    
    try {
        reset_connection();
        connection_state_.is_connected = true;
        connection_state_.retry_count = 0;
        connection_state_.last_heartbeat = std::chrono::steady_clock::now();
        return true;
    }
    catch (const std::exception& e) {
        handle_connection_error();
        return false;
    }
}

void BitMEXConnector::handle_connection_error() {
    connection_state_.is_connected = false;
    connection_state_.retry_count++;
    
    std::this_thread::sleep_for(ConnectionState::RETRY_DELAY);
} 

bool BitMEXConnector::RateLimiter::should_throttle() {
    std::lock_guard<std::mutex> lock(mutex);
    auto now = std::chrono::steady_clock::now();
    
    // Remove old requests
    while (!request_times.empty() && 
           now - request_times.front() > std::chrono::minutes(1)) {
        request_times.pop_front();
    }
    
    return request_times.size() >= MAX_REQUESTS_PER_MINUTE;
} 

void BitMEXConnector::subscribe_executions(
    const std::function<void(const ExecutionUpdate&)>& callback) {
    
    execution_callback_ = callback;
    
    // Subscribe to execution topic on WebSocket
    ws_thread_.attr("subscribe")(py::str("execution"));
    
    // Add execution handler
    auto py_execution_handler = [this](const py::dict& data) {
        ExecutionUpdate update;
        update.order_id = data["orderID"].cast<int64_t>();
        update.exec_id = data["execID"].cast<std::string>();
        update.exec_price = data["price"].cast<double>();
        update.exec_quantity = data["lastQty"].cast<double>();
        update.exec_type = data["execType"].cast<std::string>();
        update.timestamp = std::chrono::system_clock::now();
        
        // Store execution
        {
            std::lock_guard<std::mutex> lock(execution_mutex_);
            execution_history_.push_back(update);
            if (execution_history_.size() > 1000) {
                execution_history_.pop_front();
            }
        }
        
        // Notify callback
        if (execution_callback_) {
            execution_callback_(update);
        }
    };
    
    ws_thread_.attr("on_execution")(py::cpp_function(py_execution_handler));
} 