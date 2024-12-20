#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "market_data.h"
#include "order_manager.h"

namespace py = pybind11;

class BitMEXConnector {
public:
    struct Config {
        std::string base_url;
        std::string symbol;
        std::string api_key;
        std::string api_secret;
        std::string order_id_prefix = "mm_bitmex_";
        bool should_ws_auth = true;
        bool post_only = false;
        int timeout = 7;
    };

    explicit BitMEXConnector(const Config& config);
    ~BitMEXConnector();

    // Market data methods
    MarketDepth get_order_book();
    void subscribe_market_data(const std::function<void(const MarketDepth&)>& callback);

    // Order management
    bool place_order(const Order& order);
    bool cancel_order(int64_t order_id);
    bool amend_order(const Order& order);
    
    struct ExecutionUpdate {
        int64_t order_id;
        std::string exec_id;
        double exec_price;
        double exec_quantity;
        std::string exec_type;
        std::chrono::system_clock::time_point timestamp;
    };

    void subscribe_executions(const std::function<void(const ExecutionUpdate&)>& callback);
    std::vector<ExecutionUpdate> get_recent_executions(size_t n = 100);

private:
    Config config_;
    py::object bitmex_instance_;
    py::object ws_thread_;
    
    void init_python();
    void convert_order_to_dict(const Order& order, py::dict& order_dict);
    Order convert_dict_to_order(const py::dict& order_dict);

    struct ConnectionState {
        std::atomic<bool> is_connected{false};
        std::atomic<int> retry_count{0};
        std::chrono::steady_clock::time_point last_heartbeat;
        static constexpr int MAX_RETRIES = 3;
        static constexpr auto RETRY_DELAY = std::chrono::seconds(5);
    };
    
    ConnectionState connection_state_;
    
    bool ensure_connection();
    void handle_connection_error();
    void reset_connection();

    struct PositionState {
        std::atomic<double> current_position{0.0};
        std::atomic<double> avg_entry_price{0.0};
        std::atomic<double> unrealized_pnl{0.0};
        std::mutex update_mutex;
    };
    
    PositionState position_state_;
    
    void update_position(const py::dict& position_data);
    double get_current_position() const { 
        return position_state_.current_position.load(); 
    }

    struct RateLimiter {
        static constexpr int MAX_REQUESTS_PER_MINUTE = 300;
        static constexpr int MAX_REQUESTS_PER_SECOND = 30;
        
        std::mutex mutex;
        std::deque<std::chrono::steady_clock::time_point> request_times;
        
        bool should_throttle();
        void add_request();
    };
    
    RateLimiter rate_limiter_;

    std::deque<ExecutionUpdate> execution_history_;
    std::mutex execution_mutex_;
    std::function<void(const ExecutionUpdate&)> execution_callback_;
}; 