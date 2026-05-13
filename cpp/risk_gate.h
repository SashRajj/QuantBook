// RiskGate: C++ core of PreTradeRiskGate.check().
//
// Pure pre-trade computation: takes the proposed orders, current
// positions, prices, equity, ADV map, and limits; returns the first
// breach as a human-readable string, or empty string for "all clear".
//
// State (the high-water mark) is held by the Python wrapper. We pass
// it in on every call so the C++ side stays trivially testable.

#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace qbexec {

// Plain config struct. `std::optional` is used to mirror Python's
// `None` for "this check disabled".
struct RiskLimitsC {
    std::optional<double> max_gross_leverage;
    std::optional<double> max_net_exposure;
    std::optional<double> max_position_pct;
    std::optional<int> max_positions;
    std::optional<double> max_adv_participation_pct;
    std::optional<double> max_daily_turnover_pct;
    std::optional<double> drawdown_kill_pct;
    std::unordered_set<std::string> blocked_symbols;
    bool strict_adv = false;
};

// Light-weight order view -- just what the risk gate looks at.
struct OrderView {
    std::string symbol;
    double qty;
};

// Run every gate against the proposed basket. Returns an empty string
// for "all clear", otherwise the first breach as a formatted string
// matching the Python implementation's wording.
std::string risk_check(
    const std::vector<OrderView>& orders,
    const std::unordered_map<std::string, double>& current_positions,
    const std::unordered_map<std::string, double>& prices,
    double equity,
    std::optional<double> hwm,
    const std::unordered_map<std::string, double>& adv_dollar,
    const RiskLimitsC& limits);

}  // namespace qbexec
