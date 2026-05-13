// RiskGate implementation. Mirrors PreTradeRiskGate.check() exactly:
// same order of checks, same floating-point math, same human-readable
// breach strings (formatted via snprintf to avoid locale surprises).

#include "risk_gate.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace qbexec {

namespace {

// snprintf wrapper that returns std::string. Sized generously; the
// breach strings are all short.
std::string format(const char* fmt, ...) {
    char buf[512];
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    return std::string(buf);
}

}  // namespace

std::string risk_check(
    const std::vector<OrderView>& orders,
    const std::unordered_map<std::string, double>& current_positions,
    const std::unordered_map<std::string, double>& prices,
    double equity,
    std::optional<double> hwm,
    const std::unordered_map<std::string, double>& adv_dollar,
    const RiskLimitsC& limits) {

    // Drawdown kill switch (fires first; can hard-flat regardless of basket).
    if (hwm.has_value() && hwm.value() > 0.0
            && limits.drawdown_kill_pct.has_value()) {
        double dd = equity / hwm.value() - 1.0;
        if (dd <= -limits.drawdown_kill_pct.value()) {
            return format(
                "drawdown kill: equity=%.2f hwm=%.2f dd=%.2f%% limit=%.2f%%",
                equity, hwm.value(), dd * 100.0,
                -limits.drawdown_kill_pct.value() * 100.0);
        }
    }

    // Blocked symbols.
    for (const auto& o : orders) {
        if (limits.blocked_symbols.count(o.symbol)) {
            return "blocked symbol in order list: " + o.symbol;
        }
    }

    // Post-trade book.
    std::unordered_map<std::string, double> post_qty = current_positions;
    for (const auto& o : orders) {
        post_qty[o.symbol] = post_qty[o.symbol] + o.qty;  // default-inits to 0
    }

    double long_notional = 0.0;
    double short_notional = 0.0;
    for (const auto& kv : post_qty) {
        auto pit = prices.find(kv.first);
        if (pit == prices.end()) continue;
        double notional = kv.second * pit->second;
        if (kv.second > 0.0) {
            long_notional += notional;
        } else if (kv.second < 0.0) {
            short_notional += -notional;  // matches Python's `-sum(...)` form
        }
    }
    double gross = long_notional + short_notional;
    double net = long_notional - short_notional;

    if (equity <= 0.0) {
        return format("non-positive equity: %.2f", equity);
    }

    if (limits.max_gross_leverage.has_value()
            && gross / equity > limits.max_gross_leverage.value()) {
        return format(
            "gross leverage breach: gross=%.0f equity=%.0f ratio=%.3f limit=%g",
            gross, equity, gross / equity, limits.max_gross_leverage.value());
    }

    if (limits.max_net_exposure.has_value()
            && std::fabs(net) / equity > limits.max_net_exposure.value()) {
        return format(
            "net exposure breach: net=%+.0f ratio=%+.3f limit=+/-%g",
            net, net / equity, limits.max_net_exposure.value());
    }

    // Per-name cap.
    if (limits.max_position_pct.has_value()) {
        for (const auto& kv : post_qty) {
            auto pit = prices.find(kv.first);
            if (pit == prices.end()) continue;
            double pos_pct = std::fabs(kv.second * pit->second) / equity;
            if (pos_pct > limits.max_position_pct.value()) {
                return format(
                    "per-name cap breach: %s pos_pct=%.3f limit=%g",
                    kv.first.c_str(), pos_pct,
                    limits.max_position_pct.value());
            }
        }
    }

    // Position count.
    if (limits.max_positions.has_value()) {
        int non_zero = 0;
        for (const auto& kv : post_qty) {
            if (std::fabs(kv.second) > 1e-9) non_zero++;
        }
        if (non_zero > limits.max_positions.value()) {
            return format("too many positions: %d > %d",
                          non_zero, limits.max_positions.value());
        }
    }

    // ADV participation per order.
    if (limits.max_adv_participation_pct.has_value()) {
        for (const auto& o : orders) {
            auto pit = prices.find(o.symbol);
            if (pit == prices.end()) continue;
            double order_notional = std::fabs(o.qty * pit->second);
            auto ait = adv_dollar.find(o.symbol);
            if (ait == adv_dollar.end() || ait->second <= 0.0) {
                if (limits.strict_adv) {
                    return format(
                        "ADV unknown for %s; cannot enforce "
                        "participation limit (set strict_adv=False to skip)",
                        o.symbol.c_str());
                }
                continue;  // soft pass
            }
            double participation = order_notional / ait->second;
            if (participation > limits.max_adv_participation_pct.value()) {
                return format(
                    "ADV participation breach: %s order_notional=%.0f "
                    "adv=%.0f participation=%.3f limit=%g",
                    o.symbol.c_str(), order_notional, ait->second,
                    participation,
                    limits.max_adv_participation_pct.value());
            }
        }
    }

    // Daily turnover.
    if (limits.max_daily_turnover_pct.has_value()) {
        double total_turnover = 0.0;
        for (const auto& o : orders) {
            auto pit = prices.find(o.symbol);
            double px = (pit == prices.end()) ? 0.0 : pit->second;
            total_turnover += std::fabs(o.qty * px);
        }
        if (total_turnover / equity > limits.max_daily_turnover_pct.value()) {
            return format(
                "daily turnover breach: turnover=%.0f ratio=%.3f limit=%g",
                total_turnover, total_turnover / equity,
                limits.max_daily_turnover_pct.value());
        }
    }

    return "";  // all clear
}

}  // namespace qbexec
