// PaperBookkeeper: C++ core of the paper broker's fill engine.
//
// Owns: cash, per-symbol positions (qty + avg basis), realized PnL,
// cumulative commissions, and the four-branch fill logic
// (open / same-side add / partial close / full-close-or-flip).
//
// Deliberately knows nothing about Order objects, idempotency, status
// transitions, or audit logging. Those concerns stay in the Python
// PaperBroker wrapper so the public API is preserved.
//
// Numerical contract: produces results identical to the Python
// reference implementation in execution.py to the last bit, given the
// same inputs and the same floating-point ordering of operations.

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace qbexec {

struct PositionRecord {
    double qty;          // signed; negative means short
    double avg_price;    // weighted-average basis
};

// Result of simulating a single submitted order against the in-memory
// price feed: enough to let the Python wrapper update the Order
// dataclass (filled_qty, avg_fill_price, status) without re-doing the
// math.
struct FillResult {
    double fill_qty;
    double fill_price;
    double commission;
};

class PaperBookkeeper {
public:
    PaperBookkeeper(double starting_cash,
                    double slippage_bps,
                    double commission_bps,
                    double fill_ratio);

    // Price feed update; merge semantics (matches Python dict.update).
    void update_prices(const std::unordered_map<std::string, double>& prices);
    void update_price(const std::string& symbol, double price);

    // Returns the last seen price; throws std::out_of_range if missing.
    double last_price(const std::string& symbol) const;
    bool has_price(const std::string& symbol) const;

    // Simulate a fill for `order_qty` on `symbol`. Applies slippage and
    // commission, mutates positions/cash/realized_pnl, and returns the
    // fill metadata the caller needs to update the Order object.
    FillResult simulate_and_apply(const std::string& symbol,
                                  double order_qty);

    // Direct fill application (used for testing and for paths that
    // already know the fill price / commission, e.g. a live broker
    // adapter that bypasses the slippage model).
    void apply_fill(const std::string& symbol,
                    double qty,
                    double price,
                    double commission);

    // Account state accessors.
    double cash() const { return cash_; }
    double realized_pnl() const { return realized_pnl_; }
    double total_commissions() const { return total_commissions_; }
    double starting_cash() const { return starting_cash_; }
    double slippage_bps() const { return slippage_bps_; }
    double commission_bps() const { return commission_bps_; }
    double fill_ratio() const { return fill_ratio_; }

    bool has_position(const std::string& symbol) const;
    PositionRecord get_position(const std::string& symbol) const;
    std::vector<std::pair<std::string, PositionRecord>> all_positions() const;

    // Marked-to-market equity. Throws if `strict` and a held position
    // has no price in `marks`.
    double equity(const std::unordered_map<std::string, double>& marks,
                  bool strict) const;

    // For internal sanity-checking and tests. Returns the residual
    // cash + sum(qty * mark) + realized_pnl - (starting_cash - total_commissions);
    // an honest implementation keeps this within float noise.
    double invariant_residual(
        const std::unordered_map<std::string, double>& marks) const;

private:
    double starting_cash_;
    double slippage_bps_;
    double commission_bps_;
    double fill_ratio_;

    double cash_;
    double realized_pnl_;
    double total_commissions_;

    std::unordered_map<std::string, PositionRecord> positions_;
    std::unordered_map<std::string, double> prices_;
};

}  // namespace qbexec
