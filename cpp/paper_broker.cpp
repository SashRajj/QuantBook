// PaperBookkeeper implementation. See header for the contract.
//
// The four-branch fill logic mirrors execution.py::PaperBroker._apply_fill
// line-for-line; we intentionally do not "improve" the order of
// operations so the floating-point output stays bit-identical.

#include "paper_broker.h"

#include <cmath>
#include <stdexcept>

namespace qbexec {

PaperBookkeeper::PaperBookkeeper(double starting_cash,
                                 double slippage_bps,
                                 double commission_bps,
                                 double fill_ratio)
    : starting_cash_(starting_cash),
      slippage_bps_(slippage_bps),
      commission_bps_(commission_bps),
      fill_ratio_(fill_ratio),
      cash_(starting_cash),
      realized_pnl_(0.0),
      total_commissions_(0.0) {
    if (!(fill_ratio > 0.0 && fill_ratio <= 1.0)) {
        throw std::invalid_argument("fill_ratio must be in (0, 1]");
    }
}

void PaperBookkeeper::update_prices(
        const std::unordered_map<std::string, double>& prices) {
    for (const auto& kv : prices) {
        prices_[kv.first] = kv.second;
    }
}

void PaperBookkeeper::update_price(const std::string& symbol, double price) {
    prices_[symbol] = price;
}

bool PaperBookkeeper::has_price(const std::string& symbol) const {
    return prices_.find(symbol) != prices_.end();
}

double PaperBookkeeper::last_price(const std::string& symbol) const {
    auto it = prices_.find(symbol);
    if (it == prices_.end()) {
        throw std::out_of_range(
            "no price for " + symbol + "; call update_prices first");
    }
    return it->second;
}

bool PaperBookkeeper::has_position(const std::string& symbol) const {
    return positions_.find(symbol) != positions_.end();
}

PositionRecord PaperBookkeeper::get_position(const std::string& symbol) const {
    auto it = positions_.find(symbol);
    if (it == positions_.end()) {
        throw std::out_of_range("no position for " + symbol);
    }
    return it->second;
}

std::vector<std::pair<std::string, PositionRecord>>
PaperBookkeeper::all_positions() const {
    std::vector<std::pair<std::string, PositionRecord>> out;
    out.reserve(positions_.size());
    for (const auto& kv : positions_) {
        out.emplace_back(kv.first, kv.second);
    }
    return out;
}

FillResult PaperBookkeeper::simulate_and_apply(const std::string& symbol,
                                               double order_qty) {
    // Mirrors PaperBroker.submit_order's fill-pricing block exactly.
    double fill_qty = order_qty * fill_ratio_;
    double px = last_price(symbol);
    double sign = (fill_qty > 0.0) ? 1.0 : -1.0;
    double slip = slippage_bps_ / 1e4 * sign;
    double fill_px = px * (1.0 + slip);
    double notional = std::fabs(fill_qty) * fill_px;
    double commission = notional * commission_bps_ / 1e4;

    apply_fill(symbol, fill_qty, fill_px, commission);
    return FillResult{fill_qty, fill_px, commission};
}

void PaperBookkeeper::apply_fill(const std::string& symbol,
                                 double qty,
                                 double price,
                                 double commission) {
    // Four-branch position-transition logic, matching execution.py:
    //   open / same-side add / partial close / full-close-or-flip.
    auto it = positions_.find(symbol);
    if (it == positions_.end()) {
        // Open: no prior position.
        positions_.emplace(symbol, PositionRecord{qty, price});
    } else {
        PositionRecord& pos = it->second;
        bool same_direction = pos.qty * qty > 0.0;
        double new_qty = pos.qty + qty;

        if (same_direction) {
            // Sign-agnostic weighted-average basis. Order of operations
            // matches the Python expression exactly:
            //     pos.avg_price = (pos.avg_price * pos.qty + price * qty) / new_qty
            pos.avg_price =
                (pos.avg_price * pos.qty + price * qty) / new_qty;
            pos.qty = new_qty;
        } else {
            bool full_close_or_flip = std::fabs(qty) >= std::fabs(pos.qty);
            double closed_qty = full_close_or_flip ? pos.qty : -qty;
            realized_pnl_ += closed_qty * (price - pos.avg_price);

            if (full_close_or_flip) {
                if (new_qty == 0.0) {
                    positions_.erase(it);
                } else {
                    pos.qty = new_qty;
                    pos.avg_price = price;  // flipped side's new basis
                }
            } else {
                pos.qty = new_qty;  // basis preserved on partial close
            }
        }
    }

    // Same ordering as Python: cash debit/credit + commission.
    cash_ -= qty * price + commission;
    total_commissions_ += commission;
}

double PaperBookkeeper::equity(
        const std::unordered_map<std::string, double>& marks,
        bool strict) const {
    double mv = 0.0;
    for (const auto& kv : positions_) {
        auto it = marks.find(kv.first);
        if (it == marks.end()) {
            if (strict) {
                throw std::out_of_range(
                    "missing price for held position " + kv.first);
            }
            continue;
        }
        mv += kv.second.qty * it->second;
    }
    return cash_ + mv;
}

double PaperBookkeeper::invariant_residual(
        const std::unordered_map<std::string, double>& marks) const {
    double mv = 0.0;
    for (const auto& kv : positions_) {
        auto it = marks.find(kv.first);
        if (it == marks.end()) continue;
        mv += kv.second.qty * it->second;
    }
    // Realised PnL is *already* baked into `cash_` (a +1000 trade
    // credits cash by +1000 directly). With prices flat at fill, the
    // honest identity is therefore
    //     cash + mv + commissions == starting_cash + realized_pnl
    // which we rearrange as `lhs - rhs` for a signed residual.
    return (cash_ + mv + total_commissions_)
           - (starting_cash_ + realized_pnl_);
}

}  // namespace qbexec
