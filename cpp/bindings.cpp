// pybind11 bindings for the qbexec_cpp module.
//
// Surface area is deliberately narrow: just enough for the Python
// wrappers in execution.py and risk.py to delegate the hot path
// (fill simulation + bookkeeping, pre-trade risk checks) to C++.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>          // std::vector, std::unordered_map, std::pair

#include "paper_broker.h"
#include "risk_gate.h"

namespace py = pybind11;
using namespace qbexec;

PYBIND11_MODULE(qbexec_cpp, m) {
    m.doc() = "C++ core for the quantresearch execution layer";

    py::class_<PositionRecord>(m, "PositionRecord")
        .def_readonly("qty", &PositionRecord::qty)
        .def_readonly("avg_price", &PositionRecord::avg_price);

    py::class_<FillResult>(m, "FillResult")
        .def_readonly("fill_qty", &FillResult::fill_qty)
        .def_readonly("fill_price", &FillResult::fill_price)
        .def_readonly("commission", &FillResult::commission);

    py::class_<PaperBookkeeper>(m, "PaperBookkeeper")
        .def(py::init<double, double, double, double>(),
             py::arg("starting_cash"),
             py::arg("slippage_bps"),
             py::arg("commission_bps"),
             py::arg("fill_ratio"))
        .def("update_prices", &PaperBookkeeper::update_prices)
        .def("update_price", &PaperBookkeeper::update_price)
        .def("last_price", &PaperBookkeeper::last_price)
        .def("has_price", &PaperBookkeeper::has_price)
        .def("simulate_and_apply", &PaperBookkeeper::simulate_and_apply,
             py::arg("symbol"), py::arg("order_qty"))
        .def("apply_fill", &PaperBookkeeper::apply_fill,
             py::arg("symbol"), py::arg("qty"),
             py::arg("price"), py::arg("commission"))
        .def("cash", &PaperBookkeeper::cash)
        .def("realized_pnl", &PaperBookkeeper::realized_pnl)
        .def("total_commissions", &PaperBookkeeper::total_commissions)
        .def("starting_cash", &PaperBookkeeper::starting_cash)
        .def("slippage_bps", &PaperBookkeeper::slippage_bps)
        .def("commission_bps", &PaperBookkeeper::commission_bps)
        .def("fill_ratio", &PaperBookkeeper::fill_ratio)
        .def("has_position", &PaperBookkeeper::has_position)
        .def("get_position", &PaperBookkeeper::get_position)
        .def("all_positions", &PaperBookkeeper::all_positions)
        .def("equity", &PaperBookkeeper::equity,
             py::arg("marks"), py::arg("strict") = true)
        .def("invariant_residual", &PaperBookkeeper::invariant_residual);

    py::class_<RiskLimitsC>(m, "RiskLimitsC")
        .def(py::init<>())
        .def_readwrite("max_gross_leverage", &RiskLimitsC::max_gross_leverage)
        .def_readwrite("max_net_exposure", &RiskLimitsC::max_net_exposure)
        .def_readwrite("max_position_pct", &RiskLimitsC::max_position_pct)
        .def_readwrite("max_positions", &RiskLimitsC::max_positions)
        .def_readwrite("max_adv_participation_pct",
                       &RiskLimitsC::max_adv_participation_pct)
        .def_readwrite("max_daily_turnover_pct",
                       &RiskLimitsC::max_daily_turnover_pct)
        .def_readwrite("drawdown_kill_pct", &RiskLimitsC::drawdown_kill_pct)
        .def_readwrite("blocked_symbols", &RiskLimitsC::blocked_symbols)
        .def_readwrite("strict_adv", &RiskLimitsC::strict_adv);

    py::class_<OrderView>(m, "OrderView")
        .def(py::init<>())
        .def(py::init([](const std::string& sym, double qty) {
            return OrderView{sym, qty};
        }))
        .def_readwrite("symbol", &OrderView::symbol)
        .def_readwrite("qty", &OrderView::qty);

    m.def("risk_check", &risk_check,
          py::arg("orders"),
          py::arg("current_positions"),
          py::arg("prices"),
          py::arg("equity"),
          py::arg("hwm"),
          py::arg("adv_dollar"),
          py::arg("limits"));
}
