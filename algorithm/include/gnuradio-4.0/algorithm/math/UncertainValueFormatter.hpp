#ifndef GNURADIO_ALGORITHM_MATH_UNCERTAINVALUEFORMATTER_HPP
#define GNURADIO_ALGORITHM_MATH_UNCERTAINVALUEFORMATTER_HPP

#include <format>
#include <ostream>

#include <gnuradio-4.0/algorithm/math/UncertainValue.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

template<gr::arithmetic_or_complex_like T>
struct std::formatter<gr::UncertainValue<T>> {
    formatter<T> value_formatter;

    constexpr auto parse(format_parse_context& ctx) { return value_formatter.parse(ctx); }

    template<typename FormatContext>
    auto format(const gr::UncertainValue<T>& uv, FormatContext& ctx) const {
        auto out = ctx.out();
        out      = std::format_to(out, "(");
        out      = value_formatter.format(uv.value, ctx);
        out      = std::format_to(out, " ± ");
        out      = value_formatter.format(uv.uncertainty, ctx);
        out      = std::format_to(out, ")");
        return out;
    }
};

namespace gr {
template<UncertainValueLike T>
std::ostream& operator<<(std::ostream& os, const T& v) {
    return os << std::format("{}", v);
}
} // namespace gr

#endif // GNURADIO_ALGORITHM_MATH_UNCERTAINVALUEFORMATTER_HPP
