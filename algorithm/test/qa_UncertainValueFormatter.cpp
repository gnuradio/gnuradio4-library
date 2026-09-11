#include <boost/ut.hpp>

#include <complex>
#include <format>
#include <print>
#include <sstream>

#include <gnuradio-4.0/algorithm/math/UncertainValueFormatter.hpp>

const boost::ut::suite uncertainValueFormatter = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;
    using namespace gr;
    using UncertainDouble  = gr::UncertainValue<double>;
    using UncertainComplex = gr::UncertainValue<std::complex<double>>;

    if (std::getenv("DISABLE_SENSITIVE_TESTS") == nullptr) {
        boost::ext::ut::cfg<override> = {.tag = {"visual", "benchmarks"}};
    }

    "std::formatter<gr::UncertainValue<T>>"_test = [] {
        expect(eq("(1.23 ± 0.45)"s, std::format("{}", UncertainDouble{1.23, 0.45})));
        expect(eq("(3.14 ± 0.01)"s, std::format("{}", UncertainDouble{3.14, 0.01})));
        expect(eq("(0 ± 0)"s, std::format("{}", UncertainDouble{0, 0})));

        expect(eq("((1+2i) ± (0.1+0.2i))"s, std::format("{}", UncertainComplex{{1, 2}, {0.1, 0.2}})));
        expect(eq("((3.14+1.59i) ± (0.01+0.02i))"s, std::format("{}", UncertainComplex{{3.14, 1.59}, {0.01, 0.02}})));
        expect(eq("(0 ± 0)"s, std::format("{}", UncertainComplex{{0, 0}, {0, 0}})));

        expect(eq("(1.230 ± 0.450)"s, std::format("{:1.3f}", UncertainDouble{1.23, 0.45})));
        expect(eq("(3.140 ± 0.010)"s, std::format("{:1.3f}", UncertainDouble{3.14, 0.01})));
        expect(eq("(0.000 ± 0.000)"s, std::format("{:1.3f}", UncertainDouble{0, 0})));

        std::stringstream ss;
        ss << UncertainDouble{1.23, 0.45};
        expect(eq("(1.23 ± 0.45)"s, ss.str()));
    };

    boost::ut::tag("visual") / "visual examples"_test = [] {
        UncertainValue uValueA{4.0, 2.0};
        UncertainValue uValueB{2.0, 1.0};
        static_assert(UncertainValueLike<decltype(uValueA)>);
        static_assert(!UncertainValueLike<double>);

        std::print("uncorrelated values:\n");
        std::print("{} + {} = {}\n", uValueA, uValueB, uValueA + uValueB);
        std::print("{} - {} = {}\n", uValueA, uValueB, uValueA - uValueB);
        std::print("{} * {} = {}\n", uValueA, uValueB, uValueA * uValueB);
        std::print("{} / {} = {}\n", uValueA, uValueB, uValueA / uValueB);

        std::print("mixed-regular values:\n");
        std::print("{} + {} = {}\n", uValueA, uValueB.value, uValueA + uValueB.value);
        std::print("{} - {} = {}\n", uValueA, uValueB.value, uValueA - uValueB.value);
        std::print("{} * {} = {}\n", uValueA, uValueB.value, uValueA * uValueB.value);
        std::print("{} / {} = {}\n", uValueA, uValueB.value, uValueA / uValueB.value);

        using namespace std::complex_literals;
        UncertainValue uValueAC{4. + 1i, +1. + 1i};
        UncertainValue uValueBC{2. - 1i, +2. + 2i};

        std::print("uncorrelated values - complex:\n");
        std::print("{} + {} = {}\n", uValueAC, uValueBC, uValueAC + uValueBC);
        std::print("{} - {} = {}\n", uValueAC, uValueBC, uValueAC - uValueBC);
        std::print("{} * {} = {}\n", uValueAC, uValueBC, uValueAC * uValueBC);
        std::print("{} / {} = {}\n", uValueAC, uValueBC, uValueAC / uValueBC);
    };
};

int main() { /* tests are statically executed */ }
