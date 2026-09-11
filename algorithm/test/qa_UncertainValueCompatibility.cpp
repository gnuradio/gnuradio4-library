#include <boost/ut.hpp>

#include <format>

#include <gnuradio-4.0/meta/UncertainValue.hpp>

const boost::ut::suite uncertainValueCompatibility = [] {
    using namespace boost::ut;
    using namespace std::literals::string_literals;

    "deprecated UncertainValue include forwards type and formatter"_test = [] {
        const gr::UncertainValue<double> value{1.0, 0.1};
        expect(eq("(1 ± 0.1)"s, std::format("{}", value)));
    };
};

int main() { /* tests are statically executed */ }
