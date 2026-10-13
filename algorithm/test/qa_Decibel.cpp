#include <boost/ut.hpp>

#include <cmath>
#include <limits>
#include <tuple>

#include <gnuradio-4.0/algorithm/math/Decibel.hpp>

using namespace boost::ut;
using gr::math::amplitudeToDb;
using gr::math::kDbFloor;
using gr::math::kDbFloorAmplitude;
using gr::math::kDbFloorPower;
using gr::math::toDb;

const boost::ut::suite<"Decibel"> decibelTests = [] {
    "the two floors are one level"_test = [] {
        expect(approx(10. * std::log10(kDbFloorPower), kDbFloor, 1e-9));
        expect(approx(20. * std::log10(kDbFloorAmplitude), kDbFloor, 1e-9));
        expect(approx(kDbFloorAmplitude * kDbFloorAmplitude, kDbFloorPower, 1e-40));
    };

    "a ratio above the floor is its exact dB value"_test = []<typename F> {
        expect(eq(toDb(F{1}), F{0}));
        expect(approx(toDb(F{100}), F{20}, static_cast<F>(1e-5)));
        expect(approx(toDb(static_cast<F>(0.5)), F{10} * std::log10(static_cast<F>(0.5)), static_cast<F>(1e-6)));
        expect(eq(amplitudeToDb(F{1}), F{0}));
        expect(approx(amplitudeToDb(F{100}), F{40}, static_cast<F>(1e-5)));
        expect(approx(amplitudeToDb(static_cast<F>(0.5)), F{20} * std::log10(static_cast<F>(0.5)), static_cast<F>(1e-6)));
    } | std::tuple<float, double>{};

    "zero, a negative, negative infinity and a value below the floor read as the floor"_test = []<typename F> {
        const F floor = static_cast<F>(kDbFloor);
        for (const F linear : {F{0}, -F{0}, F{-1}, -std::numeric_limits<F>::infinity(), static_cast<F>(kDbFloorPower / 10.), std::numeric_limits<F>::denorm_min()}) {
            expect(eq(toDb(linear), floor)) << "power" << linear;
        }
        for (const F amplitude : {F{0}, -F{0}, F{-1}, -std::numeric_limits<F>::infinity(), static_cast<F>(kDbFloorAmplitude / 10.), std::numeric_limits<F>::denorm_min()}) {
            expect(eq(amplitudeToDb(amplitude), floor)) << "amplitude" << amplitude;
        }
    } | std::tuple<float, double>{};

    "the floor ratio itself reads at the floor level"_test = []<typename F> {
        expect(approx(toDb(static_cast<F>(kDbFloorPower)), static_cast<F>(kDbFloor), static_cast<F>(1e-3)));
        expect(approx(amplitudeToDb(static_cast<F>(kDbFloorAmplitude)), static_cast<F>(kDbFloor), static_cast<F>(1e-3)));
    } | std::tuple<float, double>{};

    "a NaN stays a NaN"_test = []<typename F> {
        expect(std::isnan(toDb(std::numeric_limits<F>::quiet_NaN())));
        expect(std::isnan(amplitudeToDb(std::numeric_limits<F>::quiet_NaN())));
    } | std::tuple<float, double>{};
};

int main() { /* not needed for UT */ }
