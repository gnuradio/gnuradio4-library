#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <map>
#include <numbers>
#include <print>
#include <string>
#include <vector>

#include <gnuradio-4.0/algorithm/signal/ToneGenerator.hpp>

using namespace boost::ut;

const boost::ut::suite toneGeneratorTests = [] {
    using gr::signal::ToneGenerator;
    using gr::signal::ToneType;

    "numerical equivalence with existing SignalGenerator test vectors"_test = [] {
        // exact same parameters and expected values as qa_sources.cpp "SignalGenerator test"
        // sample_rate=2048, frequency=256, amplitude=1, offset=2, phase=pi/4
        constexpr std::size_t N      = 16;
        constexpr double      offset = 2.;

        struct WaveformCase {
            ToneType            type;
            std::vector<double> expected; // at amplitude=1, offset=0
        };

        // clang-format off
        const std::vector<WaveformCase> cases{
            {ToneType::Const,    {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.}},
            {ToneType::Sin,      {0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0., 0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0.}},
            {ToneType::Cos,      {0.707106, 0., -0.707106, -1., -0.7071067, 0., 0.707106, 1., 0.707106, 0., -0.707106, -1., -0.707106, 0., 0.707106, 1.}},
            {ToneType::Square,   {1., 1., 1., -1., -1., -1., -1., 1., 1., 1., 1., -1., -1., -1., -1., 1.}},
            {ToneType::Saw,      {0.25, 0.5, 0.75, -1., -0.75, -0.5, -0.25, 0., 0.25, 0.5, 0.75, -1., -0.75, -0.5, -0.25, 0.}},
            {ToneType::Triangle, {0.5, 1., 0.5, 0., -0.5, -1., -0.5, 0., 0.5, 1., 0.5, 0., -0.5, -1., -0.5, 0.}},
        };
        // clang-format on

        for (const auto& [type, expected] : cases) {
            ToneGenerator<double> gen;
            gen.configure(type, 256., 2048., std::numbers::pi / 4., 1., offset);

            for (std::size_t i = 0; i < N; ++i) {
                const double val = gen.generateSample();
                const double exp = expected[i] + offset;
                expect(approx(exp, val, 1e-5)) << std::format("type={} i={} expected={} got={}", static_cast<int>(type), i, exp, val);
            }
        }
    };

    "continuity across multiple fill calls"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::Sin, 100., 1000., 0., 1., 0.);

        std::vector<double> block1(50);
        std::vector<double> block2(50);
        gen.fill(block1);
        gen.fill(block2);

        // generate reference in one shot
        ToneGenerator<double> ref;
        ref.configure(ToneType::Sin, 100., 1000., 0., 1., 0.);
        std::vector<double> full(100);
        ref.fill(full);

        for (std::size_t i = 0; i < 50; ++i) {
            expect(eq(block1[i], full[i])) << std::format("block1 mismatch at {}", i);
        }
        for (std::size_t i = 0; i < 50; ++i) {
            expect(eq(block2[i], full[50 + i])) << std::format("block2 mismatch at {}", i);
        }
    };

    "reset restarts waveform"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::Sin, 100., 1000., 0., 1., 0.);

        std::vector<double> first(10);
        gen.fill(first);
        gen.reset();
        std::vector<double> afterReset(10);
        gen.fill(afterReset);

        for (std::size_t i = 0; i < first.size(); ++i) {
            expect(eq(first[i], afterReset[i])) << std::format("reset mismatch at {}", i);
        }
    };

    "float precision"_test = [] {
        ToneGenerator<float> gen;
        gen.configure(ToneType::Sin, 256.f, 2048.f, std::numbers::pi_v<float> / 4.f, 1.f, 0.f);

        const float val = gen.generateSample();
        expect(approx(static_cast<double>(val), 0.707106, 1e-4)) << std::format("float sin(pi/4) = {}", val);
    };

    "fillComplex Sin produces analytic signal"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::Sin, 100., 1000., 0., 1., 0.);

        constexpr std::size_t             N = 10;
        std::vector<std::complex<double>> complexOut(N);
        gen.fillComplex(complexOut);

        // reference: real part should match scalar generateSample
        ToneGenerator<double> ref;
        ref.configure(ToneType::Sin, 100., 1000., 0., 1., 0.);

        for (std::size_t i = 0; i < N; ++i) {
            const double realRef = ref.generateSample();
            expect(approx(complexOut[i].real(), realRef, 1e-12)) << std::format("complex real mismatch at {}", i);
        }

        // magnitude should be ~amplitude for all samples (analytic signal property)
        for (std::size_t i = 0; i < N; ++i) {
            expect(approx(std::abs(complexOut[i]), 1.0, 1e-12)) << std::format("complex magnitude at {} = {}", i, std::abs(complexOut[i]));
        }
    };

    "fillComplex Cos produces analytic signal"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::Cos, 100., 1000., 0., 2., 0.);

        constexpr std::size_t             N = 10;
        std::vector<std::complex<double>> complexOut(N);
        gen.fillComplex(complexOut);

        ToneGenerator<double> ref;
        ref.configure(ToneType::Cos, 100., 1000., 0., 2., 0.);

        for (std::size_t i = 0; i < N; ++i) {
            const double realRef = ref.generateSample();
            expect(approx(complexOut[i].real(), realRef, 1e-12)) << std::format("cos complex real mismatch at {}", i);
            expect(approx(std::abs(complexOut[i]), 2.0, 1e-12)) << std::format("cos complex magnitude at {}", i);
        }
    };

    "fillComplex non-sinusoidal has zero imaginary"_test = [] {
        for (auto type : {ToneType::Const, ToneType::Square, ToneType::Saw, ToneType::Triangle}) {
            ToneGenerator<double> gen;
            gen.configure(type, 100., 1000., 0., 1., 0.);

            std::vector<std::complex<double>> out(20);
            gen.fillComplex(out);

            ToneGenerator<double> ref;
            ref.configure(type, 100., 1000., 0., 1., 0.);

            for (std::size_t i = 0; i < out.size(); ++i) {
                expect(eq(out[i].imag(), 0.0)) << std::format("type={} i={} imag={}", static_cast<int>(type), i, out[i].imag());
                expect(approx(out[i].real(), ref.generateSample(), 1e-12)) << std::format("type={} i={} real mismatch", static_cast<int>(type), i);
            }
        }
    };

    "FastSin short-term precision matches Sin"_test = [] {
        ToneGenerator<double> fast;
        fast.configure(ToneType::FastSin, 256., 2048., std::numbers::pi / 4., 1., 2.);

        ToneGenerator<double> ref;
        ref.configure(ToneType::Sin, 256., 2048., std::numbers::pi / 4., 1., 2.);

        for (std::size_t i = 0; i < 200; ++i) {
            const double fastVal = fast.generateSample();
            const double refVal  = ref.generateSample();
            expect(approx(fastVal, refVal, 1e-12)) << std::format("FastSin vs Sin at {}: fast={} ref={}", i, fastVal, refVal);
        }
    };

    "FastCos short-term precision matches Cos"_test = [] {
        ToneGenerator<double> fast;
        fast.configure(ToneType::FastCos, 256., 2048., std::numbers::pi / 4., 1., 2.);

        ToneGenerator<double> ref;
        ref.configure(ToneType::Cos, 256., 2048., std::numbers::pi / 4., 1., 2.);

        for (std::size_t i = 0; i < 200; ++i) {
            const double fastVal = fast.generateSample();
            const double refVal  = ref.generateSample();
            expect(approx(fastVal, refVal, 1e-12)) << std::format("FastCos vs Cos at {}: fast={} ref={}", i, fastVal, refVal);
        }
    };

    "FastSin long-term drift remains bounded"_test = [] {
        ToneGenerator<double> fast;
        fast.configure(ToneType::FastSin, 440., 48000., 0., 1., 0.);

        ToneGenerator<double> ref;
        ref.configure(ToneType::Sin, 440., 48000., 0., 1., 0.);

        double maxError = 0.;
        for (std::size_t i = 0; i < 100'000; ++i) {
            const double fastVal = fast.generateSample();
            const double refVal  = ref.generateSample();
            maxError             = std::max(maxError, std::abs(fastVal - refVal));
        }
        expect(lt(maxError, 1e-8)) << std::format("FastSin max error after 100k samples: {:.2e}", maxError);
    };

    "FastSin fillComplex produces analytic signal"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::FastSin, 100., 1000., 0., 1., 0.);

        constexpr std::size_t             N = 100;
        std::vector<std::complex<double>> out(N);
        gen.fillComplex(out);

        for (std::size_t i = 0; i < N; ++i) {
            expect(approx(std::abs(out[i]), 1.0, 1e-12)) << std::format("FastSin magnitude at {} = {}", i, std::abs(out[i]));
        }
    };

    "FastCos fillComplex produces analytic signal"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::FastCos, 100., 1000., 0., 2., 0.);

        constexpr std::size_t             N = 100;
        std::vector<std::complex<double>> out(N);
        gen.fillComplex(out);

        ToneGenerator<double> ref;
        ref.configure(ToneType::FastCos, 100., 1000., 0., 2., 0.);

        for (std::size_t i = 0; i < N; ++i) {
            const double realRef = ref.generateSample();
            expect(approx(out[i].real(), realRef, 1e-12)) << std::format("FastCos complex real mismatch at {}", i);
            expect(approx(std::abs(out[i]), 2.0, 1e-12)) << std::format("FastCos complex magnitude at {}", i);
        }
    };

    "FastSin reset restarts waveform"_test = [] {
        ToneGenerator<double> gen;
        gen.configure(ToneType::FastSin, 100., 1000., 0., 1., 0.);

        std::vector<double> first(20);
        gen.fill(first);
        gen.reset();
        std::vector<double> afterReset(20);
        gen.fill(afterReset);

        for (std::size_t i = 0; i < first.size(); ++i) {
            expect(eq(first[i], afterReset[i])) << std::format("FastSin reset mismatch at {}", i);
        }
    };

    "FastSin float precision"_test = [] {
        ToneGenerator<float> gen;
        gen.configure(ToneType::FastSin, 256.f, 2048.f, std::numbers::pi_v<float> / 4.f, 1.f, 0.f);

        const float val = gen.generateSample();
        expect(approx(static_cast<double>(val), 0.707106, 1e-4)) << std::format("float FastSin(pi/4) = {}", val);
    };

    "all waveform types produce non-zero output"_test = [] {
        for (auto type : {ToneType::Const, ToneType::Sin, ToneType::Cos, ToneType::Square, ToneType::Saw, ToneType::Triangle, ToneType::FastSin, ToneType::FastCos}) {
            ToneGenerator<double> gen;
            gen.configure(type, 100., 1000., 0., 1., 0.);
            bool hasNonZero = false;
            for (int i = 0; i < 100; ++i) {
                if (gen.generateSample() != 0.0) {
                    hasNonZero = true;
                    break;
                }
            }
            expect(hasNonZero) << std::format("type={} produced all zeros", static_cast<int>(type));
        }
    };

    // The value type cannot hold time, and this is what that costs. At 1 MS/s the tick is 1e-6 s, which a float
    // second-accumulator rounds as soon as its ULP approaches it: within the first second the tone reads
    // 953.674 Hz, past 16.777 s the tick rounds to twice its size, and at t = 32 s the tick is below half an ULP
    // and the accumulator stops moving, freezing every waveform that reads it on DC. The generator keeps an exact
    // sample count instead, so the arm below asserts the phase after thirty seconds and the arm beside it runs the
    // accumulator that used to carry it, the way qa_SampleClock demonstrates the double's failure rather than
    // asserting it.
    "a float Sin holds its frequency past thirty seconds"_test = [] {
        constexpr float       sampleRate = 1.0e6f;
        constexpr float       frequency  = 1000.f;
        constexpr double      twoPi      = 2. * std::numbers::pi_v<double>;
        constexpr std::size_t kWindow    = 4096UZ;
        constexpr std::size_t kBlocks    = 7325UZ; // 30'003'200 samples, 30.0032 s at 1 MS/s
        constexpr std::size_t kTotal     = kBlocks * kWindow;

        ToneGenerator<float> gen;
        gen.configure(ToneType::Sin, frequency, sampleRate, 0.f, 1.f, 0.f);

        std::vector<float> window(kWindow);
        for (std::size_t block = 0; block < kBlocks; ++block) {
            gen.fill(window);
        }
        expect(eq(gen._sampleIndex, static_cast<std::uint64_t>(kTotal))) << "the count is the time, and it is exact";

        // The reference is the analytic instant of each sample: at these magnitudes the cycle count is 3.0e4 with a
        // double ULP of 3.6e-12, so std::sin of its reduced argument is exact to far below the float tolerance.
        // Correlating the window against sin and cos of that argument reads the accumulated phase directly, and the
        // correlation runs over whole cycles -- 4000 samples is four of them here -- because over a partial cycle
        // the sin-cos cross term does not vanish and would read as a phase the generator does not have.
        constexpr std::size_t kWholeCycles = 4000UZ;
        const std::size_t     first        = kTotal - kWindow;
        std::vector<double>   reference(kWindow);
        double                maxError   = 0.;
        double                inPhase    = 0.;
        double                quadrature = 0.;
        for (std::size_t i = 0; i < kWindow; ++i) {
            const double cycles = static_cast<double>(frequency) * static_cast<double>(first + i) / static_cast<double>(sampleRate);
            const double theta  = twoPi * (cycles - std::floor(cycles));
            const double got    = static_cast<double>(window[i]);
            reference[i]        = std::sin(theta);
            maxError            = std::max(maxError, std::abs(got - reference[i]));
            if (i >= kWindow - kWholeCycles) {
                inPhase += got * reference[i];
                quadrature += got * std::cos(theta);
            }
        }
        const double phaseError = std::atan2(quadrature, inPhase);
        expect(lt(maxError, 1e-4)) << std::format("worst sample error over the last 4096 of {} samples: {:.3e}", kTotal, maxError);
        expect(lt(std::abs(phaseError), 1e-5)) << std::format("accumulated phase error after 30 s: {:.3e} rad", phaseError);

        // the same stream through a float second-accumulator, which is the form this kernel used to keep
        float       elapsed    = 0.f;
        const float tick       = 1.f / sampleRate;
        double      afterOneS  = 0.;
        std::size_t stallIndex = 0UZ;
        for (std::size_t n = 0; n < kTotal; ++n) {
            const float before = elapsed;
            elapsed += tick;
            if (elapsed == before && stallIndex == 0UZ) {
                stallIndex = n;
            }
            if (n + 1UZ == 1'000'000UZ) {
                afterOneS = static_cast<double>(elapsed);
            }
        }
        expect(gt(std::abs(afterOneS - 1.), 5e-3)) << std::format("the accumulator reads {:.7f} s where one second has passed", afterOneS);
        expect(gt(stallIndex, 0UZ)) << "the accumulator is expected to stop advancing";
        expect(lt(stallIndex, kTotal)) << std::format("the accumulator stops at sample {} of {}", stallIndex, kTotal);
        expect(eq(elapsed, 32.f)) << std::format("the accumulator stands at {} s where {} s have passed", elapsed, static_cast<double>(kTotal) / static_cast<double>(sampleRate));

        // a frozen time is a frozen argument: the old form's last window is one repeated value, DC where a 1 kHz
        // tone should be, and it stands a full amplitude away from the tone the same parameters describe
        const float        omega = 2.f * std::numbers::pi_v<float> * frequency;
        std::vector<float> stalled(kWindow);
        double             oldError = 0.;
        for (std::size_t i = 0; i < kWindow; ++i) {
            stalled[i] = std::sin(omega * elapsed);
            elapsed += tick;
            oldError = std::max(oldError, std::abs(static_cast<double>(stalled[i]) - reference[i]));
        }
        const auto [low, high] = std::minmax_element(stalled.begin(), stalled.end());
        expect(eq(*low, *high)) << "the old form's last window carries no tone at all";
        expect(gt(oldError, 0.99)) << std::format("the old form's worst error over that window: {:.3e}", oldError);
        std::println("30 s at 1 MS/s: worst sample error {:.3e}, phase error {:.3e} rad; the float accumulator reads {:.7f} s at 1 s, stops at sample {}, stands at {} s and is off by {:.3f}", maxError, phaseError, afterOneS, stallIndex, elapsed, oldError);
    };

    // configure() and reset() are noexcept and seed a phasor that refuses an increment which is not finite, so
    // every argument that cannot name a tone has to be coerced where it arrives rather than thrown three frames
    // down. A sample rate of zero reaches this through a signal-generator block's own settings.
    "a rate, a frequency or a phase that names no tone leaves a constant"_test = [] {
        constexpr float kInfinity   = std::numeric_limits<float>::infinity();
        constexpr float kNotANumber = std::numeric_limits<float>::quiet_NaN();

        struct Row {
            const char* what;
            float       frequency;
            float       sampleRate;
            float       phase;
        };
        constexpr std::array<Row, 6> rows{{{"a zero sample rate", 1000.f, 0.f, 0.f}, {"a negative sample rate", 1000.f, -48000.f, 0.f}, //
            {"an infinite sample rate", 1000.f, kInfinity, 0.f}, {"a sample rate that is not a number", 1000.f, kNotANumber, 0.f},      //
            {"a frequency that is not a number", kNotANumber, 48000.f, 0.f}, {"a phase that is not a number", 1000.f, 48000.f, kNotANumber}}};

        for (const Row& row : rows) {
            for (const ToneType type : {ToneType::Sin, ToneType::FastSin, ToneType::Cos, ToneType::FastCos}) {
                ToneGenerator<float> gen;
                expect(nothrow([&] { gen.configure(type, row.frequency, row.sampleRate, row.phase, 1.f, 0.f); })) << row.what;
                expect(nothrow([&] { gen.reset(); })) << row.what;

                std::vector<float> block(64UZ, 0.f);
                expect(nothrow([&] { gen.fill(block); })) << row.what;
                bool finite = true;
                for (const float value : block) {
                    finite = finite && std::isfinite(value);
                }
                expect(that % finite) << row.what << ": every sample it produced is a number";
            }
        }

        ToneGenerator<float> silent;
        silent.configure(ToneType::Sin, 1000.f, 0.f, 0.f, 1.f, 0.f);
        expect(that % (silent._type == ToneType::Const)) << "a rate that cannot carry a tone leaves the generator on its constant";
        expect(eq(silent._stepQ64, 0ULL)) << "and with no phase increment at all";
    };
};

int main() { /* not needed for UT */ }
