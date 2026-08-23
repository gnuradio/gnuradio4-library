#include <array>
#include <cassert>
#include <format>
#include <numbers>
#include <numeric>

#include <boost/ut.hpp>

#include <gnuradio-4.0/meta/formatter.hpp>

#include <gnuradio-4.0/algorithm/fourier/fft.hpp>
#include <gnuradio-4.0/algorithm/fourier/fft_common.hpp>
#include <gnuradio-4.0/algorithm/fourier/window.hpp>

template<typename T>
std::vector<T> generateSinSample(std::size_t N, double sample_rate, double frequency, double amplitude) {
    std::vector<T> signal(N);
    for (std::size_t i = 0; i < N; i++) {
        if constexpr (gr::meta::complex_like<T>) {
            signal[i] = {static_cast<typename T::value_type>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sample_rate)), 0.};
        } else {
            signal[i] = static_cast<T>(amplitude * std::sin(2. * std::numbers::pi * frequency * static_cast<double>(i) / sample_rate));
        }
    }
    return signal;
}

template<typename TIn>
std::vector<std::complex<double>> naiveDft(const std::vector<TIn>& in) {
    const std::size_t                 n = in.size();
    std::vector<std::complex<double>> out(n);
    for (std::size_t k = 0; k < n; ++k) {
        std::complex<double> acc{};
        for (std::size_t i = 0; i < n; ++i) {
            const double angle = -2. * std::numbers::pi * static_cast<double>((i * k) % n) / static_cast<double>(n);
            acc += std::complex<double>(static_cast<double>(in[i].real()), static_cast<double>(in[i].imag())) * std::polar<double>(1., angle);
        }
        out[k] = acc;
    }
    return out;
}

template<typename TActual>
double relativeL2Error(const TActual& actual, const std::vector<std::complex<double>>& reference) {
    double num = 0.;
    double den = 0.;
    for (std::size_t k = 0; k < reference.size(); ++k) {
        const std::complex<double> a{static_cast<double>(actual[k].real()), static_cast<double>(actual[k].imag())};
        num += std::norm(a - reference[k]);
        den += std::norm(reference[k]);
    }
    return den > 0. ? std::sqrt(num / den) : std::sqrt(num);
}

template<typename T>
std::vector<T> deterministicComplexSignal(std::size_t n) {
    std::vector<T> signal(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = std::sin(0.7 * static_cast<double>(i) + 0.3) + 0.4 * std::cos(2.9 * static_cast<double>(i));
        const double y = std::cos(1.3 * static_cast<double>(i) - 0.9) - 0.2 * std::sin(0.11 * static_cast<double>(i));
        signal[i]      = T(static_cast<typename T::value_type>(x), static_cast<typename T::value_type>(y));
    }
    return signal;
}

template<gr::meta::array_or_vector_type T, gr::meta::array_or_vector_type U = T>
bool equalVectors(const T& v1, const U& v2, double tolerance = std::is_same_v<typename T::value_type, double> ? 1.e-5 : 1e-4) {
    if (v1.size() != v2.size()) {
        return false;
    }
    if constexpr (gr::meta::complex_like<typename T::value_type>) {
        return std::ranges::equal(v1, v2, [&tolerance](const auto& l, const auto& r) { return std::abs(l.real() - r.real()) < static_cast<typename T::value_type>(tolerance) && std::abs(l.imag() - r.imag()) < static_cast<typename T::value_type>(tolerance); });
    } else {
        return std::ranges::equal(v1, v2, [&tolerance](const auto& l, const auto& r) { return std::abs(static_cast<double>(l) - static_cast<double>(r)) < tolerance; });
    }
}

template<typename TInput, typename TOutput, template<typename, typename> typename TAlgo>
struct TestTypes {
    using InType   = TInput;
    using OutType  = TOutput;
    using AlgoType = TAlgo<TInput, TOutput>;
};

const boost::ut::suite<"FFT algorithms and window functions"> windowTests = [] {
    using namespace boost::ut;
    using namespace boost::ut::reflection;
    using gr::algorithm::window::create;
    using gr::algorithm::FFT;

    using ComplexTypesToTest = std::tuple<
        // complex input, same in-out precision
        TestTypes<std::complex<float>, std::complex<float>, FFT>, TestTypes<std::complex<double>, std::complex<double>, FFT>,
        // complex input, different in-out precision
        TestTypes<std::complex<float>, std::complex<double>, FFT>, TestTypes<std::complex<double>, std::complex<float>, FFT>>;

    using RealTypesToTest = std::tuple<
        // real input, same in-out precision
        TestTypes<float, std::complex<float>, FFT>, TestTypes<double, std::complex<double>, FFT>,
        // real input, different in-out precision
        TestTypes<double, std::complex<float>, FFT>, TestTypes<double, std::complex<float>, FFT>>;

    using AllTypesToTest = decltype(std::tuple_cat(std::declval<ComplexTypesToTest>(), std::declval<RealTypesToTest>()));

    "FFT algo sin tests"_test = []<typename T>() {
        typename T::AlgoType fftAlgo{};
        constexpr double     tolerance{1.e-5};
        struct TestParams {
            gr::Size_t N{1024};           // must be power of 2
            double     sample_rate{128.}; // must be power of 2 (only for the unit test for easy comparison with true result)
            double     frequency{1.};
            double     amplitude{1.};
            bool       outputInDb{false};
        };

        std::vector<TestParams> testCases = {{256, 128., 10., 5., false}, {512, 4., 1., 1., false}, {512, 32., 1., 0.1, false}, {256, 128., 10., 5., false}};
        for (const auto& t : testCases) {
            assert(std::has_single_bit(t.N));
            assert(std::has_single_bit(static_cast<std::size_t>(t.sample_rate)));

            const auto signal{generateSinSample<typename T::InType>(t.N, t.sample_rate, t.frequency, t.amplitude)};
            auto       fftResult         = fftAlgo.compute(signal);
            auto       magnitudeSpectrum = gr::algorithm::fft::computeMagnitudeSpectrum(fftResult, {.computeHalfSpectrum = true});
            auto       fullSpectrum      = gr::algorithm::fft::computeMagnitudeSpectrum(fftResult);
            auto       phase             = gr::algorithm::fft::computePhaseSpectrum(fftResult, {.outputInDeg = true, .unwrapPhase = true});
            const auto peakIndex{static_cast<std::size_t>(std::distance(magnitudeSpectrum.begin(), std::ranges::max_element(magnitudeSpectrum)))};
            const auto peakAmplitude = magnitudeSpectrum[peakIndex];
            const auto peakFrequency{static_cast<double>(peakIndex) * t.sample_rate / static_cast<double>(t.N)};

            const auto expectedAmplitude = t.outputInDb ? 20. * log10(std::abs(t.amplitude)) : t.amplitude;
            expect(approx(static_cast<double>(peakAmplitude), expectedAmplitude, tolerance)) << std::format("{} equal amplitude", type_name<T>());
            expect(approx(peakFrequency, t.frequency, tolerance)) << std::format("{} equal frequency", type_name<T>());
            expect(approx(static_cast<double>(fullSpectrum[peakIndex]), expectedAmplitude / 2., tolerance)) << std::format("{} full spectrum splits the amplitude over the mirrored pair", type_name<T>());
        }
    } | AllTypesToTest{};

    "FFT algo pattern tests"_test = []<typename T>() {
        using InType = T::InType;
        typename T::AlgoType fftAlgo{};
        constexpr double     tolerance{1.e-5};
        constexpr gr::Size_t N{16};
        static_assert(N == 16, "expected values are calculated for N == 16");

        std::vector<InType> signal(N);
        std::size_t         expectedPeakIndex{0};
        InType              expectedFft0{0., 0.};
        double              expectedPeakAmplitude{0.};
        for (std::size_t iT = 0; iT < 5; iT++) {
            if (iT == 0) {
                std::ranges::fill(signal.begin(), signal.end(), InType(0., 0.));
                expectedFft0          = {0., 0.};
                expectedPeakAmplitude = 0.;
            } else if (iT == 1) {
                std::ranges::fill(signal.begin(), signal.end(), InType(1., 0.));
                expectedFft0          = {16., 0.};
                expectedPeakAmplitude = 1.;
            } else if (iT == 2) {
                std::ranges::fill(signal.begin(), signal.end(), InType(1., 1.));
                expectedFft0          = {16., 16.};
                expectedPeakAmplitude = std::sqrt(2.);
            } else if (iT == 3) {
                std::iota(signal.begin(), signal.end(), 1);
                expectedFft0          = {136., 0.};
                expectedPeakAmplitude = 8.5;
            } else if (iT == 4) {
                int i = 0;
                std::ranges::generate(signal.begin(), signal.end(), [&i] { return InType(static_cast<typename InType::value_type>(i++ % 2), 0.); });
                expectedFft0          = {8., 0.};
                expectedPeakAmplitude = 0.5;
            }

            auto fftResult         = fftAlgo.compute(signal);
            auto magnitudeSpectrum = gr::algorithm::fft::computeMagnitudeSpectrum(fftResult);

            const auto peakIndex{static_cast<std::size_t>(std::distance(magnitudeSpectrum.begin(), std::ranges::max_element(magnitudeSpectrum)))};
            const auto peakAmplitude{magnitudeSpectrum[peakIndex]};

            expect(eq(peakIndex, expectedPeakIndex)) << std::format("<{}> equal peak index", type_name<T>());
            expect(approx(static_cast<double>(peakAmplitude), expectedPeakAmplitude, tolerance)) << std::format("<{}> equal amplitude", type_name<T>());
            expect(approx(static_cast<double>(fftResult[0].real()), static_cast<double>(expectedFft0.real()), tolerance)) << std::format("<{}> equal fft[0].real()", type_name<T>());
            expect(approx(static_cast<double>(fftResult[0].imag()), static_cast<double>(expectedFft0.imag()), tolerance)) << std::format("<{}> equal fft[0].imag()", type_name<T>());
        }
    } | ComplexTypesToTest{};

    // FFT<>::compute equals the unnormalized forward DFT for sizes that are not a power of two
    "non-power-of-two forward DFT"_test = []<typename T>() {
        using InType    = typename T::InType;
        using ValueType = typename T::OutType::value_type;
        typename T::AlgoType fftAlgo{};

        const double tolerance = std::is_same_v<ValueType, float> ? 1.e-4 : 1.e-10;
        for (const std::size_t n : {3UZ, 5UZ, 6UZ, 7UZ, 9UZ, 10UZ, 12UZ, 13UZ, 100UZ, 257UZ, 1000UZ, 1009UZ}) {
            const auto signal    = deterministicComplexSignal<InType>(n);
            const auto reference = naiveDft(signal);
            const auto result    = fftAlgo.compute(signal);

            expect(eq(result.size(), n)) << std::format("<{}> n={} output size", type_name<T>(), n);
            expect(lt(relativeL2Error(result, reference), tolerance)) << std::format("<{}> n={} relative L2 error {}", type_name<T>(), n, relativeL2Error(result, reference));
        }
    } | ComplexTypesToTest{};

    // unnormalized convention: backward(forward(x)) == N*x, across the SimdFFT, radix-2 and Bluestein paths
    "forward/backward round-trip"_test = []<typename TVal>() {
        using Cplx = std::complex<TVal>;
        gr::algorithm::FFT<Cplx, Cplx>                                     forwardFft{};
        gr::algorithm::FFT<Cplx, Cplx, gr::algorithm::Direction::Backward> backwardFft{};

        const double tolerance = std::is_same_v<TVal, float> ? 1.e-4 : 1.e-12;
        for (const std::size_t n : {8UZ, 12UZ, 16UZ, 64UZ, 100UZ, 257UZ, 1024UZ}) {
            const auto signal    = deterministicComplexSignal<Cplx>(n);
            const auto spectrum  = forwardFft.compute(signal);
            const auto roundTrip = backwardFft.compute(spectrum);

            std::vector<std::complex<double>> expected(n);
            for (std::size_t i = 0; i < n; ++i) {
                expected[i] = std::complex<double>(static_cast<double>(signal[i].real()), static_cast<double>(signal[i].imag())) * static_cast<double>(n);
            }
            expect(lt(relativeL2Error(roundTrip, expected), tolerance)) << std::format("<{}> n={} round-trip relative L2 error {}", type_name<TVal>(), n, relativeL2Error(roundTrip, expected));
        }
    } | std::tuple<float, double>();

    // one instance interleaving transform lengths matches a dedicated instance per length, bit for bit, over
    // more distinct lengths than the plan cache holds and across the SimdFFT, radix-2 and Bluestein paths
    "interleaved transform lengths"_test = []<typename T>() {
        using InType = typename T::InType;
        typename T::AlgoType interleaved{};

        constexpr std::array lengths{1024UZ, 8UZ, 1009UZ, 4096UZ, 100UZ};
        for (std::size_t round = 0UZ; round < 3UZ; ++round) {
            for (const std::size_t n : lengths) {
                const auto           signal = generateSinSample<InType>(n, static_cast<double>(n), 5., 1.);
                typename T::AlgoType dedicated{};
                const auto           expected = dedicated.compute(signal);
                const auto           actual   = interleaved.compute(signal);

                expect(eq(actual.size(), n)) << std::format("<{}> n={} output size", type_name<T>(), n);
                expect(std::ranges::equal(actual, expected)) << std::format("<{}> n={} round={} interleaved matches a dedicated instance", type_name<T>(), n, round);
            }
        }
    } | AllTypesToTest{};

    // amplitude scaling: 1/N over the full spectrum, 2/N over the half spectrum except at DC and Nyquist
    "magnitude spectrum scaling"_test = []<typename TVal>() {
        using Cplx                               = std::complex<TVal>;
        constexpr std::size_t          N         = 64UZ;
        constexpr double               tolerance = 1.e-5;
        gr::algorithm::FFT<Cplx, Cplx> fftAlgo{};

        for (const std::size_t bin : {0UZ, 1UZ, 7UZ, 32UZ}) {
            std::vector<Cplx> signal(N);
            for (std::size_t i = 0; i < N; ++i) {
                const double angle = 2. * std::numbers::pi * static_cast<double>((bin * i) % N) / static_cast<double>(N);
                signal[i]          = Cplx(static_cast<TVal>(std::cos(angle)), static_cast<TVal>(std::sin(angle)));
            }
            const auto full = gr::algorithm::fft::computeMagnitudeSpectrum(fftAlgo.compute(signal));
            expect(approx(static_cast<double>(full[bin]), 1., tolerance)) << std::format("<{}> unit complex exponential at bin {}", type_name<TVal>(), bin);
        }

        constexpr double      amplitude = 3.;
        constexpr std::size_t cosineBin = 5UZ;
        std::vector<Cplx>     cosine(N);
        for (std::size_t i = 0; i < N; ++i) {
            cosine[i] = Cplx(static_cast<TVal>(amplitude * std::cos(2. * std::numbers::pi * static_cast<double>(cosineBin * i) / static_cast<double>(N))), TVal(0));
        }
        const auto cosineSpectrum = fftAlgo.compute(cosine);
        expect(approx(static_cast<double>(gr::algorithm::fft::computeMagnitudeSpectrum(cosineSpectrum, {.computeHalfSpectrum = true})[cosineBin]), amplitude, tolerance)) << std::format("<{}> real cosine, half spectrum", type_name<TVal>());
        expect(approx(static_cast<double>(gr::algorithm::fft::computeMagnitudeSpectrum(cosineSpectrum)[cosineBin]), amplitude / 2., tolerance)) << std::format("<{}> real cosine, full spectrum", type_name<TVal>());

        const std::vector<Cplx> dc(N, Cplx(TVal(2), TVal(0)));
        const auto              dcSpectrum = fftAlgo.compute(dc);
        expect(approx(static_cast<double>(gr::algorithm::fft::computeMagnitudeSpectrum(dcSpectrum)[0]), 2., tolerance)) << std::format("<{}> DC, full spectrum", type_name<TVal>());
        expect(approx(static_cast<double>(gr::algorithm::fft::computeMagnitudeSpectrum(dcSpectrum, {.computeHalfSpectrum = true})[0]), 2., tolerance)) << std::format("<{}> DC, half spectrum", type_name<TVal>());
    } | std::tuple<float, double>();

    "Unwrap Phase tests"_test = [] {
        std::vector<double> phase = {0.2, -1., 2.5, -3.1, 0.9, -0.5, 1.2, 0.8, 1.5, -1.2, -2.7, 0.9, -0.8, -1.4, 0.6, 1.1, -1.9, 0.4, 1.3, -0.7};
        // Output generated with python numpy.unwrap(phase)
        std::vector<double> expOut = {0.2, -1., -3.78318531, -3.1, -5.38318531, -6.78318531, -5.08318531, -5.48318531, -4.78318531, -7.48318531, -8.98318531, -11.66637061, -13.36637061, -13.96637061, -11.96637061, -11.46637061, -14.46637061, -12.16637061, -11.26637061, -13.26637061};
        gr::algorithm::fft::unwrapPhase(phase);
        expect(equalVectors(phase, expOut)) << "unwrapped phases are equal";
    };

    "window pre-computed array tests"_test = []<typename T>() { // this tests regression w.r.t. changed implementations
        // Expected value for size 8
        std::array RectangularRef{1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
        std::array HammingRef{0.07672f, 0.25053218f, 0.64108455f, 0.9542833f, 0.95428324f, 0.6410846f, 0.25053206f, 0.07672f};
        std::array HannRef{0.f, 0.1882550991f, 0.611260467f, 0.950484434f, 0.950484434f, 0.611260467f, 0.1882550991f, 0.f};
        std::array BlackmanRef{0.f, 0.09045342435f, 0.4591829575f, 0.9203636181f, 0.9203636181f, 0.4591829575f, 0.09045342435f, 0.f};
        std::array BlackmanHarrisRef{0.00006f, 0.03339172348f, 0.3328335043f, 0.8893697722f, 0.8893697722f, 0.3328335043f, 0.03339172348f, 0.00006f};
        std::array BlackmanNuttallRef{0.0003628f, 0.03777576895f, 0.34272762f, 0.8918518611f, 0.8918518611f, 0.34272762f, 0.03777576895f, 0.0003628f};
        std::array ExponentialRef{0.0010032727f, 0.0072136726f, 0.0518673257f, 0.3729334040f, 0.3729334040f, 0.0518673257f, 0.0072136726f, 0.0010032727f};
        std::array FlatTopRef{0.f, -0.0358150410f, 0.0092233033f, 0.7815529111f, 0.7815529111f, 0.0092233033f, -0.0358150410f, 0.f};
        std::array NuttallRef{0.f, 0.0311427368f, 0.3264168059f, 0.8876284573f, 0.8876284573f, 0.3264168059f, 0.0311427368f, 0.f};
        std::array KaiserRef{0.5714348848f, 0.7650986027f, 0.9113132365f, 0.9899091685f, 0.9899091685f, 0.9113132365f, 0.7650986027f, 0.5714348848f};

        // check all windows for unwanted changes
        using enum gr::algorithm::window::Type;
        expect(equalVectors(create<T>(None, 8), RectangularRef)) << std::format("<{}> equal Rectangular vector {} vs. ref: {}", type_name<T>(), create<T>(None, 8), RectangularRef);
        expect(equalVectors(create<T>(Rectangular, 8), RectangularRef)) << std::format("<{}> equal Rectangular vector {} vs. ref: {}", type_name<T>(), create<T>(Rectangular, 8), RectangularRef);
        expect(equalVectors(create<T>(Hamming, 8), HammingRef)) << std::format("<{}> equal Hamming vector {} vs. ref: {}", type_name<T>(), create<T>(Hamming, 8), HammingRef);
        expect(equalVectors(create<T>(Hann, 8), HannRef)) << std::format("<{}> equal Hann vector {} vs. ref: {}", type_name<T>(), create<T>(Hann, 8), HannRef);
        expect(equalVectors(create<T>(Blackman, 8), BlackmanRef)) << std::format("<{}> equal Blackman vvector {} vs. ref: {}", type_name<T>(), create<T>(Blackman, 8), BlackmanRef);
        expect(equalVectors(create<T>(BlackmanHarris, 8), BlackmanHarrisRef)) << std::format("<{}> equal BlackmanHarris vector {} vs. ref: {}", type_name<T>(), create<T>(BlackmanHarris, 8), BlackmanHarrisRef);
        expect(equalVectors(create<T>(BlackmanNuttall, 8), BlackmanNuttallRef)) << std::format("<{}> equal BlackmanNuttall vector {} vs. ref: {}", type_name<T>(), create<T>(BlackmanNuttall, 8), BlackmanNuttallRef);
        expect(equalVectors(create<T>(Exponential, 8), ExponentialRef)) << std::format("<{}> equal Exponential vector {} vs. ref: {}", type_name<T>(), create<T>(Exponential, 8), ExponentialRef);
        expect(equalVectors(create<T>(FlatTop, 8), FlatTopRef)) << std::format("<{}> equal FlatTop vector {} vs. ref: {}", type_name<T>(), create<T>(FlatTop, 8), FlatTopRef);
        expect(equalVectors(create<T>(Nuttall, 8), NuttallRef)) << std::format("<{}> equal Nuttall vector {} vs. ref: {}", type_name<T>(), create<T>(Nuttall, 8), NuttallRef);
        expect(equalVectors(create<T>(Kaiser, 8), KaiserRef)) << std::format("<{}> equal Kaiser vector {} vs. ref: {}", type_name<T>(), create<T>(Kaiser, 8), KaiserRef);

        // test zero length
        expect(eq(create<T>(None, 0).size(), 0u)) << std::format("<{}> zero size None vectors", type_name<T>());
        expect(eq(create<T>(Rectangular, 0).size(), 0u)) << std::format("<{}> zero size Rectangular vectors", type_name<T>());
        expect(eq(create<T>(Hamming, 0).size(), 0u)) << std::format("<{}> zero size Hamming vectors", type_name<T>());
        expect(eq(create<T>(Hann, 0).size(), 0u)) << std::format("<{}> zero size Hann vectors", type_name<T>());
        expect(eq(create<T>(Blackman, 0).size(), 0u)) << std::format("<{}> zero size Blackman vectors", type_name<T>());
        expect(eq(create<T>(BlackmanHarris, 0).size(), 0u)) << std::format("<{}> zero size BlackmanHarris vectors", type_name<T>());
        expect(eq(create<T>(BlackmanNuttall, 0).size(), 0u)) << std::format("<{}> zero size BlackmanNuttall vectors", type_name<T>());
        expect(eq(create<T>(Exponential, 0).size(), 0u)) << std::format("<{}> zero size Exponential vectors", type_name<T>());
        expect(eq(create<T>(FlatTop, 0).size(), 0u)) << std::format("<{}> zero size FlatTop vectors", type_name<T>());
        expect(eq(create<T>(Nuttall, 0).size(), 0u)) << std::format("<{}> zero size Nuttall vectors", type_name<T>());
        expect(eq(create<T>(Kaiser, 0).size(), 0u)) << std::format("<{}> zero size Kaiser vectors", type_name<T>());
    } | std::tuple<float, double>();

    "basic window tests"_test = [](auto& val) {
        const auto& [window, windowName] = val;
        using enum gr::algorithm::window::Type;

        const auto w = create(window, 1024U);
        expect(eq(w.size(), 1024U));

        if (window == FlatTop || window == Blackman || window == Nuttall) {
            return; // min max out of [0, 1] by design and/or numerical corner cases
        }
        const auto [min, max] = std::ranges::minmax_element(w);
        expect(ge(*min, 0.f)) << std::format("window {} min value\n", windowName);
        expect(le(*max, 1.f)) << std::format("window {} max value\n", windowName);
    } | magic_enum::enum_entries<gr::algorithm::window::Type>();

    // every window is finite and symmetric and degenerates to a single unity tap at n == 1; at odd n the
    // center tap is unity
    "window shape invariants"_test = []<typename T>() {
        using enum gr::algorithm::window::Type;
        for (const auto& entry : magic_enum::enum_entries<gr::algorithm::window::Type>()) {
            const auto window     = entry.first;
            const auto windowName = entry.second;
            for (const std::size_t n : {1UZ, 2UZ, 3UZ, 64UZ, 65UZ}) {
                if (window == Kaiser && n == 1UZ) {
                    expect(throws<std::invalid_argument>([window] { std::ignore = create<T>(window, 1UZ); })) << "Kaiser rejects n == 1";
                    continue;
                }
                const auto w = create<T>(window, n);
                expect(eq(w.size(), n)) << std::format("<{}> {} n={} size", type_name<T>(), windowName, n);
                expect(std::ranges::all_of(w, [](T v) { return std::isfinite(v); })) << std::format("<{}> {} n={} finite", type_name<T>(), windowName, n);
                for (std::size_t i = 0; i < n; ++i) {
                    expect(approx(static_cast<double>(w[i]), static_cast<double>(w[n - 1UZ - i]), 1.e-5)) << std::format("<{}> {} n={} symmetric at {}", type_name<T>(), windowName, n, i);
                }
                if (n == 1UZ) {
                    expect(approx(static_cast<double>(w[0]), 1., 1.e-6)) << std::format("<{}> {} single tap is unity", type_name<T>(), windowName);
                }
                if ((n % 2UZ) == 1UZ) {
                    expect(approx(static_cast<double>(w[n / 2UZ]), 1., 1.e-5)) << std::format("<{}> {} n={} center tap is unity", type_name<T>(), windowName, n);
                    expect(le(static_cast<double>(*std::ranges::max_element(w)), 1. + 1.e-5)) << std::format("<{}> {} n={} bounded by unity", type_name<T>(), windowName, n);
                }
            }
        }
    } | std::tuple<float, double>();

    "window corner cases"_test = []<typename T>() {
        static_assert(not magic_enum::enum_cast<gr::algorithm::window::Type>("UnknownWindow", magic_enum::case_insensitive).has_value());
        // Hann has a single spelling in this set. The name below evaluated sin^2 over two periods across
        // its own span, which is zero at the center and peaks a bin off DC in the transform.
        static_assert(not magic_enum::enum_cast<gr::algorithm::window::Type>("HannExp", magic_enum::case_insensitive).has_value());
        expect(throws<std::invalid_argument>([] { std::ignore = create(gr::algorithm::window::Type::Kaiser, 1); })) << "invalid Kaiser window size";
        expect(throws<std::invalid_argument>([] { std::ignore = create(gr::algorithm::window::Type::Kaiser, 2, -1.f); })) << "invalid Kaiser window beta";
    } | std::tuple<float, double>();
};

int main() { /* not needed for UT */ }
