#include <array>
#include <cassert>
#include <format>
#include <limits>
#include <numbers>
#include <numeric>
#include <string_view>
#include <type_traits>

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

// Window figures of merit, all recomputed from the coefficients rather than quoted.
// The transform is |DFT(w)| zero-padded to nFft; the main lobe is bounded by descending from the transform's
// peak to the first local minimum and the peak sidelobe level is the highest point beyond it, relative to the
// peak. ENBW is N sum(w^2)/(sum w)^2 and the coherent gain is sum(w)/N -- these three answer the spectral
// analysis question. The stopband attenuation answers the FIR design question instead and is a property of the
// filter, not of the window: a sinc lowpass at 0.25 cycles/sample windowed with w and scaled to unit DC gain,
// read as the highest |H| beyond the first stopband null. The two differ by up to 17 dB and neither substitutes
// for the other.
namespace windowMetrics {

[[nodiscard]] inline std::vector<double> magnitudeSpectrum(const std::vector<double>& coefficients, std::size_t nFft) {
    static gr::algorithm::FFT<double> fft; // one instance: the plan for a given length is built once
    std::vector<double>               padded(nFft, 0.);
    std::ranges::copy(coefficients, padded.begin());
    const auto          spectrum = fft.compute(padded);
    std::vector<double> magnitude(nFft / 2UZ + 1UZ);
    for (std::size_t k = 0UZ; k < magnitude.size(); ++k) {
        magnitude[k] = std::abs(spectrum[k]);
    }
    return magnitude;
}

[[nodiscard]] inline double peakSidelobeDb(const std::vector<double>& coefficients, std::size_t nFft) {
    const auto        magnitude = magnitudeSpectrum(coefficients, nFft);
    const std::size_t peak      = static_cast<std::size_t>(std::ranges::max_element(magnitude) - magnitude.begin());
    std::size_t       edge      = peak;
    while (edge + 1UZ < magnitude.size() && magnitude[edge + 1UZ] < magnitude[edge]) {
        ++edge;
    }
    return 20. * std::log10(*std::max_element(magnitude.begin() + static_cast<std::ptrdiff_t>(edge), magnitude.end()) / magnitude[peak]);
}

[[nodiscard]] inline double enbw(const std::vector<double>& coefficients) {
    const double sum       = std::accumulate(coefficients.begin(), coefficients.end(), 0.);
    const double sumSquare = std::inner_product(coefficients.begin(), coefficients.end(), coefficients.begin(), 0.);
    return static_cast<double>(coefficients.size()) * sumSquare / (sum * sum);
}

[[nodiscard]] inline double coherentGain(const std::vector<double>& coefficients) { return std::accumulate(coefficients.begin(), coefficients.end(), 0.) / static_cast<double>(coefficients.size()); }

[[nodiscard]] inline double stopbandAttenuationDb(const std::vector<double>& coefficients, std::size_t nFft) {
    constexpr double    fc     = 0.25; // cycles/sample, the sinc's own argument and therefore the -6 dB point
    const std::size_t   n      = coefficients.size();
    const double        center = static_cast<double>(n - 1UZ) / 2.;
    std::vector<double> taps(n);
    for (std::size_t i = 0UZ; i < n; ++i) {
        const double x = 2. * fc * (static_cast<double>(i) - center);
        taps[i]        = 2. * fc * (x == 0. ? 1. : std::sin(std::numbers::pi * x) / (std::numbers::pi * x)) * coefficients[i];
    }
    const double dcGain = std::accumulate(taps.begin(), taps.end(), 0.);
    std::ranges::transform(taps, taps.begin(), [dcGain](double tap) { return tap / dcGain; });

    const auto  magnitude = magnitudeSpectrum(taps, nFft);
    std::size_t nullBin   = static_cast<std::size_t>(fc * static_cast<double>(nFft));
    while (nullBin + 1UZ < magnitude.size() && magnitude[nullBin + 1UZ] < magnitude[nullBin]) { // descend to the first stopband null
        ++nullBin;
    }
    return -20. * std::log10(*std::max_element(magnitude.begin() + static_cast<std::ptrdiff_t>(nullBin), magnitude.end()));
}

} // namespace windowMetrics

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

    // the two-argument overload hands back the caller's own buffer for an lvalue output and an owning container for an rvalue output
    "compute output ownership"_test = [] {
        using Cplx                               = std::complex<double>;
        constexpr std::size_t          N         = 8UZ;
        constexpr double               tolerance = 1.e-12;
        gr::algorithm::FFT<Cplx, Cplx> fftAlgo{};

        const std::vector<Cplx> signal(N, Cplx(1., 0.));
        std::vector<Cplx>       output(N);

        static_assert(std::is_lvalue_reference_v<decltype(fftAlgo.compute(signal, output))>);
        static_assert(!std::is_reference_v<decltype(fftAlgo.compute(signal, std::vector<Cplx>(N)))>);

        const auto& borrowed = fftAlgo.compute(signal, output);
        expect(&borrowed == &output) << "an lvalue output is returned by reference";

        const auto& owned = fftAlgo.compute(signal, std::vector<Cplx>(N));
        expect(eq(owned.size(), N)) << "an rvalue output survives the call expression";
        expect(approx(owned[0].real(), static_cast<double>(N), tolerance)) << "DC bin of a constant input";
        for (std::size_t k = 1UZ; k < N; ++k) {
            expect(approx(std::abs(owned[k]), 0., tolerance)) << std::format("bin {} of a constant input", k);
        }
    };

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
        std::array BartlettRef{0.f, 0.2857142857f, 0.5714285714f, 0.8571428571f, 0.8571428571f, 0.5714285714f, 0.2857142857f, 0.f};
        std::array WelchRef{0.f, 0.4897959184f, 0.8163265306f, 0.9795918367f, 0.9795918367f, 0.8163265306f, 0.4897959184f, 0.f};
        std::array ParzenRef{0.00390625f, 0.10546875f, 0.47265625f, 0.91796875f, 0.91796875f, 0.47265625f, 0.10546875f, 0.00390625f};
        std::array TukeyRef{0.f, 0.6112604670f, 1.f, 1.f, 1.f, 1.f, 0.6112604670f, 0.f};
        std::array GaussianRef{0.0439369336f, 0.2030327963f, 0.5632793505f, 0.9382155957f, 0.9382155957f, 0.5632793505f, 0.2030327963f, 0.0439369336f};

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
        expect(equalVectors(create<T>(Bartlett, 8), BartlettRef)) << std::format("<{}> equal Bartlett vector {} vs. ref: {}", type_name<T>(), create<T>(Bartlett, 8), BartlettRef);
        expect(equalVectors(create<T>(Welch, 8), WelchRef)) << std::format("<{}> equal Welch vector {} vs. ref: {}", type_name<T>(), create<T>(Welch, 8), WelchRef);
        expect(equalVectors(create<T>(Parzen, 8), ParzenRef)) << std::format("<{}> equal Parzen vector {} vs. ref: {}", type_name<T>(), create<T>(Parzen, 8), ParzenRef);
        expect(equalVectors(create<T>(Tukey, 8), TukeyRef)) << std::format("<{}> equal Tukey vector {} vs. ref: {}", type_name<T>(), create<T>(Tukey, 8), TukeyRef);
        expect(equalVectors(create<T>(Gaussian, 8), GaussianRef)) << std::format("<{}> equal Gaussian vector {} vs. ref: {}", type_name<T>(), create<T>(Gaussian, 8), GaussianRef);

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
        expect(eq(create<T>(Bartlett, 0).size(), 0u)) << std::format("<{}> zero size Bartlett vectors", type_name<T>());
        expect(eq(create<T>(Welch, 0).size(), 0u)) << std::format("<{}> zero size Welch vectors", type_name<T>());
        expect(eq(create<T>(Parzen, 0).size(), 0u)) << std::format("<{}> zero size Parzen vectors", type_name<T>());
        expect(eq(create<T>(Tukey, 0).size(), 0u)) << std::format("<{}> zero size Tukey vectors", type_name<T>());
        expect(eq(create<T>(Gaussian, 0).size(), 0u)) << std::format("<{}> zero size Gaussian vectors", type_name<T>());
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

    // every window is finite and degenerates to a single unity tap at n == 1; at odd n the center tap is unity,
    // which the high-pass and band-stop deltas depend on: they keep their center tap through the window
    "window shape invariants"_test = []<typename T>() {
        using enum gr::algorithm::window::Type;
        for (const auto& entry : magic_enum::enum_entries<gr::algorithm::window::Type>()) {
            const auto window     = entry.first;
            const auto windowName = entry.second;
            for (const std::size_t n : {1UZ, 2UZ, 3UZ, 64UZ, 65UZ, 1023UZ}) {
                if (window == Kaiser && n == 1UZ) {
                    expect(throws<std::invalid_argument>([window] { std::ignore = create<T>(window, 1UZ); })) << "Kaiser rejects n == 1";
                    continue;
                }
                const auto w = create<T>(window, n);
                expect(eq(w.size(), n)) << std::format("<{}> {} n={} size", type_name<T>(), windowName, n);
                expect(std::ranges::all_of(w, [](T v) { return std::isfinite(v); })) << std::format("<{}> {} n={} finite", type_name<T>(), windowName, n);
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

    // a setting stored as HannExp still names a window, and the window it names is Hann
    "HannExp names the Hann window"_test = []<typename T>() {
        const auto parsed = magic_enum::enum_cast<gr::algorithm::window::Type>("HannExp");
        expect(parsed.has_value()) << "HannExp resolves to a window";
        expect(magic_enum::enum_name(parsed.value()) == std::string_view{"HannExp"}) << "the name round-trips through the enum";

        for (const std::size_t n : {2UZ, 9UZ, 64UZ, 1023UZ}) {
            const auto aliased = create<T>(parsed.value(), n);
            const auto hann    = create<T>(gr::algorithm::window::Type::Hann, n);
            expect(std::ranges::equal(aliased, hann)) << std::format("<{}> n={} HannExp coefficients equal Hann", type_name<T>(), n);
        }
    } | std::tuple<float, double>();

    // the second half of every window is the mirrored first half, so its symmetry is bit-exact rather than
    // within a rounding of it: 1 - abs(i - M)/M evaluated at mirrored i is not bit-identical for all n, and a
    // windowed FIR is exactly linear phase only if the window is exactly symmetric
    "window symmetry is bit-exact"_test = []<typename T>() {
        for (const auto& [window, windowName] : magic_enum::enum_entries<gr::algorithm::window::Type>()) {
            for (const std::size_t n : {2UZ, 3UZ, 8UZ, 9UZ, 64UZ, 65UZ, 255UZ, 1023UZ}) {
                const auto w         = create<T>(window, n);
                bool       symmetric = true;
                for (std::size_t i = 0UZ; i < n / 2UZ; ++i) {
                    symmetric = symmetric && (w[i] == w[n - 1UZ - i]);
                }
                expect(symmetric) << std::format("<{}> {} n={} is bit-exactly symmetric", type_name<T>(), windowName, n);
            }
        }
    } | std::tuple<float, double>();

    // Bartlett, Welch and Hann are exactly zero at both ends; Parzen's ends are 2/N^3 because its radius
    // divides by N/2 rather than by N - 1, which is what the bound below allows for
    "window endpoints"_test = [] {
        using enum gr::algorithm::window::Type;
        for (const std::size_t n : {3UZ, 9UZ, 64UZ, 1023UZ}) {
            for (const auto window : {Bartlett, Welch, Hann}) {
                const auto w = create<double>(window, n);
                expect(eq(w.front(), 0.)) << std::format("{} n={} starts at zero", magic_enum::enum_name(window), n);
                expect(eq(w.back(), 0.)) << std::format("{} n={} ends at zero", magic_enum::enum_name(window), n);
            }
            const auto   parzen   = create<double>(Parzen, n);
            const double expected = 2. / (static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n));
            expect(approx(parzen.front() / expected, 1., 1e-9)) << std::format("Parzen n={} ends at 2/N^3 = {:e}, is {:e}", n, expected, parzen.front());
        }
        for (const std::size_t n : {2UZ, 9UZ, 1023UZ}) {
            expect(std::ranges::all_of(create<double>(Rectangular, n), [](double v) { return v == 1.; })) << std::format("Rectangular n={} is exactly one everywhere", n);
        }
    };

    // alpha = 0 and alpha = 1 are the endpoints of the taper: both must be reachable and both exact
    "Tukey degenerates to Rectangular and to Hann"_test = [] {
        using enum gr::algorithm::window::Type;
        for (const std::size_t n : {9UZ, 64UZ, 1023UZ}) {
            expect(std::ranges::equal(create<double>(Tukey, n, 0.), create<double>(Rectangular, n))) << std::format("Tukey(0) n={} equals Rectangular bit-exactly", n);
            expect(equalVectors(create<double>(Tukey, n, 1.), create<double>(Hann, n), 1e-12)) << std::format("Tukey(1) n={} equals Hann", n);
        }
    };

    // the shape parameter is one scalar carrying four different quantities; NaN asks for the window's own default
    "window parameter validation"_test = [] {
        using enum gr::algorithm::window::Type;
        expect(throws<std::invalid_argument>([] { std::ignore = create<double>(Tukey, 64UZ, -0.1); })) << "Tukey alpha below zero";
        expect(throws<std::invalid_argument>([] { std::ignore = create<double>(Tukey, 64UZ, 1.1); })) << "Tukey alpha above one";
        expect(throws<std::invalid_argument>([] { std::ignore = create<double>(Gaussian, 64UZ, 0.); })) << "Gaussian sigma at zero";
        expect(throws<std::invalid_argument>([] { std::ignore = create<double>(Gaussian, 64UZ, 0.51); })) << "Gaussian sigma above one half";
        expect(throws<std::invalid_argument>([] { std::ignore = create<double>(Kaiser, 64UZ, -1.); })) << "Kaiser beta below zero";
        expect(throws<std::invalid_argument>([] { std::ignore = create<double>(Exponential, 64UZ, -1.); })) << "Exponential decay below zero";

        for (const auto& [window, windowName] : magic_enum::enum_entries<gr::algorithm::window::Type>()) {
            expect(nothrow([window] { std::ignore = create<double>(window, 64UZ); })) << std::format("{} accepts a NaN parameter", windowName);
        }
        expect(std::ranges::equal(create<double>(Kaiser, 64UZ), create<double>(Kaiser, 64UZ, 1.6))) << "Kaiser defaults to beta = 1.6";
        expect(std::ranges::equal(create<double>(Tukey, 64UZ), create<double>(Tukey, 64UZ, 0.5))) << "Tukey defaults to alpha = 0.5";
        expect(std::ranges::equal(create<double>(Gaussian, 64UZ), create<double>(Gaussian, 64UZ, 0.4))) << "Gaussian defaults to sigma = 0.4";
        expect(std::ranges::equal(create<double>(Exponential, 64UZ), create<double>(Exponential, 64UZ, 60.))) << "Exponential defaults to 60 dB of decay";
        // a shallower decay is a wider window: 20 dB leaves the edges at a tenth, 60 dB at a thousandth
        expect(approx(create<double>(Exponential, 65UZ, 20.).front(), 0.1, 1e-3)) << "Exponential(20 dB) edge value";
        expect(approx(create<double>(Exponential, 65UZ, 60.).front(), 0.001, 1e-5)) << "Exponential(60 dB) edge value";
    };

    // sigma is a fraction of the half-length rather than a count of samples, so the shape -- and with it
    // the whole table row -- holds across N.
    "Gaussian sigma is scale invariant"_test = [] {
        using enum gr::algorithm::window::Type;
        const double shortWindow = windowMetrics::peakSidelobeDb(create<double>(Gaussian, 255UZ, 0.4), 1UZ << 18);
        const double longWindow  = windowMetrics::peakSidelobeDb(create<double>(Gaussian, 1023UZ, 0.4), 1UZ << 18);
        expect(approx(shortWindow, longWindow, 0.5)) << std::format("Gaussian(0.4) peak sidelobe level: {:.2f} dB at N=255 vs {:.2f} dB at N=1023", shortWindow, longWindow);
    };

    "Rectangular ENBW is exactly one"_test = [] {
        for (const std::size_t n : {2UZ, 3UZ, 8UZ, 9UZ, 64UZ, 65UZ, 255UZ, 1023UZ}) {
            expect(eq(windowMetrics::enbw(create<double>(gr::algorithm::window::Type::Rectangular, n)), 1.)) << std::format("Rectangular n={} ENBW", n);
        }
    };

    // The measured table. Every number is recomputed from the coefficients by the definitions in
    // windowMetrics, which is what catches a coefficient typo: a table carrying only ENBW and the coherent
    // gain would not, since a squared sine has the same mean and mean square as Hann over a whole period.
    // The response is read on a 2^18-point transform, which resolves each tabled level to 0.3 dB.
    "measured window table at N = 1023"_test = [] {
        struct Row {
            gr::algorithm::window::Type window;
            double                      param;
            double                      peakSidelobeDb;
            double                      enbw;
            double                      coherentGain;
            double                      attenuationDb;
        };
        using enum gr::algorithm::window::Type;
        constexpr double        own    = std::numeric_limits<double>::quiet_NaN(); // the window's own default
        static const std::array kTable = std::to_array<Row>({
            //                                PSL dB     ENBW      CG      A dB
            {Rectangular, own, /*        */ -13.26, 1.0000, 1.0000, 20.96},
            {Bartlett, own, /*           */ -26.52, 1.3346, 0.4995, 26.26},
            {Welch, own, /*              */ -21.29, 1.2012, 0.6660, 31.38},
            {Hann, own, /*               */ -31.47, 1.5015, 0.4995, 43.94},
            {Hamming, own, /*            */ -43.19, 1.3686, 0.5379, 53.40},
            {Parzen, own, /*             */ -53.05, 1.9175, 0.3750, 56.64},
            {Blackman, own, /*           */ -58.11, 1.7284, 0.4196, 75.29},
            {FlatTop, own, /*            */ -76.62, 3.7739, 0.2155, 95.70},
            {BlackmanHarris, own, /*     */ -92.01, 2.0063, 0.3584, 109.29},
            {Nuttall, own, /*            */ -93.32, 2.0232, 0.3554, 111.96},
            {BlackmanNuttall, own, /*    */ -98.16, 1.9780, 0.3632, 114.90},
            // Exponential's transform leaves its main lobe without a null bound, so neither figure is an
            // attenuation in the sense the rows above use them: it stays above -49.48 dB out to 38.4 bins, and
            // the windowed sinc's transition band is 330 taps wide. Pinned as a regression anchor.
            {Exponential, own, /*        */ -49.48, 3.4626, 0.1445, 47.27},

            {Kaiser, 0.0, /*             */ -13.26, 1.0000, 1.0000, 20.96},
            {Kaiser, 1.6, /*             */ -16.70, 1.0238, 0.8482, 26.53},
            {Kaiser, 3.0, /*             */ -23.76, 1.1372, 0.6837, 36.88},
            {Kaiser, 4.0, /*             */ -29.98, 1.2476, 0.6032, 45.36},
            {Kaiser, 5.0, /*             */ -36.73, 1.3601, 0.5443, 54.13},
            {Kaiser, 6.0, /*             */ -43.81, 1.4682, 0.4996, 63.02},
            {Kaiser, 7.0, /*             */ -51.12, 1.5705, 0.4642, 72.00},
            {Kaiser, 8.0, /*             */ -58.63, 1.6673, 0.4353, 81.08},
            {Kaiser, 9.0, /*             */ -66.30, 1.7593, 0.4112, 90.26},
            {Kaiser, 10.0, /*            */ -74.10, 1.8469, 0.3908, 99.44},
            {Kaiser, 12.0, /*            */ -89.92, 2.0111, 0.3575, 117.35},
            {Kaiser, 14.0, /*            */ -105.90, 2.1632, 0.3315, 135.06},

            // Tukey buys almost nothing in stopband attenuation until alpha is close to one -- 0.0 to 0.75 sits
            // between 21 and 25 dB, because only at exactly 1 does the taper reach the center
            {Tukey, 0.0, /*              */ -13.26, 1.0000, 1.0000, 20.96},
            {Tukey, 0.1, /*              */ -13.31, 1.0398, 0.9491, 20.99},
            {Tukey, 0.25, /*             */ -13.60, 1.1031, 0.8741, 21.15},
            {Tukey, 0.5, /*              */ -15.12, 1.2234, 0.7493, 22.03},
            {Tukey, 0.75, /*             */ -19.39, 1.3613, 0.6244, 24.89},
            {Tukey, 0.9, /*              */ -24.97, 1.4477, 0.5495, 30.05},
            {Tukey, 1.0, /*              */ -31.47, 1.5015, 0.4995, 43.94},

            {Gaussian, 0.2, /*           */ -128.86, 2.8237, 0.2504, 151.23},
            {Gaussian, 0.25, /*          */ -87.74, 2.2592, 0.3130, 106.77},
            {Gaussian, 0.3, /*           */ -64.23, 1.8857, 0.3753, 84.59},
            {Gaussian, 0.35, /*          */ -52.12, 1.6272, 0.4364, 69.53},
            {Gaussian, 0.4, /*           */ -43.30, 1.4468, 0.4947, 59.23},
            {Gaussian, 0.5, /*           */ -31.92, 1.2334, 0.5977, 45.37},
        });

        constexpr std::size_t nFft        = 1UZ << 18;
        constexpr double      dbTolerance = 0.3;

        for (const auto& row : kTable) {
            const auto        window = create<double>(row.window, 1023UZ, row.param);
            const std::string label  = std::isnan(row.param) ? std::string(magic_enum::enum_name(row.window)) : std::format("{}({})", magic_enum::enum_name(row.window), row.param);
            expect(approx(windowMetrics::enbw(window), row.enbw, 1e-3)) << std::format("{} ENBW", label);
            expect(approx(windowMetrics::coherentGain(window), row.coherentGain, 1e-4)) << std::format("{} coherent gain", label);
            expect(approx(windowMetrics::peakSidelobeDb(window, nFft), row.peakSidelobeDb, dbTolerance)) << std::format("{} peak sidelobe level", label);
            expect(approx(windowMetrics::stopbandAttenuationDb(window, nFft), row.attenuationDb, dbTolerance)) << std::format("{} windowed-sinc stopband attenuation", label);
        }
    };

    "window corner cases"_test = []<typename T>() {
        static_assert(not magic_enum::enum_cast<gr::algorithm::window::Type>("UnknownWindow", magic_enum::case_insensitive).has_value());
        expect(throws<std::invalid_argument>([] { std::ignore = create(gr::algorithm::window::Type::Kaiser, 1); })) << "invalid Kaiser window size";
        expect(throws<std::invalid_argument>([] { std::ignore = create(gr::algorithm::window::Type::Kaiser, 2, -1.f); })) << "invalid Kaiser window beta";
    } | std::tuple<float, double>();
};

int main() { /* not needed for UT */ }
