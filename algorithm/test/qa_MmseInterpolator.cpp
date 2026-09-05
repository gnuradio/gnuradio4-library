#include <boost/ut.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <print>
#include <random>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/sync/MmseInterpolator.hpp>

namespace {

using gr::sync::mmseCorrelation;
using gr::sync::mmseDerivativeTarget;
using gr::sync::mmseDifferentiatorTaps;
using gr::sync::MmseInterpolatorBank;
using gr::sync::mmseObjective;
using gr::sync::mmseTaps;
using gr::sync::mmseTarget;

using CD    = std::complex<double>;
using CF    = std::complex<float>;
using Clock = std::chrono::steady_clock;

constexpr double kPi = std::numbers::pi;

/// The published `mu = 0.5` solution, quoted to the eight decimals the QA asserts to.
constexpr double kTapsHalf[]    = {-0.00677751, 0.03945777, -0.14265809, 0.60983636, 0.60983636, -0.14265809, 0.03945777, -0.00677751};
constexpr double kTapsQuarter[] = {-0.00455932, 0.02578440, -0.08770110, 0.29100579, 0.87130542, -0.12204653, 0.03118661, -0.00517776};
constexpr double kTapsFine[]    = {-1.547003e-04, 8.537773e-04, -2.769683e-03, 7.892947e-03, 9.985335e-01, -5.410541e-03, 1.246416e-03, -1.989931e-04};

/// The interpolating differentiator at `mu = 0.5` and at `mu = 0`, the second of which is not antisymmetric.
constexpr double kDerivativeHalf[] = {-0.00172098, 0.01495866, -0.09339298, 1.21727965, -1.21727965, 0.09339298, -0.01495866, 0.00172098};
constexpr double kDerivativeZero[] = {-0.01979746, 0.10918008, -0.35369730, 1.00393832, -0.17642511, -0.69894013, 0.16037810, -0.02556926};

/// @brief `H(w) = sum_k h_k exp(-j w d_k)` with `d_k = k - L/2`, the response the objective fits.
[[nodiscard]] CD responseOf(std::span<const double> taps, double frequency) {
    const double center = 0.5 * static_cast<double>(taps.size());
    CD           acc{};
    for (std::size_t k = 0UZ; k < taps.size(); ++k) {
        acc += taps[k] * std::polar(1.0, -2.0 * kPi * frequency * (static_cast<double>(k) - center));
    }
    return acc;
}

/**
 * @brief The gradient of `J/(2W)` by Simpson quadrature of the defining integral.
 *
 * It works from the defining integral rather than from `R` or `p`, so it tests the closed forms
 * rather than the arithmetic that uses them. `derivative` selects the ideal response
 * `j*w*exp(j*w*mu)` over `exp(j*w*mu)`.
 */
[[nodiscard]] std::vector<double> quadratureGradient(std::span<const double> taps, double band, double mu, bool derivative) {
    constexpr std::size_t kSteps  = 20000UZ;
    const std::size_t     order   = taps.size();
    const double          radians = 2.0 * kPi * band;
    const double          center  = 0.5 * static_cast<double>(order);

    std::vector<double> gradient(order, 0.0);
    for (std::size_t k = 0UZ; k < order; ++k) {
        const double dk        = static_cast<double>(k) - center;
        const auto   integrand = [&](double w) {
            double residualReal = derivative ? -w * std::sin(w * mu) : std::cos(w * mu);
            double residualImag = derivative ? w * std::cos(w * mu) : std::sin(w * mu);
            for (std::size_t l = 0UZ; l < order; ++l) {
                const double dl = static_cast<double>(l) - center;
                residualReal -= taps[l] * std::cos(w * dl);
                residualImag += taps[l] * std::sin(w * dl);
            }
            return -2.0 * (residualReal * std::cos(w * dk) - residualImag * std::sin(w * dk));
        };

        double sum = integrand(0.0) + integrand(radians);
        for (std::size_t i = 1UZ; i < kSteps; ++i) {
            const double w = radians * static_cast<double>(i) / static_cast<double>(kSteps);
            sum += (i % 2UZ != 0UZ ? 4.0 : 2.0) * integrand(w);
        }
        // The integrand is even in w, so the half integral doubled is the whole one; then divided by 2W.
        gradient[k] = 2.0 * (sum * (radians / static_cast<double>(kSteps)) / 3.0) / (2.0 * radians);
    }
    return gradient;
}

[[nodiscard]] double maxAbsolute(std::span<const double> values) {
    double worst = 0.0;
    for (double v : values) {
        worst = std::max(worst, std::abs(v));
    }
    return worst;
}

/// @brief A real signal whose spectrum is exactly zero above 0.24 cycles/sample, evaluable at any real position.
class BandLimitedSignal {
public:
    explicit BandLimitedSignal(std::uint32_t seed) {
        std::mt19937                           rng(seed);
        std::uniform_real_distribution<double> phase(0.0, 2.0 * kPi);
        double                                 power = 0.0;
        for (std::size_t i = 0UZ; i < kTones; ++i) {
            _frequency[i] = 0.02 + 0.22 * static_cast<double>(i) / static_cast<double>(kTones - 1UZ);
            _phase[i]     = phase(rng);
            _amplitude[i] = 1.0 / (1.0 + static_cast<double>(i));
            power += 0.5 * _amplitude[i] * _amplitude[i];
        }
        const double scale = 1.0 / std::sqrt(power); // unit RMS, so an absolute error is a relative one
        for (double& a : _amplitude) {
            a *= scale;
        }
    }

    [[nodiscard]] double at(double t) const {
        double sum = 0.0;
        for (std::size_t i = 0UZ; i < kTones; ++i) {
            sum += _amplitude[i] * std::cos(2.0 * kPi * _frequency[i] * t + _phase[i]);
        }
        return sum;
    }

private:
    static constexpr std::size_t kTones             = 12UZ;
    double                       _frequency[kTones] = {};
    double                       _phase[kTones]     = {};
    double                       _amplitude[kTones] = {};
};

} // namespace

const boost::ut::suite<"mmse interpolator"> mmseTests = [] {
    using namespace boost::ut;

    "the normal equations are the ones the objective implies"_test = [] {
        const std::vector<double> matrix = mmseCorrelation(8, 0.25);
        for (int m = 0; m < 8; ++m) {
            const double expected = m == 0 ? 1.0 : std::sin(kPi * 0.5 * static_cast<double>(m)) / (kPi * 0.5 * static_cast<double>(m));
            expect(approx(matrix[static_cast<std::size_t>(m)], expected, 1.0e-15)) << std::format("R[0][{}] = {:.8f} against sinc({}/2)", m, matrix[static_cast<std::size_t>(m)], m);
            if (m != 0 && m % 2 == 0) {
                expect(lt(std::abs(matrix[static_cast<std::size_t>(m)]), 1.0e-15)) << std::format("R[0][{}] is an even lag and sinc of an integer vanishes", m);
            }
        }
        for (std::size_t k = 0UZ; k < 8UZ; ++k) {
            for (std::size_t l = 0UZ; l < 8UZ; ++l) {
                expect(eq(matrix[k * 8UZ + l], matrix[l * 8UZ + k])) << "R is symmetric";
                expect(eq(matrix[k * 8UZ + l], matrix[((k + 1UZ) % 8UZ) * 8UZ + ((l + 1UZ) % 8UZ)]) || (k + 1UZ == 8UZ) || (l + 1UZ == 8UZ)) << "and Toeplitz";
            }
        }
    };

    "the solution satisfies its own optimality condition"_test = [] {
        for (int nTaps : {6, 8, 12}) {
            const std::vector<double> matrix = mmseCorrelation(nTaps, 0.25);
            const std::size_t         order  = static_cast<std::size_t>(nTaps);

            double worst = 0.0;
            for (int i = 0; i <= 20; ++i) {
                const double              mu     = static_cast<double>(i) / 20.0;
                const std::vector<double> taps   = mmseTaps(nTaps, 0.25, mu);
                const std::vector<double> target = mmseTarget(nTaps, 0.25, mu);
                for (std::size_t k = 0UZ; k < order; ++k) {
                    double product = 0.0;
                    for (std::size_t l = 0UZ; l < order; ++l) {
                        product += matrix[k * order + l] * taps[l];
                    }
                    worst = std::max(worst, std::abs(product - target[k]));
                }
            }
            expect(lt(worst, 1.0e-14)) << std::format("L={}: max|R h - p| over 21 values of mu is {:.3g}", nTaps, worst);
        }
    };

    "the gradient by quadrature agrees, and it never touches R or p"_test = [] {
        for (int nTaps : {6, 8, 12}) {
            const std::vector<double> taps     = mmseTaps(nTaps, 0.25, 0.5);
            const double              gradient = maxAbsolute(quadratureGradient(taps, 0.25, 0.5, false));
            expect(lt(gradient, 1.0e-10)) << std::format("L={}, mu=0.5: max|grad J| by quadrature is {:.3g}", nTaps, gradient);
        }
        const std::vector<double> quarter = mmseTaps(8, 0.25, 0.25);
        expect(lt(maxAbsolute(quadratureGradient(quarter, 0.25, 0.25, false)), 1.0e-10));
        std::println("mmse: max|grad J| by quadrature at L=8, mu=0.5 is {:.3g}", maxAbsolute(quadratureGradient(mmseTaps(8, 0.25, 0.5), 0.25, 0.5, false)));
    };

    "the stationary point is a minimum"_test = [] {
        const std::vector<double> taps      = mmseTaps(8, 0.25, 0.5);
        const double              objective = mmseObjective(taps, 0.25, 0.5);
        expect(lt(std::abs(objective / 2.432577e-07 - 1.0), 1.0e-6)) << std::format("J(h) = {:.6e}", objective);

        std::mt19937                     rng(4U);
        std::normal_distribution<double> gaussian(0.0, 1.0);
        double                           smallest = 1e30;
        for (int trial = 0; trial < 200; ++trial) {
            std::vector<double> perturbed = taps;
            std::vector<double> direction(perturbed.size());
            double              norm = 0.0;
            for (double& d : direction) {
                d = gaussian(rng);
                norm += d * d;
            }
            norm = std::sqrt(norm);
            for (std::size_t k = 0UZ; k < perturbed.size(); ++k) {
                perturbed[k] += 1.0e-3 * direction[k] / norm;
            }
            smallest = std::min(smallest, mmseObjective(perturbed, 0.25, 0.5));
        }
        expect(gt(smallest, objective)) << std::format("the smallest of 200 perturbed objectives is {:.6e} against {:.6e}", smallest, objective);
    };

    "the three tap rows are the ones the equations give"_test = [] {
        const std::vector<double> half    = mmseTaps(8, 0.25, 0.5);
        const std::vector<double> quarter = mmseTaps(8, 0.25, 0.25);
        const std::vector<double> fine    = mmseTaps(8, 0.25, 1.0 / 128.0);
        for (std::size_t k = 0UZ; k < 8UZ; ++k) {
            expect(approx(half[k], kTapsHalf[k], 1.0e-8)) << std::format("h(0.5)[{}] = {:.8f}", k, half[k]);
            expect(approx(quarter[k], kTapsQuarter[k], 1.0e-8)) << std::format("h(0.25)[{}] = {:.8f}", k, quarter[k]);
            expect(approx(fine[k], kTapsFine[k], 1.0e-8)) << std::format("h(1/128)[{}] = {:.8f}", k, fine[k]);
        }
    };

    "the symmetry identity holds, which is what catches an index error"_test = [] {
        double worst = 0.0;
        for (int i = 0; i <= 20; ++i) {
            const double              mu       = static_cast<double>(i) / 20.0;
            const std::vector<double> forward  = mmseTaps(8, 0.25, mu);
            const std::vector<double> backward = mmseTaps(8, 0.25, 1.0 - mu);
            for (std::size_t k = 0UZ; k < forward.size(); ++k) {
                worst = std::max(worst, std::abs(forward[k] - backward[forward.size() - 1UZ - k]));
            }
        }
        expect(lt(worst, 1.0e-10)) << std::format("h(1-mu) against reverse(h(mu)) over 21 values: {:.3g}", worst);

        const std::vector<double> zero = mmseTaps(8, 0.25, 0.0);
        for (std::size_t k = 0UZ; k < zero.size(); ++k) {
            expect(lt(std::abs(zero[k] - (k == 4UZ ? 1.0 : 0.0)), 1.0e-10)) << std::format("h(0)[{}] = {:.3g}: a unit impulse at index L/2", k, zero[k]);
        }
    };

    "the interpolated sample sits at i + L/2 - 1 + mu"_test = [] {
        // The test that pins the delay convention. Everything downstream, every tag this feeds,
        // moves by one input sample if the window is booked one sample later.
        const BandLimitedSignal    signal(11U);
        const MmseInterpolatorBank bank(8, 128, 0.25, false);

        std::vector<double> samples(64);
        for (std::size_t n = 0UZ; n < samples.size(); ++n) {
            samples[n] = signal.at(static_cast<double>(n));
        }
        constexpr std::size_t kStart = 20UZ;

        for (double mu : {0.0, 0.3, 0.7}) {
            const double interpolated = bank.interpolate(samples.data() + kStart, bank.row(mu));
            const double atConvention = signal.at(static_cast<double>(kStart) + 3.0 + mu);
            const double oneEarlier   = signal.at(static_cast<double>(kStart) + 2.0 + mu);
            const double oneLater     = signal.at(static_cast<double>(kStart) + 4.0 + mu);

            expect(lt(std::abs(interpolated - atConvention), 1.0e-3)) << std::format("mu={:.2f}: {:+.8f} against the exact {:+.8f}", mu, interpolated, atConvention);
            expect(lt(std::abs(interpolated - atConvention), std::abs(interpolated - oneEarlier))) << "and closer to i+3+mu than to i+2+mu";
            expect(lt(std::abs(interpolated - atConvention), std::abs(interpolated - oneLater))) << "and than to i+4+mu";
            std::println("mmse delay: mu={:.2f} error {:.2e} on a unit-RMS signal band-limited to 0.24", mu, std::abs(interpolated - atConvention));
        }
        expect(lt(std::abs(bank.interpolate(samples.data() + kStart, bank.row(0.0)) - samples[kStart + 3UZ]), 1.0e-10)) << "at mu = 0 it is sample selection, exactly";
        expect(eq(bank.delay(), 3UZ)) << "which is L/2 - 1";
    };

    "the response error and the band-edge droop are the measured tables"_test = [] {
        struct AccuracyRow {
            int    nTaps    = 0;
            double band     = 0.0;
            double expected = 0.0;
        };
        constexpr AccuracyRow kAccuracy[] = {{6, 0.25, 6.94e-03}, {6, 1.0 / 3.0, 5.02e-02}, {8, 0.25, 1.20e-03}, {8, 1.0 / 3.0, 1.69e-02}, {12, 0.25, 3.56e-05}, {12, 1.0 / 3.0, 1.90e-03}};

        for (const AccuracyRow& row : kAccuracy) {
            double worst   = 0.0;
            double worstMu = 0.0;
            for (int i = 0; i <= 32; ++i) {
                const double              mu   = static_cast<double>(i) / 32.0;
                const std::vector<double> taps = mmseTaps(row.nTaps, row.band, mu);
                double                    here = 0.0;
                for (int j = 0; j <= 200; ++j) {
                    const double frequency = row.band * static_cast<double>(j) / 200.0;
                    here                   = std::max(here, std::abs(responseOf(taps, frequency) - std::polar(1.0, 2.0 * kPi * frequency * mu)));
                }
                if (here > worst) {
                    worst   = here;
                    worstMu = mu;
                }
            }
            expect(lt(std::abs(worst / row.expected - 1.0), 0.05)) << std::format("L={} B={:.4f}: {:.3e} against {:.3e}", row.nTaps, row.band, worst, row.expected);
            expect(approx(worstMu, 0.5, 1.0e-9)) << "and the worst mu is 0.5, in every case";
        }

        struct DroopRow {
            double frequency = 0.0;
            double magnitude = 0.0;
        };
        constexpr DroopRow kDroop[] = {{0.25, 0.99880}, {0.30, 0.97537}, {0.35, 0.88920}, {0.40, 0.69761}, {0.45, 0.38821}};

        const std::vector<double> half = mmseTaps(8, 0.25, 0.5);
        for (const DroopRow& row : kDroop) {
            expect(approx(std::abs(responseOf(half, row.frequency)), row.magnitude, 1.0e-4)) << std::format("|H({:.2f})| = {:.5f}", row.frequency, std::abs(responseOf(half, row.frequency)));
        }
        // At two samples per symbol and rolloff 0.35 the signal's own band edge is 0.3375, where
        // the interpolator is 0.73 dB down.
        const double atTwoSamplesPerSymbol = std::abs(responseOf(half, 0.3375));
        expect(lt(20.0 * std::log10(atTwoSamplesPerSymbol), -0.7)) << std::format("|H(0.3375)| = {:.5f}, {:.2f} dB", atTwoSamplesPerSymbol, 20.0 * std::log10(atTwoSamplesPerSymbol));
        expect(gt(20.0 * std::log10(atTwoSamplesPerSymbol), -0.8));
    };

    "the differentiator comes from the same factorization and a different right-hand side"_test = [] {
        const std::vector<double> half = mmseDifferentiatorTaps(8, 0.25, 0.5);
        for (std::size_t k = 0UZ; k < half.size(); ++k) {
            expect(approx(half[k], kDerivativeHalf[k], 1.0e-8)) << std::format("h_d(0.5)[{}] = {:.8f}", k, half[k]);
        }
        expect(lt(maxAbsolute(quadratureGradient(half, 0.25, 0.5, true)), 1.0e-10)) << "its own objective's gradient, by quadrature";

        const std::vector<double> zero = mmseDifferentiatorTaps(8, 0.25, 0.0);
        for (std::size_t k = 0UZ; k < zero.size(); ++k) {
            expect(approx(zero[k], kDerivativeZero[k], 1.0e-8)) << std::format("h_d(0)[{}] = {:.8f}", k, zero[k]);
        }
        double sum = 0.0;
        for (double tap : zero) {
            sum += tap;
        }
        // The fit leaves H(0) free, so there is a small residual DC response. Asserting == 0 would
        // be asserting a property the objective was never given.
        expect(lt(std::abs(sum), 1.0e-3)) << std::format("sum h_d(0) = {:.3e}", sum);
        expect(gt(std::abs(sum), 1.0e-5)) << "and it is not zero either";

        double worst = 0.0;
        for (int i = 0; i <= 20; ++i) {
            const double              mu       = static_cast<double>(i) / 20.0;
            const std::vector<double> forward  = mmseDifferentiatorTaps(8, 0.25, mu);
            const std::vector<double> backward = mmseDifferentiatorTaps(8, 0.25, 1.0 - mu);
            for (std::size_t k = 0UZ; k < forward.size(); ++k) {
                worst = std::max(worst, std::abs(forward[k] + backward[forward.size() - 1UZ - k]));
            }
        }
        expect(lt(worst, 1.0e-9)) << std::format("h_d(1-mu) against -reverse(h_d(mu)): {:.3g}", worst);
    };

    "the differentiator differentiates"_test = [] {
        const MmseInterpolatorBank bank(8, 128, 0.25, true);
        for (double frequency : {0.05, 0.10, 0.20}) {
            std::vector<double> samples(64);
            for (std::size_t n = 0UZ; n < samples.size(); ++n) {
                samples[n] = std::cos(2.0 * kPi * frequency * static_cast<double>(n) + 0.4);
            }
            for (double mu : {0.0, 0.25, 0.5, 0.75}) {
                constexpr std::size_t kStart   = 20UZ;
                const double          position = static_cast<double>(kStart) + 3.0 + mu;
                const double          slope    = bank.differentiate(samples.data() + kStart, bank.row(mu));
                // d/dn cos(2 pi f n + phi) = 2 pi f cos(2 pi f n + phi + pi/2)
                const double expected = 2.0 * kPi * frequency * std::cos(2.0 * kPi * frequency * position + 0.4 + kPi / 2.0);
                expect(lt(std::abs(slope - expected), 0.01 * 2.0 * kPi * frequency)) << std::format("f={:.2f} mu={:.2f}: {:+.6f} against {:+.6f}", frequency, mu, slope, expected);
            }
        }
    };

    "the bank is the same solve, reversed, on a grid of 129 rows"_test = [] {
        const MmseInterpolatorBank bank(8, 128, 0.25, true);
        expect(eq(bank.size(), 8UZ));
        expect(eq(bank.steps(), 128UZ));
        expect(that % bank.hasDerivative());

        for (std::size_t row : {0UZ, 1UZ, 32UZ, 64UZ, 100UZ, 128UZ}) {
            const double                 mu          = static_cast<double>(row) / 128.0;
            const std::vector<double>    reference   = mmseTaps(8, 0.25, mu);
            const std::vector<double>    slope       = mmseDifferentiatorTaps(8, 0.25, mu);
            const std::span<const float> stored      = bank.tapsFor(row);
            const std::span<const float> storedSlope = bank.derivativeTapsFor(row);
            for (std::size_t k = 0UZ; k < 8UZ; ++k) {
                expect(eq(stored[k], static_cast<float>(reference[7UZ - k]))) << std::format("row {} tap {}: the bank holds the solve reversed, bit for bit", row, k);
                expect(eq(storedSlope[k], static_cast<float>(slope[7UZ - k])));
            }
        }

        expect(eq(bank.row(0.0), 0UZ));
        expect(eq(bank.row(1.0), 128UZ));
        expect(eq(bank.row(0.5), 64UZ));
        expect(eq(bank.row(0.5 + 1.0 / 512.0), 64UZ)) << "round to nearest, so the quantization error never exceeds half a step";
        expect(eq(bank.row(0.5 + 3.0 / 512.0), 65UZ));
        expect(eq(bank.row(-0.3), 0UZ)) << "outside [0,1] is a caller error and is clamped, never wrapped";
        expect(eq(bank.row(1.7), 128UZ));
    };

    "row selection is nearest with ties up, over the whole quantization domain"_test = [] {
        // `row` adds a half and truncates rather than calling nearbyint, which cannot be inlined.
        // Off a tie the two are the same row; on one they name the two equally distant rows, and
        // this test pins which of the two is chosen.
        const auto reference = [](double mu, std::size_t nSteps) -> std::size_t {
            const double scaled = std::nearbyint(mu * static_cast<double>(nSteps));
            if (!(scaled > 0.0)) {
                return 0UZ;
            }
            return scaled < static_cast<double>(nSteps) ? static_cast<std::size_t>(scaled) : nSteps;
        };

        constexpr std::size_t kSweep = 40000UZ;
        for (int steps : {8, 32, 128, 1024}) {
            const MmseInterpolatorBank bank(8, steps, 0.25, false);
            const auto                 nSteps = static_cast<std::size_t>(steps);

            std::size_t offTie      = 0UZ;
            std::size_t skippedTie  = 0UZ;
            std::size_t disagreeing = 0UZ;
            double      worst       = 0.0;
            for (std::size_t j = 0UZ; j <= kSweep; ++j) {
                const double      mu       = static_cast<double>(j) / static_cast<double>(kSweep);
                const double      scaled   = mu * static_cast<double>(nSteps);
                const std::size_t selected = bank.row(mu);

                worst = std::max(worst, std::abs(static_cast<double>(selected) / static_cast<double>(nSteps) - mu));
                if (std::abs(scaled - std::nearbyint(scaled)) == 0.5) {
                    ++skippedTie;
                    continue;
                }
                ++offTie;
                if (selected != reference(mu, nSteps)) {
                    ++disagreeing;
                    if (disagreeing <= 3UZ) {
                        expect(eq(selected, reference(mu, nSteps))) << std::format("nSteps={} mu={:.17g}: row {} against nearbyint's {}", nSteps, mu, selected, reference(mu, nSteps));
                    }
                }
            }
            expect(eq(disagreeing, 0UZ)) << std::format("nSteps={}: {} of {} off-tie values disagree with nearbyint", nSteps, disagreeing, offTie);
            expect(gt(offTie, kSweep / 2UZ)) << "more than half the swept values are off-tie";
            expect(le(worst, 0.5 / static_cast<double>(nSteps))) << std::format("nSteps={}: worst |row/nSteps - mu| is {:.3g} against the half-step bound {:.3g}", nSteps, worst, 0.5 / static_cast<double>(nSteps));
            std::println("mmse row: nSteps={} — {} off-tie values agree with nearbyint exactly, {} ties skipped, worst quantization {:.3g} of a sample", nSteps, offTie, skippedTie, worst);

            // Every exact tie, `mu = (2k+1)/(2*nSteps)`, which a power-of-two `nSteps` scales exactly.
            std::size_t notUp             = 0UZ;
            std::size_t movedByTheTieRule = 0UZ;
            for (std::size_t k = 0UZ; k < nSteps; ++k) {
                const double mu = (2.0 * static_cast<double>(k) + 1.0) / (2.0 * static_cast<double>(nSteps));
                if (bank.row(mu) != k + 1UZ) {
                    ++notUp;
                }
                if (reference(mu, nSteps) != k + 1UZ) {
                    ++movedByTheTieRule;
                }
            }
            expect(eq(notUp, 0UZ)) << std::format("nSteps={}: {} of {} ties do not go up", nSteps, notUp, nSteps);
            expect(eq(movedByTheTieRule, (nSteps + 1UZ) / 2UZ)) << "nearbyint rounds a tie to even, so it differs on exactly the ties whose lower row is even";
        }

        const MmseInterpolatorBank bank(8, 128, 0.25, false);
        expect(eq(bank.row(0.5 / 128.0), 1UZ)) << "the smallest tie there is";
        expect(eq(bank.row(1.5 / 128.0), 2UZ)) << "and one nearbyint would have agreed on";
        expect(eq(bank.row(std::nan("")), 0UZ)) << "NaN selects row 0";
        expect(eq(bank.row(1.0e300), 128UZ)) << "and the clamp is total, however far outside [0,1] the caller is";
        expect(eq(bank.row(-1.0e300), 0UZ));
    };

    "a dot product's accumulation order is a function of L alone"_test = [] {
        const MmseInterpolatorBank            bank(8, 128, 0.25, false);
        std::mt19937                          rng(19U);
        std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);

        std::vector<CF> window(8);
        for (CF& v : window) {
            v = CF{uniform(rng), uniform(rng)};
        }
        for (std::size_t row : {0UZ, 7UZ, 64UZ, 128UZ}) {
            const std::span<const float> taps = bank.tapsFor(row);
            CF                           reference{};
            for (std::size_t k = 0UZ; k < 8UZ; ++k) {
                reference += taps[k] * window[k];
            }
            expect(eq(bank.interpolate(window.data(), row), reference)) << "the same sum in the same order, bit for bit";
        }
    };

    "the conditioning is why the solve is in double"_test = [] {
        const std::vector<double>                      matrix = mmseCorrelation(8, 0.25);
        const gr::sync::detail::SymmetricFactorization factored(matrix, 8UZ);

        const auto normalize = [](std::vector<double>& v) {
            double norm = 0.0;
            for (double x : v) {
                norm += x * x;
            }
            norm = std::sqrt(norm);
            for (double& x : v) {
                x /= norm;
            }
            return norm;
        };

        std::vector<double> vector(8UZ, 1.0);
        std::vector<double> product(8UZ);
        double              largest = 0.0;
        for (int iteration = 0; iteration < 400; ++iteration) {
            for (std::size_t k = 0UZ; k < 8UZ; ++k) {
                product[k] = 0.0;
                for (std::size_t l = 0UZ; l < 8UZ; ++l) {
                    product[k] += matrix[k * 8UZ + l] * vector[l];
                }
            }
            largest = normalize(product);
            vector  = product;
        }

        std::vector<double> inverseVector(8UZ, 1.0);
        std::vector<double> solved(8UZ);
        double              inverseLargest = 0.0;
        for (int iteration = 0; iteration < 400; ++iteration) {
            factored.solve(inverseVector, solved);
            inverseLargest = normalize(solved);
            inverseVector  = solved;
        }

        const double condition = largest * inverseLargest;
        expect(lt(std::abs(condition / 6.19e4 - 1.0), 0.01)) << std::format("cond(R) = {:.4g} at L=8, B=0.25 — eleven digits kept in double, five in float", condition);
    };

    "degenerate designs throw"_test = [] {
        expect(throws<std::invalid_argument>([] { (void)mmseTaps(7, 0.25, 0.5); })) << "an odd tap count has no center pair to straddle";
        expect(throws<std::invalid_argument>([] { (void)mmseTaps(0, 0.25, 0.5); }));
        expect(throws<std::invalid_argument>([] { (void)mmseTaps(8, 0.0, 0.5); }));
        expect(throws<std::invalid_argument>([] { (void)mmseTaps(8, 0.5, 0.5); })) << "the whole band is not a design band";
        expect(throws<std::invalid_argument>([] { (void)MmseInterpolatorBank(8, 100, 0.25); })) << "nSteps must be a power of two";
        expect(nothrow([] { (void)MmseInterpolatorBank(12, 256, 1.0 / 3.0); }));
    };

    "the bank is regenerated from the normal equations at construction"_test = [] {
        // The bank is solved rather than tabulated, which is what keeps its parameters free. The cost is
        // recorded rather than asserted on: it is a setup-path figure and belongs to the machine that read it.
        double best = 1e30;
        for (int repeat = 0; repeat < 20; ++repeat) {
            const auto                 start = Clock::now();
            const MmseInterpolatorBank bank(8, 128, 0.25, true);
            const double               microseconds = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count()) / 1000.0;
            expect(eq(bank.size(), 8UZ));
            best = std::min(best, microseconds);
        }
        std::println("mmse bank: L=8, nSteps=128, with derivative, generated from the normal equations in {:.1f} us — setup path, not per sample", best);
    };
};

int main() { /* tests are automatically registered and run */ }
