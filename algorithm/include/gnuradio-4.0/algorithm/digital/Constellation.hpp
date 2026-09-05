#ifndef GNURADIO_ALGORITHM_CONSTELLATION_HPP
#define GNURADIO_ALGORITHM_CONSTELLATION_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

/**
 * @brief The map between integer symbols and points in the complex plane, and the two inverses a
 * receiver needs: the hard decision and the per-bit soft decision.
 *
 * One value type holds the points, the labeling, the normalization and the decision strategy the
 * geometry admits. The strategy -- `SquareQam` for square Gray-labeled QAM, `Psk` for equally spaced
 * points on one circle, `Nearest` for anything else -- is resolved once, at construction, from the
 * points themselves, so nothing dispatches per symbol.
 *
 * Conventions:
 *
 * - A label is the block's output byte, MSB first. Bit index 0 of `softDecisions`' output is the most
 *   significant bit of the label.
 * - One means positive. BPSK symbol 1 is `+1`; QPSK is `2*(Im>0) + (Re>0)`; within a square-QAM axis
 *   field the most significant bit is the sign.
 * - A positive soft value means the bit is one: `ln(P(b=1|z)/P(b=0|z))`. Much of the coding
 *   literature writes the ratio the other way up; anything crossing that boundary needs one negation.
 * - A noise power is linear, never dB, and it is the total complex noise power `N0`.
 * - Gray labeling means, operationally, that every pair of points at the minimum distance has labels
 *   differing in exactly one bit. All six factory constellations satisfy it.
 *
 * The point sets. PSK is `c[gray(g) ^ labelXor] = exp(j*(phaseOffset + 2*pi*g/M))`; square QAM
 * (`M = 4^k`) carries `ma = m/2` bits per axis over `L = 2^ma` levels `-(L-1) .. +(L-1)`, the level at
 * position `i` taking the axis label `gray(i)`, and the symbol label being `(labelQ << ma) | labelI`.
 * `bpsk()` and `qpsk()` take `labelXor` 1 and 3, which reproduce the labeling deployed
 * implementations transmit and expect; changing them breaks interoperability. No such settled
 * convention exists for 8PSK, so its rotation and label offset are parameters, and a receiver talking
 * to particular equipment states that equipment's own.
 *
 * Normalization scales the points once, at construction: `Power`, the default, by `sqrt(mean |c|^2)`,
 * `Amplitude` by `mean |c|`, `None` not at all. The two agree for any constant-modulus constellation
 * and differ for QAM. `Power` is the default because every noise figure, every `Es/N0`, every detector
 * gain in the synchronization family and every SNR estimate is a power. Normalization fixes the
 * transmitter's reference and is not an AGC: `hardDecision` is scale-invariant for PSK but not for
 * QAM, and every soft decision compares a distance against `N0`, so an AGC holding the input at the
 * constellation's own scale is a precondition of both.
 *
 * `Nearest` is the definition of the hard decision and resolves a tie to the lowest symbol; the closed
 * forms resolve theirs the same way. Two ties are decided by floating point rather than by that rule,
 * because the slicer rounds in the raw lattice while `Nearest` measures against the normalized points:
 * the two place a boundary other than an axis zero a couple of ulp apart, and at the origin of a
 * constant-modulus constellation the closed form answers `sectorSymbol[0]`, the symbol at the
 * constellation's own phase offset, while `Nearest` decides on the last bit of each point's magnitude.
 * Both are deterministic.
 *
 * Soft decisions. `MaxLog` is the default and is what the closed forms compute:
 *
 * ```
 *   L_k(z) ~= ( min_{s: bit_k(s)=0} |z - c[s]|^2 - min_{s: bit_k(s)=1} |z - c[s]|^2 ) / N0
 * ```
 *
 * It is exact for BPSK and QPSK, where each bit's two candidate subsets each hold exactly one nearest
 * point, and its error is about `ln 2` elsewhere, the worst case being a tie between the two nearest
 * points of one subset. `Exact` computes the true LLR from the full `M`-point sums, which do not
 * shorten the way the max-log minima do. Under `MaxLog`, `sign(L_k)` is bit `k` of
 * `hardDecision`; that does not hold for `Exact`, where a distant cluster of same-bit points can
 * outweigh a nearer single one.
 *
 * Rotation order. Gray labeling minimizes bit errors per symbol error; rotation-ordered labeling,
 * `c[(r+1) mod M] = c[r] * exp(j*2*pi/M)`, is what differential coding needs, and for `M >= 4` the two
 * are different labelings of the same points. `rotationToGray()` and `grayToRotation()` are the two
 * permutations and exist only for a constellation whose points lie on a circle, since rotation order
 * is meaningless off one. For the `psk(M, phi0, k)` family `rotationToGray(r) == gray(r) ^ k`.
 */
namespace gr::digital {

/// @brief Which reference scale the stored points carry. `Power` is the default; see the file comment.
enum class Normalization { Power, Amplitude, None };

/// @brief Which decision the geometry admits, resolved once at construction and never per symbol.
enum class DecisionStrategy { Psk, SquareQam, Nearest };

/// @brief `MaxLog` needs no transcendental and is exact for BPSK and QPSK; `Exact` is the true LLR.
enum class SoftAlgorithm { MaxLog, Exact };

[[nodiscard]] inline constexpr std::uint32_t grayEncode(std::uint32_t index) noexcept { return index ^ (index >> 1U); }

[[nodiscard]] inline constexpr std::uint32_t grayDecode(std::uint32_t code) noexcept {
    std::uint32_t index = code;
    for (std::uint32_t shift = 1U; shift < 32U; shift <<= 1U) {
        index ^= index >> shift;
    }
    return index;
}

template<std::floating_point F>
class Constellation {
public:
    using value_type = std::complex<F>;

    static constexpr std::size_t kMaxPoints = 256UZ;

    static constexpr std::size_t kMaxAxisBits   = 4UZ;
    static constexpr std::size_t kMaxAxisLevels = 16UZ;

    [[nodiscard]] static Constellation bpsk(Normalization mode = Normalization::Power) { return psk(2UZ, F{0}, 1U, mode); }
    [[nodiscard]] static Constellation qpsk(Normalization mode = Normalization::Power) { return psk(4UZ, static_cast<F>(std::numbers::pi / 4.0), 3U, mode); }
    [[nodiscard]] static Constellation psk8(Normalization mode = Normalization::Power) { return psk(8UZ, F{0}, 0U, mode); }

    [[nodiscard]] static Constellation psk(std::size_t arity, F phaseOffset = F{0}, std::uint8_t labelXor = 0U, Normalization mode = Normalization::Power) {
        requirePowerOfTwo(arity, "psk");
        std::vector<std::complex<F>> raw(arity);
        for (std::uint32_t g = 0U; g < arity; ++g) {
            raw[(grayEncode(g) ^ labelXor) & (arity - 1UZ)] = unitPhasor(phaseOffset, g, arity);
        }
        return Constellation(std::move(raw), mode, phaseOffset, true);
    }

    [[nodiscard]] static Constellation qam(std::size_t arity, Normalization mode = Normalization::Power) {
        requirePowerOfTwo(arity, "qam");
        const std::size_t bits = bitCount(arity);
        if (bits % 2UZ != 0UZ) {
            throw std::invalid_argument("gr::digital::Constellation::qam: arity must be a power of four, got " + std::to_string(arity));
        }
        const std::size_t            axisBits   = bits / 2UZ;
        const std::size_t            axisLevels = 1UZ << axisBits;
        std::vector<std::complex<F>> raw(arity);
        for (std::uint32_t iQ = 0U; iQ < axisLevels; ++iQ) {
            for (std::uint32_t iI = 0U; iI < axisLevels; ++iI) {
                const auto label = static_cast<std::size_t>((grayEncode(iQ) << axisBits) | grayEncode(iI));
                raw[label]       = std::complex<F>(rawLevel(iI, axisLevels), rawLevel(iQ, axisLevels));
            }
        }
        return Constellation(std::move(raw), mode, F{0}, false);
    }

    [[nodiscard]] static Constellation custom(std::span<const std::complex<F>> pointsInSymbolOrder, Normalization mode = Normalization::Power) {
        requirePowerOfTwo(pointsInSymbolOrder.size(), "custom");
        return Constellation(std::vector<std::complex<F>>(pointsInSymbolOrder.begin(), pointsInSymbolOrder.end()), mode, F{0}, false);
    }

    [[nodiscard]] std::size_t                      size() const noexcept { return arity_; }
    [[nodiscard]] std::size_t                      bitsPerSymbol() const noexcept { return bits_; }
    [[nodiscard]] DecisionStrategy                 strategy() const noexcept { return strategy_; }
    [[nodiscard]] Normalization                    normalization() const noexcept { return normalization_; }
    [[nodiscard]] F                                normalizationScale() const noexcept { return normalizationScale_; }
    [[nodiscard]] F                                minimumDistance() const noexcept { return minimumDistance_; }
    [[nodiscard]] std::size_t                      rotationalSymmetry() const noexcept { return onCircle_ ? arity_ : (strategy_ == DecisionStrategy::SquareQam ? 4UZ : 1UZ); }
    [[nodiscard]] std::span<const std::complex<F>> points() const noexcept { return points_; }

    /// @brief The point for `symbol`, masked to the arity: a stream byte can never read past the list.
    [[nodiscard]] std::complex<F> point(std::uint8_t symbol) const noexcept { return points_[static_cast<std::size_t>(symbol & symbolMask_)]; }

    /**
     * @brief The two permutations between rotation order and this constellation's labeling.
     *
     * 256 entries each, identity beyond the arity, and present only where the points lie on a circle.
     * `rotationToGray(r)` is the symbol at angle `phaseOffset + r*2*pi/M`.
     */
    [[nodiscard]] std::optional<std::array<std::uint8_t, 256>> rotationToGray() const {
        if (!onCircle_) {
            return std::nullopt;
        }
        return sectorSymbol_;
    }

    [[nodiscard]] std::optional<std::array<std::uint8_t, 256>> grayToRotation() const {
        if (!onCircle_) {
            return std::nullopt;
        }
        return symbolSector_;
    }

    [[nodiscard]] std::uint8_t hardDecision(std::complex<F> z) const noexcept {
        switch (strategy_) {
        case DecisionStrategy::SquareQam: return decideSquareQam(z);
        case DecisionStrategy::Psk: return decidePsk(z);
        default: return decideNearest(z);
        }
    }

    /// @brief The definition: the label of the nearest point, ties to the lowest symbol.
    [[nodiscard]] std::uint8_t nearestDecision(std::complex<F> z) const noexcept { return decideNearest(z); }

    void hardDecisions(std::span<const std::complex<F>> in, std::span<std::uint8_t> out) const {
        if (out.size() < in.size()) {
            throw std::invalid_argument("gr::digital::Constellation::hardDecisions: output span is shorter than the input");
        }
        switch (strategy_) {
        case DecisionStrategy::SquareQam:
            for (std::size_t i = 0UZ; i < in.size(); ++i) {
                out[i] = decideSquareQam(in[i]);
            }
            return;
        case DecisionStrategy::Psk:
            for (std::size_t i = 0UZ; i < in.size(); ++i) {
                out[i] = decidePsk(in[i]);
            }
            return;
        default:
            for (std::size_t i = 0UZ; i < in.size(); ++i) {
                out[i] = decideNearest(in[i]);
            }
            return;
        }
    }

    /// @brief `bitsPerSymbol()` log-likelihood ratios, MSB first, by the cheapest route the geometry allows.
    void softDecisions(std::complex<F> z, F noisePower, std::span<F> out, SoftAlgorithm algorithm = SoftAlgorithm::MaxLog) const {
        checkSoftArguments(noisePower, out.size());
        softDecisionsUnchecked(z, noisePower, out, algorithm);
    }

    void softDecisions(std::span<const std::complex<F>> in, F noisePower, std::span<F> out, SoftAlgorithm algorithm = SoftAlgorithm::MaxLog) const {
        checkSoftArguments(noisePower, bits_);
        if (out.size() < in.size() * bits_) {
            throw std::invalid_argument("gr::digital::Constellation::softDecisions: output span holds fewer than bitsPerSymbol() values per symbol");
        }
        for (std::size_t i = 0UZ; i < in.size(); ++i) {
            softDecisionsUnchecked(in[i], noisePower, out.subspan(i * bits_, bits_), algorithm);
        }
    }

    /**
     * @brief The full `M`-point definition, which the closed forms are checked against.
     *
     * Also the only route for `Exact`, whose sums do not shorten the way the max-log minima do.
     */
    void softDecisionsExhaustive(std::complex<F> z, F noisePower, std::span<F> out, SoftAlgorithm algorithm = SoftAlgorithm::MaxLog) const {
        checkSoftArguments(noisePower, out.size());
        exhaustive(z, noisePower, out, algorithm);
    }

private:
    Constellation(std::vector<std::complex<F>>&& raw, Normalization mode, F phaseOffset, bool pskFamily) : points_(std::move(raw)), normalization_(mode) {
        arity_      = points_.size();
        bits_       = bitCount(arity_);
        symbolMask_ = static_cast<std::uint8_t>(arity_ - 1UZ);

        double sumMagnitude = 0.0;
        double sumPower     = 0.0;
        for (const auto& c : points_) {
            sumMagnitude += std::hypot(static_cast<double>(c.real()), static_cast<double>(c.imag()));
            sumPower += static_cast<double>(c.real()) * static_cast<double>(c.real()) + static_cast<double>(c.imag()) * static_cast<double>(c.imag());
        }
        const double meanMagnitude = sumMagnitude / static_cast<double>(arity_);
        const double rmsMagnitude  = std::sqrt(sumPower / static_cast<double>(arity_));
        const double multiplier    = mode == Normalization::Power ? 1.0 / rmsMagnitude : (mode == Normalization::Amplitude ? 1.0 / meanMagnitude : 1.0);
        if (!(multiplier > 0.0) || !std::isfinite(multiplier)) {
            throw std::invalid_argument("gr::digital::Constellation: the point set has no positive scale to normalize by");
        }
        normalizationScale_ = static_cast<F>(multiplier);
        for (auto& c : points_) {
            c *= normalizationScale_;
        }

        minimumDistance_ = std::numeric_limits<F>::infinity();
        for (std::size_t a = 0UZ; a + 1UZ < arity_; ++a) {
            for (std::size_t b = a + 1UZ; b < arity_; ++b) {
                minimumDistance_ = std::min(minimumDistance_, std::abs(points_[a] - points_[b]));
            }
        }

        detectCircle(phaseOffset, pskFamily);
        const bool square = detectSquareQam();
        strategy_         = square ? DecisionStrategy::SquareQam : (onCircle_ ? DecisionStrategy::Psk : DecisionStrategy::Nearest);
    }

    static void requirePowerOfTwo(std::size_t arity, const char* what) {
        if (arity < 2UZ || arity > kMaxPoints || (arity & (arity - 1UZ)) != 0UZ) {
            throw std::invalid_argument(std::string("gr::digital::Constellation::") + what + ": arity must be a power of two in [2, 256], got " + std::to_string(arity));
        }
    }

    [[nodiscard]] static std::size_t bitCount(std::size_t arity) noexcept {
        std::size_t bits = 0UZ;
        while ((1UZ << bits) < arity) {
            ++bits;
        }
        return bits;
    }

    [[nodiscard]] static F rawLevel(std::uint32_t position, std::size_t axisLevels) noexcept { return static_cast<F>(2.0 * static_cast<double>(position) - static_cast<double>(axisLevels - 1UZ)); }

    /**
     * @brief `exp(j*(phaseOffset + 2*pi*index/arity))`, exact on every eighth of a turn.
     *
     * `std::cos` and `std::sin` of a quarter turn are neither zero nor one, and the resulting 1-ulp
     * asymmetry displaces the BPSK and QPSK nearest-point boundaries off the axes, where the closed
     * forms keep them, so the two disagree over a thin wedge. Reducing the angle to eighths of a turn
     * and reading the eight exact values keeps the point set symmetric under both sign flips and the
     * coordinate swap, which is what makes the boundaries agree.
     */
    [[nodiscard]] static std::complex<F> unitPhasor(F phaseOffset, std::uint32_t index, std::size_t arity) noexcept {
        constexpr double kQuarterPi = std::numbers::pi / 4.0;
        constexpr F      kHalfRoot  = static_cast<F>(0.707106781186547524400844362104849);

        const double eighths = static_cast<double>(phaseOffset) / kQuarterPi + 8.0 * static_cast<double>(index) / static_cast<double>(arity);
        if (std::abs(eighths - std::round(eighths)) < 1.0e-12) {
            switch (static_cast<std::uint32_t>((std::llround(eighths) % 8LL + 8LL) % 8LL)) {
            case 0U: return {F{1}, F{0}};
            case 1U: return {kHalfRoot, kHalfRoot};
            case 2U: return {F{0}, F{1}};
            case 3U: return {-kHalfRoot, kHalfRoot};
            case 4U: return {F{-1}, F{0}};
            case 5U: return {-kHalfRoot, -kHalfRoot};
            case 6U: return {F{0}, F{-1}};
            default: return {kHalfRoot, -kHalfRoot};
            }
        }
        const F angle = phaseOffset + static_cast<F>(2.0 * std::numbers::pi * static_cast<double>(index) / static_cast<double>(arity));
        return {std::cos(angle), std::sin(angle)};
    }

    void detectCircle(F factoryPhase, bool pskFamily) {
        onCircle_ = false;
        sectorSymbol_.fill(0U);
        symbolSector_.fill(0U);

        const auto step   = static_cast<F>(2.0 * std::numbers::pi / static_cast<double>(arity_));
        const F    radius = std::abs(points_[0]);
        if (!(radius > F{0})) {
            return;
        }
        for (const auto& c : points_) {
            if (std::abs(std::abs(c) - radius) > static_cast<F>(1e-6) * radius) {
                return;
            }
        }

        F phase = factoryPhase;
        if (!pskFamily) {
            F angle = std::atan2(points_[0].imag(), points_[0].real());
            if (angle < F{0}) {
                angle += static_cast<F>(2.0 * std::numbers::pi);
            }
            phase = std::fmod(angle, step);
        }

        std::array<bool, kMaxPoints> seen{};
        for (std::uint32_t s = 0U; s < arity_; ++s) {
            const F    offset   = std::atan2(points_[s].imag(), points_[s].real()) - phase;
            const auto g        = static_cast<std::uint32_t>((std::llround(offset / step) % static_cast<long long>(arity_) + static_cast<long long>(arity_)) % static_cast<long long>(arity_));
            const F    residual = std::remainder(offset - step * static_cast<F>(g), static_cast<F>(2.0 * std::numbers::pi));
            if (std::abs(residual) > static_cast<F>(1e-6) || seen[g]) {
                return;
            }
            seen[g]          = true;
            sectorSymbol_[g] = static_cast<std::uint8_t>(s);
            symbolSector_[s] = static_cast<std::uint8_t>(g);
        }
        for (std::size_t i = arity_; i < kMaxPoints; ++i) {
            sectorSymbol_[i] = static_cast<std::uint8_t>(i);
            symbolSector_[i] = static_cast<std::uint8_t>(i);
        }

        onCircle_    = true;
        phaseCosine_ = std::cos(phase);
        phaseSine_   = std::sin(phase);
        sectorScale_ = static_cast<F>(static_cast<double>(arity_) / (2.0 * std::numbers::pi));
        binaryTie_   = std::min(sectorSymbol_[0], sectorSymbol_[1]);
    }

    [[nodiscard]] bool detectSquareQam() {
        if (bits_ % 2UZ != 0UZ || bits_ / 2UZ > kMaxAxisBits) {
            return false;
        }
        axisBits_   = bits_ / 2UZ;
        axisLevels_ = 1UZ << axisBits_;

        F extent = F{0};
        for (const auto& c : points_) {
            extent = std::max({extent, std::abs(c.real()), std::abs(c.imag())});
        }
        if (!(extent > F{0})) {
            return false;
        }
        axisScale_   = static_cast<F>(axisLevels_ - 1UZ) / extent;
        axisSpacing_ = F{2} / axisScale_;

        const F tolerance = static_cast<F>(1e-6) * axisSpacing_;
        for (std::uint32_t s = 0U; s < arity_; ++s) {
            const std::uint32_t iI       = grayDecode(s & static_cast<std::uint32_t>(axisLevels_ - 1UZ));
            const std::uint32_t iQ       = grayDecode(s >> axisBits_);
            const auto          expected = std::complex<F>(rawLevel(iI, axisLevels_) / axisScale_, rawLevel(iQ, axisLevels_) / axisScale_);
            if (std::abs(points_[s] - expected) > tolerance) {
                return false;
            }
        }

        for (std::uint32_t i = 0U; i < levelCount(); ++i) {
            axisGray_[i]    = static_cast<std::uint8_t>(grayEncode(i));
            axisLevel_[i]   = rawLevel(i, axisLevels_) / axisScale_;
            axisTieDown_[i] = static_cast<std::uint8_t>(i > 0U && grayEncode(i - 1U) < grayEncode(i) ? 1U : 0U);
        }
        axisHalfScale_ = axisScale_ * F{0.5};
        axisCenter_    = static_cast<F>(axisLevels_ / 2UZ);
        axisTop_       = static_cast<F>(axisLevels_ - 1UZ);
        return true;
    }

    /**
     * @brief The level index for one axis, with the offset added last.
     *
     * The levels sit at odd raw coordinates and the boundaries between them at even ones, so the index is
     * `floor(x*scale/2)` plus the center level `L/2`, and the coordinate is exactly on a boundary when that
     * half is an integer. Halving the scale is exact and `L/2` is an integer, so both facts survive one
     * rounding of one product. Folding the offset into the product, as `(x*scale + (L-1))*0.5 + 0.5`,
     * rounds every `x` within an ulp of `L-1` onto the tie, which then steps a positive coordinate to
     * the negative level.
     */
    [[nodiscard]] std::uint32_t axisIndex(F x) const noexcept {
        const F    half    = x * axisHalfScale_;
        const F    below   = std::floor(half);
        const F    level   = below + axisCenter_;
        const F    clamped = std::min(std::max(level, F{0}), axisTop_);
        const auto index   = static_cast<std::uint32_t>(clamped) & (kMaxAxisLevels - 1U);
        const bool tied    = half == below && level == clamped;
        return (tied ? index - static_cast<std::uint32_t>(axisTieDown_[index]) : index) & (kMaxAxisLevels - 1U);
    }

    [[nodiscard]] std::uint8_t decideSquareQam(std::complex<F> z) const noexcept { return static_cast<std::uint8_t>((axisGray_[axisIndex(z.imag())] << axisBits_) | axisGray_[axisIndex(z.real())]); }

    [[nodiscard]] std::uint8_t decidePsk(std::complex<F> z) const noexcept {
        const F real = z.real() * phaseCosine_ + z.imag() * phaseSine_;
        if (arity_ == 2UZ) {
            return real > F{0} ? sectorSymbol_[0] : (real < F{0} ? sectorSymbol_[1] : binaryTie_);
        }
        const F imag      = z.imag() * phaseCosine_ - z.real() * phaseSine_;
        F       angle     = std::atan2(imag, real);
        angle             = angle < F{0} ? angle + static_cast<F>(2.0 * std::numbers::pi) : angle;
        const F    scaled = angle * sectorScale_ + F{0.5};
        const auto sector = scaled >= F{0} ? static_cast<std::uint32_t>(scaled) : 0U;
        return sectorSymbol_[sector & (static_cast<std::uint32_t>(arity_) - 1U)];
    }

    [[nodiscard]] std::uint8_t decideNearest(std::complex<F> z) const noexcept {
        std::uint8_t best     = 0U;
        F            bestDist = std::numeric_limits<F>::infinity();
        for (std::uint32_t s = 0U; s < arity_; ++s) {
            const std::complex<F> difference = z - points_[s];
            const F               distance   = difference.real() * difference.real() + difference.imag() * difference.imag();
            if (distance < bestDist) {
                bestDist = distance;
                best     = static_cast<std::uint8_t>(s);
            }
        }
        return best;
    }

    void checkSoftArguments(F noisePower, std::size_t outputSize) const {
        if (!(noisePower > F{0})) {
            throw std::invalid_argument("gr::digital::Constellation::softDecisions: noise power must be positive and linear, got " + std::to_string(static_cast<double>(noisePower)));
        }
        if (outputSize < bits_) {
            throw std::invalid_argument("gr::digital::Constellation::softDecisions: output span holds fewer than bitsPerSymbol() values");
        }
    }

    void softDecisionsUnchecked(std::complex<F> z, F noisePower, std::span<F> out, SoftAlgorithm algorithm) const {
        if (algorithm == SoftAlgorithm::MaxLog && strategy_ == DecisionStrategy::SquareQam) {
            const std::size_t axisBits = std::min(axisBits_, kMaxAxisBits);
            axisSoftDecisions(z.imag(), noisePower, out.subspan(0UZ, axisBits));
            axisSoftDecisions(z.real(), noisePower, out.subspan(axisBits, axisBits));
            return;
        }
        if (arity_ == 2UZ) {
            const std::complex<F> difference = points_[1] - points_[0];
            const F               bias       = norm2(points_[0]) - norm2(points_[1]);
            out[0UZ]                         = (bias + F{2} * (z.real() * difference.real() + z.imag() * difference.imag())) / noisePower;
            return;
        }
        exhaustive(z, noisePower, out, algorithm);
    }

    void axisSoftDecisions(F x, F noisePower, std::span<F> out) const noexcept {
        const F spacing = axisSpacing_;
        if (axisLevels_ == 2UZ) {
            out[0UZ] = F{2} * spacing * x / noisePower;
            return;
        }
        if (axisLevels_ == 4UZ) {
            const F magnitude = std::abs(x);
            const F high      = magnitude <= spacing ? F{2} * spacing * x : F{4} * spacing * x - F{2} * spacing * spacing * (x < F{0} ? F{-1} : F{1});
            out[0UZ]          = high / noisePower;
            out[1UZ]          = F{2} * spacing * (spacing - magnitude) / noisePower;
            return;
        }
        const std::uint32_t levels = levelCount();
        std::array<F, 16>   distance; // NOLINT(cppcoreguidelines-pro-type-member-init) written before read, `levels` entries
        for (std::uint32_t i = 0U; i < levels; ++i) {
            const F delta = x - axisLevel_[i];
            distance[i]   = delta * delta;
        }
        const std::size_t axisBits = std::min(axisBits_, kMaxAxisBits);
        for (std::size_t k = 0UZ; k < axisBits; ++k) {
            F          zero = std::numeric_limits<F>::infinity();
            F          one  = std::numeric_limits<F>::infinity();
            const auto bit  = static_cast<std::uint32_t>(axisBits - 1UZ - k);
            for (std::uint32_t i = 0U; i < levels; ++i) {
                F& target = ((axisGray_[i] >> bit) & 1U) != 0U ? one : zero;
                target    = std::min(target, distance[i]);
            }
            out[k] = (zero - one) / noisePower;
        }
    }

    void exhaustive(std::complex<F> z, F noisePower, std::span<F> out, SoftAlgorithm algorithm) const {
        const std::uint32_t       count = pointCount();
        std::array<F, kMaxPoints> distance; // NOLINT(cppcoreguidelines-pro-type-member-init) written before read, `count` entries
        F                         smallest = std::numeric_limits<F>::infinity();
        for (std::uint32_t s = 0U; s < count; ++s) {
            const std::complex<F> difference = z - points_[s];
            distance[s]                      = difference.real() * difference.real() + difference.imag() * difference.imag();
            smallest                         = std::min(smallest, distance[s]);
        }

        if (algorithm == SoftAlgorithm::MaxLog) {
            for (std::size_t k = 0UZ; k < bits_; ++k) {
                F zero = std::numeric_limits<F>::infinity();
                F one  = std::numeric_limits<F>::infinity();
                for (std::uint32_t s = 0U; s < count; ++s) {
                    F& target = bitOf(s, k) != 0U ? one : zero;
                    target    = std::min(target, distance[s]);
                }
                out[k] = (zero - one) / noisePower;
            }
            return;
        }

        std::array<F, kMaxPoints> weight; // NOLINT(cppcoreguidelines-pro-type-member-init) written before read
        for (std::uint32_t s = 0U; s < count; ++s) {
            weight[s] = std::exp(-(distance[s] - smallest) / noisePower);
        }
        for (std::size_t k = 0UZ; k < bits_; ++k) {
            F zero = F{0};
            F one  = F{0};
            for (std::uint32_t s = 0U; s < count; ++s) {
                F& target = bitOf(s, k) != 0U ? one : zero;
                target += weight[s];
            }
            out[k] = std::log(one) - std::log(zero);
        }
    }

    [[nodiscard]] std::uint32_t bitOf(std::uint32_t symbol, std::size_t bit) const noexcept { return (symbol >> (bits_ - 1UZ - bit)) & 1U; }

    [[nodiscard]] std::uint32_t levelCount() const noexcept { return static_cast<std::uint32_t>(std::min(axisLevels_, kMaxAxisLevels)); }

    [[nodiscard]] std::uint32_t pointCount() const noexcept { return static_cast<std::uint32_t>(std::min(arity_, kMaxPoints)); }

    [[nodiscard]] static F norm2(std::complex<F> c) noexcept { return c.real() * c.real() + c.imag() * c.imag(); }

    std::vector<std::complex<F>> points_;
    Normalization                normalization_;
    std::size_t                  arity_{};
    std::size_t                  bits_{};
    std::uint8_t                 symbolMask_{};
    F                            normalizationScale_{};
    F                            minimumDistance_{};
    DecisionStrategy             strategy_{DecisionStrategy::Nearest};

    bool                          onCircle_{false};
    F                             phaseCosine_{1};
    F                             phaseSine_{0};
    F                             sectorScale_{0};
    std::uint8_t                  binaryTie_{0};
    std::array<std::uint8_t, 256> sectorSymbol_{};
    std::array<std::uint8_t, 256> symbolSector_{};

    std::size_t                  axisBits_{};
    std::size_t                  axisLevels_{};
    F                            axisScale_{1};
    F                            axisSpacing_{1};
    F                            axisHalfScale_{0};
    F                            axisCenter_{0};
    F                            axisTop_{0};
    std::array<std::uint8_t, 16> axisGray_{};
    std::array<std::uint8_t, 16> axisTieDown_{};
    std::array<F, 16>            axisLevel_{};
};

/**
 * @brief Builds a constellation from the plain values that name one: the family, its size, and its options.
 *
 * The parameters a caller carries as settings, resolved in one place, so that every consumer of a named constellation
 * agrees on what a name means. `arity` is read by `psk` and `qam` only, `phaseOffset` and `labelXor` by `psk` only,
 * and `interleavedPoints` by `custom` only.
 */
template<std::floating_point F>
[[nodiscard]] Constellation<F> constellationFromName(std::string_view name, std::size_t arity, F phaseOffset, std::uint8_t labelXor, std::span<const F> interleavedPoints, std::string_view normalizationName) {
    const Normalization mode = normalizationName == "power" ? Normalization::Power : normalizationName == "amplitude" ? Normalization::Amplitude : normalizationName == "none" ? Normalization::None : throw std::invalid_argument("gr::digital::constellationFromName: normalization must be 'power', 'amplitude' or 'none', got '" + std::string(normalizationName) + "'");

    if (name == "bpsk") {
        return Constellation<F>::bpsk(mode);
    }
    if (name == "qpsk") {
        return Constellation<F>::qpsk(mode);
    }
    if (name == "psk8") {
        return Constellation<F>::psk8(mode);
    }
    if (name == "psk") {
        return Constellation<F>::psk(arity, phaseOffset, labelXor, mode);
    }
    if (name == "qam") {
        return Constellation<F>::qam(arity, mode);
    }
    if (name == "custom") {
        if (interleavedPoints.empty() || interleavedPoints.size() % 2UZ != 0UZ) {
            throw std::invalid_argument("gr::digital::constellationFromName: 'custom' needs points as interleaved re,im pairs, got " + std::to_string(interleavedPoints.size()) + " values");
        }
        std::vector<std::complex<F>> list(interleavedPoints.size() / 2UZ);
        for (std::size_t i = 0UZ; i < list.size(); ++i) {
            list[i] = std::complex<F>(interleavedPoints[2UZ * i], interleavedPoints[2UZ * i + 1UZ]);
        }
        return Constellation<F>::custom(std::span<const std::complex<F>>(list), mode);
    }
    throw std::invalid_argument("gr::digital::constellationFromName: name must be 'bpsk', 'qpsk', 'psk8', 'psk', 'qam' or 'custom', got '" + std::string(name) + "'");
}

} // namespace gr::digital
#endif // GNURADIO_ALGORITHM_CONSTELLATION_HPP
