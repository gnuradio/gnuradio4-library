#ifndef GNURADIO_ALGORITHM_FEC_CONVOLUTIONAL_HPP
#define GNURADIO_ALGORITHM_FEC_CONVOLUTIONAL_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

/**
 * Non-recursive, non-systematic convolutional codes of rate 1/n, with a terminated
 * maximum-likelihood Viterbi decoder.
 *
 * A code is a constraint length K, three through nine, and n generator polynomials, two through
 * four of them, each K bits wide and conventionally spelled in octal. Polynomial bit i, counting
 * from the least significant, taps the input delayed by i steps: bit 0 is the current input bit
 * and bit K-1 the oldest the shift register holds. One input step emits one bit per polynomial,
 * in the order the polynomials are given.
 *
 * That convention has an anchor which needs no implementation to check against. Drive the encoder
 * with a single one followed by zeros and step i emits bit i of each polynomial, so the impulse
 * response read least significant bit first is the polynomials themselves. This is what a
 * generator polynomial means, and it is the property against which a code's spelling can be
 * checked against any published statement of a standard code.
 *
 * Frames are terminated. The encoder appends K-1 zero bits after the information, so k
 * information bits become (k + K - 1) * n coded bits and the trellis both starts and ends in the
 * zero state. One record is one such frame, which is what makes the decode exact: the decoder
 * holds the whole trellis and tracks back from the state termination guarantees, so its answer is
 * the maximum-likelihood path over the frame rather than the survivor of a truncated window.
 *
 * The decoder takes a hard-decision word of bit items or a soft-decision word of one float per
 * coded bit, the sign carrying the bit and the magnitude the confidence. Its branch metric is the
 * correlation sum, so any consistent scaling of the inputs scales the metric and leaves every
 * decision where it was; a demodulator hands over whatever amplitude it has without a
 * normalization step or a quantization table.
 *
 * A Viterbi decoder always answers. The trellis has a best path through any received word, so
 * there is no validity signal to report and nothing is refused. What the result carries instead
 * is the distance between the received word and the winning path re-encoded, which is the
 * decoder's own account, in bits, of what the channel did.
 */
namespace gr::fec {

/// Constraint lengths the code family accepts. Below three there is no code; above nine the
/// trellis grows past what the exhaustive checks beside this header can afford to sweep.
inline constexpr std::size_t kConvMinConstraintLength = 3UZ;
inline constexpr std::size_t kConvMaxConstraintLength = 9UZ;

/// Generator counts the code family accepts, which are the rates 1/2 through 1/4.
inline constexpr std::size_t kConvMinPolynomials = 2UZ;
inline constexpr std::size_t kConvMaxPolynomials = 4UZ;

/**
 * A code: its constraint length and its generator polynomials.
 *
 * Build one through `configure`, which is where the family's limits are enforced. A code whose
 * `configure` refused is left unconfigured rather than half built, so a caller that ignores the
 * answer encodes nothing rather than encoding under a code nobody named.
 */
struct ConvolutionalCode {
    std::size_t                                    constraintLength = 0UZ; ///< K, the register's width in input bits
    std::size_t                                    polynomialCount  = 0UZ; ///< n, the coded bits one input step emits
    std::array<std::uint32_t, kConvMaxPolynomials> polynomials{};          ///< the generators, in output order

    /// Accept @p length and @p generators as a code, or refuse them and leave this unconfigured.
    ///
    /// A generator wider than K bits taps a delay the register does not hold, a zero generator
    /// emits a constant output that carries nothing, and two equal generators emit the same bit
    /// twice for a rate the code does not have. None of the three is a code, so each is refused
    /// here rather than left to produce an answer that looks like one.
    [[nodiscard]] constexpr bool configure(std::size_t length, std::span<const std::uint32_t> generators) noexcept {
        constraintLength = 0UZ;
        polynomialCount  = 0UZ;
        polynomials      = {};

        if (length < kConvMinConstraintLength || length > kConvMaxConstraintLength) {
            return false;
        }
        if (generators.size() < kConvMinPolynomials || generators.size() > kConvMaxPolynomials) {
            return false;
        }
        const std::uint32_t past = std::uint32_t{1U} << length;
        for (std::size_t i = 0UZ; i < generators.size(); ++i) {
            if (generators[i] == 0U || generators[i] >= past) {
                return false;
            }
            for (std::size_t j = 0UZ; j < i; ++j) {
                if (generators[i] == generators[j]) {
                    return false;
                }
            }
        }

        constraintLength = length;
        polynomialCount  = generators.size();
        std::copy(generators.begin(), generators.end(), polynomials.begin());
        return true;
    }

    [[nodiscard]] constexpr bool configured() const noexcept { return constraintLength != 0UZ; }

    /// Trellis states, one per pattern of the K-1 input bits the register remembers.
    [[nodiscard]] constexpr std::size_t states() const noexcept { return configured() ? (1UZ << (constraintLength - 1UZ)) : 0UZ; }
};

/// The coded bits one step emits from @p state and @p input, output j in bit j.
///
/// The state holds the K-1 previous input bits, its bit j the input delayed j+1 steps, so the
/// window the polynomials tap is the current bit with the state above it.
[[nodiscard]] inline constexpr std::uint32_t convolutionalBranch(const ConvolutionalCode& code, std::uint32_t state, std::uint32_t input) noexcept {
    const std::uint32_t window = (state << 1U) | (input & 1U);

    std::uint32_t coded = 0U;
    for (std::size_t j = 0UZ; j < code.polynomialCount; ++j) {
        coded |= static_cast<std::uint32_t>(std::popcount(code.polynomials[j] & window) & 1) << j;
    }
    return coded;
}

/// The state @p input leaves the register in, coming from @p state.
[[nodiscard]] inline constexpr std::uint32_t convolutionalNextState(const ConvolutionalCode& code, std::uint32_t state, std::uint32_t input) noexcept {
    const std::uint32_t mask = static_cast<std::uint32_t>(code.states() - 1UZ);
    return ((state << 1U) | (input & 1U)) & mask;
}

/// Coded bits a terminated frame of @p infoBits information bits occupies.
[[nodiscard]] inline constexpr std::size_t convolutionalEncodedBits(const ConvolutionalCode& code, std::size_t infoBits) noexcept { return code.configured() ? (infoBits + code.constraintLength - 1UZ) * code.polynomialCount : 0UZ; }

/**
 * @brief How a decoded record's ends are treated.
 *
 * `Terminated` is a frame: the encoder appended `K-1` zero tail bits, the trellis starts and ends
 * in the zero state, and the tail is removed from the information. `Open` is a record cut out of a
 * continuous convolutional stream: no state is known at either end unless a caller supplies the
 * initial one, the traceback starts from the best-metric final state, and every step carries an
 * information bit. The last `K-1` information bits of an open decode are decided without the
 * future that would resolve them, so a caller that needs them reliable extracts `K-1` steps past
 * its payload and discards them.
 */
enum class ConvTermination : std::uint8_t { Terminated, Open };

/// Information bits a frame of @p codedBits coded bits carries under @p termination, or zero when
/// that length is not a frame. Terminated: a nonzero multiple of n holding the K-1 tail steps plus
/// at least one information step. Open: any nonzero multiple of n, every step an information bit.
[[nodiscard]] inline constexpr std::size_t convolutionalInfoBits(const ConvolutionalCode& code, std::size_t codedBits, ConvTermination termination) noexcept {
    if (!code.configured() || codedBits == 0UZ || codedBits % code.polynomialCount != 0UZ) {
        return 0UZ;
    }
    const std::size_t steps = codedBits / code.polynomialCount;
    if (termination == ConvTermination::Open) {
        return steps;
    }
    return steps < code.constraintLength ? 0UZ : steps - (code.constraintLength - 1UZ);
}

/// Information bits a terminated frame of @p codedBits coded bits carries, or zero when that
/// length is not a frame: it must be a nonzero multiple of n and hold at least the K-1 tail steps
/// plus one information step.
[[nodiscard]] inline constexpr std::size_t convolutionalInfoBits(const ConvolutionalCode& code, std::size_t codedBits) noexcept { return convolutionalInfoBits(code, codedBits, ConvTermination::Terminated); }

/// Encode @p info as one terminated frame into @p out, one bit per item with the low bit
/// significant, and answer the coded bits written.
///
/// The K-1 zero bits that return the register to the zero state are appended here rather than
/// left to the caller, because they are what makes the frame a frame and the decode exact.
/// Nothing is written and zero is answered when the code is unconfigured or @p out is too short
/// for the frame `convolutionalEncodedBits` names.
inline std::size_t convolutionalEncode(const ConvolutionalCode& code, std::span<const std::uint8_t> info, std::span<std::uint8_t> out) noexcept {
    const std::size_t coded = convolutionalEncodedBits(code, info.size());
    if (coded == 0UZ || out.size() < coded) {
        return 0UZ;
    }

    const std::size_t steps = info.size() + code.constraintLength - 1UZ;
    std::uint32_t     state = 0U;
    for (std::size_t t = 0UZ; t < steps; ++t) {
        const std::uint32_t input = (t < info.size()) ? (info[t] & 1U) : 0U;
        const std::uint32_t word  = convolutionalBranch(code, state, input);
        for (std::size_t j = 0UZ; j < code.polynomialCount; ++j) {
            out[t * code.polynomialCount + j] = static_cast<std::uint8_t>((word >> j) & 1U);
        }
        state = convolutionalNextState(code, state, input);
    }
    return coded;
}

/// What a decode reports. There is no validity flag, because a Viterbi decode has no refusal.
struct ViterbiResult {
    std::size_t distance = 0UZ;  ///< bits between the received word, sign-sliced when soft, and the winning path re-encoded
    float       metric   = 0.0F; ///< the winning path's accumulated branch metric: a Hamming distance when hard, a correlation when soft
};

/**
 * A whole-frame maximum-likelihood decoder for one code.
 *
 * The decoder owns its working storage — the branch table, the two path-metric arrays and the
 * survivor decisions of every step — and grows it to the longest frame it has been asked for, so
 * a chain that settles on a frame length stops allocating after its first record.
 *
 * The trellis is walked forward over every step of the frame, each state keeping the better of
 * its two incoming branches. Under `Terminated` the traceback starts from the zero state because
 * termination guarantees the transmitted path ended there; under `Open` it starts from the
 * best-metric final state, the ends being unknown for a record cut out of a continuous stream.
 * No survivor window is applied and no truncation depth is chosen: the whole frame is in hand,
 * so the path traced back is the maximum-likelihood path over it rather than an approximation
 * of one. An open decode's trellis converges after roughly `5K` input steps from no start
 * knowledge, and its last `K-1` information bits lack the future that would resolve them — the
 * two ends a caller sizes its record around.
 *
 * Both entry points want the information span the frame carries, exactly the length
 * `convolutionalInfoBits` names for the received length. A pair of spans that do not describe a
 * frame of this code is answered with a zero result and nothing written, which is the same
 * refusal `convolutionalEncode` makes and for the same reason.
 */
class ViterbiDecoder {
public:
    ViterbiDecoder() = default;
    explicit ViterbiDecoder(const ConvolutionalCode& code) { configure(code); }

    /// Take @p code, sizing the branch table and the metrics for it. An unconfigured code, or an
    /// `initialState` naming a state the code does not have, leaves the decoder unconfigured and
    /// every decode then answers a zero result. `initialState` is meaningful under `Open` only:
    /// a terminated frame's initial state is the zero state by construction.
    bool configure(const ConvolutionalCode& code, ConvTermination termination = ConvTermination::Terminated, std::optional<std::uint32_t> initialState = std::nullopt) {
        _code         = {};
        _states       = 0UZ;
        _half         = 0UZ;
        _termination  = ConvTermination::Terminated;
        _initialState = std::nullopt;
        _traceback.clear(); // sized by the state count, so it cannot outlive a code change
        _words.clear();
        if (!code.configured()) {
            return false;
        }
        if (initialState.has_value() && *initialState >= code.states()) {
            return false;
        }

        _code         = code;
        _states       = code.states();
        _half         = _states >> 1U;
        _termination  = termination;
        _initialState = termination == ConvTermination::Open ? initialState : std::nullopt;

        _outputs.assign(2UZ * _states, 0U);
        for (std::size_t state = 0UZ; state < _states; ++state) {
            for (std::size_t input = 0UZ; input < 2UZ; ++input) {
                _outputs[(state << 1U) | input] = convolutionalBranch(_code, static_cast<std::uint32_t>(state), static_cast<std::uint32_t>(input));
            }
        }
        _metric.assign(_states, 0.0F);
        _next.assign(_states, 0.0F);
        return true;
    }

    [[nodiscard]] const ConvolutionalCode& code() const noexcept { return _code; }
    [[nodiscard]] ConvTermination          termination() const noexcept { return _termination; }

    /// Decode a received word of bit items, the low bit of each item significant.
    [[nodiscard]] ViterbiResult decodeHard(std::span<const std::uint8_t> received, std::span<std::uint8_t> info) {
        const std::size_t steps = prepare(received.size(), info.size());
        if (steps == 0UZ) {
            return {};
        }

        const std::size_t n = _code.polynomialCount;
        for (std::size_t t = 0UZ; t < steps; ++t) {
            std::uint32_t word = 0U;
            for (std::size_t j = 0UZ; j < n; ++j) {
                word |= (received[t * n + j] & 1U) << j;
            }
            _words[t] = word;
        }

        // The branch metric is the Hamming distance between the branch's output and the received
        // step, so the winning path metric is the distance of the whole word from that path.
        const float metric = runTrellis(steps, [this, n](std::size_t t, std::span<float> cost) noexcept {
            const std::uint32_t word = _words[t];
            for (std::size_t pattern = 0UZ; pattern < (1UZ << n); ++pattern) {
                cost[pattern] = static_cast<float>(std::popcount(static_cast<std::uint32_t>(pattern) ^ word));
            }
        });

        ViterbiResult result;
        result.distance = traceback(steps, info);
        result.metric   = metric;
        return result;
    }

    /// Decode a received word of one float per coded bit, positive carrying a one and the
    /// magnitude the confidence, zero a pure erasure.
    [[nodiscard]] ViterbiResult decodeSoft(std::span<const float> received, std::span<std::uint8_t> info) {
        const std::size_t steps = prepare(received.size(), info.size());
        if (steps == 0UZ) {
            return {};
        }

        const std::size_t n = _code.polynomialCount;
        for (std::size_t t = 0UZ; t < steps; ++t) {
            std::uint32_t word = 0U;
            for (std::size_t j = 0UZ; j < n; ++j) {
                if (received[t * n + j] > 0.0F) {
                    word |= std::uint32_t{1U} << j;
                }
            }
            _words[t] = word; // the sign-sliced word, which is what the reported distance is against
        }

        // The branch metric is the correlation between the branch's output and the received step,
        // which the trellis minimizes as its negative so that one traversal serves both decoders.
        const float metric = runTrellis(steps, [received, n](std::size_t t, std::span<float> cost) noexcept {
            for (std::size_t pattern = 0UZ; pattern < (1UZ << n); ++pattern) {
                float sum = 0.0F;
                for (std::size_t j = 0UZ; j < n; ++j) {
                    const float value = received[t * n + j];
                    sum += (((pattern >> j) & 1UZ) != 0UZ) ? value : -value;
                }
                cost[pattern] = -sum;
            }
        });

        ViterbiResult result;
        result.distance = traceback(steps, info);
        result.metric   = -metric; // reported the way the convention states it, as a correlation
        return result;
    }

private:
    /// A metric no reachable path can hold, standing for the states the first K-1 steps cannot
    /// have entered. Adding branch metrics to it leaves it far above any of them.
    static constexpr float kUnreachable = 1.0e30F;

    /// Check the spans against the code, size the working storage for the frame, and answer the
    /// frame's trellis steps, or zero when the spans are not a frame of this code.
    [[nodiscard]] std::size_t prepare(std::size_t codedBits, std::size_t infoBits) {
        const std::size_t carried = convolutionalInfoBits(_code, codedBits, _termination);
        if (carried == 0UZ || carried != infoBits) {
            return 0UZ;
        }

        const std::size_t steps = codedBits / _code.polynomialCount;
        if (_words.size() < steps) {
            _words.resize(steps);
        }
        if (_traceback.size() < steps * _states) {
            _traceback.resize(steps * _states);
        }
        return steps;
    }

    /// Walk the trellis forward, keeping one survivor per state per step, and answer the winning
    /// final state's metric: the zero state's when terminated, the best state's when open.
    /// @p stepCosts fills a branch metric per output pattern of a step.
    template<typename FStepCosts>
    [[nodiscard]] float runTrellis(std::size_t steps, FStepCosts&& stepCosts) {
        std::array<float, (1UZ << kConvMaxPolynomials)> cost{};

        if (_termination == ConvTermination::Terminated) {
            std::fill(_metric.begin(), _metric.end(), kUnreachable);
            _metric[0UZ] = 0.0F; // the frame starts in the zero state, which is what termination buys
        } else if (_initialState.has_value()) {
            std::fill(_metric.begin(), _metric.end(), kUnreachable);
            _metric[*_initialState] = 0.0F; // the caller knows where the stream's path stood
        } else {
            std::fill(_metric.begin(), _metric.end(), 0.0F); // no start knowledge: every state equally likely
        }

        for (std::size_t t = 0UZ; t < steps; ++t) {
            stepCosts(t, std::span(cost));

            const std::size_t base = t * _states;
            for (std::size_t state = 0UZ; state < _states; ++state) {
                // A state names the input that entered it in its low bit and all but the oldest
                // bit of its predecessor above, so its two predecessors differ only in that bit.
                const std::size_t input = state & 1UZ;
                const std::size_t lower = state >> 1U;
                const std::size_t upper = lower | _half;

                const float from0 = _metric[lower] + cost[_outputs[(lower << 1U) | input]];
                const float from1 = _metric[upper] + cost[_outputs[(upper << 1U) | input]];

                const bool survivor      = from1 < from0;
                _next[state]             = survivor ? from1 : from0;
                _traceback[base + state] = survivor ? std::uint8_t{1U} : std::uint8_t{0U};
            }
            _metric.swap(_next);
        }
        return _metric[winningState()];
    }

    /// Where the traceback starts: the zero state when terminated, the best-metric state when open.
    [[nodiscard]] std::size_t winningState() const noexcept {
        if (_termination == ConvTermination::Terminated) {
            return 0UZ;
        }
        return static_cast<std::size_t>(std::min_element(_metric.begin(), _metric.end()) - _metric.begin());
    }

    /// Walk the survivors back from the winning final state, writing the information bits and
    /// accumulating the distance between the received word and the path being read out.
    [[nodiscard]] std::size_t traceback(std::size_t steps, std::span<std::uint8_t> info) const noexcept {
        std::size_t distance = 0UZ;
        std::size_t state    = winningState();

        for (std::size_t t = steps; t-- > 0UZ;) {
            const std::size_t   input    = state & 1UZ;
            const std::size_t   previous = (state >> 1U) | ((_traceback[t * _states + state] != 0U) ? _half : 0UZ);
            const std::uint32_t emitted  = _outputs[(previous << 1U) | input];

            distance += static_cast<std::size_t>(std::popcount(emitted ^ _words[t]));
            if (t < info.size()) { // the last K-1 steps are the tail, which carries no information
                info[t] = static_cast<std::uint8_t>(input);
            }
            state = previous;
        }
        return distance;
    }

    ConvolutionalCode            _code{};
    std::size_t                  _states       = 0UZ;
    std::size_t                  _half         = 0UZ; ///< the bit a state's two predecessors differ in
    ConvTermination              _termination  = ConvTermination::Terminated;
    std::optional<std::uint32_t> _initialState = std::nullopt; ///< Open only: where the stream's path is known to stand

    std::vector<std::uint32_t> _outputs;   ///< branch outputs, indexed by state and input
    std::vector<float>         _metric;    ///< path metric per state, this step
    std::vector<float>         _next;      ///< path metric per state, the step being built
    std::vector<std::uint8_t>  _traceback; ///< the survivor's choice of predecessor, per step and state
    std::vector<std::uint32_t> _words;     ///< the received word per step, hard or sign-sliced
};

} // namespace gr::fec

#endif // GNURADIO_ALGORITHM_FEC_CONVOLUTIONAL_HPP
