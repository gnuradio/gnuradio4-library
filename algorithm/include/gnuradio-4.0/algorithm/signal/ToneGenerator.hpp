#ifndef GNURADIO_ALGORITHM_TONE_GENERATOR_HPP
#define GNURADIO_ALGORITHM_TONE_GENERATOR_HPP

#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstdint>
#include <numbers>
#include <span>

#include <gnuradio-4.0/algorithm/signal/Phasor.hpp>

namespace gr::signal {

enum class ToneType : int { Const, Sin, Cos, Square, Saw, Triangle, FastSin, FastCos };

/**
 * @brief Stateful oscillator producing Const/Sin/Cos/Square/Saw/Triangle/FastSin/FastCos waveforms.
 *
 * Sin/Cos call std::sin/std::cos per sample (high precision, no drift).
 * FastSin/FastCos step interleaved recursive phasors instead: one complex multiply a sample, amortized over the
 * lane group, and a re-seed from an exact phase on a fixed cadence to bound the drift.
 * output(t) = A * waveform(2*pi*f*t + phase) + O.
 * A frequency at or below zero, a sample rate that is not finite and positive, and a phase that is not finite
 * are coerced at configure time rather than refused; the generator then holds its constant value.
 *
 * Time is a sample count and never a running float. An accumulator holding elapsed seconds in the value type
 * loses the tick as soon as its ULP approaches it: at F = float and 1 MS/s the 1e-6 s tick already rounds to
 * 9.5367e-7 s at t = 1 s, so the tone reads 953.674 Hz instead of 1 kHz; past 16.777 s the tick rounds to
 * 1.9073e-6 s and the tone doubles; and at t = 32 s the tick is below half an ULP, the accumulator stands
 * still, and every waveform that reads it freezes on DC. A sample count has none of that: it is exact, and
 * the instant is formed from it at the point of use.
 *
 * That instant is a phase reduced to one cycle, and it is reduced in fixed point because there the reduction
 * is free: the cycles per sample are held as Q64 cycles, the product with the sample count is exact, and the
 * whole cycles leave in the wrap of the 64-bit multiply rather than in a subtraction that would first have to
 * form -- and then cancel -- a number the size of the run. Two error terms follow from that, and neither is
 * an accumulation. The step is the requested ratio rounded to a Q64 cycle, so the frequency it realizes is
 * within 2^-65 cycles/sample of f/fs -- the quotient is taken as a two-part double, one fused multiply-add
 * recovering what the division dropped, so the rounding is of the ratio itself and not of its nearest double,
 * which for f/fs above 2^-12 puts the step nearer the request than a double can express it. The phase at
 * sample n is then exact from that step, and only its conversion to double rounds: under 2^-55 cycles, or
 * 1.7e-16 rad, at every index. What is left is the departure from the ideal rational, n * 2^-65 cycles: one
 * ULP at 2*pi after 4096 samples, 5.1e-12 rad over the 30 s at 1 MS/s the qa runs, 1.5e-8 rad over a day of
 * it and 1.5e-3 rad over 285 years of it. The transcendental is evaluated in F, so F = float rounds the
 * argument once more, to 2.4e-7 rad; that is the floor the Phasor's re-seed carries as well, and it likewise
 * does not grow with stream length.
 */
template<std::floating_point F>
struct ToneGenerator {
    /// phasors generated per block in the real path: large enough to amortize the call, small enough to stay hot
    static constexpr std::size_t kBlock = 256UZ;

    ToneType _type      = ToneType::Sin;
    F        _frequency = F(1);
    F        _amplitude = F(1);
    F        _offset    = F(0);
    F        _phase     = F(0);

    /// elapsed time, as the exact count of samples emitted since the last reset()
    std::uint64_t _sampleIndex = 0ULL;

    // The instant is formed from that count in fixed point, so that reducing it to one cycle is the wrap the
    // arithmetic performs anyway rather than a subtraction of two large numbers.
    std::uint64_t _stepQ64          = 0ULL; ///< frac(f/sampleRate) in Q64 cycles, set in configure()
    std::uint64_t _originQ64        = 0ULL; ///< frac(phase/(2*pi)) in Q64 cycles, set in configure()
    double        _radiansPerSample = 0.;   ///< 2*pi*f/sampleRate, the increment the phasor path is seeded with

    // Recursive phasor state for FastSin/FastCos
    Phasor<F> _tone{}; ///< the shared phasor: one recurrence in the tree, with a bounded phase error

    /// Every argument that cannot name a tone is coerced here rather than refused. This setter is `noexcept`
    /// and is reached from a block's settings, and the phasor it seeds refuses an increment that is not finite,
    /// so a sample rate of zero would otherwise cross three `noexcept` frames as a throw. A rate that is not
    /// finite and positive, or a frequency that is not finite, leaves the generator on its constant value the way
    /// a frequency at or below zero already does; a phase that is not finite is taken as zero.
    void configure(ToneType type, F frequency, F sampleRate, F phase, F amplitude, F offset) noexcept {
        const double f     = static_cast<double>(frequency);
        const double fs    = static_cast<double>(sampleRate);
        const double start = static_cast<double>(phase);

        const bool usableRate  = std::isfinite(fs) && fs > 0.;
        const bool usableTone  = usableRate && std::isfinite(f) && f > 0.;
        const bool usablePhase = std::isfinite(start);

        _frequency = frequency;
        _amplitude = amplitude;
        _offset    = offset;
        _phase     = usablePhase ? phase : F(0);
        _type      = usableTone ? type : ToneType::Const;

        // the quotient and the part it dropped: one fused multiply-add recovers f - hi*fs exactly, so hi + lo is
        // the ratio to ~2^-106 and the step is rounded from the ratio itself rather than from its nearest double
        const double hi = usableTone ? f / fs : 0.;
        const double lo = usableTone ? std::fma(-hi, fs, f) / fs : 0.;

        _stepQ64          = toFixedCycles(hi, lo);
        _originQ64        = toFixedCycles(usablePhase ? start / kTwoPi : 0., 0.);
        _radiansPerSample = kTwoPi * hi;
        initPhasor();
    }

    void reset() noexcept {
        _sampleIndex = 0ULL;
        initPhasor();
    }

    [[nodiscard]] constexpr F generateSample() noexcept {
        if (_type == ToneType::FastSin || _type == ToneType::FastCos) {
            const std::complex<F> phasor = stepPhasor();
            return _amplitude * (_type == ToneType::FastSin ? phasor.imag() : phasor.real()) + _offset;
        }
        const F value = computeSample();
        advanceState();
        return value;
    }

    void fill(std::span<F> out) noexcept {
        switch (_type) {
        case ToneType::FastSin: fillPhasor(out, [](F /*pr*/, F pi) { return pi; }); return;
        case ToneType::FastCos: fillPhasor(out, [](F pr, F /*pi*/) { return pr; }); return;
        default:
            for (auto& sample : out) {
                sample = generateSample();
            }
            return;
        }
    }

    [[nodiscard]] constexpr std::complex<F> generateComplexSample() noexcept {
        std::complex<F> result;

        switch (_type) {
        case ToneType::Sin: {
            const F theta = thetaNow();
            result        = std::complex<F>(_amplitude * std::sin(theta) + _offset, -_amplitude * std::cos(theta));
            break;
        }
        case ToneType::Cos: {
            const F theta = thetaNow();
            result        = std::complex<F>(_amplitude * std::cos(theta) + _offset, _amplitude * std::sin(theta));
            break;
        }
        case ToneType::FastSin: {
            // analytic signal of sin: {sin(t), -cos(t)} from phasor = exp(j*t)
            const std::complex<F> phasor = stepPhasor();
            return std::complex<F>(_amplitude * phasor.imag() + _offset, -_amplitude * phasor.real());
        }
        case ToneType::FastCos: {
            // analytic signal of cos: {cos(t), sin(t)} from phasor = exp(j*t)
            const std::complex<F> phasor = stepPhasor();
            return std::complex<F>(_amplitude * phasor.real() + _offset, _amplitude * phasor.imag());
        }
        default: result = std::complex<F>(computeSample(), F(0)); break;
        }
        advanceState();
        return result;
    }

    void fillComplex(std::span<std::complex<F>> out) noexcept {
        switch (_type) {
        case ToneType::FastSin:
            // analytic signal of sin: {sin(θ), -cos(θ)}
            fillPhasorComplex(out, [](F pr, F pi) { return std::complex<F>(pi, -pr); });
            return;
        case ToneType::FastCos:
            // analytic signal of cos: {cos(θ), sin(θ)}
            fillPhasorComplex(out, [](F pr, F pi) { return std::complex<F>(pr, pi); });
            return;
        default:
            for (auto& sample : out) {
                sample = generateComplexSample();
            }
            return;
        }
    }

private:
    static constexpr double kTwoPi = 2. * std::numbers::pi_v<double>;

    template<typename ExtractComponent>
    void fillPhasor(std::span<F> out, ExtractComponent extract) noexcept {
        const F           amp = _amplitude;
        const F           off = _offset;
        const std::size_t n   = out.size();

        // The phasors come from the shared generator, which advances sixteen interleaved lanes and re-seeds from an
        // exact phase on a fixed cadence. Only one component of each is wanted here, so they are taken a block at a
        // time through a buffer small enough to stay in cache rather than one span the width of the call.
        std::array<std::complex<F>, kBlock> block{};
        for (std::size_t at = 0UZ; at < n; at += kBlock) {
            const std::size_t take = std::min(kBlock, n - at);
            _tone.fill(std::span<std::complex<F>>(block.data(), take));
            for (std::size_t i = 0UZ; i < take; ++i) {
                out[at + i] = amp * extract(block[i].real(), block[i].imag()) + off;
            }
        }
        _sampleIndex += n;
    }

    template<typename ExtractComponent>
    void fillPhasorComplex(std::span<std::complex<F>> out, ExtractComponent extract) noexcept {
        const F           amp = _amplitude;
        const F           off = _offset;
        const std::size_t n   = out.size();

        // the output is already complex, so the phasors are generated into it and rewritten in place
        _tone.fill(out);
        for (std::size_t i = 0UZ; i < n; ++i) {
            const std::complex<F> value = extract(out[i].real(), out[i].imag());
            out[i]                      = std::complex<F>(amp * value.real() + off, amp * value.imag());
        }
        _sampleIndex += n;
    }

    /// The phasor of the sample about to be emitted; the generator is left standing past it, so it is the only
    /// state the fast waveforms carry and a scalar call and a filled span advance it identically.
    [[nodiscard]] std::complex<F> stepPhasor() noexcept {
        std::complex<F> phasor{};
        _tone.fill(std::span<std::complex<F>>(&phasor, 1UZ));
        ++_sampleIndex;
        return phasor;
    }

    void initPhasor() noexcept { _tone.configure(_radiansPerSample, static_cast<double>(_phase)); }

    constexpr void advanceState() noexcept { ++_sampleIndex; }

    /// `cycles` and the finer `extra` beside it -- one phase in cycles -- rounded to the nearest Q64 cycle
    [[nodiscard]] static std::uint64_t toFixedCycles(double cycles, double extra) noexcept {
        const double fraction = cycles - std::floor(cycles); // [0, 1); the whole cycles are what fixed point drops
        const double scaled   = fraction * 0x1p64;           // exact: scaling by a power of two moves no bits
        if (!(scaled < 0x1p64)) {                            // a fraction that rounded up to a whole cycle, or not finite
            return 0ULL;
        }
        // floor is exact on a double and so is the subtraction, so what remains below the whole Q64 cycles is
        // carried without loss and rounded together with `extra` rather than each of them rounding on its own
        const double integral = std::floor(scaled);
        const double leftover = (scaled - integral) + extra * 0x1p64;
        return static_cast<std::uint64_t>(integral) + static_cast<std::uint64_t>(std::llround(leftover));
    }

    /// The instant of the sample about to be emitted, as a fraction of a cycle in [-0.5, 0.5).
    ///
    /// The Q64 product is exact and its overflow is the reduction: the whole cycles leave in the wrap of the
    /// 64-bit multiply, so no large number is ever formed and nothing cancels, and the phase is a pure function
    /// of the sample index however the stream was walked to reach it. Reading the result as signed puts the
    /// argument in [-pi, pi), which is where the transcendental wants it. Only the conversion to double rounds,
    /// and it does not grow with elapsed time: it is under 2^-55 cycles, 1.7e-16 rad, at every index.
    [[nodiscard]] constexpr double cyclesNow() const noexcept {
        const std::uint64_t phase = _originQ64 + _stepQ64 * _sampleIndex;
        return static_cast<double>(static_cast<std::int64_t>(phase)) * 0x1p-64;
    }

    /// the reduced phase in radians, rounded to the value type once, at the transcendental
    [[nodiscard]] constexpr F thetaNow() const noexcept { return static_cast<F>(kTwoPi * cyclesNow()); }

    [[nodiscard]] constexpr F computeSample() const noexcept {
        const double cycles = cyclesNow();
        const F      theta  = static_cast<F>(kTwoPi * cycles);
        const F      cycle  = static_cast<F>(cycles);

        switch (_type) {
        case ToneType::Sin: return _amplitude * std::sin(theta) + _offset;
        case ToneType::Cos: return _amplitude * std::cos(theta) + _offset;
        case ToneType::FastSin: // handled by generateSample, which owns the recurrence
        case ToneType::FastCos: return _amplitude * std::sin(theta) + _offset;
        case ToneType::Const: return _amplitude + _offset;
        case ToneType::Square:
            // the reduced phase is signed, so the half-cycle test is its sign: [0, 0.5) is the first half of the
            // cycle and [-0.5, 0) the second. It is taken before the value type rounds it, so a phase a hair under
            // the half cycle stays in the first half rather than rounding into the second.
            return (cycles < 0.) ? -_amplitude + _offset : _amplitude + _offset;
        case ToneType::Saw: return _amplitude * (F(2) * (cycle - std::floor(cycle + F(0.5)))) + _offset;
        case ToneType::Triangle: {
            return _amplitude * (F(4) * std::abs(cycle - std::floor(cycle + F(0.75)) + F(0.25)) - F(1)) + _offset;
        }
        }
        return F(0);
    }
};

} // namespace gr::signal

#endif // GNURADIO_ALGORITHM_TONE_GENERATOR_HPP
