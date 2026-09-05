#ifndef GNURADIO_ALGORITHM_PHASOR_HPP
#define GNURADIO_ALGORITHM_PHASOR_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <numbers>
#include <span>
#include <stdexcept>

namespace gr::signal {

/**
 * @brief Stateful unit phasor: generates, or mixes into a stream, exp(j*phase) at a fixed or a per-sample
 * phase increment.
 *
 * Fixed-increment generation (fill/mix) runs kLanes interleaved phasors, each advanced by
 * (e^{j*increment})^kLanes, so the sample loop carries no dependency between neighbors and vectorizes.
 * Every kReseedInterval samples the lane vector is re-seeded from an exact phase, which bounds the
 * recurrence's error by the re-seed rounding instead of letting it walk: the phase error at any output sample
 * does not grow with stream length, which is the kernel's accuracy contract.
 *
 * The lane vector and the position within the re-seed interval are state carried across calls, so a stream
 * split into calls anywhere runs the identical sequence of operations and reproduces bit for bit. Exactly one
 * site seeds the lanes and one advances them, because two copies of the same arithmetic may compile
 * differently and contraction applied at one site alone would break that guarantee. configure(), setPhase(),
 * setIncrement() and advance() re-anchor the grid at the current phase, starting a new realization there.
 *
 * Modulated generation (fillModulated/mixModulated) takes one increment per sample and cannot use the lane
 * recurrence, so it accumulates phase scalar-wise in double. Its phase sequence is the prefix sum of the
 * increments reduced to (-pi, pi] after each step, which likewise makes each phase a pure function of the
 * previous one and so bit-identical however the stream is split. A modulated call re-anchors the grid at its
 * end phase.
 *
 * All phase state is double regardless of F: a value_type accumulator rounds the re-seed phase once per call,
 * which for F = float is ~2.4e-7 rad and dominates every other error over a stream.
 *
 * Units are radians and radians/sample throughout. Hz belongs to blocks, which know a sample rate.
 */
template<std::floating_point F>
struct Phasor {
    using value_type = F;
    using Complex    = std::complex<F>;

    /// interleaved generation lanes: the sample loop advances kLanes phasors that depend on none of their
    /// neighbors, which is what lets it vectorize
    static constexpr std::size_t kLanes = 16UZ;

    /// Re-seed cadence, a whole number of lane groups. It sets the longest chain of rounded multiplications
    /// between two exact seeds, which is the dominant error term in float, so a short cadence is what holds
    /// the phase error near its floor; the seed costs one sin/cos per cadence, which at 256 stays below a
    /// hundredth of the per-sample work.
    static constexpr std::size_t kReseedInterval = 256UZ;
    static_assert(kReseedInterval % kLanes == 0UZ, "a lane group must never straddle a re-seed boundary");

    double      _seedPhase{0.}; ///< exact phase, reduced to (-pi, pi], at the start of the current interval
    double      _increment{0.}; ///< radians/sample
    std::size_t _position{0UZ}; ///< samples since that interval start, in [0, kReseedInterval)

    /// The generation lanes, carried across calls so a split stream runs the identical sequence of
    /// operations. Two properties of this storage are load-bearing for throughput, each worth a factor of
    /// two or more. The lanes are interleaved complex, so a fill is a contiguous block copy rather than a
    /// per-sample interleave of two arrays. And generation works on a local copy: a member array shares its
    /// element type with the output span, so a store to the output must be assumed to alias the lanes and
    /// force a reload every sample, while a local whose address never escapes cannot.
    struct Lanes {
        std::array<Complex, kLanes> v{};
    };
    Lanes _lanes{};
    bool  _lanesValid{false};

    /// The chirp recurrence: the phasor, the rotator that advances it, and whether the pair is live. A chirp's
    /// increment is not constant, so the lane machinery above does not serve it; these are carried across calls
    /// for the same reason the lanes are, and are seeded only at offset 0 of a re-seed interval, so the sequence
    /// of operations depends on the stream position alone.
    std::complex<double> _chirpPhasor{1., 0.};
    std::complex<double> _chirpRotator{1., 0.};
    bool                 _chirpValid{false};

    [[nodiscard]] static double wrapPhase(double phase) noexcept { return std::remainder(phase, 2. * std::numbers::pi_v<double>); }

    void configure(double increment, double initialPhase) {
        if (!std::isfinite(increment) || !std::isfinite(initialPhase)) {
            throw std::invalid_argument("gr::signal::Phasor: increment and initial phase must both be finite");
        }
        _increment = increment;
        anchor(initialPhase);
    }

    /// The increment changes, the phase runs on: a frequency change advances the phase from where it stands
    /// rather than restarting it. The grid re-anchors on the current phase, so the new increment takes effect
    /// from the next sample.
    void setIncrement(double increment) {
        if (!std::isfinite(increment)) {
            throw std::invalid_argument("gr::signal::Phasor: increment must be finite");
        }
        const double current = phase();
        _increment           = increment;
        anchor(current);
    }

    void setPhase(double newPhase) {
        if (!std::isfinite(newPhase)) {
            throw std::invalid_argument("gr::signal::Phasor: phase must be finite");
        }
        anchor(newPhase);
    }

    [[nodiscard]] double phase() const noexcept { return wrapPhase(_seedPhase + static_cast<double>(_position) * _increment); }
    [[nodiscard]] double increment() const noexcept { return _increment; }

    /// Moves the phase by nSamples increments and re-anchors the generation grid there.
    void advance(std::ptrdiff_t nSamples) noexcept { anchor(phase() + static_cast<double>(nSamples) * _increment); }

    /// out[k] = exp(j*(phase + k*increment)); leaves the phase advanced by out.size() increments
    void fill(std::span<Complex> out) noexcept { generate<false>({}, out); }

    /// out[k] = in[k] * exp(j*(phase + k*increment)); leaves the phase advanced by the sample count
    void mix(std::span<const Complex> in, std::span<Complex> out) noexcept { generate<true>(in, out.first(std::min(in.size(), out.size()))); }

    /// out[k] = exp(j*phase_k) with phase_{k+1} = phase_k + increments[k] — the CPM/FM entry point
    template<std::floating_point FIncrement>
    void fillModulated(std::span<const FIncrement> increments, std::span<Complex> out) noexcept {
        modulate<false>(increments, {}, out);
    }

    /// the same, multiplied into a stream — the time-varying-CFO entry point
    template<std::floating_point FIncrement>
    void mixModulated(std::span<const FIncrement> increments, std::span<const Complex> in, std::span<Complex> out) noexcept {
        modulate<true>(increments, in, out);
    }

    /// @brief Fills `out` with a linear chirp: the increment starts at the current one and grows by
    /// `incrementStep` radians per sample, per sample.
    ///
    /// The schedule a linear frequency sweep produces is arithmetic, so the phasor needs no transcendental per
    /// sample: it is multiplied by a rotator which is itself multiplied by a constant. Set the starting increment
    /// with `setIncrement` before the first call; a chirp and a plain fill are two state machines over the same
    /// phase, so a given Phasor runs one or the other, not both at once.
    void fillChirp(double incrementStep, std::span<Complex> out) noexcept { chirp<false>(incrementStep, {}, out); }

    /// @brief Multiplies `in` by that chirp into `out`, over `min(in.size(), out.size())` samples.
    void mixChirp(double incrementStep, std::span<const Complex> in, std::span<Complex> out) noexcept { chirp<true>(incrementStep, in, out.first(std::min(in.size(), out.size()))); }

private:
    [[nodiscard]] double intervalPhase() const noexcept { return static_cast<double>(kReseedInterval) * _increment; }

    void anchor(double newPhase) noexcept {
        _seedPhase  = wrapPhase(newPhase);
        _position   = 0UZ;
        _lanesValid = false;
        _chirpValid = false;
    }

    /// the one place lanes are built, always at offset 0 of an interval
    void seedLanes(Lanes& lanes) const noexcept {
        const std::complex<double> seed   = std::polar(1., _seedPhase);
        const std::complex<double> step   = std::polar(1., _increment);
        const F                    stepRe = static_cast<F>(step.real());
        const F                    stepIm = static_cast<F>(step.imag());

        lanes.v[0UZ] = Complex(static_cast<F>(seed.real()), static_cast<F>(seed.imag()));
        for (std::size_t w = 1UZ; w < kLanes; ++w) {
            const F phRe = lanes.v[w - 1UZ].real();
            const F phIm = lanes.v[w - 1UZ].imag();
            lanes.v[w]   = Complex(phRe * stepRe - phIm * stepIm, phRe * stepIm + phIm * stepRe);
        }
    }

    /// the one place lanes are advanced, by one whole group
    void advanceLanes(Lanes& lanes) const noexcept {
        const std::complex<double> laneStep = std::polar(1., static_cast<double>(kLanes) * _increment);
        const F                    stepRe   = static_cast<F>(laneStep.real());
        const F                    stepIm   = static_cast<F>(laneStep.imag());
        for (std::size_t w = 0UZ; w < kLanes; ++w) {
            const F phRe = lanes.v[w].real();
            const F phIm = lanes.v[w].imag();
            lanes.v[w]   = Complex(phRe * stepRe - phIm * stepIm, phRe * stepIm + phIm * stepRe);
        }
    }

    /// crosses a group boundary: a whole interval re-seeds exactly, anything else steps the lanes
    void crossGroup(Lanes& lanes) noexcept {
        if (_position == kReseedInterval) {
            _seedPhase = wrapPhase(_seedPhase + intervalPhase());
            _position  = 0UZ;
            seedLanes(lanes);
        } else {
            advanceLanes(lanes);
        }
    }

    /// Seeds the chirp pair at offset 0 of the current interval, where the exact phase is `_seedPhase` and the
    /// exact increment is `_increment`.
    void seedChirp() noexcept {
        _chirpPhasor  = std::polar(1., _seedPhase);
        _chirpRotator = std::polar(1., _increment);
        _chirpValid   = true;
    }

    /// Rolls the interval forward by one whole cadence, exactly: the phase advances by the sum of the schedule
    /// over the interval, and the increment by the interval's worth of steps.
    void rollChirpInterval(double incrementStep) noexcept {
        constexpr double m = static_cast<double>(kReseedInterval);
        _seedPhase         = wrapPhase(_seedPhase + m * _increment + incrementStep * m * (m - 1.) * 0.5);
        _increment += m * incrementStep;
        _position = 0UZ;
        seedChirp();
    }

    template<bool kMix>
    void chirp(double incrementStep, std::span<const Complex> in, std::span<Complex> out) noexcept {
        const std::size_t nSamples = out.size();
        if (nSamples == 0UZ) {
            return;
        }
        if (!_chirpValid) {
            seedChirp();
        }
        const std::complex<double> stepRotator = std::polar(1., incrementStep);

        for (std::size_t k = 0UZ; k < nSamples; ++k) {
            if (_position == kReseedInterval) {
                rollChirpInterval(incrementStep);
            }
            const Complex phasor(static_cast<F>(_chirpPhasor.real()), static_cast<F>(_chirpPhasor.imag()));
            if constexpr (kMix) {
                out[k] = in[k] * phasor;
            } else {
                out[k] = phasor;
            }
            _chirpPhasor *= _chirpRotator;
            _chirpRotator *= stepRotator;
            ++_position;
        }
    }

    template<bool kMix>
    void generate(std::span<const Complex> in, std::span<Complex> out) noexcept {
        const std::size_t nSamples = out.size();
        if (nSamples == 0UZ) {
            return;
        }

        const auto emit = [&](const Lanes& source, std::size_t index, std::size_t lane) noexcept {
            if constexpr (kMix) {
                const F re   = in[index].real();
                const F im   = in[index].imag();
                const F phRe = source.v[lane].real();
                const F phIm = source.v[lane].imag();
                out[index]   = Complex(re * phRe - im * phIm, re * phIm + im * phRe);
            } else {
                out[index] = source.v[lane];
            }
        };

        // A run that ends inside the current lane group is emitted from the lane vector in place. The local copy
        // below is what lets a long fill vectorize, and it is kLanes complex loads and stores whether the call
        // asks for one sample or a thousand -- which for a caller taking one sample at a time is the whole of the
        // per-sample work. Nothing else differs: the lanes are read here and seeded and advanced nowhere but in
        // the two functions that own that arithmetic.
        const std::size_t start = _position % kLanes;
        if (_lanesValid && nSamples < kLanes - start) {
            for (std::size_t j = 0UZ; j < nSamples; ++j) {
                emit(_lanes, j, start + j);
            }
            _position += nSamples;
            return;
        }

        Lanes lanes = _lanes;
        if (!_lanesValid) {
            seedLanes(lanes);
            _lanesValid = true;
        }

        std::size_t produced = 0UZ;
        while (produced < nSamples) {
            const std::size_t lane0 = _position % kLanes;
            if (lane0 == 0UZ && nSamples - produced >= kLanes) {
                // a whole group: no dependency between the kLanes iterations, so this is the loop that
                // vectorizes, and it is the common case
                for (std::size_t w = 0UZ; w < kLanes; ++w) {
                    emit(lanes, produced + w, w);
                }
                produced += kLanes;
                _position += kLanes;
                crossGroup(lanes);
            } else {
                const std::size_t take = std::min(nSamples - produced, kLanes - lane0);
                for (std::size_t j = 0UZ; j < take; ++j) {
                    emit(lanes, produced + j, lane0 + j);
                }
                produced += take;
                _position += take;
                if (_position % kLanes == 0UZ) {
                    crossGroup(lanes);
                }
                // a group left unfinished keeps its lanes; the next call resumes at lane0
            }
        }

        _lanes = lanes;
    }

    template<bool kMix, std::floating_point FIncrement>
    void modulate(std::span<const FIncrement> increments, std::span<const Complex> in, std::span<Complex> out) noexcept {
        std::size_t nSamples = std::min(increments.size(), out.size());
        if constexpr (kMix) {
            nSamples = std::min(nSamples, in.size());
        }

        constexpr double twoPi = 2. * std::numbers::pi_v<double>;
        double           value = phase();
        for (std::size_t k = 0UZ; k < nSamples; ++k) {
            const F phRe = static_cast<F>(std::cos(value));
            const F phIm = static_cast<F>(std::sin(value));
            if constexpr (kMix) {
                const F re = in[k].real();
                const F im = in[k].imag();
                out[k]     = Complex(re * phRe - im * phIm, re * phIm + im * phRe);
            } else {
                out[k] = Complex(phRe, phIm);
            }
            value += static_cast<double>(increments[k]);
            // reduce every step rather than once per call, so the phase sequence depends only on the previous
            // phase and the increment and stays bit-identical however the stream is split
            if (value > std::numbers::pi_v<double> || value < -std::numbers::pi_v<double>) {
                value = std::remainder(value, twoPi);
            }
        }
        anchor(value);
    }
};

} // namespace gr::signal

#endif // GNURADIO_ALGORITHM_PHASOR_HPP
