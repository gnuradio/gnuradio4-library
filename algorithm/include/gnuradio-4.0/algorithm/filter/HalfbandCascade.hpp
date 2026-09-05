#ifndef GNURADIO_HALFBAND_CASCADE_HPP
#define GNURADIO_HALFBAND_CASCADE_HPP

#include <algorithm>
#include <complex>
#include <cstddef>
#include <span>
#include <vector>

/**
 * @brief A cascade of halving FIR stages, shaped by two measured findings.
 *
 * Header-only and free of dependencies beyond the standard library: the arithmetic below is a
 * plain loop rather than a kernel call.
 *
 * A dot product per output is bounded by the per-output overhead rather than by the multiplies.
 * Measured across decimations from 2 to 512, the time per input sample held flat while the designed
 * multiplies per input sample fell by a third. Taking sixteen outputs together removes the horizontal sum at
 * the end of each dot product: a tap becomes a scalar broadcast over thirty-two consecutive
 * floats, sixteen outputs' worth of real and imaginary parts side by side, and every lane
 * carries its own running total. That alone was worth about 3.5x.
 *
 * A stage whose two edges stand symmetrically about a quarter of its own rate is a halfband, its
 * impulse response `sin(pi t/2)/(pi t)`, and every tap at an even offset from the center is zero.
 * Measured at -328 dB under the peak, which is `sin` of a rounded multiple of pi, so they are
 * skipped by index and never by testing against zero. Keeping the two phases of the input apart
 * makes the surviving taps a contiguous run; skipping the zeros turns the other phase into a
 * single scalar multiply. Given the block, that is worth another 3.2x to 5.6x; the saving is the
 * block's to give.
 *
 * A stage writes its output already split for the stage below, so the input is taken apart once
 * at the top of the cascade.
 *
 * The shape holds for a halving cascade. A single-stage decimation by 25 over 137 taps costs 1.8x
 * as much rewritten this way as it does as an ordinary dot product per output, because splitting
 * into 25 phases is 25 strided gathers over the whole full-rate stream and costs more than the
 * horizontal sums it removes. Use this where the decimation is 2 per stage; measure before
 * assuming it anywhere else.
 */
namespace gr::filter {

class HalfbandCascade {
public:
    using CF = std::complex<float>;

    /**
     * @brief One stage per tap set, each halving the rate.
     *
     * Every set must be an odd-length halfband, its cutoff exactly a quarter of its own input
     * rate, or the zeros this skips are not zeros and the output is a different filter. Nothing
     * here checks that; the designs that feed it are pinned where they are built.
     */
    explicit HalfbandCascade(std::span<const std::vector<float>> stages) {
        _stage.reserve(stages.size());
        for (const std::vector<float>& t : stages) {
            _stage.push_back(Stage::of(t));
        }
        _even.resize(_stage.size());
        _odd.resize(_stage.size());
        _phase.assign(_stage.size(), 0UZ);
    }

    /**
     * @brief Start every stage as though silence preceded its input.
     *
     * Without this a stage produces nothing until it holds a whole window, so a cascade fed L
     * samples returns fewer than L/2^stages and its stream is offset by the group delay. With it,
     * the first push returns exactly the count the decimation implies and output zero sits where a
     * filter started on a zeroed history would put it, which is what a block with a fixed
     * input-to-output ratio has to deliver.
     *
     * The prefix is `taps - 2` rather than `taps - 1`: an output reads its window forward from the
     * sample it is handed, and one fewer zero lines that up with a window read backward from the
     * newest sample.
     *
     * A caller that feeds its own leading silence must not call this, or the two paddings compound.
     *
     * The length used is the one the stage was designed with, kept rather than reconstructed from the
     * taps that survive: a design whose center index is odd puts its first and last taps at an odd offset
     * from the center, so both survive and it keeps one tap more than a design whose center index is
     * even. Inverting the live count therefore reads two taps long for those, at 19, 23 and 27 taps, and
     * such a stage primes one sample too deep and hands back one output more than the decimation implies.
     */
    void primeWithSilence() {
        for (std::size_t s = 0UZ; s < _stage.size(); ++s) {
            const std::size_t taps = _stage[s].designed;
            if (taps < 2UZ) {
                continue;
            }
            const std::vector<CF> zeros(taps - 2UZ, CF{});
            room(s, zeros.size()).fill(std::span<const CF>(zeros));
        }
    }

    /// @brief Forget the history: the next push() starts a new stream.
    void reset() {
        for (std::vector<CF>& v : _even) {
            v.clear();
        }
        for (std::vector<CF>& v : _odd) {
            v.clear();
        }
        _phase.assign(_phase.size(), 0UZ);
    }

    [[nodiscard]] std::size_t stages() const noexcept { return _stage.size(); }

    /// @brief The multiplies an output actually costs, against the tap count it was designed with: a
    /// little over half, the rest being by zero.
    [[nodiscard]] std::size_t liveTaps(std::size_t stage) const noexcept { return _stage[stage].bulk.size() + 1UZ; }

    /**
     * @brief Filter @p in and append what comes out to @p out.
     *
     * The stream produced is the one a single call over the whole input would have produced, to the
     * bit: each stage keeps the two phases of whatever it has not consumed, and every output is
     * summed in the same order over the taps whichever path computed it.
     */
    std::size_t push(std::span<const CF> in, std::vector<CF>& out) {
        const std::size_t before = out.size();
        const std::size_t n      = _stage.size();
        if (n == 0UZ) {
            out.insert(out.end(), in.begin(), in.end());
            return out.size() - before;
        }

        // The one place the stream is taken apart; every stage below is handed its two phases
        // already separated by the stage above it.
        room(0UZ, in.size()).fill(in);

        for (std::size_t s = 0UZ; s < n; ++s) {
            const Stage&           h  = _stage[s];
            const std::vector<CF>& xe = _even[s];
            const std::vector<CF>& xo = _odd[s];

            std::size_t made = 0UZ;
            if (xe.size() > h.maxEven && xo.size() > h.maxOdd) {
                made = std::min(xe.size() - h.maxEven, xo.size() - h.maxOdd);
            }

            const CF*    along  = (h.bulkOnOdd ? xo : xe).data();
            const CF*    single = (h.bulkOnOdd ? xe : xo).data() + h.centerAt;
            const float* bulk   = h.bulk.data();
            const auto   taps   = static_cast<unsigned int>(h.bulk.size());
            const float  center = h.center;
            const auto*  xf     = reinterpret_cast<const float*>(along);

            const bool        last = s + 1UZ == n;
            const std::size_t at   = out.size();
            Room              dst;
            CF*               into = nullptr;
            if (last) {
                out.resize(at + made);
                into = out.data() + at;
            } else {
                dst = room(s + 1UZ, made);
            }

            const auto emit = [&](std::size_t j, CF y) {
                y += center * single[j];
                if (last) {
                    into[j] = y;
                } else {
                    dst.place(j, y);
                }
            };

            constexpr std::size_t kLanes = 16UZ;
            std::size_t           j      = 0UZ;
            for (; j + kLanes <= made; j += kLanes) {
                float acc[2UZ * kLanes] = {};
                for (unsigned int k = 0; k < taps; ++k) {
                    const float  w = bulk[k];
                    const float* p = xf + 2UZ * (j + k);
                    for (std::size_t u = 0UZ; u < 2UZ * kLanes; ++u) {
                        acc[u] += p[u] * w;
                    }
                }
                for (std::size_t u = 0UZ; u < kLanes; ++u) {
                    emit(j + u, CF{acc[2UZ * u], acc[2UZ * u + 1UZ]});
                }
            }
            // The tail sums in the same order over k, which is a correctness property: one
            // delivered sample must not differ in its last bits according to where a block
            // boundary fell.
            for (; j < made; ++j) {
                float acc[2] = {};
                for (unsigned int k = 0; k < taps; ++k) {
                    const float  w = bulk[k];
                    const float* p = xf + 2UZ * (j + k);
                    acc[0] += p[0] * w;
                    acc[1] += p[1] * w;
                }
                emit(j, CF{acc[0], acc[1]});
            }

            if (made > 0UZ) {
                // Output j reads both phases from offset j, so one output consumes one sample of
                // each.
                _even[s].erase(_even[s].begin(), _even[s].begin() + static_cast<std::ptrdiff_t>(made));
                _odd[s].erase(_odd[s].begin(), _odd[s].begin() + static_cast<std::ptrdiff_t>(made));
            }
        }
        return out.size() - before;
    }

private:
    /// @brief One halving stage, taken apart into the taps that carry anything.
    struct Stage {
        std::vector<float> bulk; /// the surviving taps, less the center tap
        float              center    = 0.0f;
        bool               bulkOnOdd = false; /// which phase @c bulk reads
        std::size_t        centerAt  = 0UZ;   /// the center tap's offset into the other phase
        std::size_t        maxEven   = 0UZ;
        std::size_t        maxOdd    = 0UZ;
        std::size_t        designed  = 0UZ; /// the length it was built with, which the surviving taps do not determine

        [[nodiscard]] static Stage of(const std::vector<float>& taps) {
            Stage      s;
            const auto n   = static_cast<int>(taps.size());
            const int  mid = (n - 1) / 2;
            s.center       = taps[static_cast<std::size_t>(mid)];
            s.designed     = taps.size();

            // The surviving taps have the parity the center tap does not: an offset from the center
            // is odd exactly where the index and the center index disagree.
            const int first = (mid % 2 == 0) ? 1 : 0;
            for (int i = first; i < n; i += 2) {
                s.bulk.push_back(taps[static_cast<std::size_t>(i)]);
            }
            s.bulkOnOdd = first == 1;
            // x[2j + i] is the even phase at j + i/2 for even i and the odd phase at j + (i-1)/2
            // for odd i, and integer division gives both.
            s.centerAt = static_cast<std::size_t>(mid / 2);

            const std::size_t bulkMax = s.bulk.empty() ? 0UZ : s.bulk.size() - 1UZ;
            s.maxOdd                  = s.bulkOnOdd ? bulkMax : s.centerAt;
            s.maxEven                 = s.bulkOnOdd ? s.centerAt : bulkMax;
            return s;
        }
    };

    /// @brief Room for a batch at one stage, as a pointer into each phase and the parity the first
    /// of them takes.
    struct Room {
        CF*         even  = nullptr;
        CF*         odd   = nullptr;
        std::size_t first = 0UZ;

        /**
         * @brief Where sample @p i of the batch goes.
         *
         * @c first is also the bias between the phases: a batch beginning on the odd phase reaches
         * its first even slot one position later, and counting both from the same place would put
         * every even sample one ahead.
         */
        void place(std::size_t i, CF x) const {
            const std::size_t at = first + i;
            if ((at % 2UZ) == 0UZ) {
                even[at / 2UZ - first] = x;
            } else {
                odd[at / 2UZ] = x;
            }
        }

        /// @brief Take a whole span apart at once, a pair at a time, so the two phases are two
        /// walks of fixed shape rather than a parity test per sample.
        void fill(std::span<const CF> src) const {
            std::size_t i = 0UZ;
            std::size_t e = 0UZ;
            std::size_t o = 0UZ;
            if (first == 1UZ && !src.empty()) {
                odd[o++] = src[0];
                i        = 1UZ;
            }
            for (; i + 1UZ < src.size(); i += 2UZ) {
                even[e++] = src[i];
                odd[o++]  = src[i + 1UZ];
            }
            if (i < src.size()) {
                even[e] = src[i];
            }
        }
    };

    Room room(std::size_t s, std::size_t n) {
        Room              r;
        const std::size_t p     = _phase[s];
        const std::size_t nEven = (p == 0UZ) ? (n + 1UZ) / 2UZ : n / 2UZ;
        const std::size_t atE   = _even[s].size();
        const std::size_t atO   = _odd[s].size();
        _even[s].resize(atE + nEven);
        _odd[s].resize(atO + n - nEven);
        r.even    = _even[s].data() + atE;
        r.odd     = _odd[s].data() + atO;
        r.first   = p;
        _phase[s] = (p + n) % 2UZ;
        return r;
    }

    std::vector<Stage>           _stage;
    std::vector<std::vector<CF>> _even, _odd;
    std::vector<std::size_t>     _phase;
};

} // namespace gr::filter

#endif // GNURADIO_HALFBAND_CASCADE_HPP
