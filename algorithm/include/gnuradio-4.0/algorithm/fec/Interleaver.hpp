#ifndef GNURADIO_ALGORITHM_FEC_INTERLEAVER_HPP
#define GNURADIO_ALGORITHM_FEC_INTERLEAVER_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <span>
#include <stdexcept>
#include <vector>

namespace gr::fec {

/// Which permutation family an `Interleaver` carries. The three differ in how the map is stated, not in what
/// they do with it, and only one of them is stream-shaped.
enum class InterleaverKind : std::uint8_t {
    Block,         ///< rows x cols, written row-major and read column-major
    Convolutional, ///< Forney/Ramsey type II: `branches` delay lines and a commutator, state across calls
    Permutation    ///< an explicit table, whatever a standard publishes
};

/**
 * @brief The three interleaver families over one vocabulary: an index map, or a set of delay lines.
 *
 * An interleaver moves items and never reads them, so the whole kernel is addressing arithmetic. The two framed
 * families resolve to an explicit map and apply it as a gather — `out[i] = in[map[i]]` — which is the form that
 * costs one load and one store per item and no branch; the convolutional family has no finite map and is a set
 * of ring buffers instead. Nothing here allocates once it is configured, and nothing here throws off the sample
 * path: every refusal happens at construction, where it can name the parameter that caused it.
 *
 * **`block`** is the rectangular interleaver: `rows * cols` items written row-major and read column-major,
 * `out[c * rows + r] = in[r * cols + c]`. Its contract is the spread: two items adjacent at the input land
 * exactly `rows` apart at the output, which is what turns a burst of `rows` channel errors into single errors
 * spread across the frame.
 *
 * **`convolutional`** is the Forney/Ramsey type-II structure: `branches` (B) delay lines, branch `b` holding
 * `b * unitDelay` (M) cells, and a commutator that steps one branch per item on both sides. A cell holds its
 * item for one whole revolution of the commutator, so a delay of `b * M` cells is a delay of `b * M * B` items
 * of stream, and the map is `k -> k + (k mod B) * M * B` — injective, because the shift is a multiple of B and
 * therefore leaves `k mod B` alone. The complementary set `(B - 1 - b) * M` inverts it, and since a branch pair
 * sums to `(B - 1) * M` cells whatever `b` is, the end-to-end delay is the same for every item and comes to
 * exactly `B * (B - 1) * M` items. That constant is `latency()`, and it is the one number a consumer needs to
 * line the recovered stream back up.
 *
 * This is the one family here whose state crosses a call boundary: it is stream-shaped and the record boundary
 * is transparent to it. Splitting the same stream into different chunks gives the identical output, because the
 * cell contents and the commutator position are the whole state and neither depends on where a call ended.
 * During the initial fill the delay lines emit `fillValue()` rather than swallowing items, so the output is
 * 1:1 with the input from the first item and the latency is visible instead of hidden.
 *
 * **`permutation`** is an explicit table, for every standard whose interleaver is just a published vector. The
 * table is the gather: `out[i] = in[table[i]]`. It is validated as a true permutation — every index below the
 * frame size appearing exactly once — and its inverse is computed there and then, so deinterleaving is the same
 * gather against a second table rather than a search. No table is a default; the caller supplies one.
 *
 * **The item type is a parameter.** The families do not read values, so `std::uint8_t` for hard bits or symbols
 * and `float` for soft decisions are the same code; a soft-decision consumer that had to pack to bytes first
 * would be quantizing precisely the information the decoder behind the deinterleaver exists to use.
 *
 * **The `block` kind takes an output window.** Several framings interleave a coded block, read out a
 * contiguous stretch of the result and spend the rest of the frame on synchronization — the AO-40 shape. With
 * a window of `(offset, length)`, `interleave()` emits only that stretch of the column-major readout and
 * `deinterleave()` takes it back, putting each item where it came from and leaving every position the window
 * did not cover at `fillValue()`. For soft items that value is the erasure a decoder wants to be handed; for
 * hard items it is whatever the caller says a missing item reads as. A full window is the ordinary case and
 * costs nothing: the map is then a bijection, the inverse is another gather, and no fill is written.
 */
template<typename T>
class Interleaver {
public:
    /// A frame ceiling. It bounds the convolutional family through its latency, which is the item count its
    /// cells hold, and it is what keeps a QA sweep over shapes cheap.
    static constexpr std::size_t kMaxFrame = std::size_t{1} << 20;

    /// The rectangular interleaver over a whole frame.
    [[nodiscard]] static Interleaver block(std::size_t rows, std::size_t cols) { return block(rows, cols, 0UZ, rows * cols, T{}); }

    /// The rectangular interleaver read out through a window: `length` items of the column-major readout
    /// starting at `offset`. `fillValue` is what `deinterleave()` writes where the window reached nothing.
    [[nodiscard]] static Interleaver block(std::size_t rows, std::size_t cols, std::size_t windowOffset, std::size_t windowLength, T fillValue) {
        if (rows == 0UZ) {
            throw std::invalid_argument("gr::fec::Interleaver: rows is 0 — a block interleaver needs at least one row");
        }
        if (cols == 0UZ) {
            throw std::invalid_argument("gr::fec::Interleaver: cols is 0 — a block interleaver needs at least one column");
        }
        if (rows > kMaxFrame / cols) {
            throw std::invalid_argument(std::format("gr::fec::Interleaver: rows {} times cols {} is beyond the frame ceiling of {} items", rows, cols, kMaxFrame));
        }

        const std::size_t frame = rows * cols;
        if (windowLength == 0UZ) {
            throw std::invalid_argument("gr::fec::Interleaver: window_length is 0 — a window that reads nothing is not a readout");
        }
        if (windowOffset >= frame) {
            throw std::invalid_argument(std::format("gr::fec::Interleaver: window_offset {} is at or past the {}-item frame", windowOffset, frame));
        }
        if (windowLength > frame - windowOffset) {
            throw std::invalid_argument(std::format("gr::fec::Interleaver: window_length {} from window_offset {} runs past the {}-item frame", windowLength, windowOffset, frame));
        }

        Interleaver self;
        self._kind      = InterleaverKind::Block;
        self._rows      = rows;
        self._cols      = cols;
        self._frameSize = frame;
        self._fillValue = fillValue;

        self._map.resize(windowLength);
        for (std::size_t i = 0UZ; i < windowLength; ++i) {
            // The `j`-th item of the readout is row `j % rows` of column `j / rows`.
            const std::size_t j = windowOffset + i;
            self._map[i]        = (j % rows) * cols + (j / rows);
        }
        self.finishFramed();
        return self;
    }

    /// The Forney/Ramsey structure: `branches` delay lines, branch `b` holding `b * unitDelay` cells.
    [[nodiscard]] static Interleaver convolutional(std::size_t branches, std::size_t unitDelay, T fillValue = T{}) {
        if (branches < 2UZ) {
            throw std::invalid_argument(std::format("gr::fec::Interleaver: branches is {} — below two branches there is nothing to interleave", branches));
        }
        if (unitDelay < 1UZ) {
            throw std::invalid_argument("gr::fec::Interleaver: unit_delay is 0 — every branch would hold no cells");
        }
        if (branches - 1UZ > kMaxFrame / branches / unitDelay) {
            throw std::invalid_argument(std::format("gr::fec::Interleaver: branches {} and unit_delay {} give a latency beyond the ceiling of {} items", branches, unitDelay, kMaxFrame));
        }

        Interleaver self;
        self._kind      = InterleaverKind::Convolutional;
        self._branches  = branches;
        self._unitDelay = unitDelay;
        self._fillValue = fillValue;
        // The two directions are two independent sets of lines with complementary depths, because one object
        // that offered both from a single set would have them overwrite each other's cells.
        self._forward = makeLines(branches, [unitDelay](std::size_t b) noexcept { return b * unitDelay; }, fillValue);
        self._reverse = makeLines(branches, [branches, unitDelay](std::size_t b) noexcept { return (branches - 1UZ - b) * unitDelay; }, fillValue);
        return self;
    }

    /// An explicit table, read as the gather `out[i] = in[table[i]]`.
    [[nodiscard]] static Interleaver permutation(std::span<const std::size_t> table) {
        if (table.empty()) {
            throw std::invalid_argument("gr::fec::Interleaver: table is empty — the permutation kind has no default table");
        }
        if (table.size() > kMaxFrame) {
            throw std::invalid_argument(std::format("gr::fec::Interleaver: table holds {} entries, beyond the frame ceiling of {} items", table.size(), kMaxFrame));
        }

        std::vector<bool> seen(table.size(), false);
        for (std::size_t i = 0UZ; i < table.size(); ++i) {
            if (table[i] >= table.size()) {
                throw std::invalid_argument(std::format("gr::fec::Interleaver: table entry {} is {}, outside a {}-item frame", i, table[i], table.size()));
            }
            if (seen[table[i]]) {
                throw std::invalid_argument(std::format("gr::fec::Interleaver: table entry {} repeats index {} — the table is not a permutation", i, table[i]));
            }
            seen[table[i]] = true;
        }

        Interleaver self;
        self._kind      = InterleaverKind::Permutation;
        self._frameSize = table.size();
        self._map.assign(table.begin(), table.end());
        self.finishFramed();
        return self;
    }

    [[nodiscard]] InterleaverKind kind() const noexcept { return _kind; }
    [[nodiscard]] std::size_t     rows() const noexcept { return _rows; }
    [[nodiscard]] std::size_t     cols() const noexcept { return _cols; }
    [[nodiscard]] std::size_t     branches() const noexcept { return _branches; }
    [[nodiscard]] std::size_t     unitDelay() const noexcept { return _unitDelay; }
    [[nodiscard]] T               fillValue() const noexcept { return _fillValue; }

    /// Items one frame takes at the interleaver input. Zero for the convolutional kind, which has no frame: it
    /// is stream-shaped and any split of the stream is as good as any other.
    [[nodiscard]] std::size_t frameSize() const noexcept { return _frameSize; }

    /// Items one frame produces at the interleaver output: the window's length for a windowed block, and
    /// `frameSize()` otherwise.
    [[nodiscard]] std::size_t interleavedSize() const noexcept { return _map.size(); }

    /// The end-to-end interleave-to-deinterleave delay in items: `B * (B - 1) * M` for the convolutional kind,
    /// and zero for the framed kinds, which produce a frame from a frame.
    [[nodiscard]] std::size_t latency() const noexcept { return _branches * (_branches - 1UZ) * _unitDelay; }

    /// The resolved gather for the framed kinds: `out[i] = in[indexMap()[i]]`. Empty for the convolutional kind.
    [[nodiscard]] std::span<const std::size_t> indexMap() const noexcept { return _map; }

    /// The inverse gather, present only where the map covers the whole frame.
    [[nodiscard]] std::span<const std::size_t> inverseMap() const noexcept { return _inverse; }

    /// Whether the map reaches every position of the frame. A windowed block does not, and its deinterleave
    /// leaves the positions it did not reach at `fillValue()`.
    [[nodiscard]] bool covers() const noexcept { return _covers; }

    /**
     * @brief Interleaves as many whole frames as `in` and `out` both hold, and returns the items written.
     *
     * The framed kinds consume `frameSize()` and produce `interleavedSize()` per frame; the convolutional kind
     * is 1:1 and consumes and produces `min(in.size(), out.size())`. A caller that hands over a partial frame
     * gets the whole frames back and nothing else — what to do about the remainder is the block's decision, not
     * the kernel's, and the kernel has no way to report a drop that a return value does not already say.
     */
    std::size_t interleave(std::span<const T> in, std::span<T> out) noexcept {
        if (_kind == InterleaverKind::Convolutional) {
            return runLines(_forward, in, out);
        }

        // The tables are read through local pointers: an item type of `std::uint8_t` may alias anything, so a
        // store through it forces the compiler to re-read whatever it cannot prove is elsewhere, and a local
        // whose address never escapes is the one thing it can prove.
        const std::size_t* const map    = _map.data();
        const std::size_t        width  = _map.size();
        const std::size_t        frames = std::min(in.size() / _frameSize, out.size() / width);
        for (std::size_t f = 0UZ; f < frames; ++f) {
            const T* src = in.data() + f * _frameSize;
            T*       dst = out.data() + f * width;
            for (std::size_t i = 0UZ; i < width; ++i) {
                dst[i] = src[map[i]];
            }
        }
        return frames * width;
    }

    /// @brief The inverse of `interleave()`, with the input and output frame sizes exchanged.
    std::size_t deinterleave(std::span<const T> in, std::span<T> out) noexcept {
        if (_kind == InterleaverKind::Convolutional) {
            return runLines(_reverse, in, out);
        }

        const std::size_t* const map     = _map.data();
        const std::size_t* const inverse = _inverse.data();
        const std::size_t        width   = _map.size();
        const std::size_t        frame   = _frameSize;
        const std::size_t        frames  = std::min(in.size() / width, out.size() / frame);
        for (std::size_t f = 0UZ; f < frames; ++f) {
            const T* src = in.data() + f * width;
            T*       dst = out.data() + f * frame;
            if (_covers) {
                for (std::size_t i = 0UZ; i < frame; ++i) {
                    dst[i] = src[inverse[i]];
                }
            } else {
                // The window's map is its own scatter, so the partial case needs no second table: fill the frame
                // with the erasure value and write back the positions the window did reach.
                std::fill_n(dst, frame, _fillValue);
                for (std::size_t i = 0UZ; i < width; ++i) {
                    dst[map[i]] = src[i];
                }
            }
        }
        return frames * frame;
    }

    /// Returns the delay lines to their initial fill and the commutators to branch zero. The framed kinds hold
    /// no state and this does nothing to them.
    void reset() noexcept {
        for (DelayLines* lines : {&_forward, &_reverse}) {
            std::ranges::fill(lines->cells, _fillValue);
            std::ranges::fill(lines->head, 0UZ);
            lines->commutator = 0UZ;
        }
    }

private:
    /// One direction's delay lines: every branch's cells in one allocation, addressed through a prefix table, so
    /// the sample path touches three vectors and never a vector of vectors.
    struct DelayLines {
        std::vector<T>           cells{};
        std::vector<std::size_t> offset{}; ///< branches + 1 prefix offsets into `cells`
        std::vector<std::size_t> head{};   ///< the cell each branch reads and then overwrites
        std::size_t              commutator{0UZ};
    };

    InterleaverKind          _kind{InterleaverKind::Block};
    std::size_t              _rows{0UZ};
    std::size_t              _cols{0UZ};
    std::size_t              _branches{0UZ};
    std::size_t              _unitDelay{0UZ};
    std::size_t              _frameSize{0UZ};
    T                        _fillValue{};
    std::vector<std::size_t> _map{};     ///< interleave gather, one entry per output item
    std::vector<std::size_t> _inverse{}; ///< deinterleave gather, present only when `_covers`
    bool                     _covers{true};
    DelayLines               _forward{};
    DelayLines               _reverse{};

    Interleaver() = default;

    /// Marks whether the map is a bijection and, if it is, inverts it once so that deinterleaving is a gather
    /// rather than a scatter into a frame that first had to be cleared.
    void finishFramed() {
        _covers = _map.size() == _frameSize;
        if (!_covers) {
            return;
        }
        _inverse.resize(_frameSize);
        for (std::size_t i = 0UZ; i < _frameSize; ++i) {
            _inverse[_map[i]] = i;
        }
    }

    template<typename FDepth>
    [[nodiscard]] static DelayLines makeLines(std::size_t branches, FDepth depthOf, T fillValue) {
        DelayLines lines;
        lines.offset.resize(branches + 1UZ);
        lines.head.assign(branches, 0UZ);
        std::size_t total = 0UZ;
        for (std::size_t b = 0UZ; b < branches; ++b) {
            lines.offset[b] = total;
            total += depthOf(b);
        }
        lines.offset[branches] = total;
        lines.cells.assign(total, fillValue);
        return lines;
    }

    /// The commutator walk. A branch of zero depth passes its item straight through — writing and reading a
    /// zero-length ring is not a thing to special-case in the loop, it is a thing the loop must not do at all.
    [[nodiscard]] std::size_t runLines(DelayLines& lines, std::span<const T> in, std::span<T> out) noexcept {
        // Locals for the same reason as above: three vectors' data pointers, held where a byte store cannot
        // reach them, instead of three reloads per item.
        T* const                 cells    = lines.cells.data();
        const std::size_t* const offset   = lines.offset.data();
        std::size_t* const       head     = lines.head.data();
        const std::size_t        branches = _branches;

        const std::size_t n = std::min(in.size(), out.size());
        std::size_t       b = lines.commutator;
        for (std::size_t k = 0UZ; k < n; ++k) {
            const std::size_t begin = offset[b];
            const std::size_t depth = offset[b + 1UZ] - begin;
            if (depth == 0UZ) {
                out[k] = in[k];
            } else {
                const std::size_t slot   = head[b];
                const T           oldest = cells[begin + slot];
                cells[begin + slot]      = in[k];
                out[k]                   = oldest;
                head[b]                  = (slot + 1UZ == depth) ? 0UZ : slot + 1UZ;
            }
            ++b;
            if (b == branches) {
                b = 0UZ;
            }
        }
        lines.commutator = b;
        return n;
    }
};

} // namespace gr::fec

#endif // GNURADIO_ALGORITHM_FEC_INTERLEAVER_HPP
