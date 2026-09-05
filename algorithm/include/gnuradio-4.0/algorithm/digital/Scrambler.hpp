#ifndef GNURADIO_ALGORITHM_SCRAMBLER_HPP
#define GNURADIO_ALGORITHM_SCRAMBLER_HPP

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/BitPacking.hpp>

/**
 * @brief The two scrambler families over GF(2), from one register definition and three recursions.
 *
 * A scrambler is a shift register of `degree` bits and a set of feedback delays. For a bit stream
 * `b` the feedback at position `k` is `f(b, k) = XOR over j in taps of b[k - j]`, and that is the
 * only expression here. The three modes differ in which stream the register holds and in nothing
 * else:
 *
 * - `Additive` runs `s[k] = f(s, k)` and emits `in[k] XOR s[k]`, so the register holds the
 *   generator's own past output and the sequence is a function of the taps and the seed alone.
 *   Error multiplication is exactly one and the whitening does not depend on the data, but the two
 *   ends must agree on the phase: one bit of phase error leaves the descrambled stream agreeing
 *   with the truth at chance.
 * - `MultiplicativeScramble` runs `out[k] = in[k] XOR f(out, k)`, so the register holds the block's
 *   own past output.
 * - `MultiplicativeDescramble` runs `out[k] = in[k] XOR f(in, k)`, so the register holds the past
 *   input. Every bit in it is received data once `degree` bits have arrived, so the output is
 *   correct from position `degree` onward whatever the register started as, from any offset into
 *   the stream. The price is error multiplication of `1 + tapCount` and a data-dependent failure
 *   mode: an input chosen as `in[k] = f(out, k)` holds the scrambled output at a constant for as
 *   long as it continues, from any seed.
 *
 * The two multiplicative modes invert each other exactly from position zero when both registers
 * start from the same `degree` bits, and `Additive` is its own inverse.
 *
 * **Taps are delays, not exponents.** A three-term feedback polynomial is written both as a
 * characteristic polynomial `x^n + x^m + 1`, whose recursion is `s[k] = s[k-(n-m)] XOR s[k-n]` and
 * whose written exponents are not delays, and as a delay-operator polynomial `1 + X^m + X^n`, whose
 * written exponents are delays. The two readings give reciprocal delay sets and therefore
 * time-reversed sequences, and the standards do not agree on which reading their spelling means:
 * CCSDS 131.0-B's `x^8 + x^7 + x^5 + x^3 + 1` is characteristic, giving delays `{1,3,5,8}`, while
 * IEEE 802.11's `x^7 + x^4 + 1` is a delay reading, giving `{4,7}`. Each is confirmed by the
 * sequence its own standard publishes. `TapMask` therefore carries delays and nothing else, and the
 * named constants below carry the reading that reproduces the published check value.
 *
 * **The output tap is not a parameter.** Reading a different stage of a Fibonacci register shifts
 * the sequence's phase by a fixed number of steps and nothing else, and a phase shift is a seed. One
 * convention is fixed here: the seed supplies `degree` bits, leftmost first, in time order. Under
 * `Additive` those are the first `degree` bits of the sequence; under the two multiplicative modes
 * they are the `degree` bits assumed to precede the stream, oldest first. Both readings pack into
 * `seed` the same way, leftmost into the most significant of the `degree` significant bits.
 *
 * An item is `BitPacking.hpp`'s item: a `std::uint8_t` carrying `bitsPerItem` significant bits in
 * its low positions, whose stream order inside the item is `MsbFirst` or `LsbFirst`. Bits above
 * `bitsPerItem` are masked away on input and left zero on output, so no stream value can fail.
 * `BitOrder` is that header's enumeration rather than a second one with the same two members.
 *
 * `configureScrambler()` validates, derives everything the sample path indexes, and is the only function here
 * that can throw. It throws `std::invalid_argument`, not a graph exception, because this header
 * depends on the standard library alone. `scramble()` and `resetScrambler()` are `noexcept`, allocate
 * nothing, and hold no state beyond the configuration they are handed.
 *
 * Three shapes carry the sample path, one per mode, and they differ because the data dependence
 * differs. `Additive`'s sequence does not depend on the data at all, so it is generated for a whole
 * chunk before an input bit is read and the scrambling proper is an XOR over two spans; the sequence
 * comes either from a bit table of one full period, built in `configureScrambler()` for `degree <= 16` and at
 * most 8193 bytes, or from a recurrence that produces `min(8, minDelay)` bits per branch-free step,
 * since no bit produced within such a step is needed by that same step. `MultiplicativeScramble`
 * reads its own output and takes the same stepped form at a loop-carried depth of `minDelay`.
 * `MultiplicativeDescramble` reads only its input, so it has no loop-carried dependency whatever:
 * at eight bits per item it is a shifted-span XOR over a rolling 64-bit window of received bits,
 * `tapCount` shifts and XORs per byte. `detail::scrambleReference()` is the bit-at-a-time definition
 * all of them are measured against; every form leaves identical state, so they are interchangeable
 * at any item boundary.
 *
 * Two things a published agreement can fix that a recursion cannot are here as well. A **profile** is
 * a name for the four settings the generator is built from — taps, seed, item width and bit order —
 * and for the family they belong to, so a self-synchronizing agreement cannot be run additively by
 * accident; it is never a default and it names nothing else. An **explicit sequence** is a mask the
 * caller supplies outright instead of walking a recursion for it, which is what most vendor whitening
 * is published as; it is the table form with the table handed over, so it costs the same per item and
 * changes no other behavior.
 *
 * The multiplicative pair also carries an output inversion and a forced-transition rule, off by
 * default and dispatched to rather than folded in. Both are stated at `configureMonitor()`, including
 * the run-length bound the rule cannot give: the pair stays a bijection on bit streams, so no property
 * of the output holds over all inputs, and what it does give is a firing bound and the destruction of
 * the plain recursion's lock-up input.
 */
namespace gr::digital {

/// @brief Feedback delays as a bitmask: bit `j-1` set means delay `j` participates.
using TapMask = std::uint64_t;

/// @brief The largest delay in @p taps, which is the register width. Zero for an empty set.
[[nodiscard]] constexpr std::uint8_t degreeOf(TapMask taps) noexcept { return static_cast<std::uint8_t>(std::bit_width(taps)); }

/// @brief The smallest delay in @p taps, which bounds the width of a branch-free step.
[[nodiscard]] constexpr std::uint8_t minDelayOf(TapMask taps) noexcept { return static_cast<std::uint8_t>(std::countr_zero(taps) + 1); }

/// @brief How many delays @p taps names, which is one less than a multiplicative descrambler's error multiplication.
[[nodiscard]] constexpr std::uint8_t tapCountOf(TapMask taps) noexcept { return static_cast<std::uint8_t>(std::popcount(taps)); }

namespace detail {

/// @brief @p value in decimal, usable while the enclosing function is being constant-evaluated.
[[nodiscard]] constexpr std::string decimalText(std::uint64_t value) {
    std::array<char, 20> digits{};
    std::size_t          length = 0UZ;
    do {
        digits[length++] = static_cast<char>('0' + static_cast<char>(value % 10ULL));
        value /= 10ULL;
    } while (value != 0ULL);

    std::string text;
    while (length != 0UZ) {
        text.push_back(digits[--length]);
    }
    return text;
}

} // namespace detail

/**
 * @brief The delays named in @p list, given as decimal integers separated by commas and optional spaces.
 *
 * Every delay is in `[1, 64]` and distinct, and at least one is required. Anything else throws
 * `std::invalid_argument` quoting @p list, which is the whole of the validation a tap set needs: a
 * delay list has no exponent to misread.
 */
[[nodiscard]] constexpr TapMask tapsFromDelayList(std::string_view list) {
    const auto reject = [list](std::string_view reason) { return std::invalid_argument("gr::digital::tapsFromDelayList: " + std::string(reason) + ", got '" + std::string(list) + "'"); };

    TapMask     taps  = 0ULL;
    std::size_t index = 0UZ;
    while (true) {
        while (index < list.size() && list[index] == ' ') {
            ++index;
        }
        const std::size_t start = index;
        unsigned          delay = 0U;
        while (index < list.size() && list[index] >= '0' && list[index] <= '9') {
            delay = delay * 10U + static_cast<unsigned>(list[index] - '0');
            if (delay > 64U) {
                throw reject("a delay above 64");
            }
            ++index;
        }
        if (index == start) {
            throw reject("the list must name at least one decimal delay");
        }
        if (delay == 0U) {
            throw reject("a delay of zero");
        }

        const TapMask bit = 1ULL << (delay - 1U);
        if ((taps & bit) != 0ULL) {
            throw reject("a repeated delay");
        }
        taps |= bit;

        while (index < list.size() && list[index] == ' ') {
            ++index;
        }
        if (index == list.size()) {
            return taps;
        }
        if (list[index] != ',') {
            throw reject("a character that is neither a digit, a comma nor a space");
        }
        ++index;
    }
}

/// @brief The delays of @p taps in ascending order, in the spelling `tapsFromDelayList` reads.
[[nodiscard]] inline std::string delayListFromTaps(TapMask taps) {
    std::string text;
    for (TapMask rest = taps; rest != 0ULL; rest &= rest - 1ULL) {
        if (!text.empty()) {
            text.push_back(',');
        }
        text += detail::decimalText(static_cast<std::uint64_t>(std::countr_zero(rest) + 1));
    }
    return text;
}

/**
 * @brief The `degree` bits of @p bits, leftmost into the most significant of the `degree` significant bits.
 *
 * The reading is the same for every mode — a run of `degree` bits in time order — and only where
 * those bits sit differs. Exactly @p degree characters are required, each `'0'` or `'1'`; anything
 * else throws `std::invalid_argument` quoting @p bits.
 */
[[nodiscard]] constexpr std::uint64_t seedFromBitString(std::string_view bits, std::uint8_t degree) {
    if (bits.size() != static_cast<std::size_t>(degree)) {
        throw std::invalid_argument("gr::digital::seedFromBitString: the seed must be exactly " + detail::decimalText(degree) + " bits, got '" + std::string(bits) + "'");
    }

    std::uint64_t seed = 0ULL;
    for (const char bit : bits) {
        if (bit != '0' && bit != '1') {
            throw std::invalid_argument("gr::digital::seedFromBitString: the seed must be '0' and '1' characters, got '" + std::string(bits) + "'");
        }
        seed = (seed << 1U) | static_cast<std::uint64_t>(bit == '1' ? 1U : 0U);
    }
    return seed;
}

/**
 * @brief The delay sets of the standards that publish one, as interoperability constants.
 *
 * Each is an agreement between two ends rather than a free choice, so each is named here for a
 * caller who wants the identifier instead of the digits, and none of them is a default. Where a
 * standard publishes a sequence, the reading recorded is the one that regenerates it and the
 * reciprocal reading's bytes are given beside it.
 */
namespace standard {

/// @brief CCSDS 131.0-B pseudo-randomizer, `h(x) = x^8 + x^7 + x^5 + x^3 + 1` read as a characteristic polynomial.
/// Seed `"11111111"`, period 255. The sequence opens `FF 48 0E C0 9A 0D 70 BC`, whose first 40 bits are the value the
/// standard publishes; the reciprocal reading `"3,5,7,8"` opens `FF 1A AF 66 52` and does not.
inline constexpr TapMask ccsds131 = tapsFromDelayList("1,3,5,8");

/// @brief CCSDS 131.0-B-5 10.4.1's primary pseudo-randomizer, `h(x) = x^17 + x^14 + 1` read as a characteristic
/// polynomial, period 131071. It is the primary sequence of the current issue; 10.4.2 keeps the degree-8 `ccsds131`
/// above only for backward compatibility with legacy systems, with a note that the shorter period can put spectral
/// lines at 1/255 of the symbol rate. The standard prints its seed as the register loading `"11000111000111000"` and
/// the register flushes out low stage first, so the seed naming this sequence under this header's time-order
/// convention is the reverse, `"00011100011100011"` — the DVB entry below has the same relationship to its figure.
/// The sequence opens `1C 71 B9 1B A9 BA 84 57`, whose first 40 bits are the value 10.4.3 note 2 publishes; the
/// reciprocal reading `"14,17"` opens `1C 71 FF FF 00 03 80 0F` from the same seed, parting at the third byte, so a
/// swapped reading is visible against the published value.
inline constexpr TapMask ccsds131_17 = tapsFromDelayList("3,17");

/// @brief The CC11xx / SX12xx data-whitening sequence, `x^9 + x^5 + 1` read as a characteristic polynomial. Seed
/// `"111111111"`, period 511, eight bits per item taken `LsbFirst`, under which the sequence opens
/// `FF E1 1D 9A ED 85 33 24 EA 7A D2 39 70 97 57 0A` — the sequence the vendor documentation describes, so taps, seed
/// and bit order are confirmed together by one string rather than one at a time. `MsbFirst` on the same taps opens
/// `FF 87 B8 59 B7 A1 CC 24`, and the reciprocal reading `"5,9"` opens `FF C1 FB E8 4C 90 72 8B` least significant bit
/// first and `FF 83 DF 17 32 09 4E D1` most significant.
inline constexpr TapMask pn9 = tapsFromDelayList("4,9");

/// @brief The Silicon Laboratories Si4463 (EZRadioPRO) whitening tap set, which is `pn9`'s. The chip's PN engine is a
/// 16-bit LFSR whose PN9 mode wires the tap mask `PKT_WHT_POLY = 0x0108`, seeded from `PKT_WHT_SEED`, default
/// `0xFFFF`, with the output taken at bit 0 (`PKT_WHT_BIT_NUM = 0`, direction FORWARD) — Silicon Laboratories'
/// Si446x revC2A API property documentation, corroborated by WDS-generated `radio_config.h` property lists. Only the
/// nine effective stages enter the recursion, so the seed is all ones however wide the register, and the Galois
/// realization of that mask emits `pn9`'s sequence advanced five steps rather than `pn9`'s own phase — which is why
/// the chip's whitening and a CC11xx's are incompatible despite one polynomial. Under this header's convention the
/// seed is `"111100001"` and the sequence opens `0F EF D0 6C 2F 9C 21 51` least significant bit first, which is what
/// the property document's tap mask, seed and output bit give once the Galois realization is written out.
inline constexpr TapMask si4463 = pn9;

/// @brief The G3RUH 9600-baud packet-radio scrambler, `1 + X^12 + X^17` read as a delay-operator polynomial, which is
/// the descrambler recursion `out[k] = in[k] XOR in[k-12] XOR in[k-17]` the published modem description states.
/// Period 131071, error multiplication 3, convergence in 17 bits. It is self-synchronizing, so it belongs to the
/// multiplicative pair and its seed does not affect correctness. That published recursion is what fixes the reading:
/// the same polynomial is also written `x^17 + x^12 + 1`, whose characteristic reading is the reciprocal `"5,17"`, and
/// both are primitive with period 131071, so the recursion is the only thing that separates them. From an all-ones
/// seed the additive form of `"12,17"` opens `FF FF 80 07 C0 7F E7 C1` and `"5,17"` opens `FF FF 83 E0 C7 CE 13 7C`.
inline constexpr TapMask g3ruh = tapsFromDelayList("12,17");

/// @brief DVB energy dispersal, ETSI EN 300 744 and EN 300 421, `1 + X^14 + X^15`. The standard loads
/// `100101010000000` into its fifteen stages and takes the output from the adder combining stages 14 and 15, so the
/// first bit emitted is the feedback bit and not the register's contents; the seed naming that same sequence under this
/// header's convention is `"000000111111011"`. Period 32767, sequence `03 F6 08 34 30 B8 A3 93`, whose opening bits are
/// the ones the standard's figure prints. Emitting the register's contents instead opens `95 01 7E 07 04 12 18 6C`, and
/// the reciprocal reading `"1,15"` opens `03 F7 FA B5 59 B3 22 44`.
inline constexpr TapMask dvb = tapsFromDelayList("14,15");

/// @brief IEEE Std 802.11, `x^7 + x^4 + 1` read as a delay polynomial; the OFDM PHY uses it additively and the DSSS and
/// HR/DSSS PHYs self-synchronizingly, which is why the family is a use and not a property of the polynomial.
/// Additive seed `"0000111"`, period 127, sequence `0E F2 C9 02 26 2E B6 0C`, reproducing all 127 bits the standard
/// publishes for the all-ones initial state; the reciprocal reading `"3,7"` opens `0F E3 B1 4B EA 85 BC E5`.
inline constexpr TapMask ieee80211 = tapsFromDelayList("4,7");

/// @brief ITU-T V-series GSTN data modems, form A, `1 + x^-18 + x^-23`; the negative exponents are delays, so there is
/// no reading to resolve. Self-synchronizing in the V-series, period 8388607 used additively, `FF FF FE 00 00 7C 00 1F`
/// from an all-ones seed. This is V.32's `GPC`: the calling station scrambles with it and descrambles with `itu_5_23`.
inline constexpr TapMask itu_18_23 = tapsFromDelayList("18,23");

/// @brief ITU-T V-series GSTN data modems, form B, `1 + x^-5 + x^-23`. Period 8388607, `FF FF FE 0F 83 E3 07 3E` from an
/// all-ones seed. This is V.32's `GPA`: the answering station scrambles with it and descrambles with `itu_18_23`.
inline constexpr TapMask itu_5_23 = tapsFromDelayList("5,23");

/// @brief ITU-T V.22 and V.22bis, `1 + x^-14 + x^-17`. Period 131071, `FF FF 80 01 C0 07 E0 1C` from an all-ones seed.
inline constexpr TapMask itu_14_17 = tapsFromDelayList("14,17");

/// @brief IEEE Std 802.3 10GBASE-R 64B/66B, `1 + x^39 + x^58`, self-synchronizing and the widest named entry. The
/// standard publishes no sequence to check a seed against; a 64B/66B link agrees on the polynomial alone.
inline constexpr TapMask ieee8023_64b66b = tapsFromDelayList("39,58");

} // namespace standard

/// @brief Which stream the register holds, which is the whole of the difference between the three.
enum class ScramblerMode : std::uint8_t { Additive, MultiplicativeScramble, MultiplicativeDescramble };

/// @brief Where an additive generator's bits come from: a bit table of one full period, or the recurrence itself.
enum class SequenceForm : std::uint8_t { Table, Recurrence };

/// @brief Where an additive mask comes from: the feedback recursion, or a sequence the caller supplied outright.
enum class SequenceSource : std::uint8_t { Lfsr, Explicit };

/// @brief Which family a named profile belongs to. This is a property of the agreement, not of its taps: `"4,7"` is
/// IEEE 802.11's additive OFDM scrambler and its self-synchronizing DSSS one, and a profile names which.
enum class ProfileFamily : std::uint8_t { Additive, Multiplicative };

/**
 * @brief A named generator: the settings a published agreement fixes, and nothing else.
 *
 * A profile carries `taps`, `seed`, `bitsPerItem` and `order` and no other setting — no reset policy, no rate and no
 * port shape — so naming one cannot change anything but the sequence. It is never a default: a caller that names no
 * profile gets none. `unrecorded` is empty for a usable profile and otherwise names the document the missing constant
 * has to come from, which is how a profile whose numbers have not been confirmed stays visible instead of shipping
 * a guess.
 */
struct ScramblerProfile {
    std::string_view name;
    ProfileFamily    family;
    TapMask          taps;
    std::uint64_t    seed;
    std::uint8_t     bitsPerItem;
    BitOrder         order;
    std::string_view unrecorded;
};

/// @brief The named profiles, in the order `profileByName` lists them when it does not find one.
inline constexpr std::array<ScramblerProfile, 5> kProfiles{{
    {"ccsds131", ProfileFamily::Additive, standard::ccsds131, seedFromBitString("11111111", 8U), 8U, BitOrder::MsbFirst, ""},                 //
    {"ccsds131_17", ProfileFamily::Additive, standard::ccsds131_17, seedFromBitString("00011100011100011", 17U), 8U, BitOrder::MsbFirst, ""}, //
    {"pn9", ProfileFamily::Additive, standard::pn9, seedFromBitString("111111111", 9U), 8U, BitOrder::LsbFirst, ""},                          //
    {"si4463", ProfileFamily::Additive, standard::si4463, seedFromBitString("111100001", 9U), 8U, BitOrder::LsbFirst, ""},                    //
    {"g3ruh", ProfileFamily::Multiplicative, standard::g3ruh, 0ULL, 8U, BitOrder::MsbFirst, ""}                                               //
}};

/// @brief The profile named @p name, or `std::invalid_argument` listing every name there is.
///
/// A profile that resolves may still be unusable: `unrecorded` is what says so, and `applyProfile` is what refuses.
[[nodiscard]] inline const ScramblerProfile& profileByName(std::string_view name) {
    for (const ScramblerProfile& entry : kProfiles) {
        if (entry.name == name) {
            return entry;
        }
    }

    std::string known;
    for (const ScramblerProfile& entry : kProfiles) {
        if (!known.empty()) {
            known += ", ";
        }
        known += entry.name;
    }
    throw std::invalid_argument("gr::digital::profileByName: unknown profile '" + std::string(name) + "', the names are " + known);
}

/**
 * @brief One configured scrambler: the five settings, what `configureScrambler()` derives from them, and the register.
 *
 * The derived members are written by `configureScrambler()` and are never set directly, with one exception:
 * `form` selects between two generators that agree bit for bit and leave identical state, so a
 * caller measuring the two may move it at any item boundary. Everything the sample path indexes is
 * fixed here, so no allocation, lock, division or modulo runs per item.
 */
struct ScramblerConfig {
    TapMask       taps        = 0ULL;
    std::uint64_t seed        = 0ULL;
    ScramblerMode mode        = ScramblerMode::Additive;
    std::uint8_t  bitsPerItem = 8U;
    BitOrder      order       = BitOrder::MsbFirst;

    std::uint8_t                 degree      = 0U;   ///< the largest delay, and the register width
    std::uint8_t                 minDelay    = 0U;   ///< the smallest delay
    std::uint8_t                 tapCount    = 0U;   ///< how many delays participate
    std::uint8_t                 stepBits    = 1U;   ///< `min(8, minDelay)`, the widest branch-free step
    std::uint64_t                regMask     = 0ULL; ///< the low `degree` bits
    std::uint64_t                initialReg  = 0ULL; ///< the register `resetScrambler()` restores
    std::array<std::uint8_t, 64> delays      = {};   ///< the delays in ascending order, `tapCount` of them
    SequenceForm                 form        = SequenceForm::Recurrence;
    std::vector<std::uint8_t>    table       = {};  ///< one period plus eight wrap bits, empty unless the table form applies
    std::size_t                  tablePeriod = 0UZ; ///< the sequence period the table holds, zero when there is none

    SequenceSource source         = SequenceSource::Lfsr; ///< whether the mask is walked or was supplied
    bool           sequenceRepeat = true;                 ///< under `Explicit`, whether the sequence tiles an epoch or covers its first `S` bits

    bool          invertOutput = false; ///< `v`, complementing every bit of the scrambled stream
    std::uint32_t forceAfter   = 0U;    ///< `N`, the forcing counter's modulus; zero disables the rule and its cost
    std::uint8_t  monitorP     = 0U;    ///< the nearer monitored delay `p`
    std::uint8_t  monitorQ     = 0U;    ///< the further monitored delay `q`

    std::uint64_t reg           = 0ULL; ///< the last `degree` bits of the relevant stream, delay 1 in bit 0
    std::size_t   phase         = 0UZ;  ///< the sequence phase in bits, table form only
    std::uint64_t monitorWindow = 0ULL; ///< the last 64 bits of the transmitted stream, delay 1 in bit 0
    std::uint32_t forceCounter  = 0U;   ///< `c`, items since the monitored comparison was last non-zero

    std::size_t nUnscrambledItems  = 0UZ; ///< items an explicit sequence could not cover, counted over the configuration's life
    std::size_t nForcedTransitions = 0UZ; ///< times the forcing term fired, counted over the configuration's life
};

/// @brief Puts the register back to the seed and the sequence phase, the monitor and the forcing counter back to the
/// start of an epoch. Allocates nothing. The two totals are not epoch state and survive, since what they report is what
/// the configuration did over its whole life.
///
/// The monitor's window starts empty rather than at the seed. The seed is a history of the register's own stream and
/// there is no transmitted history before the first item; more to the point, an empty window is the same at both ends
/// whatever either seeded, so the two forcing counters agree from item zero even when the seeds do not, which is what
/// makes the descrambler the scrambler's exact inverse rather than its inverse after convergence.
inline void resetScrambler(ScramblerConfig& cfg) noexcept {
    cfg.reg           = cfg.initialReg;
    cfg.phase         = 0UZ;
    cfg.monitorWindow = 0ULL;
    cfg.forceCounter  = 0U;
}

/**
 * @brief Sets @p cfg to the stated scrambler, deriving the register width, the step width, the table and the seeded register.
 *
 * An empty tap set, a `bitsPerItem` outside `[1, 8]` and a seed carrying bits above `degree` throw
 * `std::invalid_argument` and leave @p cfg as it was, so a rejected setting cannot half-apply. An
 * all-zero seed is accepted here and is a dead generator under `Additive`, producing an all-zero
 * sequence for ever; refusing it belongs to whatever declares the setting, since it is a legitimate
 * starting history for the two multiplicative modes.
 *
 * The table form is chosen for `Additive` at `degree <= 16`, where one full period plus eight wrap
 * bits is at most 8193 bytes, and the walk that finds the period is capped so that a tap set with no
 * period inside that bound falls back to the recurrence rather than growing the table. The seed is
 * turned into the register the pure recursion starts from: under `Additive` the seed names the first
 * `degree` bits of the sequence, so the recursion is run backwards `degree` steps to recover the
 * history that produces them, and the sample path then carries no prologue at all.
 */
inline void configureScrambler(ScramblerConfig& cfg, TapMask taps, std::uint64_t seed, ScramblerMode mode, unsigned bitsPerItem, BitOrder order) {
    if (taps == 0ULL) {
        throw std::invalid_argument("gr::digital::configureScrambler: the tap set must name at least one delay");
    }
    if (bitsPerItem < 1U || bitsPerItem > 8U) {
        throw std::invalid_argument("gr::digital::configureScrambler: bitsPerItem must be in [1, 8], got " + detail::decimalText(bitsPerItem));
    }

    const std::uint8_t  degree  = degreeOf(taps);
    const std::uint64_t regMask = degree == 64U ? ~0ULL : ((1ULL << degree) - 1ULL);
    if ((seed & ~regMask) != 0ULL) {
        throw std::invalid_argument("gr::digital::configureScrambler: the seed carries bits above degree " + detail::decimalText(degree) + ", got " + detail::decimalText(seed));
    }

    // the register the pure recursion starts from; under Additive the seed is the sequence's own first bits, so the
    // recursion runs backwards degree steps, each step recovering s[k-1-degree] from s[k-1] and the shorter delays
    std::uint64_t initialReg = seed & regMask;
    if (mode == ScramblerMode::Additive) {
        const std::uint64_t backward = ((taps << 1U) | 1ULL) & regMask;
        for (std::uint8_t step = 0U; step < degree; ++step) {
            const std::uint64_t bit = static_cast<std::uint64_t>(std::popcount(initialReg & backward) & 1);
            initialReg              = (initialReg >> 1U) | (bit << (degree - 1U));
        }
    }

    std::vector<std::uint8_t> table;
    std::size_t               tablePeriod = 0UZ;
    if (mode == ScramblerMode::Additive && degree <= 16U) {
        constexpr std::size_t kPeriodCap = 65535UZ;

        std::uint64_t walk = initialReg;
        for (std::size_t step = 1UZ; step <= kPeriodCap; ++step) {
            walk = ((walk << 1U) | static_cast<std::uint64_t>(std::popcount(walk & taps) & 1)) & regMask;
            if (walk == initialReg) {
                tablePeriod = step;
                break;
            }
        }
        if (tablePeriod != 0UZ) {
            // one period, plus the eight wrap bits an unaligned read of the last phase needs
            table.assign((tablePeriod + 15UZ) / 8UZ, std::uint8_t{0});
            walk = initialReg;
            for (std::size_t step = 0UZ; step < tablePeriod + 8UZ; ++step) {
                const unsigned bit = static_cast<unsigned>(std::popcount(walk & taps)) & 1U;
                table[step >> 3U]  = static_cast<std::uint8_t>(table[step >> 3U] | (bit << (7UZ - (step & 7UZ))));
                walk               = ((walk << 1U) | static_cast<std::uint64_t>(bit)) & regMask;
            }
        }
    }

    cfg.taps        = taps;
    cfg.seed        = seed & regMask;
    cfg.mode        = mode;
    cfg.bitsPerItem = static_cast<std::uint8_t>(bitsPerItem);
    cfg.order       = order;
    cfg.degree      = degree;
    cfg.minDelay    = minDelayOf(taps);
    cfg.tapCount    = tapCountOf(taps);
    cfg.stepBits    = std::min(std::uint8_t{8}, cfg.minDelay);
    cfg.regMask     = regMask;
    cfg.initialReg  = initialReg;
    cfg.form        = tablePeriod != 0UZ ? SequenceForm::Table : SequenceForm::Recurrence;
    cfg.table       = std::move(table);
    cfg.tablePeriod = tablePeriod;

    // a configuration is whole or it is nothing: the mask source, the forced-transition group and both totals go back
    // to their defaults here, so no field of a previous configuration can survive into this one
    cfg.source             = SequenceSource::Lfsr;
    cfg.sequenceRepeat     = true;
    cfg.invertOutput       = false;
    cfg.forceAfter         = 0U;
    cfg.monitorP           = 0U;
    cfg.monitorQ           = 0U;
    cfg.nUnscrambledItems  = 0UZ;
    cfg.nForcedTransitions = 0UZ;

    cfg.delays        = {};
    std::uint8_t slot = 0U;
    for (TapMask rest = taps; rest != 0ULL; rest &= rest - 1ULL) {
        cfg.delays[slot++] = static_cast<std::uint8_t>(std::countr_zero(rest) + 1);
    }

    resetScrambler(cfg);
}

/// @brief Sets @p cfg to the generator @p profile names, in @p mode.
///
/// The profile fills `taps`, `seed`, `bitsPerItem` and `order` and nothing else, so everything `configureScrambler()` derives is
/// derived here in the same way and by the same call. Two refusals are the point of the function: a profile whose
/// family is not @p mode's throws naming both, because a self-synchronizing agreement run additively is a silent wrong
/// answer rather than an error; and a profile whose constants are not recorded throws naming the document they are in,
/// because the alternative is a default nobody stated.
inline void applyProfile(ScramblerConfig& cfg, std::string_view name, ScramblerMode mode) {
    const ScramblerProfile& entry  = profileByName(name);
    const ProfileFamily     wanted = mode == ScramblerMode::Additive ? ProfileFamily::Additive : ProfileFamily::Multiplicative;
    if (entry.family != wanted) {
        const std::string_view family = entry.family == ProfileFamily::Additive ? "additive" : "multiplicative (self-synchronizing)";
        const std::string_view asked  = wanted == ProfileFamily::Additive ? "additive" : "multiplicative";
        throw std::invalid_argument("gr::digital::applyProfile: profile '" + std::string(name) + "' is " + std::string(family) + " and cannot be used " + std::string(asked));
    }
    if (!entry.unrecorded.empty()) {
        throw std::invalid_argument("gr::digital::applyProfile: profile '" + std::string(name) + "' cannot be applied because " + std::string(entry.unrecorded));
    }

    configureScrambler(cfg, entry.taps, entry.seed, mode, static_cast<unsigned>(entry.bitsPerItem), entry.order);
}

/**
 * @brief The bit sequence @p text spells, as hexadecimal byte pairs separated by optional spaces.
 *
 * A whitening mask is bytes and is published as bytes, so it is spelled in hexadecimal rather than as the `'0'` and
 * `'1'` characters `taps` and `seed` use. The returned bytes hold the sequence most significant bit first, so bit `i`
 * of the sequence is bit `7 - (i mod 8)` of byte `i / 8`. Anything that is not a hexadecimal digit or a space, an odd
 * number of digits, and an empty sequence throw `std::invalid_argument` quoting @p text.
 */
[[nodiscard]] inline std::vector<std::uint8_t> sequenceFromHex(std::string_view text) {
    const auto reject = [text](std::string_view reason) { return std::invalid_argument("gr::digital::sequenceFromHex: " + std::string(reason) + ", got '" + std::string(text) + "'"); };

    std::vector<std::uint8_t> bytes;
    unsigned                  pending = 0U;
    unsigned                  digits  = 0U;
    for (const char c : text) {
        if (c == ' ') {
            continue;
        }
        unsigned value = 0U;
        if (c >= '0' && c <= '9') {
            value = static_cast<unsigned>(c - '0');
        } else if (c >= 'A' && c <= 'F') {
            value = static_cast<unsigned>(c - 'A') + 10U;
        } else if (c >= 'a' && c <= 'f') {
            value = static_cast<unsigned>(c - 'a') + 10U;
        } else {
            throw reject("a character that is neither a hexadecimal digit nor a space");
        }

        pending = (pending << 4U) | value;
        if (++digits == 2U) {
            bytes.push_back(static_cast<std::uint8_t>(pending));
            pending = 0U;
            digits  = 0U;
        }
    }
    if (digits != 0U) {
        throw reject("an odd number of hexadecimal digits");
    }
    if (bytes.empty()) {
        throw reject("the sequence must name at least one byte");
    }
    return bytes;
}

/**
 * @brief Sets @p cfg to an additive scrambler whose mask is @p sequence rather than a recursion's output.
 *
 * An explicit sequence *is* the table form with the table supplied instead of walked, so `tablePeriod` becomes the
 * sequence length `S` in bits and the same table generator serves it unchanged. `taps`, `degree`, `minDelay`,
 * `stepBits` and `regMask` are all zero and the recurrence path is unreachable; the alternation with a tap set is
 * structural rather than checked, since this function takes no taps and `configureScrambler()` takes no sequence.
 *
 * `repeat` states what happens when an epoch outlasts the sequence. `true` tiles it, exactly as a period-`S` recursion
 * would, and cannot lose whitening. `false` covers the first `S` bits of the epoch and leaves everything after them
 * unscrambled, counting each such item in `nUnscrambledItems` — the block is 1:1 and cannot drop an item, so what it
 * can do about an overrun is count it and say so. An item straddling `S` is scrambled with the bits that remain and is
 * not counted; at one bit per item, or where `bitsPerItem` divides `S`, no item straddles.
 *
 * @p sequence must hold between 1 and 2^20 bits, most significant bit first, and `bitsPerItem` must be in `[1, 8]`.
 * Anything else throws `std::invalid_argument` and leaves @p cfg as it was.
 */
inline void configureExplicit(ScramblerConfig& cfg, std::span<const std::uint8_t> sequence, std::size_t bitCount, unsigned bitsPerItem, BitOrder order, bool repeat) {
    constexpr std::size_t kMaxBits = 1UZ << 20;

    if (bitCount == 0UZ || bitCount > kMaxBits) {
        throw std::invalid_argument("gr::digital::configureExplicit: the sequence must hold between 1 and 1048576 bits, got " + detail::decimalText(bitCount));
    }
    if (bitCount > sequence.size() * 8UZ) {
        throw std::invalid_argument("gr::digital::configureExplicit: the sequence holds " + detail::decimalText(sequence.size() * 8UZ) + " bits, fewer than the " + detail::decimalText(bitCount) + " claimed");
    }
    if (bitsPerItem < 1U || bitsPerItem > 8U) {
        throw std::invalid_argument("gr::digital::configureExplicit: bitsPerItem must be in [1, 8], got " + detail::decimalText(bitsPerItem));
    }

    // one period plus the eight wrap bits an unaligned read of the last phase needs, which are the sequence's own first
    // eight under `repeat` and are masked away without being read under `!repeat`
    std::vector<std::uint8_t> table((bitCount + 15UZ) / 8UZ, std::uint8_t{0});
    for (std::size_t bit = 0UZ; bit < bitCount + 8UZ; ++bit) {
        const std::size_t from  = bit < bitCount ? bit : bit - bitCount;
        const unsigned    value = (static_cast<unsigned>(sequence[from >> 3U]) >> (7UZ - (from & 7UZ))) & 1U;
        table[bit >> 3U]        = static_cast<std::uint8_t>(table[bit >> 3U] | (value << (7UZ - (bit & 7UZ))));
    }

    cfg.taps        = 0ULL;
    cfg.seed        = 0ULL;
    cfg.mode        = ScramblerMode::Additive;
    cfg.bitsPerItem = static_cast<std::uint8_t>(bitsPerItem);
    cfg.order       = order;
    cfg.degree      = 0U;
    cfg.minDelay    = 0U;
    cfg.tapCount    = 0U;
    cfg.stepBits    = 0U;
    cfg.regMask     = 0ULL;
    cfg.initialReg  = 0ULL;
    cfg.delays      = {};
    cfg.form        = SequenceForm::Table;
    cfg.table       = std::move(table);
    cfg.tablePeriod = bitCount;

    cfg.source             = SequenceSource::Explicit;
    cfg.sequenceRepeat     = repeat;
    cfg.invertOutput       = false;
    cfg.forceAfter         = 0U;
    cfg.monitorP           = 0U;
    cfg.monitorQ           = 0U;
    cfg.nUnscrambledItems  = 0UZ;
    cfg.nForcedTransitions = 0UZ;

    resetScrambler(cfg);
}

/**
 * @brief Adds the output inversion and the forced-transition rule to a configured multiplicative scrambler or descrambler.
 *
 * Written out, with `y` the transmitted stream, `f(y,k)` the feedback over the tap delays and `v` the inversion:
 * `F[k] = 1` exactly when the counter reads `N - 1`; the scrambler emits `y[k] = v XOR x[k] XOR f(y,k) XOR F[k]` and
 * the descrambler recovers `x[k] = v XOR y[k] XOR f(y,k) XOR F[k]`; then both compute `d = y[k-p] XOR y[k-q]` and set
 * the counter to zero if `d` is one and to `(c + 1) mod N` otherwise.
 *
 * **The monitor reads only the transmitted stream**, which is the whole reason the rule stays invertible: both ends see
 * `y`, so both compute the same counter and the same `F[k]` with no side channel, and the descrambler is still exactly
 * the scrambler's inverse. What follows from that is also what the rule cannot promise — the pair is a bijection on bit
 * streams, so for every output, the constant one included, there is exactly one input producing it, and no property of
 * the output is guaranteed over all inputs. What the rule does guarantee is that the block never emits `N` consecutive
 * items over which the monitored comparison is zero without injecting a complement at the `N`-th, and that the plain
 * recursion's lock-up input no longer holds the output at a constant.
 *
 * @p forceAfter is `N`, in `[2, 4096]`, or zero to disable the rule entirely; @p monitorDelays names `p` and `q` as two
 * distinct decimal delays in the spelling `tapsFromDelayList` reads, and is required when `N` is non-zero and refused
 * when it is zero, since a setting describing a disabled mechanism is a settings error rather than a no-op. Call this
 * after `configureScrambler()`, which puts the whole group back to disabled.
 */
inline void configureMonitor(ScramblerConfig& cfg, bool invertOutput, std::uint32_t forceAfter, std::string_view monitorDelays) {
    if (forceAfter != 0U && (forceAfter < 2U || forceAfter > 4096U)) {
        throw std::invalid_argument("gr::digital::configureMonitor: forceAfter must be zero or in [2, 4096], got " + detail::decimalText(forceAfter));
    }
    if (forceAfter == 0U && !monitorDelays.empty()) {
        throw std::invalid_argument("gr::digital::configureMonitor: monitorDelays describes a disabled mechanism when forceAfter is zero, got '" + std::string(monitorDelays) + "'");
    }
    if (cfg.mode == ScramblerMode::Additive && (invertOutput || forceAfter != 0U)) {
        throw std::invalid_argument("gr::digital::configureMonitor: the output inversion and the forced-transition rule belong to the multiplicative pair, which has feedback for them to act on");
    }

    std::uint8_t nearer  = 0U;
    std::uint8_t further = 0U;
    if (forceAfter != 0U) {
        const TapMask delays = tapsFromDelayList(monitorDelays);
        if (std::popcount(delays) != 2) {
            throw std::invalid_argument("gr::digital::configureMonitor: monitorDelays must name exactly two distinct delays, got '" + std::string(monitorDelays) + "'");
        }
        nearer  = minDelayOf(delays);
        further = degreeOf(delays);
    }

    cfg.invertOutput = invertOutput;
    cfg.forceAfter   = forceAfter;
    cfg.monitorP     = nearer;
    cfg.monitorQ     = further;
    resetScrambler(cfg);
}

namespace detail {

/// @brief Bit `i` of the index at bit `7-i`, which turns an `LsbFirst` item into a field and back again.
inline constexpr std::array<std::uint8_t, 256> kReversedByte = [] {
    std::array<std::uint8_t, 256> reversed{};
    for (std::size_t value = 0UZ; value < reversed.size(); ++value) {
        unsigned bits = 0U;
        for (unsigned bit = 0U; bit < 8U; ++bit) {
            bits |= ((static_cast<unsigned>(value) >> bit) & 1U) << (7U - bit);
        }
        reversed[value] = static_cast<std::uint8_t>(bits);
    }
    return reversed;
}();

/// @brief The `width` significant bits of @p item as a field holding the stream's first bit at position `width-1`.
/// Its own inverse, so the same call turns a field back into an item.
[[nodiscard]] inline std::uint8_t streamField(std::uint8_t item, unsigned width, BitOrder order) noexcept {
    const unsigned field = static_cast<unsigned>(item) & ((1U << width) - 1U);
    return order == BitOrder::MsbFirst ? static_cast<std::uint8_t>(field) : kReversedByte[static_cast<std::size_t>(field) << (8U - width)];
}

/// @brief The next @p count feedback bits, oldest at position `count-1`, read from @p reg alone.
/// Correct exactly while `count <= minDelay`, where no bit of a batch is a delay of another.
[[nodiscard]] inline std::uint32_t feedbackBits(std::uint64_t reg, TapMask taps, unsigned count) noexcept {
    std::uint32_t bits = 0U;
    for (unsigned step = 0U; step < count; ++step) {
        bits = (bits << 1U) | (static_cast<std::uint32_t>(std::popcount((taps >> step) & reg)) & 1U);
    }
    return bits;
}

/// @brief Carries the sequence phase over @p items for a form that does not read it, so every form leaves the same state.
inline void advancePhase(ScramblerConfig& cfg, std::size_t items) noexcept {
    if (cfg.tablePeriod != 0UZ) {
        cfg.phase = (cfg.phase + items * static_cast<std::size_t>(cfg.bitsPerItem)) % cfg.tablePeriod;
    }
}

/// @brief Sequence items from the recurrence, `min(stepBits, bitsPerItem)` bits per branch-free step.
inline void generateRecurrence(ScramblerConfig& cfg, std::span<std::uint8_t> sequence) noexcept {
    const unsigned      width = static_cast<unsigned>(cfg.bitsPerItem);
    const unsigned      step  = std::min(static_cast<unsigned>(cfg.stepBits), width);
    const TapMask       taps  = cfg.taps;
    const std::uint64_t mask  = cfg.regMask;
    std::uint64_t       reg   = cfg.reg;

    for (std::uint8_t& item : sequence) {
        std::uint32_t field = 0U;
        for (unsigned left = width; left != 0U;) {
            const unsigned      count = std::min(step, left);
            const std::uint32_t bits  = feedbackBits(reg, taps, count);
            reg                       = ((reg << count) | bits) & mask;
            field                     = (field << count) | bits;
            left -= count;
        }
        item = streamField(static_cast<std::uint8_t>(field), width, cfg.order);
    }

    cfg.reg = reg;
    advancePhase(cfg, sequence.size());
}

/// @brief Sequence items from the table, one unaligned read of `bitsPerItem` bits per item and a compare-and-subtract on the phase.
inline void generateTable(ScramblerConfig& cfg, std::span<std::uint8_t> sequence) noexcept {
    const unsigned      width  = static_cast<unsigned>(cfg.bitsPerItem);
    const std::uint32_t field  = (1U << width) - 1U;
    const std::size_t   period = cfg.tablePeriod;
    const std::uint8_t* buffer = cfg.table.data();
    const std::uint64_t mask   = cfg.regMask;
    std::size_t         phase  = cfg.phase;
    std::uint64_t       reg    = cfg.reg;

    for (std::uint8_t& item : sequence) {
        const std::uint32_t window = (static_cast<std::uint32_t>(buffer[phase >> 3U]) << 8U) | static_cast<std::uint32_t>(buffer[(phase >> 3U) + 1UZ]);
        const std::size_t   shift  = 16UZ - (phase & 7UZ) - static_cast<std::size_t>(width);
        const std::uint32_t bits   = (window >> shift) & field;

        item = streamField(static_cast<std::uint8_t>(bits), width, cfg.order);
        reg  = ((reg << width) | bits) & mask;
        phase += width;
        while (phase >= period) {
            phase -= period;
        }
    }

    cfg.phase = phase;
    cfg.reg   = reg;
}

/// @brief An explicit sequence that does not repeat: the table read once, then a zero mask and a counted item.
/// The phase does not wrap, so it says how far into the epoch the stream has gone rather than where in a period it is.
inline void generateExplicitOnce(ScramblerConfig& cfg, std::span<std::uint8_t> sequence) noexcept {
    const unsigned      width  = static_cast<unsigned>(cfg.bitsPerItem);
    const std::uint32_t field  = (1U << width) - 1U;
    const std::size_t   length = cfg.tablePeriod;
    const std::uint8_t* buffer = cfg.table.data();
    const BitOrder      order  = cfg.order;
    std::size_t         phase  = cfg.phase;
    std::size_t         missed = 0UZ;

    for (std::uint8_t& item : sequence) {
        std::uint32_t bits = 0U;
        if (phase >= length) {
            ++missed;
        } else {
            const std::uint32_t window = (static_cast<std::uint32_t>(buffer[phase >> 3U]) << 8U) | static_cast<std::uint32_t>(buffer[(phase >> 3U) + 1UZ]);
            const std::size_t   shift  = 16UZ - (phase & 7UZ) - static_cast<std::size_t>(width);
            const std::size_t   left   = length - phase;
            bits                       = (window >> shift) & field;
            if (left < static_cast<std::size_t>(width)) {
                // the item straddles the end of the sequence: keep the bits it covers and drop the wrap bits behind them
                bits &= field ^ ((1U << (static_cast<unsigned>(width) - static_cast<unsigned>(left))) - 1U);
            }
        }
        item = streamField(static_cast<std::uint8_t>(bits), width, order);
        phase += width;
    }

    cfg.phase = phase;
    cfg.nUnscrambledItems += missed;
}

/// @brief The additive mode: a chunk of sequence generated before an input bit is read, then an XOR over two spans.
inline void scrambleAdditive(ScramblerConfig& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out) noexcept {
    constexpr std::size_t kChunk = 256UZ;

    std::array<std::uint8_t, kChunk> sequence{};
    const std::uint8_t               itemMask = static_cast<std::uint8_t>((1U << cfg.bitsPerItem) - 1U);
    const std::size_t                count    = std::min(in.size(), out.size());

    for (std::size_t base = 0UZ; base < count; base += kChunk) {
        const std::size_t block = std::min(kChunk, count - base);
        if (cfg.source == SequenceSource::Explicit && !cfg.sequenceRepeat) {
            generateExplicitOnce(cfg, std::span<std::uint8_t>{sequence.data(), block});
        } else if (cfg.form == SequenceForm::Table) {
            generateTable(cfg, std::span<std::uint8_t>{sequence.data(), block});
        } else {
            generateRecurrence(cfg, std::span<std::uint8_t>{sequence.data(), block});
        }
        for (std::size_t i = 0UZ; i < block; ++i) {
            out[base + i] = static_cast<std::uint8_t>((in[base + i] ^ sequence[i]) & itemMask);
        }
    }
}

/// @brief The multiplicative scrambler: the feedback reads the block's own output, so the steps are `min(stepBits, bitsPerItem)` bits deep.
inline void scrambleMultiplicative(ScramblerConfig& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out) noexcept {
    const unsigned      width = static_cast<unsigned>(cfg.bitsPerItem);
    const unsigned      step  = std::min(static_cast<unsigned>(cfg.stepBits), width);
    const TapMask       taps  = cfg.taps;
    const std::uint64_t mask  = cfg.regMask;
    const std::size_t   count = std::min(in.size(), out.size());
    std::uint64_t       reg   = cfg.reg;

    for (std::size_t i = 0UZ; i < count; ++i) {
        const std::uint32_t source = streamField(in[i], width, cfg.order);
        std::uint32_t       field  = 0U;
        for (unsigned left = width; left != 0U;) {
            const unsigned      taken = std::min(step, left);
            const std::uint32_t bits  = ((source >> (left - taken)) & ((1U << taken) - 1U)) ^ feedbackBits(reg, taps, taken);
            reg                       = ((reg << taken) | bits) & mask;
            field                     = (field << taken) | bits;
            left -= taken;
        }
        out[i] = streamField(static_cast<std::uint8_t>(field), width, cfg.order);
    }

    cfg.reg = reg;
}

/// @brief The multiplicative descrambler at any item width: the feedback reads only the input, so the item's own bits
/// complete the window and the `bitsPerItem` feedback bits are independent of one another.
inline void descrambleItems(ScramblerConfig& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out) noexcept {
    const unsigned      width = static_cast<unsigned>(cfg.bitsPerItem);
    const TapMask       taps  = cfg.taps;
    const std::uint64_t mask  = cfg.regMask;
    const std::size_t   count = std::min(in.size(), out.size());
    std::uint64_t       reg   = cfg.reg;

    for (std::size_t i = 0UZ; i < count; ++i) {
        const std::uint64_t source   = static_cast<std::uint64_t>(streamField(in[i], width, cfg.order));
        std::uint32_t       feedback = 0U;
        for (unsigned bit = 0U; bit < width; ++bit) {
            const std::uint64_t inside = (taps & ((1ULL << bit) - 1ULL)) << (width - bit);
            const unsigned      parity = (static_cast<unsigned>(std::popcount((taps >> bit) & reg)) ^ static_cast<unsigned>(std::popcount(inside & source))) & 1U;
            feedback                   = (feedback << 1U) | parity;
        }
        reg    = ((reg << width) | source) & mask;
        out[i] = streamField(static_cast<std::uint8_t>(static_cast<std::uint32_t>(source) ^ feedback), width, cfg.order);
    }

    cfg.reg = reg;
}

/// @brief The multiplicative descrambler at eight bits per item and `degree <= 56`: a rolling 64-bit window of received
/// bits, from which each delay's byte is one shift and one mask, so the whole item is `tapCount` shifted XORs.
inline void descrambleBytes(ScramblerConfig& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out) noexcept {
    const std::uint8_t* delays = cfg.delays.data();
    const std::size_t   taps   = static_cast<std::size_t>(cfg.tapCount);
    const std::size_t   count  = std::min(in.size(), out.size());
    const BitOrder      order  = cfg.order;
    std::uint64_t       window = cfg.reg;

    for (std::size_t i = 0UZ; i < count; ++i) {
        const std::uint32_t source = streamField(in[i], 8U, order);
        window                     = (window << 8U) | static_cast<std::uint64_t>(source);

        std::uint32_t value = source;
        for (std::size_t tap = 0UZ; tap < taps; ++tap) {
            value ^= static_cast<std::uint32_t>(window >> delays[tap]) & 0xFFU;
        }
        out[i] = streamField(static_cast<std::uint8_t>(value), 8U, order);
    }

    cfg.reg = window & cfg.regMask;
}

/// @brief The multiplicative pair with the output inversion and the forced-transition rule, one bit at a time.
///
/// The counter is loop-carried at a depth of one bit, so this form has no stepped variant and is the reason the rule is
/// dispatched to rather than folded into the plain paths: a chain that leaves `invertOutput` false and `forceAfter`
/// zero never reaches it and pays nothing for it. Both modes run the same loop over the same transmitted stream `y`,
/// which is the block's output when scrambling and its input when descrambling, and that is the whole of their
/// difference here.
inline void scrambleMonitored(ScramblerConfig& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out) noexcept {
    const unsigned      width    = static_cast<unsigned>(cfg.bitsPerItem);
    const TapMask       taps     = cfg.taps;
    const std::uint64_t mask     = cfg.regMask;
    const BitOrder      order    = cfg.order;
    const bool          received = cfg.mode == ScramblerMode::MultiplicativeDescramble;
    const unsigned      invert   = cfg.invertOutput ? 1U : 0U;
    const std::uint32_t modulus  = cfg.forceAfter;
    const unsigned      shiftP   = cfg.monitorP == 0U ? 0U : static_cast<unsigned>(cfg.monitorP) - 1U;
    const unsigned      shiftQ   = cfg.monitorQ == 0U ? 0U : static_cast<unsigned>(cfg.monitorQ) - 1U;
    const std::size_t   count    = std::min(in.size(), out.size());
    std::uint64_t       reg      = cfg.reg;
    std::uint64_t       window   = cfg.monitorWindow;
    std::uint32_t       counter  = cfg.forceCounter;
    std::size_t         fired    = 0UZ;

    for (std::size_t i = 0UZ; i < count; ++i) {
        unsigned item = 0U;
        for (unsigned bit = 0U; bit < width; ++bit) {
            const unsigned position = order == BitOrder::MsbFirst ? width - 1U - bit : bit;
            const unsigned feedback = static_cast<unsigned>(std::popcount(reg & taps)) & 1U;
            const unsigned source   = (static_cast<unsigned>(in[i]) >> position) & 1U;
            const unsigned force    = modulus != 0U && counter == modulus - 1U ? 1U : 0U;
            const unsigned result   = invert ^ source ^ feedback ^ force;
            const unsigned sent     = received ? source : result;

            fired += force;
            if (modulus != 0U) {
                // the comparison reads the bits already transmitted, so both ends compute it from the same history
                const unsigned differs = static_cast<unsigned>((window >> shiftP) ^ (window >> shiftQ)) & 1U;
                counter                = differs != 0U ? 0U : (counter + 1U) % modulus;
            }
            window = (window << 1U) | static_cast<std::uint64_t>(sent);
            reg    = ((reg << 1U) | static_cast<std::uint64_t>(sent)) & mask;
            item |= result << position;
        }
        out[i] = static_cast<std::uint8_t>(item);
    }

    cfg.reg           = reg;
    cfg.monitorWindow = window;
    cfg.forceCounter  = counter;
    cfg.nForcedTransitions += fired;
}

/// @brief The definition itself, one bit at a time through one feedback per bit, for every mode. The fast forms are
/// measured against this and agree with it bit for bit and in state.
inline void scrambleReference(ScramblerConfig& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out) noexcept {
    const unsigned      width = static_cast<unsigned>(cfg.bitsPerItem);
    const TapMask       taps  = cfg.taps;
    const std::uint64_t mask  = cfg.regMask;
    const ScramblerMode mode  = cfg.mode;
    const std::size_t   count = std::min(in.size(), out.size());
    std::uint64_t       reg   = cfg.reg;

    for (std::size_t i = 0UZ; i < count; ++i) {
        unsigned item = 0U;
        for (unsigned bit = 0U; bit < width; ++bit) {
            const unsigned position = cfg.order == BitOrder::MsbFirst ? width - 1U - bit : bit;
            const unsigned feedback = static_cast<unsigned>(std::popcount(reg & taps)) & 1U;
            const unsigned source   = (static_cast<unsigned>(in[i]) >> position) & 1U;
            const unsigned result   = source ^ feedback;
            const unsigned fed      = mode == ScramblerMode::Additive ? feedback : (mode == ScramblerMode::MultiplicativeScramble ? result : source);

            reg = ((reg << 1U) | static_cast<std::uint64_t>(fed)) & mask;
            item |= result << position;
        }
        out[i] = static_cast<std::uint8_t>(item);
    }

    cfg.reg = reg;
    advancePhase(cfg, count);
}

} // namespace detail

/**
 * @brief Scrambles @p in into @p out item for item, advancing the register.
 *
 * The spans are the same length; a caller that supplies more of either converts only the items both
 * hold. Each item carries `cfg.bitsPerItem` significant bits, unused high bits of an input item are
 * masked rather than rejected, and the output's unused high bits are zero. No allocation, no branch
 * on data, no division and no throw, and there is no tail, flush or epilogue because the block is
 * one output item per input item and holds nothing back.
 */
inline void scramble(ScramblerConfig& cfg, std::span<const std::uint8_t> in, std::span<std::uint8_t> out) noexcept {
    const bool monitored = cfg.invertOutput || cfg.forceAfter != 0U;
    switch (cfg.mode) {
    case ScramblerMode::Additive: detail::scrambleAdditive(cfg, in, out); return;
    case ScramblerMode::MultiplicativeScramble:
        if (monitored) {
            detail::scrambleMonitored(cfg, in, out);
        } else {
            detail::scrambleMultiplicative(cfg, in, out);
        }
        return;
    default:
        if (monitored) {
            detail::scrambleMonitored(cfg, in, out);
        } else if (cfg.bitsPerItem == 8U && cfg.degree <= 56U) {
            detail::descrambleBytes(cfg, in, out);
        } else {
            detail::descrambleItems(cfg, in, out);
        }
        return;
    }
}

} // namespace gr::digital

#endif // GNURADIO_ALGORITHM_SCRAMBLER_HPP
