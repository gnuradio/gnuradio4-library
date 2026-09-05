#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <numbers>
#include <print>
#include <span>
#include <tuple>
#include <vector>

#include <gnuradio-4.0/algorithm/fec/Convolutional.hpp>

/*
 * The convention these tests hold the code to is the impulse response: driven with a single one,
 * the encoder must emit each generator polynomial's bits least significant first. That identity is
 * the definition of a generator polynomial rather than a restatement of the encoder, so it is what
 * pins the spelling of a published code onto this implementation without consulting another one.
 *
 * Everything after it follows from termination. A frame ends in the zero state, so the traceback
 * starts there and the path it reads out is the maximum-likelihood path over the whole frame. The
 * clean round trip proves the trellis inverts the encoder; the injected errors prove the reported
 * distance is an account of the received word rather than a promise about the answer; and the
 * bit-error rate against the published curves proves the search finds the best path and not merely
 * a good one, since a decoder settling for a neighboring path would lose the coding gain those
 * curves measure.
 */
namespace {

using gr::fec::ConvolutionalCode;
using gr::fec::convolutionalEncode;
using gr::fec::convolutionalEncodedBits;
using gr::fec::convolutionalInfoBits;
using gr::fec::ViterbiDecoder;
using gr::fec::ViterbiResult;

/// A code from its constraint length and its generators, which the tests spell in octal.
[[nodiscard]] ConvolutionalCode codeOf(std::size_t length, std::initializer_list<std::uint32_t> generators) {
    ConvolutionalCode code;
    boost::ut::expect(code.configure(length, std::span<const std::uint32_t>(generators.begin(), generators.size())));
    return code;
}

/// The classic constraint length 7 rate 1/2 code, the one the published curves describe.
[[nodiscard]] ConvolutionalCode classicCode() { return codeOf(7UZ, {0171U, 0133U}); }

std::uint64_t rng = 0x2545F4914F6CDD1DULL;

[[nodiscard]] std::uint64_t next() {
    rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return rng >> 17U;
}

[[nodiscard]] std::vector<std::uint8_t> randomBits(std::size_t count) {
    std::vector<std::uint8_t> bits(count);
    for (std::size_t i = 0UZ; i < count; ++i) {
        bits[i] = static_cast<std::uint8_t>(next() & 1ULL);
    }
    return bits;
}

/// One terminated frame of @p info under @p code.
[[nodiscard]] std::vector<std::uint8_t> encodeFrame(const ConvolutionalCode& code, std::span<const std::uint8_t> info) {
    std::vector<std::uint8_t> coded(convolutionalEncodedBits(code, info.size()));
    boost::ut::expect(boost::ut::eq(convolutionalEncode(code, info, coded), coded.size()));
    return coded;
}

/// The bits of @p coded as soft values, a one becoming +1 and a zero -1.
[[nodiscard]] std::vector<float> asSoft(std::span<const std::uint8_t> coded) {
    std::vector<float> values(coded.size());
    for (std::size_t i = 0UZ; i < coded.size(); ++i) {
        values[i] = ((coded[i] & 1U) != 0U) ? 1.0F : -1.0F;
    }
    return values;
}

/// Positions at which two equal-length bit words disagree.
[[nodiscard]] std::size_t disagreements(std::span<const std::uint8_t> a, std::span<const std::uint8_t> b) {
    std::size_t count = 0UZ;
    for (std::size_t i = 0UZ; i < a.size() && i < b.size(); ++i) {
        count += ((a[i] & 1U) != (b[i] & 1U)) ? 1UZ : 0UZ;
    }
    return count;
}

/// A code as a table row: the constraint length and the generators, octal as they are published.
struct Spelling {
    std::size_t                    length;
    std::size_t                    count;
    std::array<std::uint32_t, 3UZ> generators;
};

[[nodiscard]] ConvolutionalCode codeOf(const Spelling& spelling) {
    ConvolutionalCode code;
    boost::ut::expect(code.configure(spelling.length, std::span<const std::uint32_t>(spelling.generators).first(spelling.count)));
    return code;
}

/// The standard rate 1/2 and rate 1/3 codes at each constraint length the round trip sweeps.
constexpr std::array<Spelling, 8UZ> kSpellings{{
    {3UZ, 2UZ, {07U, 05U, 0U}},
    {3UZ, 3UZ, {07U, 05U, 03U}},
    {5UZ, 2UZ, {023U, 035U, 0U}},
    {5UZ, 3UZ, {025U, 033U, 037U}},
    {7UZ, 2UZ, {0171U, 0133U, 0U}},
    {7UZ, 3UZ, {0133U, 0171U, 0165U}},
    {9UZ, 2UZ, {0753U, 0561U, 0U}},
    {9UZ, 3UZ, {0557U, 0663U, 0711U}},
}};

/**
 * A seeded Gaussian source: a xorshift generator feeding the Box-Muller transform.
 *
 * The channel a bit-error rate is measured on has to be the same channel on every run and on every
 * machine, so the noise is built here from an integer state rather than drawn from a library whose
 * sequence is free to change under it.
 */
class Awgn {
public:
    explicit Awgn(std::uint64_t seed) noexcept : _state(seed | 1ULL) {}

    /// A standard normal deviate. Box-Muller makes two at a time and the second is kept.
    [[nodiscard]] double normal() noexcept {
        if (_hasSpare) {
            _hasSpare = false;
            return _spare;
        }
        const double radius = std::sqrt(-2.0 * std::log(uniform()));
        const double angle  = 2.0 * std::numbers::pi * uniform();
        _spare              = radius * std::sin(angle);
        _hasSpare           = true;
        return radius * std::cos(angle);
    }

private:
    /// A value in (0, 1], the open end held away from zero so the logarithm stays finite.
    [[nodiscard]] double uniform() noexcept {
        _state ^= _state << 13U;
        _state ^= _state >> 7U;
        _state ^= _state << 17U;
        return static_cast<double>((_state >> 11U) + 1ULL) * 0x1.0p-53;
    }

    std::uint64_t _state;
    bool          _hasSpare = false;
    double        _spare    = 0.0;
};

/// What one pass over the channel cost, in information bits each of the three receivers got wrong.
struct Trial {
    std::size_t bits    = 0UZ;
    std::size_t hard    = 0UZ;
    std::size_t soft    = 0UZ;
    std::size_t uncoded = 0UZ;

    [[nodiscard]] double rate(std::size_t errors) const { return static_cast<double>(errors) / static_cast<double>(bits); }
};

/**
 * Send @p frames terminated frames through an additive white Gaussian noise channel at @p ebN0Db
 * and count what each receiver got wrong.
 *
 * The signaling is antipodal with unit bit energy, so a coded bit arrives as +1 or -1 plus noise
 * of standard deviation sqrt(N0/2). A code spends one bit's energy over n coded bits, which is why
 * the coded noise carries the rate and the uncoded reference — the same information bit sent with
 * the whole of its energy — does not. Termination adds K-1 steps to a frame, a fraction of a
 * percent at the frame length used here, and is not charged against the rate, because the
 * published curves describe the unterminated code.
 *
 * The hard and the soft decoder see one noise realization between them, so their error counts are
 * comparable rather than merely similar. Both the information and the noise come from @p seed
 * alone, so a trial's numbers are the same whatever ran before it in the file.
 */
[[nodiscard]] Trial runChannel(const ConvolutionalCode& code, std::size_t frames, std::size_t infoBits, double ebN0Db, std::uint64_t seed) {
    const double rate         = 1.0 / static_cast<double>(code.polynomialCount);
    const double ebN0         = std::pow(10.0, ebN0Db / 10.0);
    const double sigma        = 1.0 / std::sqrt(2.0 * rate * ebN0);
    const double uncodedSigma = 1.0 / std::sqrt(2.0 * ebN0);

    const std::size_t codedBits = convolutionalEncodedBits(code, infoBits);

    Awgn                      noise(seed);
    std::uint64_t             source = seed;
    ViterbiDecoder            decoder(code);
    std::vector<std::uint8_t> info(infoBits);
    std::vector<std::uint8_t> coded(codedBits);
    std::vector<std::uint8_t> hardWord(codedBits);
    std::vector<float>        softWord(codedBits);
    std::vector<std::uint8_t> fromHard(infoBits);
    std::vector<std::uint8_t> fromSoft(infoBits);

    Trial trial;
    for (std::size_t frame = 0UZ; frame < frames; ++frame) {
        for (std::size_t i = 0UZ; i < infoBits; ++i) {
            source  = source * 6364136223846793005ULL + 1442695040888963407ULL;
            info[i] = static_cast<std::uint8_t>((source >> 33U) & 1ULL);
        }
        boost::ut::expect(boost::ut::eq(convolutionalEncode(code, info, coded), codedBits));

        for (std::size_t i = 0UZ; i < codedBits; ++i) {
            const double value = (((coded[i] & 1U) != 0U) ? 1.0 : -1.0) + sigma * noise.normal();
            softWord[i]        = static_cast<float>(value);
            hardWord[i]        = static_cast<std::uint8_t>((value > 0.0) ? 1U : 0U);
        }

        std::ignore = decoder.decodeHard(hardWord, fromHard);
        std::ignore = decoder.decodeSoft(softWord, fromSoft);

        for (std::size_t i = 0UZ; i < infoBits; ++i) {
            trial.hard += (fromHard[i] != info[i]) ? 1UZ : 0UZ;
            trial.soft += (fromSoft[i] != info[i]) ? 1UZ : 0UZ;

            const double bare  = (((info[i] & 1U) != 0U) ? 1.0 : -1.0) + uncodedSigma * noise.normal();
            const auto   slice = static_cast<std::uint8_t>((bare > 0.0) ? 1U : 0U);
            trial.uncoded += (slice != info[i]) ? 1UZ : 0UZ;
        }
        trial.bits += infoBits;
    }
    return trial;
}

/*
 * The operating points the bit-error-rate case measures at, and the published curves for the
 * (171, 133) constraint length 7 rate 1/2 code that they are read against. The values are the
 * classic hard- and soft-decision curves for that code, the pair a coding text plots against
 * Eb/N0.
 *
 * The envelope is a factor of four either way. A seeded run measures one realization rather than
 * an ensemble, its errors arrive in bursts of a handful of bits each so the estimator's spread at
 * this frame count is wide, and a value read off a plotted figure is itself good to about a factor
 * of two. What a factor of four still catches is every way a decoder can be wrong: a trellis
 * searching the wrong paths, a metric with a sign error or a traceback off by a step all land
 * orders of magnitude away rather than a factor of four.
 */
constexpr double kBerEnvelope = 4.0;

constexpr double      kHardEbN0Db   = 5.0;
constexpr double      kHardCurveBer = 3.0e-4;
constexpr double      kSoftEbN0Db   = 3.0;
constexpr double      kSoftCurveBer = 2.0e-4;
constexpr std::size_t kBerFrames    = 250UZ;
constexpr std::size_t kBerFrameBits = 2000UZ;

} // namespace

const boost::ut::suite<"convolutional"> convolutionalTests = [] {
    using namespace boost::ut;

    "a code is what the family accepts and nothing else"_test = [] {
        constexpr std::array<std::uint32_t, 2UZ> classic{0171U, 0133U};
        constexpr std::array<std::uint32_t, 1UZ> single{0171U};
        constexpr std::array<std::uint32_t, 5UZ> five{0171U, 0133U, 0165U, 0161U, 0147U};
        constexpr std::array<std::uint32_t, 2UZ> zero{0171U, 0U};
        constexpr std::array<std::uint32_t, 2UZ> wide{0171U, 0333U};
        constexpr std::array<std::uint32_t, 2UZ> twice{0171U, 0171U};
        constexpr std::array<std::uint32_t, 2UZ> shortest{07U, 05U};

        ConvolutionalCode code;
        expect(code.configure(7UZ, classic)) << "the classic constraint length 7 pair is a code";
        expect(code.configured());
        expect(eq(code.states(), 64UZ)) << "one state per pattern of the K-1 remembered bits";

        expect(!code.configure(2UZ, classic)) << "below three there is no code";
        expect(!code.configured()) << "a refused configure leaves nothing half built";
        expect(!code.configure(10UZ, classic)) << "above nine the family stops";
        expect(!code.configure(7UZ, single)) << "rate 1/1 is not a rate the family carries";
        expect(!code.configure(7UZ, five)) << "rate 1/5 is past the family";
        expect(!code.configure(7UZ, zero)) << "a zero generator emits nothing the code can use";
        expect(!code.configure(7UZ, wide)) << "a generator wider than K taps a delay the register has not got";
        expect(!code.configure(7UZ, twice)) << "the same generator twice is not two of them";

        expect(code.configure(3UZ, shortest)) << "the hand-computable pair";
        expect(eq(code.states(), 4UZ));
    };

    "a terminated frame's length is the code's own arithmetic"_test = [] {
        const ConvolutionalCode code = classicCode();
        expect(eq(convolutionalEncodedBits(code, 100UZ), (100UZ + 6UZ) * 2UZ));
        expect(eq(convolutionalInfoBits(code, (100UZ + 6UZ) * 2UZ), 100UZ)) << "the inverse of the same arithmetic";
        expect(eq(convolutionalInfoBits(code, 0UZ), 0UZ)) << "an empty word is not a frame";
        expect(eq(convolutionalInfoBits(code, 13UZ), 0UZ)) << "a length that is not a whole number of steps is not a frame";
        expect(eq(convolutionalInfoBits(code, 12UZ), 0UZ)) << "a frame of nothing but the tail carries no information";
        expect(eq(convolutionalInfoBits(code, 14UZ), 1UZ)) << "the shortest frame there is";
    };

    "the impulse response of a code is its generator polynomials"_test = [] {
        for (const Spelling& spelling : {kSpellings[4UZ], kSpellings[0UZ]}) { // (171,133) and (7,5)
            const ConvolutionalCode code = codeOf(spelling);

            std::vector<std::uint8_t> impulse(spelling.length, 0U);
            impulse[0UZ]                          = 1U;
            const std::vector<std::uint8_t> coded = encodeFrame(code, impulse);

            const std::size_t steps = impulse.size() + spelling.length - 1UZ;
            for (std::size_t step = 0UZ; step < steps; ++step) {
                for (std::size_t j = 0UZ; j < spelling.count; ++j) {
                    // Past the register's width the impulse has left it and the output is zero.
                    const std::uint8_t expected = (step < spelling.length) ? static_cast<std::uint8_t>((spelling.generators[j] >> step) & 1U) : std::uint8_t{0U};
                    expect(eq(coded[step * spelling.count + j], expected)) << "K" << spelling.length << "step" << step << "polynomial" << j;
                }
            }
        }
    };

    "a clean frame round-trips exactly through both decoders"_test = [] {
        for (const Spelling& spelling : kSpellings) {
            const ConvolutionalCode         code  = codeOf(spelling);
            const std::vector<std::uint8_t> info  = randomBits(120UZ);
            const std::vector<std::uint8_t> coded = encodeFrame(code, info);
            expect(eq(coded.size(), (120UZ + spelling.length - 1UZ) * spelling.count)) << "K" << spelling.length;

            ViterbiDecoder            decoder(code);
            std::vector<std::uint8_t> decoded(info.size());

            const ViterbiResult hard = decoder.decodeHard(coded, decoded);
            expect(std::ranges::equal(decoded, info)) << "K" << spelling.length << "rate 1 /" << spelling.count << "hard";
            expect(eq(hard.distance, 0UZ)) << "a clean word lies on a path of the code";
            expect(eq(static_cast<std::size_t>(hard.metric), 0UZ)) << "and cost that path nothing";

            std::ranges::fill(decoded, std::uint8_t{0U});
            const ViterbiResult soft = decoder.decodeSoft(asSoft(coded), decoded);
            expect(std::ranges::equal(decoded, info)) << "K" << spelling.length << "rate 1 /" << spelling.count << "soft";
            expect(eq(soft.distance, 0UZ));
            expect(eq(static_cast<std::size_t>(soft.metric), coded.size())) << "every coded bit correlated with the winning path";
        }
    };

    // Scattered errors inside the code's budget are corrected, and the reported distance is exactly
    // the count injected.
    "scattered errors are corrected and counted exactly"_test = [] {
        const ConvolutionalCode         code  = classicCode();
        const std::vector<std::uint8_t> info  = randomBits(200UZ);
        const std::vector<std::uint8_t> clean = encodeFrame(code, info);

        constexpr std::array<std::size_t, 4UZ> positions{7UZ, 101UZ, 205UZ, 309UZ}; // spaced far past the code's memory
        std::vector<std::uint8_t>              damaged = clean;
        for (const std::size_t p : positions) {
            damaged[p] = static_cast<std::uint8_t>(damaged[p] ^ 1U);
        }

        ViterbiDecoder            decoder(code);
        std::vector<std::uint8_t> decoded(info.size());

        const ViterbiResult hard = decoder.decodeHard(damaged, decoded);
        expect(std::ranges::equal(decoded, info)) << "the information survives four scattered errors";
        expect(eq(hard.distance, positions.size())) << "the distance is the account of what the channel did";

        std::ranges::fill(decoded, std::uint8_t{0U});
        const ViterbiResult soft = decoder.decodeSoft(asSoft(damaged), decoded);
        expect(std::ranges::equal(decoded, info)) << "the soft decoder reads the same flips as full-confidence ones";
        expect(eq(soft.distance, positions.size()));
    };

    // A burst past the code's budget may cost information, and what is asserted is that the reported
    // distance is a true account of the word in hand.
    "a burst past the code's budget still reports the word it decoded against"_test = [] {
        const ConvolutionalCode         code  = classicCode();
        const std::vector<std::uint8_t> info  = randomBits(200UZ);
        std::vector<std::uint8_t>       burst = encodeFrame(code, info);

        constexpr std::size_t start  = 120UZ;
        constexpr std::size_t length = 8UZ; // past half the code's free distance of ten
        for (std::size_t i = 0UZ; i < length; ++i) {
            burst[start + i] = static_cast<std::uint8_t>(burst[start + i] ^ 1U);
        }

        ViterbiDecoder            decoder(code);
        std::vector<std::uint8_t> decoded(info.size());
        const ViterbiResult       result = decoder.decodeHard(burst, decoded);

        // Re-encoding what came out and counting the disagreement reaches the decoder's own
        // statement without the decoder: the distance is the received word against the winning
        // path, whether or not that path is the transmitted one.
        const std::vector<std::uint8_t> reencoded = encodeFrame(code, decoded);
        expect(eq(result.distance, disagreements(burst, reencoded))) << "the distance accounts for the word actually in hand";
        expect(eq(static_cast<std::size_t>(result.metric), result.distance)) << "the hard path metric is that distance";
        expect(le(result.distance, length)) << "the winning path is at least as close as the transmitted one";
    };

    "spans that do not describe a frame of the code are answered with nothing"_test = [] {
        const ConvolutionalCode         code  = classicCode();
        const std::vector<std::uint8_t> info  = randomBits(40UZ);
        const std::vector<std::uint8_t> coded = encodeFrame(code, info);

        ViterbiDecoder            decoder(code);
        std::vector<std::uint8_t> decoded(info.size());
        std::vector<std::uint8_t> wrongSize(info.size() + 1UZ);

        expect(eq(decoder.decodeHard(coded, wrongSize).distance, 0UZ)) << "an information span the frame cannot fill";
        expect(eq(decoder.decodeHard(std::span(coded).first(coded.size() - 1UZ), decoded).distance, 0UZ)) << "a word that is not a whole number of steps";
        expect(eq(ViterbiDecoder{}.decodeHard(coded, decoded).distance, 0UZ)) << "a decoder that was never given a code";
    };

    // The two metrics on one noise realization, and what they are both worth. The
    // operating point is one where hard decisions still pay: a rate 1/2 code spends half the energy
    // per transmitted bit, and below about four decibels the two decibels a slicer throws away are
    // more than the code wins back, so the hard-decision and uncoded curves meet there.
    "the soft decoder beats the hard decoder on the same noise"_test = [] {
        const Trial trial = runChannel(classicCode(), 40UZ, 2000UZ, 5.0, 0xA5A5F00DDEADBEEFULL);

        expect(lt(trial.soft, trial.hard)) << "the soft metric keeps the confidence a slicer throws away";
        expect(lt(trial.hard * 4UZ, trial.uncoded)) << "hard decisions are still far better than no code at all";
        expect(lt(trial.soft * 4UZ, trial.uncoded));
        expect(gt(trial.uncoded, 0UZ)) << "the operating point is one at which an uncoded link does fail";
    };

    // The bit-error rate of the classic code against its published curves; the constants above state
    // the operating points and the envelope it is read within.
    "the classic code meets its published curves at the stated operating points"_test = [] {
        const ConvolutionalCode code = classicCode();

        const Trial  hardTrial = runChannel(code, kBerFrames, kBerFrameBits, kHardEbN0Db, 0x0123456789ABCDEFULL);
        const double hardBer   = hardTrial.rate(hardTrial.hard);
        expect(gt(hardBer, kHardCurveBer / kBerEnvelope)) << "a hard-decision rate far under the curve would mean the channel is not the stated one";
        expect(lt(hardBer, kHardCurveBer * kBerEnvelope)) << "the hard-decision rate against the published curve";

        const Trial  softTrial = runChannel(code, kBerFrames, kBerFrameBits, kSoftEbN0Db, 0x0FEDCBA987654321ULL);
        const double softBer   = softTrial.rate(softTrial.soft);
        expect(gt(softBer, kSoftCurveBer / kBerEnvelope)) << "a soft-decision rate far under the curve would mean the channel is not the stated one";
        expect(lt(softBer, kSoftCurveBer * kBerEnvelope)) << "the soft-decision rate against the published curve";

        // The measured numbers are printed rather than left for a failure to reveal.
        std::println(stderr, "convolutional: hard decision at {} dB, {} errors in {} bits, BER {:.3e} against curve {:.1e}", kHardEbN0Db, hardTrial.hard, hardTrial.bits, hardBer, kHardCurveBer);
        std::println(stderr, "convolutional: soft decision at {} dB, {} errors in {} bits, BER {:.3e} against curve {:.1e}", kSoftEbN0Db, softTrial.soft, softTrial.bits, softBer, kSoftCurveBer);
    };
};

const boost::ut::suite<"open termination"> openTerminationTests = [] {
    using namespace boost::ut;
    using gr::fec::ConvTermination;

    "the open length arithmetic, and the initial-state refusal"_test = [] {
        const ConvolutionalCode code = classicCode();
        expect(eq(convolutionalInfoBits(code, 24UZ, ConvTermination::Open), 12UZ)) << "every step of an open record carries an information bit";
        expect(eq(convolutionalInfoBits(code, 24UZ), 6UZ)) << "the terminated reading is unchanged";
        expect(eq(convolutionalInfoBits(code, 2UZ, ConvTermination::Open), 1UZ)) << "one step is a record";
        expect(eq(convolutionalInfoBits(code, 3UZ, ConvTermination::Open), 0UZ)) << "a length that is not whole steps is not a record";

        ViterbiDecoder decoder;
        expect(!decoder.configure(code, ConvTermination::Open, 64U)) << "64 names a state a 64-state code does not have";
        expect(decoder.configure(code, ConvTermination::Open, 63U)) << "63 is the last state there is";
    };

    // Section 4.2's mode: a record cut out of a continuous stream, neither end's state known.
    "open records tile a stream that a terminated decode reads whole"_test = [] {
        const ConvolutionalCode         code  = classicCode();
        constexpr std::size_t           kTail = 6UZ; // K - 1
        const std::vector<std::uint8_t> info  = randomBits(600UZ);
        const std::vector<std::uint8_t> coded = encodeFrame(code, info);
        const std::size_t               steps = coded.size() / 2UZ; // 606: 600 information steps and the tail

        ViterbiDecoder            whole(code);
        std::vector<std::uint8_t> reference(info.size());
        std::ignore = whole.decodeHard(coded, reference);
        expect(std::ranges::equal(reference, info));

        // three open records, each overlapping the next by K-1 trellis steps; a record's last K-1
        // information bits lack the future that would resolve them and are discarded, the next
        // record re-decoding those positions from its own span — and the third record's discard
        // is the terminated frame's tail
        ViterbiDecoder open;
        expect(open.configure(code, ConvTermination::Open));

        struct Slice {
            std::size_t begin;
            std::size_t end;
        };
        constexpr Slice kSlices[]{{0UZ, 250UZ}, {244UZ, 460UZ}, {454UZ, 606UZ}};

        std::vector<std::uint8_t> tiled;
        for (const Slice& slice : kSlices) {
            const std::size_t         stepsHere = slice.end - slice.begin;
            std::vector<std::uint8_t> out(stepsHere);
            const ViterbiResult       result = open.decodeHard(std::span(coded).subspan(slice.begin * 2UZ, stepsHere * 2UZ), out);
            expect(eq(result.distance, 0UZ)) << "a clean slice lies on a path of the code from some initial state";
            tiled.insert(tiled.end(), out.begin(), out.begin() + static_cast<std::ptrdiff_t>(stepsHere - kTail));
        }
        expect(eq(tiled.size(), info.size()));
        expect(std::ranges::equal(tiled, info)) << "the tiles concatenate to the terminated decode, bit for bit";
        expect(eq(steps, 606UZ));
    };

    "an open decode converges to the told-the-state decode within 5K steps"_test = [] {
        const ConvolutionalCode         code  = classicCode();
        const std::vector<std::uint8_t> info  = randomBits(400UZ);
        const std::vector<std::uint8_t> coded = encodeFrame(code, info);

        // a mid-stream slice of 300 steps starting at step 100, with a few scattered errors
        constexpr std::size_t     kFrom  = 100UZ;
        constexpr std::size_t     kSteps = 300UZ;
        std::vector<std::uint8_t> slice(coded.begin() + kFrom * 2UZ, coded.begin() + (kFrom + kSteps) * 2UZ);
        for (const std::size_t at : {31UZ, 157UZ, 240UZ, 388UZ, 511UZ}) {
            slice[at] ^= std::uint8_t{1U};
        }

        // the state the encoder was in at step kFrom, replayed from the information bits
        std::uint32_t state = 0U;
        for (std::size_t t = 0UZ; t < kFrom; ++t) {
            state = gr::fec::convolutionalNextState(code, state, info[t] & 1U);
        }

        ViterbiDecoder blind;
        ViterbiDecoder told;
        expect(blind.configure(code, ConvTermination::Open));
        expect(told.configure(code, ConvTermination::Open, state));

        std::vector<std::uint8_t> fromBlind(kSteps);
        std::vector<std::uint8_t> fromTold(kSteps);
        std::ignore = blind.decodeHard(slice, fromBlind);
        std::ignore = told.decodeHard(slice, fromTold);

        constexpr std::size_t kConverged = 35UZ; // 5K at K = 7
        expect(std::ranges::equal(std::span(fromBlind).subspan(kConverged), std::span(fromTold).subspan(kConverged))) << "from step 5K on, start knowledge no longer matters";
        expect(std::ranges::equal(std::span(fromTold).first(kSteps - 6UZ), std::span(info).subspan(kFrom, kSteps - 6UZ))) << "and the told decode recovers the data through the errors";
    };
};

int main() { /* tests are automatically registered and run */ }
