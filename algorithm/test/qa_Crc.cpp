#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <format>
#include <numeric>
#include <print>
#include <random>
#include <ranges>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include <gnuradio-4.0/algorithm/digital/Crc.hpp>

namespace {

using gr::digital::Crc;
using gr::digital::detail::reverseBits;

using Clock = std::chrono::steady_clock;

struct Parameters {
    const char*   name;
    std::uint8_t  width;
    std::uint64_t polynomial;
    std::uint64_t initialValue;
    std::uint64_t finalXor;
    bool          inputReflected;
    bool          resultReflected;
    std::uint64_t check;
};

/// The RevEng catalog rows exercised here, with their published check values.
constexpr Parameters kCatalog[] = {
    {"CRC-3/GSM", 3U, 0x3ULL, 0x0ULL, 0x7ULL, false, false, 0x4ULL},                                  //
    {"CRC-5/USB", 5U, 0x05ULL, 0x1FULL, 0x1FULL, true, true, 0x19ULL},                                //
    {"CRC-6/ITU", 6U, 0x03ULL, 0x00ULL, 0x00ULL, true, true, 0x06ULL},                                //
    {"CRC-8/SMBUS", 8U, 0x07ULL, 0x00ULL, 0x00ULL, false, false, 0xF4ULL},                            //
    {"CRC-8/ROHC", 8U, 0x07ULL, 0xFFULL, 0x00ULL, true, true, 0xD0ULL},                               //
    {"CRC-12/UMTS", 12U, 0x80FULL, 0x000ULL, 0x000ULL, false, true, 0xDAFULL},                        //
    {"CRC-16/ARC", 16U, 0x8005ULL, 0x0000ULL, 0x0000ULL, true, true, 0xBB3DULL},                      //
    {"CRC-16/IBM-3740", 16U, 0x1021ULL, 0xFFFFULL, 0x0000ULL, false, false, 0x29B1ULL},               //
    {"CRC-16/KERMIT", 16U, 0x1021ULL, 0x0000ULL, 0x0000ULL, true, true, 0x2189ULL},                   //
    {"CRC-16/IBM-SDLC", 16U, 0x1021ULL, 0xFFFFULL, 0xFFFFULL, true, true, 0x906EULL},                 //
    {"CRC-16/GENIBUS", 16U, 0x1021ULL, 0xFFFFULL, 0xFFFFULL, false, false, 0xD64EULL},                //
    {"CRC-24/OPENPGP", 24U, 0x864CFBULL, 0xB704CEULL, 0x000000ULL, false, false, 0x21CF02ULL},        //
    {"CRC-32/ISO-HDLC", 32U, 0x04C11DB7ULL, 0xFFFFFFFFULL, 0xFFFFFFFFULL, true, true, 0xCBF43926ULL}, //
    {"CRC-32/BZIP2", 32U, 0x04C11DB7ULL, 0xFFFFFFFFULL, 0xFFFFFFFFULL, false, false, 0xFC891918ULL},  //
    {"CRC-32/ISCSI", 32U, 0x1EDC6F41ULL, 0xFFFFFFFFULL, 0xFFFFFFFFULL, true, true, 0xE3069283ULL},    //
    {"CRC-32/MPEG-2", 32U, 0x04C11DB7ULL, 0xFFFFFFFFULL, 0x00000000ULL, false, false, 0x0376E6E7ULL}, //
    {"CRC-64/ECMA-182", 64U, 0x42F0E1EBA9EA3693ULL, 0ULL, 0ULL, false, false, 0x6C40DF5F0B497347ULL}, //
    {"CRC-64/XZ", 64U, 0x42F0E1EBA9EA3693ULL, ~0ULL, ~0ULL, true, true, 0x995DC9BBDF1939FAULL},       //
};

[[nodiscard]] Crc kernelOf(const Parameters& p) { return Crc(p.width, p.polynomial, p.initialValue, p.finalXor, p.inputReflected, p.resultReflected); }

[[nodiscard]] constexpr std::uint64_t maskOf(std::uint8_t width) noexcept { return width == 64U ? ~0ULL : ((1ULL << width) - 1ULL); }

/// The definition, polynomial division over GF(2), as an independent reference implementation.
[[nodiscard]] std::uint64_t bitSerial(const Parameters& p, std::span<const std::uint8_t> message) {
    const std::uint64_t mask = maskOf(p.width);
    const std::uint64_t poly = p.polynomial & mask;
    std::uint64_t       reg  = p.initialValue & mask;
    for (const std::uint8_t raw : message) {
        const std::uint64_t byte = p.inputReflected ? reverseBits(raw, 8U) : raw;
        for (int k = 7; k >= 0; --k) {
            const std::uint64_t bit = (byte >> static_cast<unsigned>(k)) & 1ULL;
            const std::uint64_t msb = (reg >> (p.width - 1U)) & 1ULL;
            reg                     = (reg << 1U) & mask;
            if ((msb ^ bit) != 0ULL) {
                reg ^= poly;
            }
        }
    }
    return ((p.resultReflected ? reverseBits(reg, p.width) : reg) ^ (p.finalXor & mask)) & mask;
}

/// Implementation B: the MSB-first table with a per-byte input reflection, which is the form the
/// kernel deliberately does not use when the input is reflected. `width >= 8` only.
[[nodiscard]] std::uint64_t msbFirstTable(const Parameters& p, std::span<const std::uint8_t> message) {
    const std::uint64_t mask = maskOf(p.width);
    const std::uint64_t poly = p.polynomial & mask;
    const std::uint64_t top  = 1ULL << (p.width - 1U);

    std::array<std::uint64_t, 256> table{};
    for (std::size_t i = 0UZ; i < 256UZ; ++i) {
        std::uint64_t remainder = static_cast<std::uint64_t>(i) << (p.width - 8U);
        for (int step = 0; step < 8; ++step) {
            remainder = (remainder & top) != 0ULL ? (((remainder << 1U) ^ poly) & mask) : ((remainder << 1U) & mask);
        }
        table[i] = remainder;
    }

    std::uint64_t reg = p.initialValue & mask;
    for (const std::uint8_t raw : message) {
        const std::uint64_t byte = p.inputReflected ? reverseBits(raw, 8U) : raw;
        reg                      = ((reg << 8U) & mask) ^ table[((reg >> (p.width - 8U)) ^ byte) & 0xFFULL];
    }
    return ((p.resultReflected ? reverseBits(reg, p.width) : reg) ^ (p.finalXor & mask)) & mask;
}

[[nodiscard]] std::vector<std::uint8_t> checkMessage() { return {'1', '2', '3', '4', '5', '6', '7', '8', '9'}; }

[[nodiscard]] std::vector<std::uint8_t> appended(const Crc& crc, std::span<const std::uint8_t> message, bool bigEndian) {
    std::vector<std::uint8_t> out(message.begin(), message.end());
    const std::uint64_t       value = crc.compute(message);
    const std::size_t         bytes = crc.width() / 8U;
    for (std::size_t i = 0UZ; i < bytes; ++i) {
        const std::size_t shift = bigEndian ? (bytes - 1UZ - i) : i;
        out.push_back(static_cast<std::uint8_t>((value >> (8UZ * shift)) & 0xFFULL));
    }
    return out;
}

[[nodiscard]] std::vector<std::uint8_t> randomBytes(std::size_t count, std::uint64_t seed) {
    std::mt19937_64                              engine{seed};
    std::uniform_int_distribution<std::uint32_t> byte{0U, 255U};
    std::vector<std::uint8_t>                    out(count);
    for (std::uint8_t& value : out) {
        value = static_cast<std::uint8_t>(byte(engine));
    }
    return out;
}

} // namespace

const boost::ut::suite<"crc"> crcTests = [] {
    using namespace boost::ut;

    "all eighteen catalog check values, through the kernel and through the definition"_test = [] {
        const auto message = checkMessage();
        for (const Parameters& p : kCatalog) {
            const std::uint64_t fromKernel = kernelOf(p).compute(message);
            const std::uint64_t fromSerial = bitSerial(p, message);
            expect(eq(fromKernel, p.check)) << std::format("{}: kernel gave {:#x}, published {:#x}", p.name, fromKernel, p.check);
            expect(eq(fromSerial, p.check)) << std::format("{}: bit-serial gave {:#x}, published {:#x}", p.name, fromSerial, p.check);
            if (p.width >= 8U) {
                expect(eq(msbFirstTable(p, message), p.check)) << std::format("{}: MSB-first table with a reflected byte", p.name);
            }
        }
        expect(that % (kernelOf(kCatalog[0]).tableForm() == Crc::TableForm::LeftAligned)) << "CRC-3/GSM needs the left-aligned table, and the naive form gives 0x1 where 0x4 is published";
        expect(that % (kernelOf(kCatalog[12]).tableForm() == Crc::TableForm::Mirrored)) << "a reflected input runs the mirrored register, never a per-byte reflection";
        expect(that % (kernelOf(kCatalog[7]).tableForm() == Crc::TableForm::MsbFirst));
    };

    "the generated table's spot values"_test = [] {
        struct Row {
            std::uint8_t  width;
            std::uint64_t polynomial;
            std::uint64_t at1;
            std::uint64_t at2;
            std::uint64_t at128;
            std::uint64_t at255;
        };
        const std::array<Row, 3> rows{Row{32U, 0x04C11DB7ULL, 0x04C11DB7ULL, 0x09823B6EULL, 0x690CE0EEULL, 0xB1F740B4ULL}, //
            Row{16U, 0x1021ULL, 0x1021ULL, 0x2042ULL, 0x9188ULL, 0x1EF0ULL},                                               //
            Row{8U, 0x07ULL, 0x07ULL, 0x0EULL, 0x89ULL, 0xF3ULL}};

        for (const Row& row : rows) {
            const Crc  crc{row.width, row.polynomial};
            const auto table = crc.table();
            expect(that % (crc.tableForm() == Crc::TableForm::MsbFirst));
            expect(eq(table[1UZ], row.at1)) << std::format("w={} t[1] = {:#x}", row.width, table[1UZ]);
            expect(eq(table[2UZ], row.at2)) << std::format("w={} t[2] = {:#x}", row.width, table[2UZ]);
            expect(eq(table[128UZ], row.at128)) << std::format("w={} t[128] = {:#x}", row.width, table[128UZ]);
            expect(eq(table[255UZ], row.at255)) << std::format("w={} t[255] = {:#x}", row.width, table[255UZ]);
        }

        for (const Parameters& p : kCatalog) {
            expect(eq(kernelOf(p).table()[0UZ], 0ULL)) << std::format("{}: table[0] is zero, which is the statement that the table is linear", p.name);
        }
    };

    "initial_value is the catalog's unreflected seed"_test = [] {
        struct Row {
            const char*   name;
            std::uint64_t initialValue;
            std::uint64_t conformant;
            std::uint64_t verbatim;
        };
        const std::array<Row, 3> rows{Row{"CRC-16/RIELLO", 0xB2AAULL, 0x63D0ULL, 0xDB52ULL}, //
            Row{"CRC-16/TMS37157", 0x89ECULL, 0x26B1ULL, 0xD3CAULL},                         //
            Row{"CRC-16/A", 0xC6C6ULL, 0xBF05ULL, 0x1480ULL}};

        const auto message = checkMessage();
        for (const Row& row : rows) {
            const Crc conformant{16U, 0x1021ULL, row.initialValue, 0x0000ULL, true, true};
            expect(eq(conformant.compute(message), row.conformant)) << std::format("{}: {:#x} against the published {:#x}", row.name, conformant.compute(message), row.conformant);

            // Passing the reversed seed makes the kernel's mirrored register hold the catalog's INIT
            // verbatim, which is the state an implementation reaches by loading INIT unreversed.
            const Crc verbatimSeed{16U, 0x1021ULL, reverseBits(row.initialValue, 16U), 0x0000ULL, true, true};
            expect(eq(verbatimSeed.compute(message), row.verbatim)) << std::format("{}: a verbatim mirrored seed gives {:#x}", row.name, row.verbatim);
            expect(neq(row.conformant, row.verbatim)) << "a non-palindromic seed is where the two domains part";
        }

        for (const std::uint64_t palindrome : {0x0000ULL, 0xFFFFULL}) {
            expect(eq(palindrome, reverseBits(palindrome, 16U))) << "the seeds in common use reverse to themselves, which is why the two domains usually agree";
        }
    };

    "an over-wide polynomial, seed or final XOR means the masked one"_test = [] {
        const auto message = checkMessage();
        for (const Parameters& p : kCatalog) {
            const std::uint64_t mask = maskOf(p.width);
            if (p.width == 64U) {
                continue; // nothing is over-wide at the full width
            }
            const Crc wide{p.width, p.polynomial | (1ULL << p.width) | ~mask, p.initialValue | ~mask, p.finalXor | ~mask, p.inputReflected, p.resultReflected};
            expect(eq(wide.compute(message), p.check)) << std::format("{}: an over-wide parameter set", p.name);
            expect(eq(wide.polynomial(), p.polynomial & mask));
            expect(eq(wide.initialValue(), p.initialValue & mask));
            expect(eq(wide.finalXor(), p.finalXor & mask));
        }
    };

    "the table forms agree with the definition over random messages"_test = [] {
        std::mt19937_64                            engine{0xC12CULL};
        std::uniform_int_distribution<std::size_t> length{0UZ, 1024UZ};
        constexpr std::size_t                      kMessages = 4096UZ;

        std::vector<std::vector<std::uint8_t>> corpus;
        corpus.reserve(kMessages);
        for (std::size_t i = 0UZ; i < kMessages; ++i) {
            corpus.push_back(randomBytes(length(engine), 0x9E3779B9ULL + i));
        }

        for (const Parameters& p : kCatalog) {
            const Crc   crc            = kernelOf(p);
            std::size_t serialMismatch = 0UZ;
            std::size_t tableMismatch  = 0UZ;
            for (const auto& message : corpus) {
                const std::uint64_t reference = bitSerial(p, message);
                serialMismatch += crc.compute(message) != reference ? 1UZ : 0UZ;
                if (p.width >= 8U) {
                    tableMismatch += msbFirstTable(p, message) != reference ? 1UZ : 0UZ;
                }
            }
            expect(eq(serialMismatch, 0UZ)) << std::format("{}: {} kernel-against-definition mismatches", p.name, serialMismatch);
            expect(eq(tableMismatch, 0UZ)) << std::format("{}: {} MSB-first-table-against-definition mismatches", p.name, tableMismatch);
        }
    };

    "the regenerated reference vectors"_test = [] {
        struct Row {
            const char*   name;
            std::size_t   index;
            std::uint64_t empty;
            std::uint64_t zero;
            std::uint64_t ones;
            std::uint64_t check;
            std::uint64_t counted;
        };
        // The CRC-32 row over {0x00..0x0F} is CE CE E2 88 and the CRC-16/IBM-3740 row is 3B 37, two
        // widely published vectors. They are regenerated here from the polynomial rather than
        // transcribed.
        const std::array<Row, 7> rows{Row{"CRC-32/ISO-HDLC", 12UZ, 0x00000000ULL, 0xD202EF8DULL, 0xFF000000ULL, 0xCBF43926ULL, 0xCECEE288ULL}, //
            Row{"CRC-16/IBM-3740", 7UZ, 0xFFFFULL, 0xE1F0ULL, 0xFF00ULL, 0x29B1ULL, 0x3B37ULL},                                                //
            Row{"CRC-16/IBM-SDLC", 9UZ, 0x0000ULL, 0xF078ULL, 0xFF00ULL, 0x906EULL, 0x13E9ULL},                                                //
            Row{"CRC-8/SMBUS", 3UZ, 0x00ULL, 0x00ULL, 0xF3ULL, 0xF4ULL, 0x41ULL},                                                              //
            Row{"CRC-5/USB", 1UZ, 0x00ULL, 0x01ULL, 0x04ULL, 0x19ULL, 0x1EULL},                                                                //
            Row{"CRC-12/UMTS", 5UZ, 0x000ULL, 0x000ULL, 0x606ULL, 0xDAFULL, 0x880ULL},                                                         //
            Row{"CRC-64/XZ", 17UZ, 0x0ULL, 0x1FADA17364673F59ULL, 0xFF00000000000000ULL, 0x995DC9BBDF1939FAULL, 0x7A64E421B6985356ULL}};

        std::vector<std::uint8_t> counted(16UZ);
        std::iota(counted.begin(), counted.end(), static_cast<std::uint8_t>(0U));

        for (const Row& row : rows) {
            const Parameters& p   = kCatalog[row.index];
            const Crc         crc = kernelOf(p);
            expect(eq(std::string(p.name), std::string(row.name))) << "the row index names the parameter set it claims";

            expect(eq(crc.compute(std::span<const std::uint8_t>{}), row.empty)) << std::format("{} over the empty message", row.name);
            expect(eq(crc.compute(std::array<std::uint8_t, 1>{0x00U}), row.zero)) << std::format("{} over {{00}}", row.name);
            expect(eq(crc.compute(std::array<std::uint8_t, 1>{0xFFU}), row.ones)) << std::format("{} over {{FF}}", row.name);
            expect(eq(crc.compute(checkMessage()), row.check)) << std::format("{} over \"123456789\"", row.name);
            expect(eq(crc.compute(counted), row.counted)) << std::format("{} over {{00..0F}}", row.name);
        }
    };

    "the residues, and the append order that does not have one"_test = [] {
        struct Row {
            const char*   name;
            std::size_t   index;
            bool          bigEndian;
            std::uint64_t residue;
        };
        // A residue exists only when the append order matches the register's shift direction: a
        // reflected CRC cancels when its remainder is appended least significant byte first, an
        // unreflected one when it is appended most significant byte first.
        const std::array<Row, 5> rows{Row{"CRC-32/ISO-HDLC", 12UZ, false, 0x2144DF1CULL}, Row{"CRC-16/IBM-SDLC", 9UZ, false, 0x0F47ULL}, //
            Row{"CRC-16/IBM-3740", 7UZ, true, 0x0000ULL}, Row{"CRC-8/SMBUS", 3UZ, true, 0x00ULL},                                        //
            Row{"CRC-32/ISCSI", 14UZ, false, 0x48674BC7ULL}};

        for (const Row& row : rows) {
            const Crc crc = kernelOf(kCatalog[row.index]);
            for (const std::size_t length : {0UZ, 1UZ, 29UZ, 200UZ}) {
                const auto          message = randomBytes(length, 0x5150ULL + length);
                const std::uint64_t residue = crc.compute(appended(crc, message, row.bigEndian));
                expect(eq(residue, row.residue)) << std::format("{} {}-endian over {} bytes: {:#x} against {:#x}", row.name, row.bigEndian ? "big" : "little", length, residue, row.residue);
            }
        }

        // The other order leaves a message-dependent term, so its result varies with the message and
        // there is nothing to tabulate: the three wrong-order rows below are each one message's answer.
        struct Unmatched {
            const char* name;
            std::size_t index;
            bool        bigEndian;
        };
        const std::array<Unmatched, 3> unmatched{Unmatched{"CRC-32/ISO-HDLC", 12UZ, true}, Unmatched{"CRC-16/IBM-SDLC", 9UZ, true}, Unmatched{"CRC-16/IBM-3740", 7UZ, false}};
        for (const Unmatched& arm : unmatched) {
            const Crc                    crc = kernelOf(kCatalog[arm.index]);
            std::array<std::uint64_t, 3> seen{};
            std::size_t                  slot = 0UZ;
            for (const std::size_t length : {1UZ, 29UZ, 200UZ}) {
                seen[slot++] = crc.compute(appended(crc, randomBytes(length, 0x5150ULL + length), arm.bigEndian));
            }
            expect(that % (seen[0] != seen[1] || seen[1] != seen[2])) << std::format("{} appended {}-endian", arm.name, arm.bigEndian ? "big" : "little");
            std::println("no residue: {} appended {}-endian gives {:#x}, {:#x}, {:#x} over 1, 29 and 200 bytes", arm.name, arm.bigEndian ? "big" : "little", seen[0], seen[1], seen[2]);
        }

        const Crc sdlc = kernelOf(kCatalog[9UZ]);
        expect(eq(0x0F47ULL ^ 0xFFFFULL, 0xF0B8ULL)) << "the pre-final-XOR register of the SDLC residue is the HDLC good-FCS magic, which is what names the parameter set";
        expect(eq(sdlc.compute(appended(sdlc, randomBytes(64UZ, 3U), false)) ^ 0xFFFFULL, 0xF0B8ULL));
    };

    "the kernel has no state a second call could see"_test = [] {
        const Crc  crc     = kernelOf(kCatalog[12]);
        const auto message = checkMessage();
        const auto other   = randomBytes(97UZ, 11U);

        const std::uint64_t first = crc.compute(message);
        (void)crc.compute(other);
        expect(eq(crc.compute(message), first)) << "an interleaved call changes nothing";
        expect(eq(crc.compute(message), first));
    };

    "width is validated before anything derived from it is computed"_test = [] {
        expect(throws([] { (void)Crc(2U, 0x3ULL); })) << "width below 3 throws";
        expect(throws([] { (void)Crc(100U, 0x07ULL); })) << "width of 100 throws, and does not shift by 100 first";
        expect(throws([] { (void)Crc(65U, 0x07ULL); }));
        expect(throws([] { (void)Crc(0U, 0x07ULL); }));

        for (const std::uint8_t width : {std::uint8_t{3U}, std::uint8_t{5U}, std::uint8_t{6U}, std::uint8_t{12U}, std::uint8_t{64U}}) {
            expect(nothrow([width] { (void)Crc(width, 0x07ULL); })) << std::format("the kernel accepts width {}, which is about the register and not the message", width);
        }
    };

    "one kernel, eight threads"_test = [] {
        const Crc                  crc    = kernelOf(kCatalog[12]);
        constexpr std::size_t      kCalls = 2000UZ;
        std::array<std::size_t, 8> failures{};
        std::vector<std::jthread>  workers;
        for (std::size_t t = 0UZ; t < 8UZ; ++t) {
            workers.emplace_back([&crc, &failures, t] {
                const auto          buffer   = randomBytes(256UZ, 0xABCDULL + t);
                const std::uint64_t expected = crc.compute(buffer);
                for (std::size_t i = 0UZ; i < kCalls; ++i) {
                    failures[t] += crc.compute(buffer) != expected ? 1UZ : 0UZ;
                }
            });
        }
        workers.clear();
        expect(eq(std::accumulate(failures.begin(), failures.end(), 0UZ), 0UZ)) << "compute() is const, noexcept and mutates nothing";
    };

    // Chunk independence needs no case of its own: the blocks are record-oriented and the kernel is
    // stateless, which the case above asserts directly.
};

int main() { /* tests are automatically registered and run */ }
