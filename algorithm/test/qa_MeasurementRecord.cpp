#include <boost/ut.hpp>

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <string>

#include <gnuradio-4.0/algorithm/MeasurementRecord.hpp>

using namespace boost::ut;
using gr::measurement::makeScalarRecord;
using gr::measurement::ScalarChannel;

namespace {

/// The channels a two-quantity sink would publish; named here once so every test reads the same record shape.
constexpr std::array<ScalarChannel, 2UZ> kChannels{
    ScalarChannel{"rms", "voltage", "V", 1.25f},
    ScalarChannel{"peak", "voltage", "V", -3.5f},
};

[[nodiscard]] bool hasKey(const gr::property_map& map, std::string_view key) { return map.find(gr::property_map::key_type(key)) != map.end(); }

template<typename T>
[[nodiscard]] T valueOf(const gr::property_map& map, std::string_view key) {
    const auto entry = map.find(gr::property_map::key_type(key));
    return entry == map.end() ? T{} : entry->second.value_or(T{});
}

} // namespace

const boost::ut::suite<"MeasurementRecord"> measurementRecordTests = [] {
    "a record is one point per channel on the measurement axis"_test = [] {
        const gr::DataSet<float> ds = makeScalarRecord(std::span<const ScalarChannel>(kChannels), 48000.f, 1024ULL);

        expect(eq(ds.extents.size(), 1UZ));
        expect(eq(ds.extents[0UZ], 1));
        expect(eq(ds.axis_names.size(), 1UZ));
        expect(eq(ds.axis_names[0UZ], std::string("Measurement")));
        expect(eq(ds.axis_units[0UZ], std::string("index")));
        expect(eq(ds.axis_values.size(), 1UZ));
        expect(eq(ds.axis_values[0UZ].size(), 1UZ));
        expect(eq(ds.axis_values[0UZ][0UZ], 0.f));

        // one signal per channel, and one value per signal because the extent is one
        expect(eq(ds.signal_names.size(), kChannels.size()));
        expect(eq(ds.signal_values.size(), kChannels.size()));
        expect(eq(ds.signal_ranges.size(), kChannels.size()));
        expect(eq(ds.meta_information.size(), kChannels.size()));
        expect(eq(ds.timing_events.size(), kChannels.size()));
        expect(eq(ds.timestamp, 0));
    };

    "each channel keeps its own name, quantity, unit and value"_test = [] {
        const gr::DataSet<float> ds = makeScalarRecord(std::span<const ScalarChannel>(kChannels), 48000.f, 1024ULL);

        for (std::size_t i = 0UZ; i < kChannels.size(); ++i) {
            expect(eq(ds.signal_names[i], std::string(kChannels[i].name)));
            expect(eq(ds.signal_quantities[i], std::string(kChannels[i].quantity)));
            expect(eq(ds.signal_units[i], std::string(kChannels[i].unit)));
            expect(eq(ds.signal_values[i], kChannels[i].value));
        }
    };

    // The keys are the contract: a consumer reads these names, so a rename is a break and the test says so.
    "a finite rate states both keys"_test = [] {
        const gr::DataSet<float> ds = makeScalarRecord(std::span<const ScalarChannel>(kChannels), 48000.f, 1024ULL);

        for (const gr::property_map& meta : ds.meta_information) {
            expect(eq(meta.size(), 2UZ)) << "sample_rate and sample_start, and nothing else was asked for";
            expect(that % hasKey(meta, "sample_rate"));
            expect(eq(valueOf<float>(meta, "sample_rate"), 48000.f));
            expect(that % hasKey(meta, "sample_start"));
            expect(eq(valueOf<std::uint64_t>(meta, "sample_start"), 1024ULL));
        }
    };

    // A block measuring in symbols has no rate to state: the key is absent, which a consumer can see, rather than a
    // zero it would divide by.
    "a non-finite rate leaves the rate key out"_test = [] {
        for (const float rate : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()}) {
            const gr::DataSet<float> ds = makeScalarRecord(std::span<const ScalarChannel>(kChannels), rate, 7ULL);

            for (const gr::property_map& meta : ds.meta_information) {
                expect(eq(meta.size(), 1UZ));
                expect(that % !hasKey(meta, "sample_rate"));
                expect(eq(valueOf<std::uint64_t>(meta, "sample_start"), 7ULL));
            }
        }
    };

    // sample_start is stated for every record, including the first window of a stream, so a consumer never has to
    // guess whether a missing key means zero or means unknown.
    "sample_start is always stated, zero included"_test = [] {
        const gr::DataSet<float> ds = makeScalarRecord(std::span<const ScalarChannel>(kChannels), 48000.f, 0ULL);

        expect(that % hasKey(ds.meta_information[0UZ], "sample_start"));
        expect(eq(valueOf<std::uint64_t>(ds.meta_information[0UZ], "sample_start"), 0ULL));
    };

    // The caller's own conditions reach every channel, so a consumer reading one signal does not have to find
    // another to learn them. The reserved keys win a collision: they are what places the reading on the stream.
    "extra reaches every channel and the reserved keys stand"_test = [] {
        gr::property_map extra;
        extra["window"]       = std::string("hann");
        extra["n_averages"]   = std::uint64_t{64ULL};
        extra["sample_start"] = std::uint64_t{5ULL};

        const gr::DataSet<float> ds = makeScalarRecord(std::span<const ScalarChannel>(kChannels), 48000.f, 4096ULL, std::move(extra));

        expect(eq(ds.meta_information.size(), kChannels.size()));
        for (const gr::property_map& meta : ds.meta_information) {
            expect(eq(meta.size(), 4UZ));
            expect(eq(valueOf<std::string>(meta, "window"), std::string("hann")));
            expect(eq(valueOf<std::uint64_t>(meta, "n_averages"), 64ULL));
            expect(eq(valueOf<float>(meta, "sample_rate"), 48000.f));
            expect(eq(valueOf<std::uint64_t>(meta, "sample_start"), 4096ULL)) << "the caller's sample_start must not displace the record's own";
        }
    };

    "a single-channel record is the same record"_test = [] {
        constexpr std::array<ScalarChannel, 1UZ> one{ScalarChannel{"power", "power", "dBm", -20.f}};

        const gr::DataSet<float> ds = makeScalarRecord(std::span<const ScalarChannel>(one), 1.f, 3ULL);

        expect(eq(ds.signal_values.size(), 1UZ));
        expect(eq(ds.meta_information.size(), 1UZ));
        expect(eq(valueOf<std::uint64_t>(ds.meta_information[0UZ], "sample_start"), 3ULL));
    };

    // No channels is a degenerate but reachable call — a sink configured with nothing selected. The axis still
    // stands and the per-channel vectors are simply empty; nothing indexes past the end.
    "no channels leaves the axis standing and every channel vector empty"_test = [] {
        const gr::DataSet<float> ds = makeScalarRecord(std::span<const ScalarChannel>{}, 48000.f, 11ULL);

        expect(eq(ds.extents[0UZ], 1));
        expect(eq(ds.axis_names[0UZ], std::string("Measurement")));
        expect(eq(ds.signal_values.size(), 0UZ));
        expect(eq(ds.meta_information.size(), 0UZ));
        expect(eq(ds.timing_events.size(), 0UZ));
    };
};

int main() { /* not needed for UT */ }
