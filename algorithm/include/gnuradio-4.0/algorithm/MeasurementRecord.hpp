#ifndef GNURADIO_ALGORITHM_MEASUREMENT_RECORD_HPP
#define GNURADIO_ALGORITHM_MEASUREMENT_RECORD_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <gnuradio-4.0/DataSet.hpp>

namespace gr::measurement {

/// @brief One named quantity in a scalar measurement record.
struct ScalarChannel {
    std::string_view name;
    std::string_view quantity;
    std::string_view unit;
    float            value;
};

/**
 * @brief The record a measurement sink publishes: one value per named quantity, with the keys the tier's consumers read.
 *
 * A measurement is a record, and a record has to say what it is of and when it was taken. `sample_rate` and
 * `sample_start` are what let a consumer place the reading on the stream it came from — `sample_start` being the
 * first input sample of the window that produced it, not the moment it happened to be emitted. Everything a
 * particular sink wants to state beyond that goes in `extra`, which is copied to every channel so a consumer
 * reading one signal does not have to find another to learn the conditions.
 *
 * The axis is the measurement index rather than a physical quantity: these records carry one point each, and a
 * consumer stacking them in time reads `sample_start` for the abscissa. A block that measures in symbols rather
 * than samples has no rate to state and passes a non-finite one: the key is then absent, which says so, instead of
 * a zero that a consumer would divide by.
 */
[[nodiscard]] inline DataSet<float> makeScalarRecord(std::span<const ScalarChannel> channels, float sampleRate, std::uint64_t sampleStart, property_map extra = {}) {
    DataSet<float>    ds;
    const std::size_t n = channels.size();

    ds.extents = {1};
    ds.layout  = gr::LayoutRight{};

    ds.axis_names = {"Measurement"};
    ds.axis_units = {"index"};
    ds.axis_values.resize(1UZ);
    ds.axis_values[0UZ] = {0.f};

    ds.signal_names.reserve(n);
    ds.signal_quantities.reserve(n);
    ds.signal_units.reserve(n);
    ds.signal_values.reserve(n);
    for (const ScalarChannel& channel : channels) {
        ds.signal_names.emplace_back(channel.name);
        ds.signal_quantities.emplace_back(channel.quantity);
        ds.signal_units.emplace_back(channel.unit);
        ds.signal_values.push_back(channel.value);
    }
    ds.signal_ranges.resize(n);

    if (std::isfinite(sampleRate)) {
        extra.insert_or_assign(property_map::key_type("sample_rate"), pmt::Value(sampleRate));
    }
    extra.insert_or_assign(property_map::key_type("sample_start"), pmt::Value(sampleStart));
    // Every channel carries the same conditions. All but the last take a copy and the last takes the map itself:
    // `assign(n, std::move(extra))` would bind to `assign(size_type, const T&)` and copy n times regardless.
    ds.meta_information.resize(n);
    for (std::size_t i = 0UZ; i + 1UZ < n; ++i) {
        ds.meta_information[i] = extra;
    }
    if (n != 0UZ) {
        ds.meta_information.back() = std::move(extra);
    }
    ds.timing_events.resize(n);
    ds.timestamp = 0;
    return ds;
}

} // namespace gr::measurement

#endif // GNURADIO_ALGORITHM_MEASUREMENT_RECORD_HPP
