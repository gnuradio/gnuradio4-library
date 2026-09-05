#include <boost/ut.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <thread>

#include <gnuradio-4.0/algorithm/MeasurementSlot.hpp>

using namespace boost::ut;
using gr::measurement::MeasurementSlot;

const boost::ut::suite<"MeasurementSlot"> measurementSlotTests = [] {
    "a fresh slot reads zeros and no coverage"_test = [] {
        MeasurementSlot<2UZ> slot{};

        const auto [values, filled] = slot.read();
        expect(eq(values[0], 0.));
        expect(eq(values[1], 0.));
        expect(eq(filled, 0ULL));
    };

    "a published set reads back whole"_test = [] {
        MeasurementSlot<3UZ> slot{};

        slot.publish({1.5, -2.5, 4.0}, 7ULL);
        const auto [values, filled] = slot.read();
        expect(eq(values[0], 1.5));
        expect(eq(values[1], -2.5));
        expect(eq(values[2], 4.0));
        expect(eq(filled, 7ULL));
    };

    "the newest publication wins"_test = [] {
        MeasurementSlot<1UZ> slot{};

        slot.publish({1.0}, 1ULL);
        slot.publish({2.0}, 2ULL);
        const auto [values, filled] = slot.read();
        expect(eq(values[0], 2.0));
        expect(eq(filled, 2ULL));
    };

    // The contract worth testing is that a reader never sees half of one publication beside half of another. Every
    // published set satisfies values[i] == values[0] + i and filled == values[0], so any read that breaks those
    // relations is torn. Reads taken before the first publication are the pristine slot rather than a torn one, and
    // filled == 0 is what distinguishes them.
    //
    // The writer runs until the reader has checked its quota rather than for a fixed number of publications, so the
    // coverage is the same on every machine instead of being whatever share of a fixed run the reader won. Nothing
    // here waits on a clock: both sides count.
    //
    // The quota is sized for a runner where the two threads share a core, not for a machine with a spare one. A
    // reader that finds the counter odd spins until the writer is scheduled again, which costs a whole quantum, so a
    // quota large enough to be interesting here could take far too long there. A wrong seqlock tears on a large
    // fraction of concurrent reads rather than on a rare one, so a few thousand checks is as conclusive as many more.
    "a concurrent reader never sees a torn set"_test = [] {
        constexpr std::size_t      kChecksWanted = 2000UZ;
        MeasurementSlot<4UZ>       slot{};
        std::atomic<bool>          readerSatisfied{false};
        std::atomic<std::uint64_t> published{0ULL};
        std::size_t                checked = 0UZ;
        std::size_t                torn    = 0UZ;

        std::thread writer([&] {
            std::uint64_t k = 0ULL;
            while (!readerSatisfied.load(std::memory_order_acquire)) {
                ++k;
                const double base = static_cast<double>(k);
                slot.publish({base, base + 1., base + 2., base + 3.}, k);
            }
            published.store(k, std::memory_order_release);
        });

        while (checked < kChecksWanted) {
            const auto [values, filled] = slot.read();
            if (filled == 0ULL) {
                continue; // the slot as constructed, not a publication
            }
            ++checked;
            if (!((values[1] == values[0] + 1.) && (values[2] == values[0] + 2.) && (values[3] == values[0] + 3.) && (static_cast<double>(filled) == values[0]))) {
                ++torn;
            }
        }
        readerSatisfied.store(true, std::memory_order_release);
        writer.join();

        expect(eq(torn, 0UZ));
        expect(eq(checked, kChecksWanted));
        expect(gt(published.load(), 0ULL));

        // the last publication is whole and is the newest one
        const auto [values, filled] = slot.read();
        expect(eq(filled, published.load()));
        expect(eq(values[0], static_cast<double>(published.load())));
    };
};

int main() { /* not needed for UT */ }
