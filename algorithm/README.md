# gnuradio-algorithm

A math support library with math primitives to be used by different block types.

Included algorithms:

- Fast Fourier Transform
- uncertainty-aware values and propagated mathematical operations

Include uncertainty propagation from its library-owned path:

```cpp
#include <gnuradio-4.0/algorithm/math/UncertainValue.hpp>
```

Formatting support is opt-in:

```cpp
#include <gnuradio-4.0/algorithm/math/UncertainValueFormatter.hpp>
```

The former `<gnuradio-4.0/meta/UncertainValue.hpp>` path remains available as
a compatibility forwarding header installed by `gnuradio4-library` and is
deprecated for new code.
