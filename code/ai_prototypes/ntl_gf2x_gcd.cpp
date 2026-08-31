#include <NTL/GF2X.h>
#include <NTL/BasicThreadPool.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <psapi.h>
#endif

namespace {

std::uint64_t read_size()
{
    std::array<unsigned char, 8> bytes{};
    std::cin.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!std::cin) {
        throw std::runtime_error("Could not read polynomial length");
    }

    std::uint64_t result = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        result |= std::uint64_t{bytes[index]} << (8 * index);
    }
    return result;
}

std::vector<unsigned char> read_polynomial()
{
    const auto size = read_size();
    if (size > static_cast<std::uint64_t>((std::numeric_limits<long>::max)())) {
        throw std::runtime_error("Polynomial is too large for NTL");
    }

    std::vector<unsigned char> result(static_cast<std::size_t>(size));
    std::cin.read(reinterpret_cast<char*>(result.data()), result.size());
    if (!std::cin) {
        throw std::runtime_error("Could not read polynomial coefficients");
    }
    return result;
}

std::size_t peak_working_set()
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(
            GetCurrentProcess(),
            &counters,
            sizeof(counters))) {
        return counters.PeakWorkingSetSize;
    }
#endif
    return 0;
}

}

int main(int argc, char** argv)
{
    try {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif

        const long thread_count = argc > 1 ? std::stol(argv[1]) : 1;
        if (thread_count < 1) {
            throw std::runtime_error("Thread count must be positive");
        }
        NTL::SetNumThreads(thread_count);

        const auto left_bytes = read_polynomial();
        const auto right_bytes = read_polynomial();

        const auto conversion_started = std::chrono::steady_clock::now();
        NTL::GF2X left;
        NTL::GF2X right;
        NTL::GF2XFromBytes(
            left,
            left_bytes.data(),
            static_cast<long>(left_bytes.size()));
        NTL::GF2XFromBytes(
            right,
            right_bytes.data(),
            static_cast<long>(right_bytes.size()));
        const auto conversion_finished = std::chrono::steady_clock::now();

        NTL::GF2X result;
        NTL::GCD(result, left, right);
        const auto gcd_finished = std::chrono::steady_clock::now();

        const std::chrono::duration<double> conversion_time =
            conversion_finished - conversion_started;
        const std::chrono::duration<double> gcd_time =
            gcd_finished - conversion_finished;

        std::cout << std::setprecision(9)
                  << "{\"degree\":" << NTL::deg(result)
                  << ",\"conversion_seconds\":" << conversion_time.count()
                  << ",\"gcd_seconds\":" << gcd_time.count()
                  << ",\"peak_working_set_bytes\":" << peak_working_set()
                  << ",\"word_bits\":" << NTL_BITS_PER_LONG
                  << ",\"threads\":" << NTL::AvailableThreads()
#ifdef NTL_GF2X_LIB
                  << ",\"external_gf2x\":true"
#else
                  << ",\"external_gf2x\":false"
#endif
                  << "}\n";
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
