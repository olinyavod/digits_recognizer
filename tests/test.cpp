#include <fstream>

#include <catch2/catch_test_macros.hpp>

#include "../app/include/neural_network.hpp"

template<typename T>
auto array_size(int size)
{
	return sizeof(T) * size + sizeof(std::uint32_t);
}

TEST_CASE("Test write singel value", "[write]")
{
	std::ofstream os("test.dat", std::ios::binary);
	
	REQUIRE(nn::write(os, 1) == sizeof(int));
	REQUIRE(nn::write(os, 1U) == sizeof(long));
	REQUIRE(nn::write(os, 1.0) == sizeof(double));
	REQUIRE(nn::write(os, 1.0f) == sizeof(float));
}

TEST_CASE("Test write array values", "[write]")
{
	const int SIZE = 784;
	std::ofstream os("test.dat", std::ios::binary);

	auto* ints = new int[SIZE];
	REQUIRE(nn::write(os, ints, SIZE) == array_size<int>(SIZE));
	delete[] ints;

	auto* longs = new int[SIZE];
	REQUIRE(nn::write(os, longs, SIZE) == array_size<long>(SIZE));
	delete[] longs;

	auto* doubles = new double[SIZE];
	REQUIRE(nn::write(os, doubles, SIZE) == array_size<double>(SIZE));
	delete[] doubles;

	auto* floats = new float[SIZE];
	REQUIRE(nn::write(os, floats, SIZE) == array_size<float>(SIZE));
	delete[] floats;	
}

TEST_CASE("Read single value", "[read]")
{
	std::srand(std::time(nullptr));
	const char* file_name = "test.dat";
	
	std::ofstream os(file_name, std::ios::binary);
		
	auto int_value = std::rand();
	auto int_size = nn::write(os, int_value);

	auto long_value = static_cast<long long>(std::rand());
	auto long_size = nn::write(os, long_value);

	auto float_value = static_cast<float>(std::rand()) / RAND_MAX;
	auto float_size = nn::write(os, float_value);

	auto double_value = static_cast<double>(std::rand()) / RAND_MAX;
	auto double_size = nn::write(os, double_value);
	
	os.close();

	std::ifstream is(file_name, std::ios::binary);

	int r_int_value;
	REQUIRE(nn::read(is, &r_int_value) == int_size);
	REQUIRE(r_int_value == int_value);

	long long r_long_value;
	REQUIRE(nn::read(is, &r_long_value) == long_size);
	REQUIRE(r_long_value == long_value);

	float r_float_value;
	REQUIRE(nn::read(is, &r_float_value) == float_size);
	REQUIRE(r_float_value == float_value);

	double r_double_value;
	REQUIRE(nn::read(is, &r_double_value) == double_size);
	REQUIRE(r_double_value == double_value);

	is.close();

	
}