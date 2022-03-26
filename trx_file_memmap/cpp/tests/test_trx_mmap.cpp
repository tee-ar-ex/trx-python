#include <gtest/gtest.h>
#include "../src/trx.h"

using namespace Eigen;
using namespace trxmmap;

// TODO: Test null filenames. Maybe use MatrixBase instead of ArrayBase
// TODO: try to update test case to use GTest parameterization
TEST(TrxFileMemmap, __generate_filename_from_data)
{
	std::string filename = "mean_fa.bit";
	std::string output_fn;

	Array<int16_t, 5, 4> arr1;
	std::string exp_1 = "mean_fa.4.int16";

	output_fn = _generate_filename_from_data(arr1, filename);
	EXPECT_STREQ(output_fn.c_str(), exp_1.c_str());
	output_fn.clear();

	Array<double, 5, 4> arr2;
	std::string exp_2 = "mean_fa.4.float64";

	output_fn = _generate_filename_from_data(arr2, filename);
	EXPECT_STREQ(output_fn.c_str(), exp_2.c_str());
	output_fn.clear();

	Array<double, 5, 1> arr3;
	std::string exp_3 = "mean_fa.float64";

	output_fn = _generate_filename_from_data(arr3, filename);
	EXPECT_STREQ(output_fn.c_str(), exp_3.c_str());
	output_fn.clear();
}

TEST(TrxFileMemmap, __split_ext_with_dimensionality)
{
	std::tuple<std::string, int, std::string> output;
	const std::string fn1 = "mean_fa.float64";
	std::tuple<std::string, int, std::string> exp1("mean_fa", 1, ".float64");
	output = _split_ext_with_dimensionality(fn1);
	EXPECT_TRUE(output == exp1);

	const std::string fn2 = "mean_fa.5.int32";
	std::tuple<std::string, int, std::string> exp2("mean_fa", 5, ".int32");
	output = _split_ext_with_dimensionality(fn2);
	// std::cout << std::get<0>(output) << " " << std::get<1>(output) << " " << std::get<2>(output) << std::endl;
	EXPECT_TRUE(output == exp2);

	const std::string fn3 = "mean_fa";
	EXPECT_THROW({
		try
		{
			output = _split_ext_with_dimensionality(fn3);
		}
		catch (const std::invalid_argument &e)
		{
			EXPECT_STREQ("Invalid filename", e.what());
			throw;
		}
	},
		     std::invalid_argument);

	const std::string fn4 = "mean_fa.5.4.int32";
	EXPECT_THROW({
		try
		{
			output = _split_ext_with_dimensionality(fn4);
		}
		catch (const std::invalid_argument &e)
		{
			EXPECT_STREQ("Invalid filename", e.what());
			throw;
		}
	},
		     std::invalid_argument);

	const std::string fn5 = "mean_fa.fa";
	EXPECT_THROW({
		try
		{
			output = _split_ext_with_dimensionality(fn5);
		}
		catch (const std::invalid_argument &e)
		{
			EXPECT_STREQ("Unsupported file extension", e.what());
			throw;
		}
	},
		     std::invalid_argument);
}

TEST(TrxFileMemmap, __compute_lengths)
{
	Matrix<half, 1, 5> offsets{half(0), half(1), half(2), half(3), half(4)};
	Matrix<u_int32_t, 1, 5> lengths(trxmmap::_compute_lengths(offsets, 4));
	Matrix<u_int32_t, 1, 5> result{u_int32_t(1), u_int32_t(1), u_int32_t(1), u_int32_t(1), u_int32_t(0)};

	EXPECT_EQ(lengths, result);

	Matrix<half, 1, 5> offsets2{half(0), half(1), half(0), half(3), half(4)};
	Matrix<u_int32_t, 1, 5> lengths2(trxmmap::_compute_lengths(offsets2, 4));
	Matrix<u_int32_t, 1, 5> result2{u_int32_t(1), u_int32_t(3), u_int32_t(0), u_int32_t(1), u_int32_t(0)};

	EXPECT_EQ(lengths2, result2);

	Matrix<half, 1, 4> offsets3{half(0), half(1), half(2), half(3)};
	Matrix<u_int32_t, 1, 4> lengths3(trxmmap::_compute_lengths(offsets3, 4));
	Matrix<u_int32_t, 1, 4> result3{u_int32_t(1), u_int32_t(1), u_int32_t(1), u_int32_t(1)};

	EXPECT_EQ(lengths3, result3);

	Matrix<half, 1, 1> offsets4(half(4));
	Matrix<u_int32_t, 1, 1> lengths4(trxmmap::_compute_lengths(offsets4, 2));
	Matrix<u_int32_t, 1, 1> result4(u_int32_t(2));

	EXPECT_EQ(lengths4, result4);

	Matrix<half, 0, 0> offsets5;
	Matrix<u_int32_t, 1, 1> lengths5(trxmmap::_compute_lengths(offsets5, 2));
	Matrix<u_int32_t, 1, 1> result5(u_int32_t(0));

	EXPECT_EQ(lengths5, result5);
}

TEST(TrxFileMemmap, __is_dtype_valid)
{
	std::string ext = ".bit";
	EXPECT_TRUE(_is_dtype_valid(ext));

	std::string ext2 = ".int16";
	EXPECT_TRUE(_is_dtype_valid(ext2));

	std::string ext3 = ".float32";
	EXPECT_TRUE(_is_dtype_valid(ext3));

	std::string ext4 = ".ushort";
	EXPECT_TRUE(_is_dtype_valid(ext4));

	std::string ext5 = ".txt";
	EXPECT_FALSE(_is_dtype_valid(ext5));
}

// TEST(TrxFileMemmap, __dichotomic_search)
// {
// }

// TEST(TrxFileMemmap, __create_memmap)
// {
// }

// TEST(TrxFileMemmap, _load)
// {
// }

// TEST(TrxFileMemmap, _load_zip)
// {
// }

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
