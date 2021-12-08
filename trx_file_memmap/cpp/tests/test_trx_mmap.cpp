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
			std::cout << "hello" << std::endl;
			EXPECT_STREQ("Invalid filename", e.what());
			throw;
		}
	},
		     std::invalid_argument);
}

TEST(TrxFileMemmap, __compute_lengths)
{
}

TEST(TrxFileMemmap, __is_dtype_valid)
{
}

TEST(TrxFileMemmap, __dichotomic_search)
{
}

TEST(TrxFileMemmap, __create_memmap)
{
}

TEST(TrxFileMemmap, _load)
{
}

TEST(TrxFileMemmap, _load_zip)
{
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
