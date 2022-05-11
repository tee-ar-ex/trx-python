#include <gtest/gtest.h>
#include "../src/trx.h"
#include <typeinfo>

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
	std::tuple<std::string, int, std::string> exp1("mean_fa", 1, "float64");
	output = _split_ext_with_dimensionality(fn1);
	EXPECT_TRUE(output == exp1);

	const std::string fn2 = "mean_fa.5.int32";
	std::tuple<std::string, int, std::string> exp2("mean_fa", 5, "int32");
	output = _split_ext_with_dimensionality(fn2);
	// std::cout << std::get<0>(output) << " TEST " << std::get<1>(output) << " " << std::get<2>(output) << std::endl;
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
	Matrix<uint64_t, 5, 1> offsets{uint64_t(0), uint64_t(1), uint64_t(2), uint64_t(3), uint64_t(4)};
	Matrix<uint32_t, 5, 1> lengths(trxmmap::_compute_lengths(offsets, 4));
	Matrix<uint32_t, 5, 1> result{uint32_t(1), uint32_t(1), uint32_t(1), uint32_t(1), uint32_t(0)};

	EXPECT_EQ(lengths, result);

	Matrix<uint64_t, 5, 1> offsets2{uint64_t(0), uint64_t(1), uint64_t(0), uint64_t(3), uint64_t(4)};
	Matrix<uint32_t, 5, 1> lengths2(trxmmap::_compute_lengths(offsets2, 4));
	Matrix<uint32_t, 5, 1> result2{uint32_t(1), uint32_t(3), uint32_t(0), uint32_t(1), uint32_t(0)};

	EXPECT_EQ(lengths2, result2);

	Matrix<uint64_t, 4, 1> offsets3{uint64_t(0), uint64_t(1), uint64_t(2), uint64_t(3)};
	Matrix<uint32_t, 4, 1> lengths3(trxmmap::_compute_lengths(offsets3, 4));
	Matrix<uint32_t, 4, 1> result3{uint32_t(1), uint32_t(1), uint32_t(1), uint32_t(1)};

	EXPECT_EQ(lengths3, result3);

	Matrix<uint64_t, 1, 1> offsets4(uint64_t(4));
	Matrix<uint32_t, 1, 1> lengths4(trxmmap::_compute_lengths(offsets4, 2));
	Matrix<uint32_t, 1, 1> result4(uint32_t(2));

	EXPECT_EQ(lengths4, result4);

	Matrix<uint64_t, 0, 0> offsets5;
	Matrix<uint32_t, 1, 1> lengths5(trxmmap::_compute_lengths(offsets5, 2));
	Matrix<uint32_t, 1, 1> result5(uint32_t(0));

	EXPECT_EQ(lengths5, result5);
}

TEST(TrxFileMemmap, __is_dtype_valid)
{
	std::string ext = "bit";
	EXPECT_TRUE(_is_dtype_valid(ext));

	std::string ext2 = "int16";
	EXPECT_TRUE(_is_dtype_valid(ext2));

	std::string ext3 = "float32";
	EXPECT_TRUE(_is_dtype_valid(ext3));

	std::string ext4 = "uint8";
	EXPECT_TRUE(_is_dtype_valid(ext4));

	std::string ext5 = "txt";
	EXPECT_FALSE(_is_dtype_valid(ext5));
}

TEST(TrxFileMemmap, __dichotomic_search)
{
	Matrix<int, 1, 5> m{0, 1, 2, 3, 4};
	int result = trxmmap::_dichotomic_search(m);
	EXPECT_EQ(result, 4);

	Matrix<int, 1, 5> m2{0, 1, 0, 3, 4};
	int result2 = trxmmap::_dichotomic_search(m2);
	EXPECT_EQ(result2, 1);

	Matrix<int, 1, 5> m3{0, 1, 2, 0, 4};
	int result3 = trxmmap::_dichotomic_search(m3);
	EXPECT_EQ(result3, 2);

	Matrix<int, 1, 5> m4{0, 1, 2, 3, 4};
	int result4 = trxmmap::_dichotomic_search(m4, 1, 2);
	EXPECT_EQ(result4, 2);

	Matrix<int, 1, 5> m5{0, 1, 2, 3, 4};
	int result5 = trxmmap::_dichotomic_search(m5, 3, 3);
	EXPECT_EQ(result5, 3);

	Matrix<int, 1, 5> m6{0, 0, 0, 0, 0};
	int result6 = trxmmap::_dichotomic_search(m6, 3, 3);
	EXPECT_EQ(result6, -1);
}

TEST(TrxFileMemmap, __create_memmap)
{

	char *dirname;
	char t[] = "/tmp/trx_XXXXXX";
	dirname = mkdtemp(t);

	std::string path(dirname);
	path += "/offsets.int16";

	std::tuple<int, int> shape = std::make_tuple(3, 4);

	// Test 1: create file and allocate space assert that correct data is filled
	mio::shared_mmap_sink empty_mmap = trxmmap::_create_memmap(path, shape);
	Map<Matrix<half, 3, 4>> expected_m(reinterpret_cast<half *>(empty_mmap.data()));
	Matrix<half, 3, 4> zero_filled{{half(0), half(0), half(0), half(0)},
				       {half(0), half(0), half(0), half(0)},
				       {half(0), half(0), half(0), half(0)}};

	EXPECT_EQ(expected_m, zero_filled);

	// Test 2: edit data and compare mapped values with new mmap
	for (int i = 0; i < expected_m.size(); i++)
	{
		expected_m(i) = half(i);
	}

	mio::shared_mmap_sink filled_mmap = trxmmap::_create_memmap(path, shape);
	Map<Matrix<half, 3, 4>> real_m(reinterpret_cast<half *>(filled_mmap.data()), std::get<0>(shape), std::get<1>(shape));

	EXPECT_EQ(expected_m, real_m);
}

TEST(TrxFileMemmap, load_header)
{
	std::string path = "../../tests/data/small.trx";
	int *errorp;
	zip_t *zf = zip_open(path.c_str(), 0, errorp);
	json root = trxmmap::load_header(zf);

	// expected output
	json expected;

	expected["DIMENSIONS"] = {117, 151, 115};
	expected["NB_STREAMLINES"] = 1000;
	expected["NB_VERTICES"] = 33886;
	expected["VOXEL_TO_RASMM"] = {{-1.25, 0.0, 0.0, 72.5},
				      {0.0, 1.25, 0.0, -109.75},
				      {0.0, 0.0, 1.25, -64.5},
				      {0.0, 0.0, 0.0, 1.0}};

	EXPECT_EQ(root, expected);

	std::string expected_str = "{\"DIMENSIONS\":[117,151,115],\"NB_STREAMLINES\":1000,\"NB_VERTICES\":33886,\"VOXEL_TO_RASMM\":[[-1.25,0.0,0.0,72.5],[0.0,1.25,0.0,-109.75],[0.0,0.0,1.25,-64.5],[0.0,0.0,0.0,1.0]]}";

	EXPECT_EQ(root.dump(), expected_str);

	free(zf);
}

// TEST(TrxFileMemmap, _load)
// {
// }

// TEST(TrxFileMemmap, _load_zip)
// {
// }

// TEST(TrxFileMemmap, initialize_empty_trx)
// {
// }

// TEST(TrxFileMemmap, _create_trx_from_pointer)
// {
// }

TEST(TrxFileMemmap, load_zip)
{
	trxmmap::TrxFile<half> *trx = trxmmap::load_from_zip<half>("../../tests/data/small.trx");
	EXPECT_GT(trx->streamlines->_data.size(), 0);
}

TEST(TrxFileMemmap, TrxFile)
{
	// trxmmap::TrxFile<half> *trx = new TrxFile<half>();

	// // expected header
	// json expected;

	// expected["DIMENSIONS"] = {1, 1, 1};
	// expected["NB_STREAMLINES"] = 0;
	// expected["NB_VERTICES"] = 0;
	// expected["VOXEL_TO_RASMM"] = {{1.0, 0.0, 0.0, 0.0},
	// 			      {0.0, 1.0, 0.0, 0.0},
	// 			      {0.0, 0.0, 1.0, 0.0},
	// 			      {0.0, 0.0, 0.0, 1.0}};

	// EXPECT_EQ(trx->header, expected);

	// std::string path = "../../tests/data/small.trx";
	// int *errorp;
	// zip_t *zf = zip_open(path.c_str(), 0, errorp);
	// json root = trxmmap::load_header(zf);
	// TrxFile<half> *root_init = new TrxFile<half>();
	// root_init->header = root;

	// // TODO: test for now..

	// trxmmap::TrxFile<half> *trx_init = new TrxFile<half>(33886, 1000, root_init);
	// json init_as;

	// init_as["DIMENSIONS"] = {117, 151, 115};
	// init_as["NB_STREAMLINES"] = 1000;
	// init_as["NB_VERTICES"] = 33886;
	// init_as["VOXEL_TO_RASMM"] = {{-1.25, 0.0, 0.0, 72.5},
	// 			     {0.0, 1.25, 0.0, -109.75},
	// 			     {0.0, 0.0, 1.25, -64.5},
	// 			     {0.0, 0.0, 0.0, 1.0}};

	// EXPECT_EQ(root_init->header, init_as);
	// EXPECT_EQ(trx_init->streamlines->_data.size(), 33886 * 3);
	// EXPECT_EQ(trx_init->streamlines->_offsets.size(), 1000);
	// EXPECT_EQ(trx_init->streamlines->_lengths.size(), 1000);
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
