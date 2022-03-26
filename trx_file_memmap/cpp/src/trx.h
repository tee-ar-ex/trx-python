#ifndef TRX_H // include guard
#define TRX_H

#include <iostream>
#include <zip.h>
#include <string.h>
#include <json/json.h>
#include <algorithm>
#include <variant>
#include <math.h>
#include <Eigen/Core>
#include <filesystem>

#include <mio/mmap.hpp>
#include <mio/shared_mmap.hpp>

using namespace Eigen;

namespace trxmmap
{

	const std::vector<std::string> dtypes({"bit", "ushort", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"});

	typedef std::map<std::string, std::variant<int, MatrixXf, RowVectorXi, std::string, double>> Dict;

	struct ArraySequence
	{
		// TODO: half precision for now. maybe fix or ask later.
		Map<Matrix<half, Dynamic, Dynamic>> _data;
		Map<Matrix<uint16_t, Dynamic, 1>> _offset;
		std::vector<uint32_t> _lengths;
		mio::shared_mmap_sink mmap_pos;
		mio::shared_mmap_sink mmap_off;

		ArraySequence() : _data(NULL, 1, 1), _offset(NULL, 1, 1){};
		// Matrix<uint64_t, Dynamic, Dynamic> _offsets;
		// Matrix<uint32_t, Dynamic, Dynamic> _lenghts;
	};

	struct MMappedMatrix
	{
		Map<Matrix<half, Dynamic, Dynamic>> _matrix;
		mio::shared_mmap_sink mmap;

		MMappedMatrix() : _matrix(NULL, 1, 1){};
	};

	class TrxFile
	{
		// Access specifier
	public:
		// Data Members
		Dict header;
		ArraySequence streamlines;

		Dict groups; // vector of strings as values

		// int or float --check python float precision (singletons)
		std::map<std::string, MMappedMatrix> data_per_streamline;
		std::map<std::string, ArraySequence> data_per_vertex;
		Dict data_per_group;
		std::string _uncompressed_folder_handle;

		// Member Functions()
		// TrxFile(int nb_vertices = 0, int nb_streamlines = 0);
		TrxFile(int nb_vertices = 0, int nb_streamlines = 0, Json::Value init_as = 0, std::string reference = "");

		void _initialize_empty_trx(int nb_streamlines, int nb_vertices, TrxFile *init_as = NULL);
		static TrxFile *_create_trx_from_pointer(Json::Value header, std::map<std::string, std::tuple<int, int>> dict_pointer_size, std::string root_zip = "", std::string root = "");

	private:
		int len();
	};

	/**
	 * Converts Json header data to a Dict structure
	 * TODO: need to improve. Perhaps can just keep it in JSON format...
	 *
	 * @param[in] root a Json::Value root obtained from reading a header file with JsonCPP
	 * @param[out] header a Dict header containing the same elements as the original root
	 * */
	Dict assignHeader(Json::Value root);

	/**
	 * Returns the properly formatted datatype name
	 *
	 * @param[in] dtype the returned Eigen datatype
	 * @param[out] fmt_dtype the formatted datatype
	 *
	 * */
	std::string _get_dtype(std::string dtype);

	/**
	 * Determine whether the extension is a valid extension
	 *
	 *
	 * @param[in] ext a string consisting of the extension starting by a .
	 * @param[out] is_valid a boolean denoting whether the extension is valid.
	 *
	 * */
	bool _is_dtype_valid(std::string &ext);

	/**
	 * This function loads the header json file
	 * stored within a Zip archive
	 *
	 * @param[in] zfolder a pointer to an opened zip archive
	 * @param[out] header the JSONCpp root of the header. NULL on error.
	 *
	 * */
	Json::Value load_header(zip_t *zfolder);

	/**
	 * Load the TRX file stored within a Zip archive.
	 *
	 * @param[in] path path to Zip archive
	 * @param[out] status return 0 if success else 1
	 *
	 * */
	int load_from_zip(const char *path);

	/**
	 * Get affine and dimensions from a Nifti or Trk file (Adapted from dipy)
	 *
	 * @param[in] reference a string pointing to a NIfTI or trk file to be used as reference
	 * @param[in] affine 4x4 affine matrix
	 * @param[in] dimensions vector of size 3
	 *
	 * */
	void get_reference_info(std::string reference, const MatrixXf &affine, const RowVectorXf &dimensions);

	// template <typename Derived>
	// void _create_memmap(std::filesystem::path &filename, std::string mode = "r", std::string dtype = "float32", int offset = 0);
	std::ostream &operator<<(std::ostream &out, const TrxFile &TrxFile);
	// private:
	// template <typename DT>
	// std::string _generate_filename_from_data(const ArrayBase<DT> &arr, std::string &filename);

	void allocate_file(const std::string &path, const int size);

	/**
	 * @brief Wrapper to support empty array as memmaps
	 *
	 * @param filename filename of the file where the empty memmap should be created
	 * @param shape shape of memmapped NDArray
	 * @param mode file open mode
	 * @param dtype datatype of memmapped NDArray
	 * @param offset offset of the data within the file
	 * @return mio::shared_mmap_sink
	 */
	// TODO: ADD order??
	// TODO: change tuple to vector to support ND arrays?
	mio::shared_mmap_sink _create_memmap(std::string &filename, std::tuple<int, int> &shape, std::string mode = "r", std::string dtype = "float32", int offset = 0);

	template <typename DT>
	std::string _generate_filename_from_data(const ArrayBase<DT> &arr, const std::string filename);
	std::tuple<std::string, int, std::string> _split_ext_with_dimensionality(const std::string filename);

	/**
	 * @brief Compute the lengths from offsets and header information
	 *
	 * @tparam DT The datatype (used for the input matrix)
	 * @param[in] offsets An array of offsets
	 * @param[in] nb_vertices the number of vertices
	 * @return Matrix<uint32_t, Dynamic, Dynamic> of lengths
	 */
	template <typename DT>
	Matrix<uint32_t, Dynamic, Dynamic, RowMajor> _compute_lengths(const MatrixBase<DT> &offsets, int nb_vertices);

	/**
	 * @brief Find where data of a contiguous array is actually ending
	 *
	 * @tparam DT (the datatype)
	 * @param x Matrix of values
	 * @param l_bound lower bound index for search
	 * @param r_bound upper bound index for search
	 * @return int index at which array value is 0 (if possible), otherwise returns -1
	 */
	template <typename DT>
	int _dichotomic_search(const MatrixBase<DT> &x, int l_bound = -1, int r_bound = -1);
#include "trx.tpp"

}

#endif /* TRX_H */