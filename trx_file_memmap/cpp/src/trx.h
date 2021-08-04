#include <iostream>
#include <zip.h>
#include <string.h>
#include <json/json.h>
#include <algorithm>
#include <variant>
#include <math.h>
#include <Eigen/Core>

using namespace Eigen;

const std::vector<std::string> dtypes({"int8", "int16", "int64", "uint8", "uint16", "uint32", "uint64", "float16", "float32", "float64"});

typedef std::map<std::string, std::variant<int, MatrixXf, RowVectorXi>> Dict;

struct ArraySequence
{
	Matrix<half, Dynamic, Dynamic> _data;
	Matrix<uint64_t, Dynamic, Dynamic> _offsets;
	Matrix<uint32_t, Dynamic, Dynamic> _lenghts;
};

class TrxFile
{
	// Access specifier
public:
	// Data Members
	Dict header;
	ArraySequence streamlines;
	Dict groups;
	Dict data_per_streamline;
	Dict data_per_vertex;
	Dict data_per_group;
	std::string _uncompressed_folder_handle;

	// Member Functions()
public:
	//TrxFile(int nb_vertices = 0, int nb_streamlines = 0);
	TrxFile(int nb_vertices = 0, int nb_streamlines = 0, Json::Value init_as = 0, std::string reference = NULL);

	static TrxFile _initialize_empty_trx(int nb_streamlines, int nb_vertices, TrxFile *init_as = NULL);
	static TrxFile _create_trx_from_pointer(Json::Value header, std::map<std::string, std::tuple<int, int>> dict_pointer_size, std::string root_zip = NULL, std::string root = NULL);

private:
	int len();
};

/**
 * Determines data type size
 * Note: need to determine a better solution
 * 
 * @param[in] dtype a string consisting of the extension starting by a .
 * @param[out] size the respective size of the dtype
 *  
 * */
long dtype_size(std::string dtype);
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
void get_reference_info(std::string reference, MatrixXf affine, RowVectorXf dimensions);

template <typename Derived>
Matrix<Derived, Dynamic, Dynamic> _create_memmap(std::string filename, std::string mode = "r", RowVectorXi shape = (1), std::string dtype = "float32", int offset = 0);
std::ostream &operator<<(std::ostream &out, const TrxFile &TrxFile);