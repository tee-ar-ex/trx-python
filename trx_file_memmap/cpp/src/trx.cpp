#include "trx.h"
#include <fstream>
#include <filesystem>

//#define ZIP_DD_SIG 0x08074b50
//#define ZIP_CD_SIG 0x06054b50
using namespace Eigen;
using namespace std;

long dtype_size(std::string dtype)
{
	if (dtype.compare("int8") == 0 || dtype.compare("uint8") == 0)
		return (long)pow(2, 8);
	else if (dtype.compare("int16") == 0 || dtype.compare("uint16") == 0 || dtype.compare("float16") == 0)
		return (long)pow(2, 16);
	else if (dtype.compare("uint32") || dtype.compare("float32"))
		return (long)pow(2, 32);
	else if (dtype.compare("int64") == 0 || dtype.compare("uint64") == 0 || dtype.compare("float64") == 0)
		return (long)pow(2, 64);
	else
		return 0;
}

bool _is_dtype_valid(std::string &ext)
{
	if (ext.compare("bit") == 0)
		return true;
	else if (std::find(dtypes.begin(), dtypes.end(), ext) != dtypes.end())
		return true;
	return false;
}

Json::Value load_header(zip_t *zfolder)
{
	// load file
	zip_file_t *zh = zip_fopen(zfolder, "header.json", ZIP_FL_UNCHANGED);

	// read data from file in chunks of 255 characters until data is fully loaded
	int buff_len = 255 * sizeof(char);
	char *buffer = (char *)malloc(buff_len);
	//char *jstream, *tmpstream;
	//jstream = NULL;
	std::string jstream = "";
	zip_int64_t nbytes;
	while ((nbytes = zip_fread(zh, buffer, buff_len - 1)) > 0)
	{
		if (buffer != NULL)
		{
			jstream += string(buffer, nbytes);
		}
	}

	// convert jstream data into Json map.
	Json::Value root;
	JSONCPP_STRING err;
	Json::CharReaderBuilder builder;

	const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
	if (!reader->parse(jstream.c_str(), jstream.c_str() + jstream.size(), &root,
					   &err))
	{
		std::cout << "error" << std::endl;
		return 0;
	}

	return root;
}

int load_from_zip(const char *path)
{
	int *errorp;
	zip_t *zf = zip_open(path, 0, errorp);
	Json::Value header = load_header(zf);

	std::map<std::string, std::tuple<int, int>> file_pointer_size;
	int global_pos = 0;
	int mem_address = 0;

	int num_entries = zip_get_num_entries(zf, ZIP_FL_UNCHANGED);

	for (int i = 0; i < num_entries; ++i)
	{
		std::string elem_filename = zip_get_name(zf, i, ZIP_FL_UNCHANGED);

		size_t lastdot = elem_filename.find_last_of(".");

		if (lastdot == std::string::npos)
			continue;
		std::string ext = elem_filename.substr(lastdot + 1, std::string::npos);

		if (ext.compare("bit") == 0)
			ext = "bool";

		// get file stats
		zip_stat_t sb;

		if (zip_stat(zf, elem_filename.c_str(), ZIP_FL_UNCHANGED, &sb) != 0)
		{
			return 1;
		}

		ifstream file(path, ios::binary);
		file.seekg(global_pos);
		mem_address = global_pos;

		unsigned char signature[4] = {0};
		const unsigned char local_sig[4] = {0x50, 0x4b, 0x03, 0x04};
		file.read((char *)signature, sizeof(signature));

		if (memcmp(signature, local_sig, sizeof(signature)) == 0)
		{
			global_pos += 30;
			global_pos += sb.comp_size + elem_filename.size();
		}

		// It's the header, skip it
		if (ext.compare("json") == 0)
			continue;

		if (!_is_dtype_valid(ext))
			continue;

		file_pointer_size[elem_filename] = {mem_address, global_pos};
	}
	return 1;
}

//TODO: support FORTRAN ORDERING
template <typename Derived>
Matrix<Derived, Dynamic, Dynamic> _create_memmap(std::string filename, std::string mode, RowVectorXi shape, std::string dtype, int offset)
{
	if (dtype.compare("bool") == 0)
	{
		// maybe should do a find and replace here instead
		filename.erase(filename.end() - 4);
		filename += "bit";
	}

	if (shape[0] > 0)
	{
	}
}

static TrxFile _initialize_empty_trx(int nb_streamlines, int nb_vertices, TrxFile *init_as = NULL)
{
	// Not sure if it make sense to create a whole new object just to update the previous
	TrxFile trx = TrxFile();
	std::filesystem::path tmp_dir{std::filesystem::temp_directory_path() /= std::tmpnam(nullptr)};
	std::cout << "Temporary folder for memmaps: " << tmp_dir << std::endl;

	trx.header["NB_VERTICES"] = nb_vertices;
	trx.header["NB_STREAMLINES"] = nb_streamlines;

	if (init_as != NULL)
	{
		trx.header["VOXEL_TO_RASMM"] = init_as->header["VOXEL_TO_RASMM"];
		trx.header["DIMENSIONS"] = init_as->header["DIMENSIONS"];
	}

	return trx;
}

TrxFile::TrxFile(int nb_vertices, int nb_streamlines, Json::Value init_as, std::string reference)
{
	MatrixXf affine(4, 4);
	RowVectorXi dimensions(3);

	if (init_as != 0)
	{
		for (int i = 0; i < 4; i++)
		{
			for (int j = 0; j < 4; j++)
			{
				affine << init_as["VOXEL_TO_RASMM"][i][j].asFloat();
			}
		}

		for (int i = 0; i < 3; i++)
		{
			dimensions[i] << init_as["DIMENSIONS"][i].asUInt();
		}
	}
	else
	{
		// add logger here
		// eye matrixt
		affine << MatrixXf::Identity(4, 4);
		dimensions << 1, 1, 1;
	}

	if (nb_vertices == 0 && nb_streamlines == 0)
	{
		if (init_as != 0)
		{
			// raise error here
			exit(1);
		}

		// will remove as completely unecessary. using as placeholders
		this->header = {};
		this->streamlines = ArraySequence();
		this->_uncompressed_folder_handle = "";

		nb_vertices = 0;
		nb_streamlines = 0;
	}
	//else if (nb_vertices != NULL && nb_streamlines != NULL)

	this->header["VOXEL_TO_RASMM"] = affine;
	this->header["DIMENSIONS"] = dimensions;
	this->header["NB_VERTICES"] = nb_vertices;
	this->header["NB_STREAMLINES"] = nb_streamlines;

	//this->header["VOXEL_TO_RASMM"] = &affine;

	//affine >> this->header["VOXEL_TO_RASMM"];
	//dimensions >> this->header["DIMENSIONS"];
	//nb_vertices >> this->header["NB_VERTICES"];
	//nb_streamlines >> this->header["NB_STREAMLINES"];
}

TrxFile TrxFile::_create_trx_from_pointer(Json::Value header, std::map<std::string, std::tuple<int, int>> dict_pointer_size, std::string root_zip, std::string root)
{
	return TrxFile(0, 0);
}

void get_reference_info(std::string reference, MatrixXf affine, RowVectorXi dimensions)
{
	if (reference.find(".nii") != std::string::npos)
	{
	}
	else if (reference.find(".trk") != std::string::npos)
	{
		// TODO: Create exception class
		std::cout << "Trk reference not implemented" << std::endl;
		std::exit(1);
	}
	else
	{
		// TODO: Create exception class
		std::cout << "Trk reference not implemented" << std::endl;
		std::exit(1);
	}
}

int main(int argc, char *argv[])
{
	// "Test cases" until i create more formal ones
	int *errorp;
	char *path = strdup(argv[1]);
	zip_t *zf = zip_open(path, 0, errorp);
	Json::Value header = load_header(zf);

	std::cout << "**Reading header**" << std::endl;
	std::cout << "Dimensions: " << header["DIMENSIONS"] << std::endl;
	std::cout << "Number of streamlines: " << header["NB_STREAMLINES"] << std::endl;
	std::cout << "Number of vertices: " << header["NB_VERTICES"] << std::endl;
	std::cout << "Voxel to RASMM: " << header["VOXEL_TO_RASMM"] << std::endl;

	load_from_zip(path);
	return 1;
}
