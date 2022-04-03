#include "trx.h"
#include <fstream>
#include <typeinfo>
#include <errno.h>
#include <algorithm>
#define SYSERROR() errno

//#define ZIP_DD_SIG 0x08074b50
//#define ZIP_CD_SIG 0x06054b50
using namespace Eigen;
using namespace std;

std::string get_base(const std::string &delimeter, const std::string &str)
{
	std::string token;

	if (str.rfind(delimeter) + 1 < str.length())
	{
		token = str.substr(str.rfind(delimeter) + 1);
	}
	else
	{
		token = str;
	}
	return token;
}

std::string get_ext(const std::string &str)
{
	std::string ext = "";
	std::string delimeter = ".";

	if (str.rfind(delimeter) + 1 < str.length())
	{
		ext = str.substr(str.rfind(delimeter));
	}
	return ext;
}

namespace trxmmap
{
	// TODO: check if there's a better way
	int _sizeof_dtype(std::string dtype)
	{
		// TODO: make dtypes enum??
		auto it = std::find(dtypes.begin(), dtypes.end(), dtype);
		int index = 0;
		if (it != dtypes.end())
		{
			index = std::distance(dtypes.begin(), it);
		}

		switch (index)
		{
		case 1:
			return 1;
		case 2:
			return sizeof(uint8_t);
		case 3:
			return sizeof(uint16_t);
		case 4:
			return sizeof(uint32_t);
		case 5:
			return sizeof(uint64_t);
		case 6:
			return sizeof(int8_t);
		case 7:
			return sizeof(int16_t);
		case 8:
			return sizeof(int32_t);
		case 9:
			return sizeof(int64_t);
		case 10:
			return sizeof(float);
		case 11:
			return sizeof(double);
		default:
			return sizeof(half); // setting this as default for now but a better solution is needed
		}
	}

	std::string _get_dtype(std::string dtype)
	{
		char dt = dtype.back();
		switch (dt)
		{
		case 'b':
			return "bit";
		case 'h':
			return "uint8";
		case 't':
			return "uint16";
		case 'j':
			return "uint32";
		case 'm':
			return "uint64";
		case 'a':
			return "int8";
		case 's':
			return "int16";
		case 'i':
			return "int32";
		case 'l':
			return "int64";
		case 'f':
			return "float32";
		case 'd':
			return "float64";
		default:
			return "float16"; // setting this as default for now but a better solution is needed
		}
	}
	std::tuple<std::string, int, std::string> _split_ext_with_dimensionality(const std::string filename)
	{

		// TODO: won't work on windows and not validating OS type
		std::string base = get_base("/", filename);

		size_t num_splits = std::count(base.begin(), base.end(), '.');
		int dim;

		if (num_splits != 1 and num_splits != 2)
		{
			throw std::invalid_argument("Invalid filename");
		}

		std::string ext = get_ext(filename);

		base = base.substr(0, base.length() - ext.length());

		if (num_splits == 1)
		{
			dim = 1;
		}
		else
		{
			int pos = base.find_last_of(".");
			dim = std::stoi(base.substr(pos + 1, base.size()));
			base = base.substr(0, pos);
		}

		bool is_valid = _is_dtype_valid(ext);

		if (is_valid == false)
		{
			// TODO: make formatted string and include provided extension name
			throw std::invalid_argument("Unsupported file extension");
		}

		std::tuple<std::string, int, std::string> output{base, dim, ext};

		return output;
	}

	bool _is_dtype_valid(std::string &ext)
	{
		if (ext.compare("bit") == 0)
			return true;
		if (std::find(trxmmap::dtypes.begin(), trxmmap::dtypes.end(), ext.substr(1)) != trxmmap::dtypes.end())
			return true;
		return false;
	}

	json load_header(zip_t *zfolder)
	{
		// load file
		zip_file_t *zh = zip_fopen(zfolder, "header.json", ZIP_FL_UNCHANGED);

		// read data from file in chunks of 255 characters until data is fully loaded
		int buff_len = 255 * sizeof(char);
		char *buffer = (char *)malloc(buff_len);

		std::string jstream = "";
		zip_int64_t nbytes;
		while ((nbytes = zip_fread(zh, buffer, buff_len - 1)) > 0)
		{
			if (buffer != NULL)
			{
				jstream += string(buffer, nbytes);
			}
		}

		free(zh);
		free(buffer);

		// convert jstream data into Json.
		auto root = json::parse(jstream);
		return root;
	}

	int load_from_zip(const char *path)
	{
		int *errorp;
		zip_t *zf = zip_open(path, 0, errorp);
		json header = load_header(zf);

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

	void allocate_file(const std::string &path, const int size)
	{
		std::ofstream file(path);
		if (file.is_open())
		{
			std::string s(size, float(0));
			file << s;
			file.flush();
			file.close();
		}
		else
		{
			std::cerr << "Failed to allocate file : " << SYSERROR() << std::endl;
		}
	}

	mio::shared_mmap_sink _create_memmap(std::string &filename, std::tuple<int, int> &shape, std::string mode, std::string dtype, int offset)
	{
		if (dtype.compare("bool") == 0)
		{
			std::string ext = "bit";
			filename.replace(filename.size() - 4, 3, ext);
			filename.pop_back();
		}

		// if file does not exist, create and allocate it
		struct stat buffer;
		if (stat(filename.c_str(), &buffer) != 0)
		{
			allocate_file(filename, std::get<0>(shape) * std::get<1>(shape) * _sizeof_dtype(dtype));
		}

		// std::error_code error;
		mio::shared_mmap_sink rw_mmap(filename, 0, mio::map_entire_file);

		return rw_mmap;
	}

	// TODO: support FORTRAN ORDERING
	// template <typename Derived>

	json assignHeader(json root)
	{
		json header = root;
		// MatrixXf affine(4, 4);
		// RowVectorXi dimensions(3);

		// for (int i = 0; i < 4; i++)
		// {
		// 	for (int j = 0; j < 4; j++)
		// 	{
		// 		affine << root["VOXEL_TO_RASMM"][i][j].asFloat();
		// 	}
		// }

		// for (int i = 0; i < 3; i++)
		// {
		// 	dimensions[i] << root["DIMENSIONS"][i].asUInt();
		// }
		// header["VOXEL_TO_RASMM"] = affine;
		// header["DIMENSIONS"] = dimensions;
		// header["NB_VERTICES"] = (int)root["NB_VERTICES"].asUInt();
		// header["NB_STREAMLINES"] = (int)root["NB_STREAMLINES"].asUInt();

		return header;
	}

	// trxmmap::TrxFile *trxmmap::TrxFile::_create_trx_from_pointer(Json::Value header, std::map<std::string, std::tuple<int, int>> dict_pointer_size, std::string root_zip, std::string root)
	// {
	// 	trxmmap::TrxFile *trx = new trxmmap::TrxFile();
	// 	trx->header = trxmmap::assignHeader(header);

	// 	// positions, offsets = None, None
	// 	Map<Matrix<half, Dynamic, Dynamic>> positions(NULL, 1, 1);
	// 	Map<Matrix<uint16_t, Dynamic, 1>> offsets(NULL, 1, 1);
	// 	std::filesystem::path filename;

	// 	for (auto const &x : dict_pointer_size)
	// 	{
	// 		std::filesystem::path elem_filename = x.first;
	// 		if (root_zip.size() == 0)
	// 		{
	// 			filename = root_zip;
	// 		}
	// 		else
	// 		{
	// 			filename = elem_filename;
	// 		}

	// 		std::filesystem::path folder = elem_filename.parent_path();

	// 		// _split_ext_with_dimensionality
	// 		std::string basename = elem_filename.filename().string();

	// 		std::string tokens[4];
	// 		int idx, prev_pos, curr_pos;
	// 		prev_pos = 0;
	// 		idx = 0;

	// 		while ((curr_pos = basename.find(".")) != std::string::npos)
	// 		{
	// 			tokens[idx] = basename.substr(prev_pos, curr_pos);
	// 			prev_pos = curr_pos + 1;
	// 			idx++;
	// 		}

	// 		if (idx < 2 || idx > 3)
	// 		{
	// 			throw("Invalid filename.");
	// 		}

	// 		basename = tokens[0];
	// 		std::string ext = "." + tokens[idx - 1];
	// 		int dim;

	// 		if (idx == 2)
	// 		{
	// 			dim = 1;
	// 		}
	// 		else
	// 		{
	// 			dim = std::stoi(tokens[1]);
	// 		}
	// 		_is_dtype_valid(ext);
	// 		// function completed

	// 		if (ext.compare(".bit") == 0)
	// 		{
	// 			ext = ".bool";
	// 		}
	// 		int mem_adress = get<0>(dict_pointer_size[elem_filename]);
	// 		int size = get<1>(dict_pointer_size[elem_filename]);

	// 		// skipped the stripping of right /..not sure when it's necessary
	// 		// Also not sure how lstripping will work on windows
	// 		if (root.compare("") == 0 && folder.string().find(root) == 0)
	// 		{
	// 			string updated_fldr = folder.string();
	// 		}
	// 	}

	// 	return trx;
	// }

	void get_reference_info(std::string reference, const MatrixXf &affine, const RowVectorXi &dimensions)
	{
		// TODO: find a library to use for nifti and trk (MRtrix??)
		//  if (reference.find(".nii") != std::string::npos)
		//  {
		//  }
		if (reference.find(".trk") != std::string::npos)
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
};
// int main(int argc, char *argv[])
//{
//  Array<int16_t, 5, 4> arr;
//  trxmmap::_generate_filename_from_data(arr, "mean_fa.int16");

// // "Test cases" until i create more formal ones
// int *errorp;
// char *path = strdup(argv[1]);
// zip_t *zf = zip_open(path, 0, errorp);
// Json::Value header = load_header(zf);

// std::cout << "**Reading header**" << std::endl;
// std::cout << "Dimensions: " << header["DIMENSIONS"] << std::endl;
// std::cout << "Number of streamlines: " << header["NB_STREAMLINES"] << std::endl;
// std::cout << "Number of vertices: " << header["NB_VERTICES"] << std::endl;
// std::cout << "Voxel to RASMM: " << header["VOXEL_TO_RASMM"] << std::endl;

// load_from_zip(path);

// TrxFile trx(1, 1);

// std::cout << "Printing trk streamline data value " << trx.streamlines._data(0, 0) << endl;
// std::cout << "Printing trk streamline offset value " << trx.streamlines._offset(0, 0) << endl;
// return 1;
//}
