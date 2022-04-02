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
			allocate_file(filename, std::get<0>(shape) * std::get<1>(shape));
		}

		// std::error_code error;
		mio::shared_mmap_sink rw_mmap(filename, 0, mio::map_entire_file);

		return rw_mmap;
	}

	// TODO: support FORTRAN ORDERING
	// template <typename Derived>

	// void trxmmap::TrxFile::_initialize_empty_trx(int nb_streamlines, int nb_vertices, trxmmap::TrxFile *init_as)
	// {

	// 	// TODO: fix as tmpnam has issues with concurrency.
	// 	std::filesystem::path tmp_dir{std::filesystem::temp_directory_path() /= std::tmpnam(nullptr)};
	// 	std::filesystem::create_directory(tmp_dir);
	// 	std::cout << "Temporary folder for memmaps: " << tmp_dir << std::endl;

	// 	this->header["NB_VERTICES"] = nb_vertices;
	// 	this->header["NB_STREAMLINES"] = nb_streamlines;

	// 	if (init_as != NULL)
	// 	{
	// 		this->header["VOXEL_TO_RASMM"] = init_as->header["VOXEL_TO_RASMM"];
	// 		this->header["DIMENSIONS"] = init_as->header["DIMENSIONS"];
	// 	}

	// 	cout << "Initializing positions with dtype: "
	// 	     << "float16" << endl;
	// 	cout << "Initializing offsets with dtype: "
	// 	     << "uint16" << endl;
	// 	cout << "Initializing lengths with dtype: "
	// 	     << "uint32" << endl;

	// 	std::filesystem::path positions_filename;
	// 	positions_filename /= tmp_dir;
	// 	positions_filename /= "positions.3.float16";
	// 	Eigen::MatrixXi m(2, 1);
	// 	m << nb_vertices, 3;
	// 	// new (&trx.streamlines._data)
	// 	this->streamlines.mmap_pos = trxmmap::_create_memmap(positions_filename, m, std::string("w+"), std::string("float16"), 0);
	// 	new (&(this->streamlines._data)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(this->streamlines.mmap_pos.data()), m(0), m(1));

	// 	std::filesystem::path offsets_filename;
	// 	offsets_filename /= tmp_dir;
	// 	offsets_filename /= "offsets.uint16";
	// 	cout << "filesystem path " << offsets_filename << endl;
	// 	m << nb_streamlines, 1;
	// 	this->streamlines.mmap_off = trxmmap::_create_memmap(offsets_filename, m, std::string("w+"), std::string("uint16_t"), 0);
	// 	new (&(this->streamlines._offset)) Map<Matrix<uint16_t, Dynamic, 1>>(reinterpret_cast<uint16_t *>(this->streamlines.mmap_off.data()), m(0), m(1));

	// 	this->streamlines._lengths.resize(nb_streamlines, 0);

	// 	if (init_as != NULL)
	// 	{
	// 		if (init_as->data_per_vertex.size() > 0)
	// 		{
	// 			std::filesystem::path dpv_dirname;
	// 			dpv_dirname /= tmp_dir;
	// 			dpv_dirname /= "dpv";
	// 			std::filesystem::create_directory(dpv_dirname);
	// 		}
	// 		if (init_as->data_per_streamline.size() > 0)
	// 		{
	// 			std::filesystem::path dps_dirname;
	// 			dps_dirname /= tmp_dir;
	// 			dps_dirname /= "dps";
	// 			std::filesystem::create_directory(dps_dirname);
	// 		}

	// 		for (auto const &x : init_as->data_per_vertex)
	// 		{
	// 			int rows, cols;
	// 			Map<Matrix<half, Dynamic, Dynamic>> tmp_as = init_as->data_per_vertex[x.first]._data;
	// 			std::filesystem::path dpv_filename;
	// 			if (tmp_as.cols() == 1 || tmp_as.rows() == 1)
	// 			{
	// 				dpv_filename /= tmp_dir;
	// 				dpv_filename /= "dpv";
	// 				dpv_filename /= x.first + "." + "float16";
	// 				rows = nb_vertices;
	// 				cols = 1;
	// 			}
	// 			else
	// 			{
	// 				rows = nb_vertices;
	// 				cols = tmp_as.cols();

	// 				dpv_filename /= tmp_dir;
	// 				dpv_filename /= "dpv";
	// 				dpv_filename /= x.first + "." + std::to_string(cols) + "." + "float16";
	// 			}

	// 			cout << "Initializing " << x.first << " (dpv) with dtype: "
	// 			     << "float16" << endl;

	// 			Eigen::MatrixXi m(2, 1);
	// 			m << rows, cols;
	// 			this->data_per_vertex[x.first] = trxmmap::ArraySequence();
	// 			this->data_per_vertex[x.first].mmap_pos = trxmmap::_create_memmap(dpv_filename, m, std::string("w+"), std::string("float16"), 0);
	// 			new (&(this->data_per_vertex[x.first]._data)) Map<Matrix<uint16_t, Dynamic, Dynamic>>(reinterpret_cast<uint16_t *>(this->data_per_vertex[x.first].mmap_pos.data()), m(0), m(1));

	// 			this->data_per_vertex[x.first]._offset = this->streamlines._offset;
	// 			this->data_per_vertex[x.first]._lengths = this->streamlines._lengths;
	// 		}

	// 		for (auto const &x : init_as->data_per_streamline)
	// 		{
	// 			string dtype = "float16";
	// 			int rows, cols;
	// 			Map<Matrix<half, Dynamic, Dynamic>> tmp_as = init_as->data_per_streamline[x.first]._matrix;
	// 			std::filesystem::path dps_filename;

	// 			if (tmp_as.rows() == 1 || tmp_as.cols() == 1)
	// 			{
	// 				dps_filename /= tmp_dir;
	// 				dps_filename /= "dps";
	// 				dps_filename /= x.first + "." + dtype;
	// 				rows = nb_streamlines;
	// 			}
	// 			else
	// 			{
	// 				cols = tmp_as.cols();
	// 				rows = nb_streamlines;

	// 				dps_filename /= tmp_dir;
	// 				dps_filename /= "dps";
	// 				dps_filename /= x.first + "." + std::to_string(cols) + "." + dtype;
	// 			}

	// 			cout << "Initializing " << x.first << " (dps) with and dtype: " << dtype << endl;

	// 			Eigen::MatrixXi m(2, 1);
	// 			m << rows, cols;
	// 			this->data_per_streamline[x.first] = trxmmap::MMappedMatrix();
	// 			this->data_per_streamline[x.first].mmap = trxmmap::_create_memmap(dps_filename, m, std::string("w+"), dtype, 0);
	// 			new (&(this->data_per_streamline[x.first]._matrix)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(this->data_per_streamline[x.first].mmap.data()), m(0), m(1));
	// 		}
	// 	}

	// 	this->_uncompressed_folder_handle = tmp_dir;
	// }

	// trxmmap::TrxFile::TrxFile(int nb_vertices, int nb_streamlines, Json::Value init_as, std::string reference)
	// {
	// 	MatrixXf affine(4, 4);
	// 	RowVectorXi dimensions(3);

	// 	if (init_as != 0)
	// 	{
	// 		for (int i = 0; i < 4; i++)
	// 		{
	// 			for (int j = 0; j < 4; j++)
	// 			{
	// 				affine << init_as["VOXEL_TO_RASMM"][i][j].asFloat();
	// 			}
	// 		}

	// 		for (int i = 0; i < 3; i++)
	// 		{
	// 			dimensions[i] << init_as["DIMENSIONS"][i].asUInt();
	// 		}
	// 	}
	// 	else
	// 	{
	// 		// add logger here
	// 		// eye matrixt
	// 		affine << MatrixXf::Identity(4, 4);
	// 		dimensions << 1, 1, 1;
	// 	}

	// 	if (nb_vertices == 0 && nb_streamlines == 0)
	// 	{
	// 		if (init_as != 0)
	// 		{
	// 			// raise error here
	// 			exit(1);
	// 		}

	// 		// will remove as completely unecessary. using as placeholders
	// 		this->header = {};
	// 		// this->streamlines = ArraySequence();
	// 		this->_uncompressed_folder_handle = "";

	// 		nb_vertices = 0;
	// 		nb_streamlines = 0;
	// 	}
	// 	else if (nb_vertices > 0 && nb_streamlines > 0)
	// 	{
	// 		std::cout << "Preallocating TrxFile with size " << nb_streamlines << " streamlines and " << nb_vertices << " vertices." << std::endl;
	// 		_initialize_empty_trx(nb_streamlines, nb_vertices);

	// 		// this->streamlines._data << half(1);
	// 		this->streamlines._offset << uint16_t(1);
	// 	}

	// 	this->header["VOXEL_TO_RASMM"] = affine;
	// 	this->header["DIMENSIONS"] = dimensions;
	// 	this->header["NB_VERTICES"] = nb_vertices;
	// 	this->header["NB_STREAMLINES"] = nb_streamlines;
	// }

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
