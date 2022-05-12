template <typename DT>
void ediff1d(Matrix<DT, Dynamic, 1> &lengths, Matrix<DT, Dynamic, Dynamic> &tmp, uint32_t to_end)
{
	Map<RowVector<uint32_t, Dynamic>> v(tmp.data(), tmp.size());
	lengths.resize(v.size(), 1);

	// TODO: figure out if there's a built in way to manage this
	for (int i = 0; i < v.size() - 1; i++)
	{
		lengths(i) = v(i + 1) - v(i);
	}
	lengths(v.size() - 1) = to_end;
}

template <typename DT>
std::string _generate_filename_from_data(const ArrayBase<DT> &arr, std::string filename)
{

	int ext_pos = filename.find_last_of(".");

	if (ext_pos == 0)
	{
		throw;
	}

	std::string base = filename.substr(0, ext_pos);
	std::string ext = filename.substr(ext_pos, filename.size());

	if (ext.size() != 0)
	{
		std::cout << "WARNING: Will overwrite provided extension if needed." << std::endl;
	}

	std::string eigen_dt = typeid(arr.matrix().data()).name();
	std::string dt = _get_dtype(eigen_dt);

	int n_rows = arr.rows();
	int n_cols = arr.cols();

	std::string new_filename;
	if (n_cols == 1)
	{
		int buffsize = filename.size() + dt.size() + 2;
		char buff[buffsize];
		snprintf(buff, sizeof(buff), "%s.%s", base.c_str(), dt.c_str());
		new_filename = buff;
	}
	else
	{
		int buffsize = filename.size() + dt.size() + n_cols + 3;
		char buff[buffsize];
		snprintf(buff, sizeof(buff), "%s.%i.%s", base.c_str(), n_cols, dt.c_str());
		new_filename = buff;
	}

	return new_filename;
}

template <typename DT>
Matrix<uint32_t, Dynamic, 1> _compute_lengths(const MatrixBase<DT> &offsets, int nb_vertices)
{
	if (offsets.size() > 1)
	{
		int last_elem_pos = _dichotomic_search(offsets);
		Matrix<uint32_t, Dynamic, 1> lengths;

		if (last_elem_pos == offsets.size() - 1)
		{
			Matrix<uint32_t, Dynamic, Dynamic> tmp(offsets.template cast<u_int32_t>());
			ediff1d(lengths, tmp, uint32_t(nb_vertices - offsets(last)));
		}
		else
		{
			Matrix<uint32_t, Dynamic, Dynamic> tmp(offsets.template cast<u_int32_t>());
			tmp(last_elem_pos + 1) = uint32_t(nb_vertices);
			ediff1d(lengths, tmp, 0);
			lengths(last_elem_pos + 1) = uint32_t(0);
		}
		return lengths;
	}
	if (offsets.size() == 1)
	{
		Matrix<uint32_t, 1, 1, RowMajor> lengths(nb_vertices);
		return lengths;
	}

	Matrix<uint32_t, 1, 1, RowMajor> lengths(0);
	return lengths;
}

template <typename DT>
int _dichotomic_search(const MatrixBase<DT> &x, int l_bound, int r_bound)
{
	if (l_bound == -1 && r_bound == -1)
	{
		l_bound = 0;
		r_bound = x.size() - 1;
	}

	if (l_bound == r_bound)
	{
		int val;
		if (x(l_bound) != 0)
			val = l_bound;
		else
			val = -1;
		return val;
	}

	int mid_bound = (l_bound + r_bound + 1) / 2;

	if (x(mid_bound) == 0)
		return _dichotomic_search(x, l_bound, mid_bound - 1);
	else
		return _dichotomic_search(x, mid_bound, r_bound);
}

template <typename DT>
TrxFile<DT>::TrxFile(int nb_vertices, int nb_streamlines, const TrxFile<DT> *init_as, std::string reference)
{
	std::vector<std::vector<float>> affine(4);
	std::vector<uint16_t> dimensions(3);

	// TODO: check if there's a more efficient way to do this with Eigen
	if (init_as != NULL)
	{
		for (int i = 0; i < 4; i++)
		{
			affine[i] = {0, 0, 0, 0};
			for (int j = 0; j < 4; j++)
			{
				affine[i][j] = float(init_as->header["VOXEL_TO_RASMM"][i][j]);
			}
		}

		for (int i = 0; i < 3; i++)
		{
			dimensions[i] = uint16_t(init_as->header["DIMENSIONS"][i]);
		}
	}
	// TODO: add else if for get_reference_info
	else
	{
		// TODO: logger
		std::cout << "No reference provided, using blank space attributes, please update them later." << std::endl;

		// identity matrix
		for (int i = 0; i < 4; i++)
		{
			affine[i] = {0, 0, 0, 0};
			affine[i][i] = 1;
		}
		dimensions = {1, 1, 1};
	}

	if (nb_vertices == 0 && nb_streamlines == 0)
	{
		if (init_as != NULL)
		{
			// raise error here
			throw std::invalid_argument("Can't us init_as without declaring nb_vertices and nb_streamlines");
		}

		// TODO: logger
		std::cout << "Initializing empty TrxFile." << std::endl;
		// will remove as completely unecessary. using as placeholders
		this->header = {};

		// TODO: maybe create a matrix to map to of specified DT. Do we need this??
		// set default datatype to half
		// default data is null so will not set data. User will need configure desired datatype
		// this->streamlines = ArraySequence<half>();
		this->_uncompressed_folder_handle = "";

		nb_vertices = 0;
		nb_streamlines = 0;
	}
	else if (nb_vertices > 0 && nb_streamlines > 0)
	{
		// TODO: logger
		std::cout << "Preallocating TrxFile with size " << nb_streamlines << " streamlines and " << nb_vertices << " vertices." << std::endl;
		TrxFile<DT> *trx = _initialize_empty_trx<DT>(nb_streamlines, nb_vertices, init_as);
		this->streamlines = trx->streamlines;
		this->groups = trx->groups;
		this->data_per_streamline = trx->data_per_streamline;
		this->data_per_vertex = trx->data_per_vertex;
		this->data_per_group = trx->data_per_group;
		this->_uncompressed_folder_handle = trx->_uncompressed_folder_handle;
		this->_copy_safe = trx->_copy_safe;
	}
	else
	{
		throw std::invalid_argument("You must declare both NB_VERTICES AND NB_STREAMLINES");
	}

	this->header["VOXEL_TO_RASMM"] = affine;
	this->header["DIMENSIONS"] = dimensions;
	this->header["NB_VERTICES"] = nb_vertices;
	this->header["NB_STREAMLINES"] = nb_streamlines;

	this->_copy_safe = true;
}

template <typename DT>
TrxFile<DT> *_initialize_empty_trx(int nb_streamlines, int nb_vertices, const TrxFile<DT> *init_as)
{
	TrxFile<DT> *trx = new TrxFile<DT>();

	char *dirname;
	char t[] = "/tmp/trx_XXXXXX";
	dirname = mkdtemp(t);

	std::string tmp_dir(dirname);

	std::cout << "Temporary folder for memmaps: " << tmp_dir << std::endl;

	trx->header["NB_VERTICES"] = nb_vertices;
	trx->header["NB_STREAMLINES"] = nb_streamlines;

	std::string positions_dtype;
	std::string offsets_dtype;
	std::string lengths_dtype;

	if (init_as != NULL)
	{
		trx->header["VOXEL_TO_RASMM"] = init_as->header["VOXEL_TO_RASMM"];
		trx->header["DIMENSIONS"] = init_as->header["DIMENSIONS"];
		positions_dtype = _get_dtype(typeid(init_as->streamlines->_data).name());
		offsets_dtype = _get_dtype(typeid(init_as->streamlines->_offsets).name());
		lengths_dtype = _get_dtype(typeid(init_as->streamlines->_lengths).name());
	}
	else
	{
		positions_dtype = _get_dtype(typeid(half).name());
		offsets_dtype = _get_dtype(typeid(uint64_t).name());
		lengths_dtype = _get_dtype(typeid(uint32_t).name());
	}

	std::cout << "Initializing positions with dtype: "
		  << positions_dtype << std::endl;
	std::cout << "Initializing offsets with dtype: "
		  << offsets_dtype << std::endl;
	std::cout << "Initializing lengths with dtype: "
		  << lengths_dtype << std::endl;

	std::string positions_filename(tmp_dir);
	positions_filename += "/positions.3." + positions_dtype;

	std::tuple<int, int> shape = std::make_tuple(nb_vertices, 3);

	trx->streamlines = new ArraySequence<DT>();
	trx->streamlines->mmap_pos = trxmmap::_create_memmap(positions_filename, shape, "w+", positions_dtype);

	// TODO: find a better way to get the dtype than using all these switch cases. Also refactor into function
	// as per specifications, positions can only be floats
	if (positions_dtype.compare("float16") == 0)
	{
		new (&(trx->streamlines->_data)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
	}
	else if (positions_dtype.compare("float32") == 0)
	{
		new (&(trx->streamlines->_data)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
	}
	else
	{
		new (&(trx->streamlines->_data)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
	}

	std::string offsets_filename(tmp_dir);
	offsets_filename += "/offsets." + offsets_dtype;
	std::cout << "filesystem path " << offsets_filename << std::endl;

	std::tuple<int, int> shape_off = std::make_tuple(nb_streamlines, 1);

	trx->streamlines->mmap_off = trxmmap::_create_memmap(offsets_filename, shape_off, "w+", offsets_dtype);
	new (&(trx->streamlines->_offsets)) Map<Matrix<uint64_t, 1, Dynamic, RowMajor>>(reinterpret_cast<uint64_t *>(trx->streamlines->mmap_off.data()), std::get<0>(shape_off), std::get<1>(shape_off));

	trx->streamlines->_lengths.resize(nb_streamlines, 0);

	if (init_as != NULL)
	{
		std::string dpv_dirname;
		std::string dps_dirname;
		if (init_as->data_per_vertex.size() > 0)
		{
			dpv_dirname = tmp_dir + "/dpv/";
			mkdir(dpv_dirname.c_str(), S_IRWXU);
		}
		if (init_as->data_per_streamline.size() > 0)
		{
			dps_dirname = tmp_dir + "/dps/";
			mkdir(dps_dirname.c_str(), S_IRWXU);
		}

		for (auto const &x : init_as->data_per_vertex)
		{
			int rows, cols;
			std::string dpv_dtype = _get_dtype(typeid(init_as->data_per_vertex.find(x.first)->second->_data).name());
			Map<Matrix<DT, Dynamic, Dynamic>> tmp_as = init_as->data_per_vertex.find(x.first)->second->_data;

			std::string dpv_filename;
			if (tmp_as.rows() == 1)
			{
				dpv_filename = dpv_dirname + x.first + "." + dpv_dtype;
				rows = nb_vertices;
				cols = 1;
			}
			else
			{
				rows = nb_vertices;
				cols = tmp_as.cols();

				dpv_filename = dpv_dirname + x.first + "." + std::to_string(cols) + "." + dpv_dtype;
			}

			// TODO: logger
			std::cout << "Initializing " << x.first << " (dpv) with dtype: "
				  << dpv_dtype << std::endl;

			std::tuple<int, int> dpv_shape = std::make_tuple(rows, cols);
			trx->data_per_vertex[x.first] = new ArraySequence<DT>();
			trx->data_per_vertex[x.first]->mmap_pos = trxmmap::_create_memmap(dpv_filename, dpv_shape, "w+", dpv_dtype);
			if (dpv_dtype.compare("float16") == 0)
			{
				new (&(trx->data_per_vertex[x.first]->_data)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->data_per_vertex[x.first]->mmap_pos.data()), rows, cols);
			}
			else if (dpv_dtype.compare("float32") == 0)
			{
				new (&(trx->data_per_vertex[x.first]->_data)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->data_per_vertex[x.first]->mmap_pos.data()), rows, cols);
			}
			else
			{
				new (&(trx->data_per_vertex[x.first]->_data)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->data_per_vertex[x.first]->mmap_pos.data()), rows, cols);
			}

			trx->data_per_vertex[x.first]->_offsets = trx->streamlines->_offsets;
			trx->data_per_vertex[x.first]->_lengths = trx->streamlines->_lengths;
		}

		for (auto const &x : init_as->data_per_streamline)
		{
			std::string dps_dtype = _get_dtype(typeid(init_as->data_per_streamline.find(x.first)->second->_matrix).name());
			int rows, cols;
			Map<Matrix<DT, Dynamic, Dynamic>> tmp_as = init_as->data_per_streamline.find(x.first)->second->_matrix;

			std::string dps_filename;

			if (tmp_as.rows() == 1)
			{
				dps_filename = dps_dirname + x.first + "." + dps_dtype;
				rows = nb_streamlines;
			}
			else
			{
				cols = tmp_as.cols();
				rows = nb_streamlines;

				dps_filename = dps_dirname + x.first + "." + std::to_string(cols) + "." + dps_dtype;
			}

			// TODO: logger
			std::cout << "Initializing " << x.first << " (dps) with and dtype: " << dps_dtype << std::endl;

			std::tuple<int, int> dps_shape = std::make_tuple(rows, cols);
			trx->data_per_streamline[x.first] = new trxmmap::MMappedMatrix<DT>();
			trx->data_per_streamline[x.first]->mmap = trxmmap::_create_memmap(dps_filename, dps_shape, std::string("w+"), dps_dtype);

			if (dps_dtype.compare("float16") == 0)
			{
				new (&(trx->data_per_streamline[x.first]->_matrix)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->data_per_streamline[x.first]->mmap.data()), rows, cols);
			}
			else if (dps_dtype.compare("float32") == 0)
			{
				new (&(trx->data_per_streamline[x.first]->_matrix)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->data_per_streamline[x.first]->mmap.data()), rows, cols);
			}
			else
			{
				new (&(trx->data_per_streamline[x.first]->_matrix)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->data_per_streamline[x.first]->mmap.data()), rows, cols);
			}
		}
	}

	trx->_uncompressed_folder_handle = tmp_dir;

	return trx;
}

template <typename DT>
TrxFile<DT> *TrxFile<DT>::_create_trx_from_pointer(json header, std::map<std::string, std::tuple<long long, long long>> dict_pointer_size, std::string root_zip, std::string root)
{
	trxmmap::TrxFile<DT> *trx = new trxmmap::TrxFile<DT>();
	trx->header = header;
	trx->streamlines = new ArraySequence<DT>();

	std::string filename;

	// TODO: Fix this hack of iterating through dictionary in reverse to get main files read first
	for (auto x = dict_pointer_size.rbegin(); x != dict_pointer_size.rend(); ++x)
	{
		std::string elem_filename = x->first;

		if (root_zip.size() > 0)
		{
			filename = root_zip;
		}
		else
		{
			filename = elem_filename;
		}

		std::string folder = std::string(dirname(const_cast<char *>(strdup(elem_filename.c_str()))));

		// _split_ext_with_dimensionality
		std::tuple<std::string, int, std::string> base_tuple = _split_ext_with_dimensionality(elem_filename);
		std::string base(std::get<0>(base_tuple));
		int dim = std::get<1>(base_tuple);
		std::string ext(std::get<2>(base_tuple));

		if (ext.compare(".bit") == 0)
		{
			ext = ".bool";
		}

		long long mem_adress = std::get<0>(x->second);
		long long size = std::get<1>(x->second);

		std::string stripped = root;

		// TODO : will not work on windows
		if (stripped.rfind("/") == stripped.size() - 1)
		{
			stripped = stripped.substr(0, stripped.size() - 1);
		}

		if (root.compare("") != 0 && folder.rfind(stripped, stripped.size()) == 0)
		{
			// 1 for the first forward slash
			folder = folder.substr(1 + root.size(), folder.size() - (1 + root.size()));
		}

		if (base.compare("positions") == 0 && folder.compare(".") == 0)
		{
			if (size != int(trx->header["NB_VERTICES"]) * 3 || dim != 3)
			{

				throw std::invalid_argument("Wrong data size/dimensionality");
			}

			std::tuple<int, int> shape = std::make_tuple(trx->header["NB_VERTICES"], 3);
			trx->streamlines->mmap_pos = trxmmap::_create_memmap(filename, shape, "r+", ext.substr(1, ext.size() - 1), mem_adress);

			// TODO: find a better way to get the dtype than using all these switch cases. Also refactor into function
			// as per specifications, positions can only be floats
			if (ext.compare(".float16") == 0)
			{
				new (&(trx->streamlines->_data)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else if (ext.compare(".float32") == 0)
			{
				new (&(trx->streamlines->_data)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else
			{
				new (&(trx->streamlines->_data)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->streamlines->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
			}
		}

		else if (base.compare("offsets") == 0 && folder.compare(".") == 0)
		{
			if (size != int(trx->header["NB_STREAMLINES"]) || dim != 1)
			{

				throw std::invalid_argument("Wrong offsets size/dimensionality");
			}

			std::tuple<int, int> shape = std::make_tuple(trx->header["NB_STREAMLINES"], 1);
			trx->streamlines->mmap_off = trxmmap::_create_memmap(filename, shape, "r+", ext.substr(1, ext.size() - 1), mem_adress);

			new (&(trx->streamlines->_offsets)) Map<Matrix<uint64_t, Dynamic, 1>>(reinterpret_cast<uint64_t *>(trx->streamlines->mmap_off.data()), std::get<0>(shape), std::get<1>(shape));

			// TODO : adapt compute_lengths to accept a map
			Matrix<uint64_t, Dynamic, 1> offsets;
			offsets = trx->streamlines->_offsets;
			trx->streamlines->_lengths = _compute_lengths(offsets, int(trx->header["NB_VERTICES"]));
		}

		else if (folder.compare("dps") == 0)
		{
			std::tuple<int, int> shape;
			trx->data_per_streamline[base] = new MMappedMatrix<DT>();
			int nb_scalar = size / int(trx->header["NB_STREAMLINES"]);

			if (size % int(trx->header["NB_STREAMLINES"]) != 0 || nb_scalar != dim)
			{

				throw std::invalid_argument("Wrong dps size/dimensionality");
			}
			else
			{
				shape = std::make_tuple(trx->header["NB_STREAMLINES"], nb_scalar);
			}
			trx->data_per_streamline[base]->mmap = trxmmap::_create_memmap(filename, shape, "r+", ext.substr(1, ext.size() - 1), mem_adress);

			if (ext.compare("float16") == 0)
			{
				new (&(trx->data_per_streamline[base]->_matrix)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->data_per_streamline[base]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else if (ext.compare("float32") == 0)
			{
				new (&(trx->data_per_streamline[base]->_matrix)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->data_per_streamline[base]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else
			{
				new (&(trx->data_per_streamline[base]->_matrix)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->data_per_streamline[base]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
			}
		}

		else if (folder.compare("dpv") == 0)
		{
			std::tuple<int, int> shape;
			trx->data_per_vertex[base] = new ArraySequence<DT>();
			int nb_scalar = size / int(trx->header["NB_VERTICES"]);

			if (size % int(trx->header["NB_VERTICES"]) != 0 || nb_scalar != dim)
			{

				throw std::invalid_argument("Wrong dpv size/dimensionality");
			}
			else
			{
				shape = std::make_tuple(trx->header["NB_VERTICES"], nb_scalar);
			}
			trx->data_per_vertex[base]->mmap_pos = trxmmap::_create_memmap(filename, shape, "r+", ext.substr(1, ext.size() - 1), mem_adress);

			if (ext.compare("float16") == 0)
			{
				new (&(trx->data_per_vertex[base]->_data)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->data_per_vertex[base]->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else if (ext.compare("float32") == 0)
			{
				new (&(trx->data_per_vertex[base]->_data)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->data_per_vertex[base]->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else
			{
				new (&(trx->data_per_vertex[base]->_data)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->data_per_vertex[base]->mmap_pos.data()), std::get<0>(shape), std::get<1>(shape));
			}

			new (&(trx->data_per_vertex[base]->_offsets)) Map<Matrix<uint64_t, Dynamic, 1>>(reinterpret_cast<uint64_t *>(trx->streamlines->mmap_off.data()), std::get<0>(shape), std::get<1>(shape));
			trx->data_per_vertex[base]->_lengths = trx->streamlines->_lengths;
		}

		else if (folder.compare("dpg") == 0)
		{
			std::tuple<int, int> shape;
			trx->data_per_streamline[base] = new MMappedMatrix<DT>();

			if (size != dim)
			{

				throw std::invalid_argument("Wrong dpg size/dimensionality");
			}
			else
			{
				shape = std::make_tuple(1, size);
			}

			std::string data_name = std::string(basename(const_cast<char *>(base.c_str())));
			std::string sub_folder = std::string(basename(const_cast<char *>(folder.c_str())));

			trx->data_per_group[sub_folder][data_name] = new MMappedMatrix<DT>();
			trx->data_per_group[sub_folder][data_name]->mmap = trxmmap::_create_memmap(filename, shape, "r+", ext.substr(1, ext.size() - 1), mem_adress);

			if (ext.compare("float16") == 0)
			{
				new (&(trx->data_per_group[sub_folder][data_name]->_matrix)) Map<Matrix<half, Dynamic, Dynamic>>(reinterpret_cast<half *>(trx->data_per_group[sub_folder][data_name]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else if (ext.compare("float32") == 0)
			{
				new (&(trx->data_per_group[sub_folder][data_name]->_matrix)) Map<Matrix<float, Dynamic, Dynamic>>(reinterpret_cast<float *>(trx->data_per_group[sub_folder][data_name]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
			}
			else
			{
				new (&(trx->data_per_group[sub_folder][data_name]->_matrix)) Map<Matrix<double, Dynamic, Dynamic>>(reinterpret_cast<double *>(trx->data_per_group[sub_folder][data_name]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
			}
		}

		else if (folder.compare("groups") == 0)
		{
			std::tuple<int, int> shape;
			if (dim != 1)
			{
				throw std::invalid_argument("Wrong group dimensionality");
			}
			else
			{
				shape = std::make_tuple(size, 1);
			}
			trx->groups[base] = new MMappedMatrix<uint32_t>();
			trx->groups[base]->mmap = trxmmap::_create_memmap(filename, shape, "r+", ext.substr(1, ext.size() - 1), mem_adress);
			new (&(trx->groups[base]->_matrix)) Map<Matrix<uint32_t, Dynamic, Dynamic>>(reinterpret_cast<uint32_t *>(trx->groups[base]->mmap.data()), std::get<0>(shape), std::get<1>(shape));
		}
		else
		{
			// TODO: logger
			std::cout << elem_filename << " is not part of a valid structure." << std::endl;
		}
	}
	if (trx->streamlines->_data.size() == 0 || trx->streamlines->_offsets.size() == 0)
	{

		throw std::invalid_argument("Missing essential data.");
	}

	return trx;
}

template <typename DT>
TrxFile<DT> *load_from_zip(std::string filename)
{
	// TODO: check error values
	int *errorp;
	zip_t *zf = zip_open(filename.c_str(), 0, errorp);
	json header = load_header(zf);

	std::map<std::string, std::tuple<long long, long long>> file_pointer_size;
	long long global_pos = 0;
	long long mem_address = 0;

	int num_entries = zip_get_num_entries(zf, ZIP_FL_UNCHANGED);

	for (int i = 0; i < num_entries; ++i)
	{
		std::string elem_filename = zip_get_name(zf, i, ZIP_FL_UNCHANGED);

		size_t lastdot = elem_filename.find_last_of(".");

		if (lastdot == std::string::npos)
			continue;
		std::string ext = elem_filename.substr(lastdot + 1, std::string::npos);

		// apparently all zip directory names end with a slash. may be a better way
		if (ext.compare("json") == 0 || elem_filename.rfind("/") == elem_filename.size() - 1)
		{
			continue;
		}

		if (!_is_dtype_valid(ext))
		{
			continue;
			// maybe throw error here instead?
			// throw std::invalid_argument("The dtype is not supported");
		}

		if (ext.compare("bit") == 0)
		{
			ext = "bool";
		}

		// get file stats
		zip_stat_t sb;

		if (zip_stat(zf, elem_filename.c_str(), ZIP_FL_UNCHANGED, &sb) != 0)
		{
			return NULL;
		}

		std::ifstream file(filename, std::ios::binary);
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

		long long size = sb.comp_size / _sizeof_dtype(ext);
		file_pointer_size[elem_filename] = {mem_address, size};
	}
	return TrxFile<DT>::_create_trx_from_pointer(header, file_pointer_size, filename);
}