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
Matrix<u_int32_t, Dynamic, Dynamic, RowMajor> _compute_lengths(const MatrixBase<DT> &offsets, int nb_vertices)
{
	if (offsets.size() > 1)
	{
		int last_elem_pos = _dichotomic_search(offsets);
		Matrix<u_int32_t, 1, Dynamic, RowMajor> lengths;

		if (last_elem_pos == offsets.size() - 1)
		{
			// ediff1d
			Matrix<u_int32_t, Dynamic, Dynamic> tmp(offsets.template cast<u_int32_t>());
			Map<RowVector<u_int32_t, Dynamic>> v(tmp.data(), tmp.size());
			lengths.resize(v.rows(), v.cols());
			lengths << v(seq(1, last)) - v(seq(0, last - 1)), u_int32_t(nb_vertices - offsets(last));
		}
		else
		{
			Matrix<u_int32_t, Dynamic, Dynamic> tmp(offsets.template cast<u_int32_t>());
			tmp(last_elem_pos + 1) = u_int32_t(nb_vertices);

			// ediff1d
			Map<RowVector<u_int32_t, Dynamic>> v(tmp.data(), tmp.size());
			lengths.resize(v.rows(), v.cols());
			lengths << v(seq(1, last)) - v(seq(0, last - 1)), u_int32_t(0);
			lengths(last_elem_pos + 1) = u_int32_t(0);
		}
		return lengths;
	}
	if (offsets.size() == 1)
	{
		Matrix<u_int32_t, 1, 1, RowMajor> lengths(nb_vertices);
		return lengths;
	}

	Matrix<u_int32_t, 1, 1, RowMajor> lengths(0);
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
