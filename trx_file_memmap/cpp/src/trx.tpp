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