import spectrum_painting_data as sp_data

sp_data.convert_matlab_to_numpy(matlab_dir="data/matlab-2",
                                numpy_dir="data/numpy-2",
                                classes=["Z", "B", "W", "BW", "ZB", "ZW", "ZBW"],
                                snr_list=[30])
