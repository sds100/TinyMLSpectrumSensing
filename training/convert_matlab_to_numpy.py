import spectrum_painting_data as sp_data

sp_data.convert_matlab_to_numpy(matlab_dir="data/matlab-random",
                                numpy_dir="data/numpy-random",
                                classes=["Z", "B", "W", "BW", "ZB", "ZW", "ZBW"],
                                snr_list=[10, 20, 30])
