from source.data_reduction import DataReduction

data = DataReduction('testdata/data_comb', 'testdata/data_comb/reduced_data')

# data.reduce_bias()
data.reduce_darks()