from source.data_reduction import DataReduction

data = DataReduction('testdata/data_comb', 'testdata/data_comb/reduced_data')

print(data.check_master('bias', master='master_bias.fits'))