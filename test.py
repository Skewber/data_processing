from source.data_reduction import DataReduction

data = DataReduction('testdata/data_comb', 'testdata/data_comb/reduced_data')
print(data.imagetypes)

# data.reduce_bias()
# data.reduce_darks()
# data.reduce_flats()
# data.reduce_lights()

files = ['NGC2281-0001_lightB.fits', 'NGC2281-0002_lightB.fits', 'NGC2281-0003_lightB.fits']
for i in range(len(files)):
    files[i] = f'./testdata/data_comb/reduced_data/{files[i]}'
data.stack_light('none', filelist=[files], out_names="ngc2281_B")
# data.stack_light('none')