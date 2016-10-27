import h5py
import pickle
import pprint
# f = h5py.File("mytestfile.hdf5", "w")
#
# dset = f.create_dataset('mydata', data=[(1,2), (3,4)])
#
#
# f2 = h5py.File('mytestfile.hdf5','r')
# print f2['mydata'][:]

# with open('dtest.dat', 'wb') as outfile:
#     pickle.dump([{'a':1,'b':2},{'c':3,'d':4}], outfile)

with open('dtest.dat', 'rb') as f:
    data = pickle.load(f)
    print data