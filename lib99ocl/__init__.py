import os

PACKAGE_PATH = os.path.dirname(os.path.abspath(__file__))
LIB99OCL = PACKAGE_PATH.split('/')
LIB99OCL[-1] = 'src'
LIB99OCL = '/'.join( LIB99OCL )
# print(PACKAGE_PATH)
# print(LIB99OCL)

