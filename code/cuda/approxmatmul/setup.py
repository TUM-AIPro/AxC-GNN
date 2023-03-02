import setuptools
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setuptools.setup(
    name='apxop',
    version="0.0.15",
    author='Rodion Novkin',
    author_email='rodion.novkin@iti.uni-stuttgart.de',
    description='Approximate Operations For PyTorch Tensors',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    ext_modules=[
        CUDAExtension('apxop', [
            'apxop.cpp',
            'apxop_kernels.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

# python setup.py install --user
