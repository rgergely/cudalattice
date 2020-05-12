# CudaLattice

This project is a CUDA C++ header-only library with a set of functions for easy conversion and filtration of volumetric data sampled on CC, BCC and FCC lattices. The aim of this project is to provide a powerful and easily applicable tool for developers to utilize the advantages of alternative lattices in their projects without deep knowledge of related alternative reconstruction schemes. 

The library implements the following reconstruction schemes (lattice, filter pairs):

- CC lattice, trilinear B-spline filter
- CC lattice, cubic B-spline filter
- BCC lattice, linear (four-directional) box spline filter
- BCC lattice, trilinear B-spline filter
- BCC lattice, cubic B-spline filter
- BCC lattice, Cosine-Weighted triLinear B-spline filter (CWLB)
- BCC lattice, Cosine-Weighted triCubic B-spline filter (CWCB)
- FCC lattice, trilinear B-spline filter
- FCC lattice, cubic B-spline filter
- FCC lattice, Cosine-Weighted triLinear B-spline filter (CWLB)
- FCC lattice, Cosine-Weighted triCubic B-spline filter (CWCB)

## Usage

Before any call to the library, you have to include cudalattice/cudalattice.cuh in your C++ code. 

### Conversion (resampling)

To resample a volumetric array from CC to BCC or FCC lattice, use `cc2bcc` and `cc2fcc` template functions, respectively. They take an input CC array allocated on the CPU and convert it to another host array containing the BCC/FCC variant. 

Both functions have four parameters. The first parameter is the CC sampled 3D host array to be converted. The second parameter is a pointer of the same type, which will be directed to a new dynamic array on host holding the result of the conversion. The third parameter specifies the extents of the input array, and the fourth parameter is a pointer to a `cudaExtent` where the functions write the extents of the output array. 

Both functions have three template parameters as well. The first and second template parameters specify the reconstruction filter used on the input and output lattice, respectively. They can be given the following values:

- `Filter::TrilinearBSpline`
- `Filter::CubicBSpline`
- `Filter::LinearBoxSpline`
- `Filter::CWLB`
- `Filter::CWCB`

The resampling policy can be configured with the third template parameter. The library provides equally dense, half dense and double dense sampling policies. That is, the output will have the same amount of samples, half the number of input samples or twice the number of input samples. 

- `Resizer::PreserveDensity`
- `Resizer::HalfDense`
- `Resizer::DoubleDense`

### Filtration (reconstruction)

The result of conversion is a BCC or FCC representation of the input volumetric data. The library provides functions also for continuous recontruction of the resampled data (e.g., for visualization or simulation). 

You have to copy your array to the device and bind it to a texture object or texture reference using the CUDA API as you would do with a regular volumetric data (both texture objects and texture references are supported by CudaLattice). 

Then you can sample your texture by simply calling `bccTex3D` or `fccTex3D` template function. They work similar to the built-in `tex3D` function of the CUDA API. The first parameter is the texture object or texture reference. The next three parameters specify the 3D sample position. For some of the calculations the array extents are also necessary. Therefore, both `bccTex3D` and `fccTex3D` have three further parameters that specify the extents of the underlying array. 

Both functions have template parameters as well. You have to specify the filter used by the conversion, the type of coordinates (`Coordinates::Normalized`, `Coordinates::Unnormalized`) and the type of texture samples. In case of texture references, you also have to specify *read mode* of the texture (`cudaTextureReadMode`). 

### Contact

If you have any questions or want to discuss the topic, contact me at *gracz [at] iit [dot] bme [dot] hu*.