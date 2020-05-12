// ---------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// lattice_converter.cuh
// Definition of the lattice converter functions.
//
// Author: Gergely Ferenc, Racz
// ---------------------------------------------------------------------------

#pragma once

#ifndef _CUDALATTICE_LATTICE_CONVERTER_CUH_
#define _CUDALATTICE_LATTICE_CONVERTER_CUH_

#include <cstring>
#include <cmath>
#include <limits>

#include <cuda_runtime.h>     // CUDA runtime library
#include "helper_math.h"      // CUDA math helper (vector operations)

#include "macro.h"
#include "enum_types.h"
#include "texture_fetch.cuh"

namespace CudaLattice
{
	namespace Texture
	{
		/**
		 * Implementation of internal helper functions and classes. 
		 */
		namespace Internal
		{
			//////////////////////////////////////////////////
			//  VOLUME TYPE INFORMATIONS

			/**
			 * This class encapsulates the informations about the
			 * way of binding the volume to a texture and provides 
			 * a function to convert its values back from texture values. 
			 */
			template <typename T>
			struct VolumeTypeInfo
			{
				//static const cudaTextureReadMode readMode = cudaReadModeNormalizedFloat;

				///**
				// * Convert texture samples back to original volume values.
				// * @param sampled  - sample taken by a texture fetch
				// * @return  - original volume value
				// */
				//DEVICE FORCEINLINE
				//static T convert(float sample)
				//{
				//	return (T)(saturate(sample) * (std::numeric_limits<T>::max() - std::numeric_limits<T>::min()) + std::numeric_limits<T>::min());
				//}
			};
			
			/**
			 * This class encapsulates the informations about the
			 * way of binding the volume to a texture and provides
			 * a function to convert its values back from texture values.
			 * @specialization float
			 */
			template<>
			struct VolumeTypeInfo<float>
			{
				static const cudaTextureReadMode readMode = cudaReadModeElementType;

				/**
				 * Convert texture samples back to original volume values.
				 * @param sampled  - sample taken by a texture fetch
				 * @return  - original volume value
				 */
				DEVICE FORCEINLINE
				static float convert(float sample)
				{
					return sample;
				}
			};

			/**
			 * This class encapsulates the informations about the
			 * way of binding the volume to a texture and provides
			 * a function to convert its values back from texture values.
			 * @specialization unsigned char
			 */
			template<>
			struct VolumeTypeInfo<unsigned char>
			{
				static const cudaTextureReadMode readMode = cudaReadModeNormalizedFloat;

				/**
				 * Convert texture samples back to original volume values.
				 * @param sampled  - sample taken by a texture fetch
				 * @return  - original volume value
				 */
				DEVICE FORCEINLINE
				static unsigned char convert(float sample)
				{
					return (unsigned char)(saturate(sample) * 255.0f);
				}
			};

			/**
			 * This class encapsulates the informations about the
			 * way of binding the volume to a texture and provides
			 * a function to convert its values back from texture values.
			 * @specialization char
			 */
			template<>
			struct VolumeTypeInfo<char>
			{
				static const cudaTextureReadMode readMode = cudaReadModeNormalizedFloat;

				/**
				 * Convert texture samples back to original volume values.
				 * @param sampled  - sample taken by a texture fetch
				 * @return  - original volume value
				 */
				DEVICE FORCEINLINE
				static char convert(float sample)
				{
					return (char)(saturate(sample) * 255.0f - 128.0f);
				}
			};

			/**
			 * This class encapsulates the informations about the
			 * way of binding the volume to a texture and provides
			 * a function to convert its values back from texture values.
			 * @specialization unsigned short
			 */
			template<>
			struct VolumeTypeInfo<unsigned short>
			{
				static const cudaTextureReadMode readMode = cudaReadModeNormalizedFloat;

				/**
				 * Convert texture samples back to original volume values.
				 * @param sampled  - sample taken by a texture fetch
				 * @return  - original volume value
				 */
				DEVICE FORCEINLINE
				static unsigned short convert(float sample)
				{
					return (unsigned short)(saturate(sample) * 65535.0f);
				}
			};

			/**
			 * This class encapsulates the informations about the
			 * way of binding the volume to a texture and provides
			 * a function to convert its values back from texture values.
			 * @specialization short
			 */
			template<>
			struct VolumeTypeInfo<short>
			{
				static const cudaTextureReadMode readMode = cudaReadModeNormalizedFloat;

				/**
				 * Convert texture samples back to original volume values.
				 * @param sampled  - sample taken by a texture fetch
				 * @return  - original volume value
				 */
				DEVICE FORCEINLINE
				static short convert(float sample)
				{
					return (short)(saturate(sample) * 65535.0f - 32768.0f);
				}
			};


			//////////////////////////////////////////////////
			//  FILTER TYPE INFORMATIONS

			template <enum Filter::FilterType filter>
			struct FilterTypeInfo
			{};

			template <>
			struct FilterTypeInfo<Filter::TrilinearBSpline>
			{
				static const enum Format::TextureFormatType texture_format = Format::Blocked;
			};

			template <>
			struct FilterTypeInfo<Filter::LinearBoxSpline>
			{
				static const enum Format::TextureFormatType texture_format = Format::Layered;
			};
			

			//////////////////////////////////////////////////
			//  LATTICESIZE

			/**
			 * Policy structure for the lattice size conversion. 
			 */
			template <
				enum Lattice::LatticeType lattice_from,
				enum Lattice::LatticeType lattice_to,
				enum Resizer::ResizerType resizer
			>
			struct LatticeSizePolicy
			{};
			
			/**
			 * Policy structure for the lattice size conversion. 
			 * @specialization Resizer::PreserveDensity
			 */
			template <
				enum Lattice::LatticeType lattice_from,
				enum Lattice::LatticeType lattice_to
			>
			struct LatticeSizePolicy<lattice_from, lattice_to, Resizer::PreserveDensity>
			{
				/**
				 * Converts the extents of one lattice to another so that the
				 * resulting lattice has nearly the same density.
				 * @param from_size  - size of the original lattice
				 * @return  - extents of the other lattice
				 */
				inline static cudaExtent size(cudaExtent size_from)
				{
					const float factor = cbrtf(lattice_from) / cbrtf(lattice_to);

					return make_cudaExtent(
						(size_t)ceilf((float)size_from.width * factor),
						(size_t)ceilf((float)size_from.height * factor),
						lattice_to * (size_t)ceilf((float)size_from.depth * factor));
				}
			};

			/**
			 * Policy structure for the lattice size conversion. 
			 * @specialization Resizer::DoubleDense
			 */
			template <
				enum Lattice::LatticeType lattice_from,
				enum Lattice::LatticeType lattice_to
			>
			struct LatticeSizePolicy<lattice_from, lattice_to, Resizer::DoubleDense>
			{
				/**
				 * Converts the extents of one lattice to another so that the
				 * resulting lattice is about double as dense as the original one.
				 * @param from_size  - size of the original lattice
				 * @return  - extents of the other lattice
				 */
				inline static cudaExtent size(cudaExtent size_from)
				{
					//static const float cuberootTwo = 1.2599210498949f;
					const float factor = cbrtf(lattice_from) / cbrtf(lattice_to) * 1.2599210498949f;

					return make_cudaExtent(
						(size_t)ceilf((float)size_from.width * factor),
						(size_t)ceilf((float)size_from.height * factor),
						lattice_to * (size_t)ceilf((float)size_from.depth * factor));
				}
			};

			/**
			 * Policy structure for the lattice size conversion. 
			 * @specialization Resizer::HalfDense
			 */
			template <
				enum Lattice::LatticeType lattice_from,
				enum Lattice::LatticeType lattice_to
			>
			struct LatticeSizePolicy<lattice_from, lattice_to, Resizer::HalfDense>
			{
				/**
				 * Converts the extents of one lattice to another so that the
				 * resulting lattice is about half as dense as the original one.
				 * @param from_size  - size of the original lattice
				 * @return  - extents of the other lattice
				 */
				inline static cudaExtent size(cudaExtent size_from)
				{
					//static const float oneOverCuberootTwo = 0.7937005259841f;
					const float factor = cbrtf(lattice_from) / cbrtf(lattice_to) * 0.7937005259841f;

					return make_cudaExtent(
						(size_t)ceilf((float)size_from.width * factor),
						(size_t)ceilf((float)size_from.height * factor),
						lattice_to * (size_t)ceilf((float)size_from.depth * factor));
				}
			};


			//////////////////////////////////////////////////
			//  TEXTUREFORMAT

			template <
				enum Lattice::LatticeType lattice,
				enum Format::TextureFormatType format
			>
			struct TextureFormatPolicy
			{};

			template <enum Format::TextureFormatType format>
			struct TextureFormatPolicy<Lattice::CC, format>
			{
				DEVICE FORCEINLINE
				static float4 indexToCartesian(int x, int y, int z, int4 /*lattice_size*/)
				{
					return make_float4((float)x, (float)y, (float)z, 0.0f);
				}
			};

			template <>
			struct TextureFormatPolicy<Lattice::BCC, Format::Blocked>
			{
				DEVICE FORCEINLINE
				static float4 indexToCartesian(int x, int y, int z, int4 lattice_size)
				{
					const int level = z / (lattice_size.z / 2);
					return make_float4(
						x * 2 + level,
						y * 2 + level,
						(z - level * (lattice_size.z / 2)) * 2 + level,
						0.0f);
				}
			};

			template <>
			struct TextureFormatPolicy<Lattice::FCC, Format::Blocked>
			{
				DEVICE FORCEINLINE
				static float4 indexToCartesian(int x, int y, int z, int4 lattice_size)
				{
					const int level = z / (lattice_size.z / 4);
					return make_float4(
						x * 2 + (level & 1),
						y * 2 + (level & 2),
						(z - level * (lattice_size.z / 4)) * 2 + 
							((level & 1) ^ ((level / 2) & 1)),
						0.0f);
				}
			};

			template <>
			struct TextureFormatPolicy<Lattice::BCC, Format::Layered>
			{
				DEVICE FORCEINLINE
				static float4 indexToCartesian(int x, int y, int z, int4 /*lattice_size*/)
				{
					return make_float4(
						x * 2 + (z & 1),
						y * 2 + (z & 1),
						z,
						0.0f);
				}
			};

			template <>
			struct TextureFormatPolicy<Lattice::FCC, Format::Layered>
			{
				DEVICE FORCEINLINE
				static float4 indexToCartesian(int x, int y, int z, int4 /*lattice_size*/)
				{
					return make_float4(
						x * 2 + (z & 1),
						y * 2 + (z & 2),
						(z / 4) * 2 + ((z & 1) ^ ((z / 2) & 1)),
						0.0f);
				}
			};


			//////////////////////////////////////////////////
			//  LATTICE CONVERTER KERNEL

			/**
			 * This kernel does the conversion from an arbitrary lattice to another representation. 
			 * It uses a simple trilinear filter to have a continuous signal on the original lattice. 
			 * The output is a representation on the other lattice.
			 * @param tex_from  - the input representation as a texture
			 * @param surf_to   - the output representation as a surface
			 * @param size_to   - extents of the output representation
			 */
			template <
				typename T,
				enum Lattice::LatticeType lattice_from,  // input lattice
				enum Lattice::LatticeType lattice_to,    // output lattice
				enum Filter::FilterType filter,          // continuous reconstruction filter on the input lattice
				enum Format::TextureFormatType format    // texture format
			>
			__global__
			void latticeConvert_kernel(cudaTextureObject_t tex_from, cudaSurfaceObject_t surf_to, int4 size_to)
			{
				int x = blockIdx.x * blockDim.x + threadIdx.x;
				int y = blockIdx.y * blockDim.y + threadIdx.y;
				int z = blockIdx.z * blockDim.z + threadIdx.z;

				// check bounds
				if ((x >= size_to.x) || (y >= size_to.y) || (z >= size_to.z)) return;

				// calculate sampling distance
				const float padding = (lattice_to != Lattice::CC) * 0.5f; // additional 0.5f padding for non-CC lattices
				float4 sd = {
					1.0f / (size_to.x + padding),
					1.0f / (size_to.y + padding),
					1.0f / (size_to.z / (float)lattice_to + padding),
					0.0f };

				// grab the integer Cartesian coordinates of the fetch position
				// and calculate the real normalized texture coordinates
				float4 coords = TextureFormatPolicy<lattice_to, format>::indexToCartesian(x, y, z, size_to);
				float4 fetch = (sd * coords + sd) * 0.5f;

				float value = Tex3D<float, lattice_from, filter, Coordinates::Normalized>::fetch(tex_from, fetch.x, fetch.y, fetch.z);
				surf3Dwrite(VolumeTypeInfo<T>::convert(value), surf_to, x*sizeof(T), y, z);
			}
		}


		////////////////////////////////////////
		//  CC to BCC converter (simple)
		////////////////////////////////////////

		template <
			enum Filter::FilterType filter_input  = Filter::TrilinearBSpline,  // continuous reconstruction filter on the input lattice
			enum Filter::FilterType filter_output = Filter::TrilinearBSpline,  // continuous reconstruction filter on the output lattice
			enum Resizer::ResizerType resizer = Resizer::PreserveDensity,      // lattice resizer policy
			typename T
		>
		cudaError_t cc2bcc(T* ccData, T** bccData, cudaExtent ccSize, cudaExtent* bccSizeOut)
		{
			cudaError_t error;
			cudaArray_t d_ccData, d_bccData;
			cudaTextureObject_t ccTex;
			cudaSurfaceObject_t bccSurf;

			// create 3D array for the input
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
			error = cudaMalloc3DArray(&d_ccData, &channelDesc, ccSize);
			if (error != cudaSuccess) return error;

			// copy input data to 3D array
			cudaMemcpy3DParms copyParams;
			std::memset(&copyParams, 0, sizeof(cudaMemcpy3DParms));
			copyParams.srcPtr = make_cudaPitchedPtr(ccData, ccSize.width*sizeof(T), ccSize.width, ccSize.height);
			copyParams.dstArray = d_ccData;
			copyParams.extent = make_cudaExtent(ccSize.width, ccSize.height, ccSize.depth);
			copyParams.kind = cudaMemcpyHostToDevice;
			error = cudaMemcpy3D(&copyParams);
			if (error != cudaSuccess) return error;

			// create texture object for the input data
			cudaResourceDesc ccResDesc;
			std::memset(&ccResDesc, 0, sizeof(cudaResourceDesc));
			ccResDesc.resType = cudaResourceTypeArray;
			ccResDesc.res.array.array = d_ccData;

			cudaTextureDesc ccTexDesc;
			std::memset(&ccTexDesc, 0, sizeof(cudaTextureDesc));
			ccTexDesc.addressMode[0] = ccTexDesc.addressMode[1] = ccTexDesc.addressMode[2] = cudaAddressModeClamp;
			ccTexDesc.filterMode = cudaFilterModeLinear;
			ccTexDesc.normalizedCoords = true;
			ccTexDesc.readMode = Internal::VolumeTypeInfo<T>::readMode;

			error = cudaCreateTextureObject(&ccTex, &ccResDesc, &ccTexDesc, NULL);
			if (error != cudaSuccess) return error;


			// create 3D array for the output
			const cudaExtent bccSize = Internal::LatticeSizePolicy<Lattice::CC, Lattice::BCC, resizer>::size(ccSize);
			*bccSizeOut = bccSize;
			error = cudaMalloc3DArray(&d_bccData, &channelDesc, bccSize, cudaArraySurfaceLoadStore);
			if (error != cudaSuccess) return error;

			// create surface object for the output data
			cudaResourceDesc bccResDesc;
			std::memset(&bccResDesc, 0, sizeof(cudaResourceDesc));
			bccResDesc.resType = cudaResourceTypeArray;
			bccResDesc.res.array.array = d_bccData;

			error = cudaCreateSurfaceObject(&bccSurf, &bccResDesc);
			if (error != cudaSuccess) return error;


			const dim3 blockDim(8, 8, 4); // 8*8*4=256
			const dim3 gridDim(
				((uint)bccSize.width - 1) / blockDim.x + 1,
				((uint)bccSize.height - 1) / blockDim.y + 1,
				((uint)bccSize.depth - 1) / blockDim.z + 1);
			const int4 bccSize_ = {(int)bccSize.width, (int)bccSize.height, (int)bccSize.depth, 0};
			Internal::latticeConvert_kernel<T, Lattice::CC, Lattice::BCC, filter_input, Internal::FilterTypeInfo<filter_output>::texture_format> <<<gridDim, blockDim>>> (ccTex, bccSurf, bccSize_);
            cudaDeviceSynchronize();
            error = cudaGetLastError();
            if (error != cudaSuccess) return error;


			// copy output data to 3D array
			*bccData = new T[bccSize.width * bccSize.height * bccSize.depth];

			std::memset(&copyParams, 0, sizeof(cudaMemcpy3DParms));
			copyParams.srcArray = d_bccData;
			copyParams.dstPtr = make_cudaPitchedPtr(*bccData, bccSize.width*sizeof(T), bccSize.width, bccSize.height);
			copyParams.extent = bccSize;
			copyParams.kind = cudaMemcpyDeviceToHost;
			error = cudaMemcpy3D(&copyParams);
			if (error != cudaSuccess) return error;

			error = cudaDestroySurfaceObject(bccSurf);
			if (error != cudaSuccess) return error;
			error = cudaDestroyTextureObject(ccTex);
			if (error != cudaSuccess) return error;
			error = cudaFreeArray(d_ccData);
			if (error != cudaSuccess) return error;
			error = cudaFreeArray(d_bccData);
			if (error != cudaSuccess) return error;

			return cudaSuccess;
		}


		////////////////////////////////////////
		//  CC to FCC converter (simple)
		////////////////////////////////////////

		template <
			enum Filter::FilterType filter_input  = Filter::TrilinearBSpline,  // continuous reconstruction filter on the input lattice
			enum Filter::FilterType filter_output = Filter::TrilinearBSpline,  // continuous reconstruction filter on the output lattice
			enum Resizer::ResizerType resizer = Resizer::PreserveDensity,      // lattice resizer policy
			typename T
		>
		cudaError_t cc2fcc(T* ccData, T** fccData, cudaExtent ccSize, cudaExtent* fccSizeOut)
		{
			cudaError_t error;
			cudaArray_t d_ccData, d_fccData;
			cudaTextureObject_t ccTex;
			cudaSurfaceObject_t fccSurf;

			// create 3D array for the input
			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
			error = cudaMalloc3DArray(&d_ccData, &channelDesc, ccSize);
			if (error != cudaSuccess) return error;

			// copy input data to 3D array
			cudaMemcpy3DParms copyParams;
			std::memset(&copyParams, 0, sizeof(cudaMemcpy3DParms));
			copyParams.srcPtr = make_cudaPitchedPtr(ccData, ccSize.width*sizeof(T), ccSize.width, ccSize.height);
			copyParams.dstArray = d_ccData;
			copyParams.extent = make_cudaExtent(ccSize.width, ccSize.height, ccSize.depth);
			copyParams.kind = cudaMemcpyHostToDevice;
			error = cudaMemcpy3D(&copyParams);
			if (error != cudaSuccess) return error;

			// create texture object for the input data
			cudaResourceDesc ccResDesc;
			std::memset(&ccResDesc, 0, sizeof(cudaResourceDesc));
			ccResDesc.resType = cudaResourceTypeArray;
			ccResDesc.res.array.array = d_ccData;

			cudaTextureDesc ccTexDesc;
			std::memset(&ccTexDesc, 0, sizeof(cudaTextureDesc));
			ccTexDesc.addressMode[0] = ccTexDesc.addressMode[1] = ccTexDesc.addressMode[2] = cudaAddressModeClamp;
			ccTexDesc.filterMode = cudaFilterModeLinear;
			ccTexDesc.normalizedCoords = true;
			ccTexDesc.readMode = Internal::VolumeTypeInfo<T>::readMode;

			error = cudaCreateTextureObject(&ccTex, &ccResDesc, &ccTexDesc, NULL);
			if (error != cudaSuccess) return error;


			// create 3D array for the output
			const cudaExtent fccSize = Internal::LatticeSizePolicy<Lattice::CC, Lattice::FCC, resizer>::size(ccSize);
			*fccSizeOut = fccSize;
			error = cudaMalloc3DArray(&d_fccData, &channelDesc, fccSize, cudaArraySurfaceLoadStore);
			if (error != cudaSuccess) return error;

			// create surface object for the output data
			cudaResourceDesc fccResDesc;
			std::memset(&fccResDesc, 0, sizeof(cudaResourceDesc));
			fccResDesc.resType = cudaResourceTypeArray;
			fccResDesc.res.array.array = d_fccData;

			error = cudaCreateSurfaceObject(&fccSurf, &fccResDesc);
			if (error != cudaSuccess) return error;


			const dim3 blockDim(8, 8, 4); // 8*8*4=256
			const dim3 gridDim(
				((uint)fccSize.width - 1) / blockDim.x + 1,
				((uint)fccSize.height - 1) / blockDim.y + 1,
				((uint)fccSize.depth - 1) / blockDim.z + 1);
			const int4 fccSize_ = {(int)fccSize.width, (int)fccSize.height, (int)fccSize.depth, 0};
			Internal::latticeConvert_kernel<T, Lattice::CC, Lattice::FCC, filter_input, Internal::FilterTypeInfo<filter_output>::texture_format> <<<gridDim, blockDim>>> (ccTex, fccSurf, fccSize_);
            cudaDeviceSynchronize();
            error = cudaGetLastError();
            if (error != cudaSuccess) return error;


			// copy output data to 3D array
			*fccData = new T[fccSize.width * fccSize.height * fccSize.depth];

			std::memset(&copyParams, 0, sizeof(cudaMemcpy3DParms));
			copyParams.srcArray = d_fccData;
			copyParams.dstPtr = make_cudaPitchedPtr(*fccData, fccSize.width*sizeof(T), fccSize.width, fccSize.height);
			copyParams.extent = fccSize;
			copyParams.kind = cudaMemcpyDeviceToHost;
			error = cudaMemcpy3D(&copyParams);
			if (error != cudaSuccess) return error;

			error = cudaDestroySurfaceObject(fccSurf);
			if (error != cudaSuccess) return error;
			error = cudaDestroyTextureObject(ccTex);
			if (error != cudaSuccess) return error;
			error = cudaFreeArray(d_ccData);
			if (error != cudaSuccess) return error;
			error = cudaFreeArray(d_fccData);
			if (error != cudaSuccess) return error;

			return cudaSuccess;
		}
	}
}

#endif /* _CUDALATTICE_LATTICE_CONVERTER_CUH_ */