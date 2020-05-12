// ---------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// texture_fetch.cuh
// Definition of texture fetch functions.
//
// Author: Gergely Ferenc, Racz
// ---------------------------------------------------------------------------

#pragma once

#ifndef _CUDALATTICE_TEXTURE_FETCH_CUH_
#define _CUDALATTICE_TEXTURE_FETCH_CUH_

#include <cstring>

#include <cuda_runtime.h>     // CUDA runtime library
#include "helper_math.h"      // CUDA math helper (vector operations)

#include "macro.h"
#include "utility.h"
#include "enum_types.h"

namespace CudaLattice
{
    namespace Texture
    {
        /**
         * Implementation of internal helper functions and classes. 
         */
        namespace Internal
        {
            //inline HOST_DEVICE float lambda(float* newValue = nullptr)
            //{
            //    static __constant__ float c_lambda;
            //    #ifndef __CUDA_ARCH__
                  //  if (newValue != nullptr)
            //        {
            //            cudaError_t error = cudaMemcpyToSymbol(c_lambda, newValue, sizeof(float));
            //            if (error != cudaSuccess) return -1.0f;
            //        }
            //        return 1.0f;
            //    #else
            //        return c_lambda;
            //    #endif
            //}

            ////////////////////////////////////////////////
            //  LOW LEVEL TEXTURE FETCH HELPER FUNCTIONS
            ////////////////////////////////////////////////

            //template <
            //    typename T,
            //    enum cudaTextureReadMode mode
            //>
            //struct TextureFetchTypeInfo
            //{
            //    typedef T type;
            //};

            //template <typename T>
            //struct TextureFetchTypeInfo<T, cudaReadModeNormalizedFloat>
            //{
            //    typedef float type;
            //};

            struct float_texture_fetches_are_supported_only {};

            template <
                typename ValueType,
                enum cudaTextureReadMode mode
            >
            DEVICE FORCEINLINE
            ValueType textureFetch3D(texture<ValueType, cudaTextureType3D, mode> texRef, float x, float y, float z)
            {
                return tex3D(texRef, x, y, z);
            }

            template <typename ValueType>
            DEVICE FORCEINLINE
            ValueType textureFetch3D(cudaTextureObject_t texObj, float x, float y, float z)
            {
                return tex3D<ValueType>(texObj, x, y, z);
            }


            ///////////////////////////////
            //  CUBIC B-SPLINE WEIGHTS
            ///////////////////////////////

            // Inline calculation of the bspline convolution weights, without conditional statements
            template<typename T> 
            DEVICE FORCEINLINE
            void bspline_weights(T fraction, T& w0, T& w1, T& w2, T& w3)
            {
                const T one_frac = 1.0f - fraction;
                const T squared = fraction * fraction;
                const T one_sqd = one_frac * one_frac;

                w0 = 1.0f/6.0f * one_sqd * one_frac;
                w1 = 2.0f/3.0f - 0.5f * squared * (2.0f-fraction);
                w2 = 2.0f/3.0f - 0.5f * one_sqd * (2.0f-one_frac);
                w3 = 1.0f/6.0f * squared * fraction;
            }
        }


        ///////////////////////////////
        //  TEXTURE FETCH CLASSES
        ///////////////////////////////

        template <
            typename ValueType,
            enum Lattice::LatticeType lattice,
            enum Filter::FilterType filter,
            enum Coordinates::CoordinateType coordinates
        >
        struct Tex3D
        {};

        template <
            typename ValueType,
            enum Coordinates::CoordinateType coordinates
        >
        struct Tex3D<ValueType, Lattice::CC, Filter::TrilinearBSpline, coordinates>
        {
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float /*w*/, float /*h*/, float /*d*/)
            {
                return fetch(tex, x, y, z);
            }

            /**
                * Fetches the given CC texture at the specified position.
                * This implementation assumes that the texture is configured to use
                * hardware trilinear support and simply delegates the call to the CUDA runtime. 
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z)
            {
                return Internal::textureFetch3D<ValueType>(tex, x, y, z);
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>
        {
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float /*w*/, float /*h*/, float /*d*/)
            {
                return fetch(tex, x, y, z);
            }

            /**
                * Fetches the given CC texture at the specified position.
                * This implementation uses the efficient GPU implementation of the tricubic 
                * B-spline filter proposed by M. Hadwiger et al. in 2005.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z)
            {
                // shift the coordinate from [0,extent] to [-0.5, extent-0.5]
                const float3 coord_grid {x - 0.5f, y - 0.5f, z - 0.5f};
                const float3 index = floorf(coord_grid);
                const float3 fraction = coord_grid - index;
                float3 w0, w1, w2, w3;
                Internal::bspline_weights(fraction, w0, w1, w2, w3);

                const float3 g0 = w0 + w1;
                const float3 g1 = w2 + w3;
                const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
                const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

                // fetch the eight linear interpolations
                // weighting and fetching is interleaved for performance and stability reasons
                auto tex000 = Internal::textureFetch3D<ValueType>(tex, h0.x, h0.y, h0.z);
                auto tex100 = Internal::textureFetch3D<ValueType>(tex, h1.x, h0.y, h0.z);
                tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
                auto tex010 = Internal::textureFetch3D<ValueType>(tex, h0.x, h1.y, h0.z);
                auto tex110 = Internal::textureFetch3D<ValueType>(tex, h1.x, h1.y, h0.z);
                tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
                tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
                auto tex001 = Internal::textureFetch3D<ValueType>(tex, h0.x, h0.y, h1.z);
                auto tex101 = Internal::textureFetch3D<ValueType>(tex, h1.x, h0.y, h1.z);
                tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
                auto tex011 = Internal::textureFetch3D<ValueType>(tex, h0.x, h1.y, h1.z);
                auto tex111 = Internal::textureFetch3D<ValueType>(tex, h1.x, h1.y, h1.z);
                tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
                tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

                return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Normalized>
        {
            /**
                * Fetches the given CC texture at the specified position.
                * This implementation uses the efficient GPU implementation of the tricubic 
                * B-spline filter proposed by M. Hadwiger et al. in 2005.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float w, float h, float d)
            {
                // shift the coordinate from [0,extent] to [-0.5, extent-0.5]
                const float3 coord_grid {x * w - 0.5f, y * h - 0.5f, z * d - 0.5f};
                const float3 index = floorf(coord_grid);
                const float3 fraction = coord_grid - index;
                float3 w0, w1, w2, w3;
                Internal::bspline_weights(fraction, w0, w1, w2, w3);

                const float3 g0 = w0 + w1;
                const float3 g1 = w2 + w3;
                const float3 h0 = (w1 / g0) - 0.5f + index;  //h0 = w1/g0 - 1, move from [-0.5, extent-0.5] to [0, extent]
                const float3 h1 = (w3 / g1) + 1.5f + index;  //h1 = w3/g1 + 1, move from [-0.5, extent-0.5] to [0, extent]

                // fetch the eight linear interpolations
                // weighting and fetching is interleaved for performance and stability reasons
                auto tex000 = Internal::textureFetch3D<ValueType>(tex, h0.x, h0.y, h0.z);
                auto tex100 = Internal::textureFetch3D<ValueType>(tex, h1.x, h0.y, h0.z);
                tex000 = g0.x * tex000 + g1.x * tex100;  //weigh along the x-direction
                auto tex010 = Internal::textureFetch3D<ValueType>(tex, h0.x, h1.y, h0.z);
                auto tex110 = Internal::textureFetch3D<ValueType>(tex, h1.x, h1.y, h0.z);
                tex010 = g0.x * tex010 + g1.x * tex110;  //weigh along the x-direction
                tex000 = g0.y * tex000 + g1.y * tex010;  //weigh along the y-direction
                auto tex001 = Internal::textureFetch3D<ValueType>(tex, h0.x, h0.y, h1.z);
                auto tex101 = Internal::textureFetch3D<ValueType>(tex, h1.x, h0.y, h1.z);
                tex001 = g0.x * tex001 + g1.x * tex101;  //weigh along the x-direction
                auto tex011 = Internal::textureFetch3D<ValueType>(tex, h0.x, h1.y, h1.z);
                auto tex111 = Internal::textureFetch3D<ValueType>(tex, h1.x, h1.y, h1.z);
                tex011 = g0.x * tex011 + g1.x * tex111;  //weigh along the x-direction
                tex001 = g0.y * tex001 + g1.y * tex011;  //weigh along the y-direction

                return (g0.z * tex000 + g1.z * tex001);  //weigh along the z-direction
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::BCC, Filter::LinearBoxSpline, Coordinates::Unnormalized>
        {
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float /*w*/, float /*h*/, float /*d*/)
            {
                return fetch(tex, x, y, z);
            }

            /**
                * Fetches the given BCC texture at the specified position, using unnormalized coordinates.
                * This implementation uses the efficient GPU implementation of the linear box spline
                * proposed by A. Entezari et al. in 2010.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE
            static ValueType fetch(TextureType tex, float x, float y, float z)
            {
                // conversion between 3D coordinates and texture index
                const float4 convert = { 0.5f, 0.5f, 1.0f, 0.0f };
                const float4 shift = { 0.5f, 0.5f, 0.5f, 0.0f };

                float4 P1, P2, P3, P4;
                float D1, D2, D3, D4;

                // transfroms position to Cartesian coordinate system
                //float4 posOS = { x * (1.0f - 0.5f / w), y * (1.0f - 0.5f / h), z * (1.0f - 1.0f / d) * 0.5f, 0.0f };
                float4 posOS = { x - 0.25f, y - 0.25f, z * 0.5f - 0.25f, 0.0f };
                    
                // convert Cartesian coordinate to BCC coordinate
                // (the division by 2 is bundled into the coordinate transformation above)
                float4 abc = { posOS.x + posOS.y, posOS.x + posOS.z, posOS.y + posOS.z, 0.0f };

                // truncate this point which results 
                // in the first of the four neighbors
                float4 floors = floorf(abc);

                // shift position into the origin 
                // of the BCC coordinate system
                abc -= floors;

                // transform the first neighbor back to 
                // the Cartesian coordinate system
                P1 = make_float4(
                    floors.x + floors.y - floors.z,
                    floors.x - floors.y + floors.z,
                    -floors.x + floors.y + floors.z,
                    0.0f);
                // the second neighbor is found immediately
                P2 = P1 + make_float4(1.0f, 1.0f, 1.0f, 0.0f);
                // assume case: mymax = alpha
                P3 = P1 + make_float4(1.0f, 1.0f, -1.0f, 0.0f);
                // assume case: mymin = gamma
                P4 = P1 + make_float4(2.0f, 0.0f, 0.0f, 0.0f);

                // sorting
                float4 sorting = { 1.0f, 0.0f, 0.0f, 0.0f };
                sorting.y = max(abc.x, max(abc.y, abc.z)); // max
                sorting.z = min(abc.x, min(abc.y, abc.z)); // min
                sorting.w = (abc.x + abc.y + abc.z) - sorting.y - sorting.z; // mid

                // get the missing two neighbors
                P3 += (sorting.y == abc.y) * make_float4(0.0f, -2.0f, 2.0f, 0.0f)
                    + (sorting.y == abc.z) * make_float4(-2.0f, 0.0f, 2.0f, 0.0f);
                P4 += (sorting.z == abc.x) * make_float4(-2.0f, 0.0f, 2.0f, 0.0f)
                    + (sorting.z == abc.y) * make_float4(-2.0f, 2.0f, 0.0f, 0.0f);

                // convert 3D coordinates to texture index
                P1 = (P1 + shift) * convert;
                P2 = (P2 + shift) * convert;
                P3 = (P3 + shift) * convert;
                P4 = (P4 + shift) * convert;

                // four texture lookups
                D1 = Internal::textureFetch3D<ValueType>(tex, P1.x, P1.y, P1.z);
                D2 = Internal::textureFetch3D<ValueType>(tex, P2.x, P2.y, P2.z);
                D3 = Internal::textureFetch3D<ValueType>(tex, P3.x, P3.y, P3.z);
                D4 = Internal::textureFetch3D<ValueType>(tex, P4.x, P4.y, P4.z);

                // interpolate using barycentric coordinates
                float4 values = { D1, D3 - D1, D2 - D4, D4 - D3 };
                return dot(sorting, values);
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::BCC, Filter::LinearBoxSpline, Coordinates::Normalized>
        {
            /**
                * Fetches the given BCC texture at the specified position, using normalized coordinates.
                * This implementation uses the efficient GPU implementation of the linear box spline
                * proposed by A. Entezari et al. in 2010.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE
            static ValueType fetch(TextureType tex, float x, float y, float z, float w, float h, float d)
            {
                // conversion between 3D coordinates and texture index
                // instead of having two terms of conversion, it is packed into one vector
                //const float4 convert = { 0.5f, 0.5f, 1.0f, 0.0f };
                //const float4 oneOverVoxels = { 1.0f / w, 1.0f / h, 1.0f / d, 0.0f };
                const float4 convert = { 1.0f / (2.0f * w), 1.0f / (2.0f * h), 1.0f / d };
                const float4 shift = { 0.5f, 0.5f, 0.5f, 0.0f };

                float4 P1, P2, P3, P4;
                float D1, D2, D3, D4;

                // transfroms position to Cartesian coordinate system
                //float4 posOS = { x * (w - 0.5f), y * (h - 0.5f), z * (d * 0.5f - 0.5f), 0.0f };
                float4 posOS = { x * w - 0.25f, y * h - 0.25f, z * d * 0.5f - 0.25f, 0.0f };

                // convert Cartesian coordinate to BCC coordinate
                // (the division by 2 is bundled into the coordinate transformation above)
                float4 abc = { posOS.x + posOS.y, posOS.x + posOS.z, posOS.y + posOS.z, 0.0f };

                // truncate this point which results 
                // in the first of the four neighbors
                float4 floors = floorf(abc);

                // shift position into the origin 
                // of the BCC coordinate system
                abc -= floors;

                // transform the first neighbor back to 
                // the Cartesian coordinate system
                P1 = make_float4(
                    floors.x + floors.y - floors.z,
                    floors.x - floors.y + floors.z,
                    -floors.x + floors.y + floors.z,
                    0.0f);
                // the second neighbor is found immediately
                P2 = P1 + make_float4(1.0f, 1.0f, 1.0f, 0.0f);
                // assume case: mymax = alpha
                P3 = P1 + make_float4(1.0f, 1.0f, -1.0f, 0.0f);
                // assume case: mymin = gamma
                P4 = P1 + make_float4(2.0f, 0.0f, 0.0f, 0.0f);

                // sorting
                float4 sorting = { 1.0f, 0.0f, 0.0f, 0.0f };
                sorting.y = max(abc.x, max(abc.y, abc.z)); // max
                sorting.z = min(abc.x, min(abc.y, abc.z)); // min
                sorting.w = (abc.x + abc.y + abc.z) - sorting.y - sorting.z; // mid

                // get the missing two neighbors
                P3 += (sorting.y == abc.y) * make_float4(0.0f, -2.0f, 2.0f, 0.0f)
                    + (sorting.y == abc.z) * make_float4(-2.0f, 0.0f, 2.0f, 0.0f);
                P4 += (sorting.z == abc.x) * make_float4(-2.0f, 0.0f, 2.0f, 0.0f)
                    + (sorting.z == abc.y) * make_float4(-2.0f, 2.0f, 0.0f, 0.0f);

                // convert 3D coordinates to texture index
                P1 = (P1 + shift) * convert;
                P2 = (P2 + shift) * convert;
                P3 = (P3 + shift) * convert;
                P4 = (P4 + shift) * convert;

                // four texture lookups
                D1 = Internal::textureFetch3D<ValueType>(tex, P1.x, P1.y, P1.z);
                D2 = Internal::textureFetch3D<ValueType>(tex, P2.x, P2.y, P2.z);
                D3 = Internal::textureFetch3D<ValueType>(tex, P3.x, P3.y, P3.z);
                D4 = Internal::textureFetch3D<ValueType>(tex, P4.x, P4.y, P4.z);

                // interpolate using barycentric coordinates
                float4 values = { D1, D3 - D1, D2 - D4, D4 - D3 };
                return dot(sorting, values);
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::BCC, Filter::TrilinearBSpline, Coordinates::Unnormalized>
        {
            /**
                * Fetches the given BCC texture at the specified position, using unnormalized coordinates.
                * This implementation uses the simple trilinear B-spline with blocked CC texture representation
                * and two linear texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float /*w*/, float /*h*/, float d)
            {
                const auto D1 = Internal::textureFetch3D<ValueType>(tex, x + 0.25f, y + 0.25f, z * 0.5f + 0.25f);
                const auto D2 = Internal::textureFetch3D<ValueType>(tex, x - 0.25f, y - 0.25f, (z + d) * 0.5f - 0.25f);
                return (D1 + D2) * 0.5f;
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::BCC, Filter::TrilinearBSpline, Coordinates::Normalized>
        {
            /**
                * Fetches the given BCC texture at the specified position, using normalized coordinates.
                * This implementation uses the simple trilinear B-spline with blocked CC texture representation
                * and two linear texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float w, float h, float d)
            {
                const float shift_x = 0.25f / w;
                const float shift_y = 0.25f / h;
                const float shift_z = 0.25f / d;

                const auto D1 = Internal::textureFetch3D<ValueType>(tex, x + shift_x, y + shift_y, z * 0.5f + shift_z);
                const auto D2 = Internal::textureFetch3D<ValueType>(tex, x - shift_x, y - shift_y, (z + 1.0f) * 0.5f - shift_z);
                return (D1 + D2) * 0.5f;
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::BCC, Filter::CubicBSpline, Coordinates::Unnormalized>
        {
            /**
                * Fetches the given BCC texture at the specified position, using unnormalized coordinates.
                * This implementation uses the tricubic B-spline with blocked CC texture representation
                * and two 2*8 trilinear texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float /*w*/, float /*h*/, float d)
            {
                const auto D1 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>::fetch(tex, x + 0.25f, y + 0.25f, z * 0.5f + 0.25f);
                const auto D2 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>::fetch(tex, x - 0.25f, y - 0.25f, (z + d) * 0.5f - 0.25f);
                return (D1 + D2) * 0.5f;
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::BCC, Filter::CWLB, Coordinates::Unnormalized>
        {
            /**
                * Fetches the given BCC texture at the specified position, using unnormalized coordinates.
                * This implementation uses the Cosine-Weighted triLinear B-spline with blocked CC texture representation
                * and two linear texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float /*w*/, float /*h*/, float d, float lambda = 1.0f)
            {
                const float W = 0.5f + lambda * (
                    cos(6.2831853071796f * (x - 0.25f)) + 
                    cos(6.2831853071796f * (y - 0.25f)) + 
                    cos(6.2831853071796f * (z - 0.25f))) * 0.1666666666667f;

                const auto D1 = Internal::textureFetch3D<ValueType>(tex, x + 0.25f, y + 0.25f, z * 0.5f + 0.25f);
                const auto D2 = Internal::textureFetch3D<ValueType>(tex, x - 0.25f, y - 0.25f, (z + d) * 0.5f - 0.25f);
                return D1 * W + D2 * (1.0f - W);
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::BCC, Filter::CWCB, Coordinates::Unnormalized>
        {
            /**
                * Fetches the given BCC texture at the specified position, using unnormalized coordinates.
                * This implementation uses the Cosine-Weighted triCubic B-spline with blocked CC texture representation
                * and two cubic texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float /*w*/, float /*h*/, float d, float lambda = 1.0f)
            {
                const float W = 0.5f + lambda * (
                    cos(6.2831853071796f * (x - 0.25f)) + 
                    cos(6.2831853071796f * (y - 0.25f)) + 
                    cos(6.2831853071796f * (z - 0.25f))) * 0.1666666666667f;

                const auto D1 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>::fetch(tex, x + 0.25f, y + 0.25f, z * 0.5f + 0.25f);
                const auto D2 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>::fetch(tex, x - 0.25f, y - 0.25f, (z + d) * 0.5f - 0.25f);
                return D1 * W + D2 * (1.0f - W);
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::BCC, Filter::CWLB, Coordinates::Normalized>
        {
            /**
                * Fetches the given BCC texture at the specified position, using normalized coordinates.
                * This implementation uses the Cosine-Weighted triLinear B-spline with blocked CC texture representation
                * and two linear texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float w, float h, float d, float lambda = 1.0f)
            {
                const float shift_x = 0.25f / w;
                const float shift_y = 0.25f / h;
                const float shift_z = 0.25f / d;

                const float W = 0.5f + lambda * (
                    cos(6.2831853071796f * (x * w - 0.25f)) + 
                    cos(6.2831853071796f * (y * h - 0.25f)) + 
                    cos(6.2831853071796f * (z * d - 0.25f))) * 0.1666666666667f;

                const auto D1 = Internal::textureFetch3D<ValueType>(tex, x + shift_x, y + shift_y, z * 0.5f + shift_z);
                const auto D2 = Internal::textureFetch3D<ValueType>(tex, x - shift_x, y - shift_y, (z + 1.0f) * 0.5f - shift_z);
                return D1 * W + D2 * (1.0f - W);
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::BCC, Filter::CWCB, Coordinates::Normalized>
        {
            /**
                * Fetches the given BCC texture at the specified position, using normalized coordinates.
                * This implementation uses the Cosine-Weighted triCubic B-spline with blocked CC texture representation
                * and two cubic texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float w, float h, float d, float lambda = 1.0f)
            {
                const float shift_x = 0.25f / w;
                const float shift_y = 0.25f / h;
                const float shift_z = 0.25f / d;

                const float W = 0.5f + lambda * (
                    cos(6.2831853071796f * (x * w - 0.25f)) + 
                    cos(6.2831853071796f * (y * h - 0.25f)) + 
                    cos(6.2831853071796f * (z * d - 0.25f))) * 0.1666666666667f;

                const auto D1 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Normalized>::fetch(tex, x + shift_x, y + shift_y, z * 0.5f + shift_z);
                const auto D2 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Normalized>::fetch(tex, x - shift_x, y - shift_y, (z + 1.0f) * 0.5f - shift_z);
                return D1 * W + D2 * (1.0f - W);
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::FCC, Filter::TrilinearBSpline, Coordinates::Unnormalized>
        {
            /**
                * Fetches the given FCC texture at the specified position, using unnormalized coordinates.
                * This implementation uses the simple trilinear B-spline with blocked CC texture representation
                * and two linear texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float /*w*/, float /*h*/, float d)
            {
                const auto D1 = Internal::textureFetch3D<ValueType>(tex, x + 0.25f, y + 0.25f, z * 0.25f + 0.25f);
                const auto D2 = Internal::textureFetch3D<ValueType>(tex, x - 0.25f, y + 0.25f, (z + d) * 0.25f - 0.25f);
                const auto D3 = Internal::textureFetch3D<ValueType>(tex, x + 0.25f, y - 0.25f, (z + d + d) * 0.25f - 0.25f);
                const auto D4 = Internal::textureFetch3D<ValueType>(tex, x - 0.25f, y - 0.25f, (z + d + d + d) * 0.25f + 0.25f);
                return (D1 + D2 + D3 + D4) * 0.25f;
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::FCC, Filter::TrilinearBSpline, Coordinates::Normalized>
        {
            /**
                * Fetches the given FCC texture at the specified position, using normalized coordinates.
                * This implementation uses the simple trilinear B-spline with blocked CC texture representation
                * and two linear texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float w, float h, float d)
            {
                const float shift_x = 0.25f / w;
                const float shift_y = 0.25f / h;
                const float shift_z = 0.25f / d;
                    
                const auto D1 = Internal::textureFetch3D<ValueType>(tex, x + shift_x, y + shift_y, z * 0.25f + shift_z);
                const auto D2 = Internal::textureFetch3D<ValueType>(tex, x - shift_x, y + shift_y, (z + 1.0f) * 0.25f - shift_z);
                const auto D3 = Internal::textureFetch3D<ValueType>(tex, x + shift_x, y - shift_y, (z + 2.0f) * 0.25f - shift_z);
                const auto D4 = Internal::textureFetch3D<ValueType>(tex, x - shift_x, y - shift_y, (z + 3.0f) * 0.25f + shift_z);
                return (D1 + D2 + D3 + D4) * 0.25f;
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::FCC, Filter::CubicBSpline, Coordinates::Unnormalized>
        {
            /**
                * Fetches the given FCC texture at the specified position, using unnormalized coordinates.
                * This implementation uses the tricubic B-spline with blocked CC texture representation
                * and two 4*8 trilinear texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float /*w*/, float /*h*/, float d)
            {
                const auto D1 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>::fetch(tex, x + 0.25f, y + 0.25f, z * 0.25f + 0.25f);
                const auto D2 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>::fetch(tex, x - 0.25f, y + 0.25f, (z + d) * 0.25f - 0.25f);
                const auto D3 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>::fetch(tex, x + 0.25f, y - 0.25f, (z + d + d) * 0.25f - 0.25f);
                const auto D4 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>::fetch(tex, x - 0.25f, y - 0.25f, (z + d + d + d) * 0.25f + 0.25f);
                return (D1 + D2 + D3 + D4) * 0.25f;
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::FCC, Filter::CWLB, Coordinates::Unnormalized>
        {
            /**
                * Fetches the given FCC texture at the specified position, using unnormalized coordinates.
                * This implementation uses the Cosine-Weighted triLinear B-spline with blocked CC texture representation
                * and four linear texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float /*w*/, float /*h*/, float d, float lambda = 1.0f)
            {
                // cosine terms of the weight
                const float Cx = cos(6.2831853071796f * (x - 0.25f));
                const float Cy = cos(6.2831853071796f * (y - 0.25f));
                const float Cz = cos(6.2831853071796f * (z - 0.25f));

                // trilinear texture fetches
                const auto D1 = Internal::textureFetch3D<ValueType>(tex, x + 0.25f, y + 0.25f, z * 0.25f + 0.25f);
                const auto D2 = Internal::textureFetch3D<ValueType>(tex, x - 0.25f, y + 0.25f, (z + d) * 0.25f - 0.25f);
                const auto D3 = Internal::textureFetch3D<ValueType>(tex, x + 0.25f, y - 0.25f, (z + d + d) * 0.25f - 0.25f);
                const auto D4 = Internal::textureFetch3D<ValueType>(tex, x - 0.25f, y - 0.25f, (z + d + d + d) * 0.25f + 0.25f);

                // return the weighted sum of the components
                return 0.25f * (
                    ((+ Cx + Cy + Cz) * lambda + 1.0f) * D1 +
                    ((- Cx + Cy - Cz) * lambda + 1.0f) * D2 +
                    ((+ Cx - Cy - Cz) * lambda + 1.0f) * D3 +
                    ((- Cx - Cy + Cz) * lambda + 1.0f) * D4);
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::FCC, Filter::CWCB, Coordinates::Unnormalized>
        {
            /**
                * Fetches the given FCC texture at the specified position, using unnormalized coordinates.
                * This implementation uses the Cosine-Weighted triCubic B-spline with blocked CC texture representation
                * and four cubic texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float /*w*/, float /*h*/, float d, float lambda = 1.0f)
            {
                // cosine terms of the weight
                const float Cx = cos(6.2831853071796f * (x - 0.25f));
                const float Cy = cos(6.2831853071796f * (y - 0.25f));
                const float Cz = cos(6.2831853071796f * (z - 0.25f));

                // trilinear texture fetches
                const auto D1 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>::fetch(tex, x + 0.25f, y + 0.25f, z * 0.25f + 0.25f);
                const auto D2 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>::fetch(tex, x - 0.25f, y + 0.25f, (z + d) * 0.25f - 0.25f);
                const auto D3 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>::fetch(tex, x + 0.25f, y - 0.25f, (z + d + d) * 0.25f - 0.25f);
                const auto D4 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Unnormalized>::fetch(tex, x - 0.25f, y - 0.25f, (z + d + d + d) * 0.25f + 0.25f);

                // return the weighted sum of the components
                return 0.25f * (
                    ((+ Cx + Cy + Cz) * lambda + 1.0f) * D1 +
                    ((- Cx + Cy - Cz) * lambda + 1.0f) * D2 +
                    ((+ Cx - Cy - Cz) * lambda + 1.0f) * D3 +
                    ((- Cx - Cy + Cz) * lambda + 1.0f) * D4);
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::FCC, Filter::CWLB, Coordinates::Normalized>
        {
            /**
                * Fetches the given FCC texture at the specified position, using normalized coordinates.
                * This implementation uses the Cosine-Weighted triLinear B-spline with blocked CC texture representation
                * and four linear texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float w, float h, float d, float lambda = 1.0f)
            {
                const float shift_x = 0.25f / w;
                const float shift_y = 0.25f / h;
                const float shift_z = 0.25f / d;

                // cosine terms of the weight
                const float Cx = cos(6.2831853071796f * (x * w - 0.25f));
                const float Cy = cos(6.2831853071796f * (y * h - 0.25f));
                const float Cz = cos(6.2831853071796f * (z * d - 0.25f));

                // trilinear texture fetches
                const auto D1 = Internal::textureFetch3D<ValueType>(tex, x + shift_x, y + shift_y, z * 0.25f + shift_z);
                const auto D2 = Internal::textureFetch3D<ValueType>(tex, x - shift_x, y + shift_y, (z + 1.0f) * 0.25f - shift_z);
                const auto D3 = Internal::textureFetch3D<ValueType>(tex, x + shift_x, y - shift_y, (z + 2.0f) * 0.25f - shift_z);
                const auto D4 = Internal::textureFetch3D<ValueType>(tex, x - shift_x, y - shift_y, (z + 3.0f) * 0.25f + shift_z);

                // return the weighted sum of the components
                return 0.25f * (
                    ((+ Cx + Cy + Cz) * lambda + 1.0f) * D1 +
                    ((- Cx + Cy - Cz) * lambda + 1.0f) * D2 +
                    ((+ Cx - Cy - Cz) * lambda + 1.0f) * D3 +
                    ((- Cx - Cy + Cz) * lambda + 1.0f) * D4);
            }
        };

        template <typename ValueType>
        struct Tex3D<ValueType, Lattice::FCC, Filter::CWCB, Coordinates::Normalized>
        {
            /**
                * Fetches the given FCC texture at the specified position, using normalized coordinates.
                * This implementation uses the Cosine-Weighted triCubic B-spline with blocked CC texture representation
                * and four cubic texture fetches.
                * @param tex      - texture reference or object
                * @param x, y, z  - coordinates of the texture fetch position
                * @param w, h, d  - extents of the texture (array)
                * @return  - value of the texture at the specified position
                */
            template <typename TextureType>
            DEVICE FORCEINLINE
            static ValueType fetch(TextureType tex, float x, float y, float z, float w, float h, float d, float lambda = 1.0f)
            {
                const float shift_x = 0.25f / w;
                const float shift_y = 0.25f / h;
                const float shift_z = 0.25f / d;

                // cosine terms of the weight
                const float Cx = cos(6.2831853071796f * (x * w - 0.25f));
                const float Cy = cos(6.2831853071796f * (y * h - 0.25f));
                const float Cz = cos(6.2831853071796f * (z * d - 0.25f));

                // trilinear texture fetches
                const auto D1 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Normalized>::fetch(tex, x + shift_x, y + shift_y, z * 0.25f + shift_z, w, h, d);
                const auto D2 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Normalized>::fetch(tex, x - shift_x, y + shift_y, (z + 1.0f) * 0.25f - shift_z, w, h, d);
                const auto D3 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Normalized>::fetch(tex, x + shift_x, y - shift_y, (z + 2.0f) * 0.25f - shift_z, w, h, d);
                const auto D4 = Tex3D<ValueType, Lattice::CC, Filter::CubicBSpline, Coordinates::Normalized>::fetch(tex, x - shift_x, y - shift_y, (z + 3.0f) * 0.25f + shift_z, w, h, d);

                // return the weighted sum of the components
                return 0.25f * (
                    ((+ Cx + Cy + Cz) * lambda + 1.0f) * D1 +
                    ((- Cx + Cy - Cz) * lambda + 1.0f) * D2 +
                    ((+ Cx - Cy - Cz) * lambda + 1.0f) * D3 +
                    ((- Cx - Cy + Cz) * lambda + 1.0f) * D4);
            }
        };


        /////////////////////////////////////////////////////////
        //  Texture fetch functions for Texture Reference API
        /////////////////////////////////////////////////////////

        template <
            enum Filter::FilterType filter = Filter::TrilinearBSpline,
            enum Coordinates::CoordinateType coordinates = Coordinates::Normalized,
            typename ValueType,
            enum cudaTextureReadMode mode
        >
        DEVICE FORCEINLINE
        ValueType ccTex3D(texture<ValueType, cudaTextureType3D, mode> texRef, float x, float y, float z, float /*w*/, float /*h*/, float /*d*/)
        {
            return Tex3D<ValueType, Lattice::CC, filter, coordinates>::fetch(texRef, x, y, z);
        }

        template <
            enum Filter::FilterType filter = Filter::TrilinearBSpline,
            enum Coordinates::CoordinateType coordinates = Coordinates::Normalized,
            typename ValueType,
            enum cudaTextureReadMode mode
        >
        DEVICE FORCEINLINE
        ValueType bccTex3D(texture<ValueType, cudaTextureType3D, mode> texRef, float x, float y, float z, float w, float h, float d)
        {
            return Tex3D<ValueType, Lattice::BCC, filter, coordinates>::fetch(texRef, x, y, z, w, h, d);
        }

        template <
            enum Filter::FilterType filter = Filter::TrilinearBSpline,
            enum Coordinates::CoordinateType coordinates = Coordinates::Normalized,
            typename ValueType,
            enum cudaTextureReadMode mode
        >
        DEVICE FORCEINLINE
        ValueType fccTex3D(texture<ValueType, cudaTextureType3D, mode> texRef, float x, float y, float z, float w, float h, float d)
        {
            return Tex3D<ValueType, Lattice::FCC, filter, coordinates>::fetch(texRef, x, y, z, w, h, d);
        }


        /////////////////////////////////////////////////////////
        //  Texture fetch functions for Texture Object API
        /////////////////////////////////////////////////////////

        template <
            typename ValueType,
            enum Filter::FilterType filter = Filter::TrilinearBSpline,
            enum Coordinates::CoordinateType coordinates = Coordinates::Normalized
        >
        DEVICE FORCEINLINE
        ValueType ccTex3D(cudaTextureObject_t texObj, float x, float y, float z, float /*w*/, float /*h*/, float /*d*/)
        {
            return Tex3D<ValueType, Lattice::CC, filter, coordinates>::fetch(texObj, x, y, z);
        }

        template <
            typename ValueType,
            enum Filter::FilterType filter = Filter::TrilinearBSpline,
            enum Coordinates::CoordinateType coordinates = Coordinates::Normalized
        >
        DEVICE FORCEINLINE
        ValueType bccTex3D(cudaTextureObject_t texObj, float x, float y, float z, float w, float h, float d)
        {
            return Tex3D<ValueType, Lattice::BCC, filter, coordinates>::fetch(texObj, x, y, z, w, h, d);
        }

        template <
            typename ValueType,
            enum Filter::FilterType filter = Filter::TrilinearBSpline,
            enum Coordinates::CoordinateType coordinates = Coordinates::Normalized
        >
        DEVICE FORCEINLINE
        ValueType fccTex3D(cudaTextureObject_t texObj, float x, float y, float z, float w, float h, float d)
        {
            return Tex3D<ValueType, Lattice::FCC, filter, coordinates>::fetch(texObj, x, y, z, w, h, d);
        }
    }
}

#endif /* _CUDALATTICE_TEXTURE_FETCH_CUH_ */