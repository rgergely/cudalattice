// ---------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// enum_types.cuh
// Definition of the lattice converter enumeration types 
// and the type information structures.
//
// Author: Gergely Ferenc, Racz
// ---------------------------------------------------------------------------

#pragma once

#ifndef _CUDALATTICE_ENUM_TYPES_
#define _CUDALATTICE_ENUM_TYPES_

namespace CudaLattice
{
    namespace Texture
    {
        namespace Lattice { enum LatticeType { CC = 1, BCC = 2, FCC = 4 }; }
        namespace Filter { enum FilterType { TrilinearBSpline, CubicBSpline, LinearBoxSpline, CWLB, CWCB }; }
        namespace Coordinates { enum CoordinateType { Normalized, Unnormalized }; }
        namespace Format { enum TextureFormatType { Blocked, Layered }; }
        namespace Resizer { enum ResizerType { PreserveDensity, DoubleDense, HalfDense }; }


        //////////////////////////////////////////////////
        //  LATTICE TYPE INFORMATIONS

        //template <enum Lattice::LatticeType lattice>
        //struct LatticeTypeInfo
        //{};

        //template <>
        //struct LatticeTypeInfo<Lattice::CC>
        //{
        //    static const int num_cc_comp = 1;
        //};

        //template <>
        //struct LatticeTypeInfo<Lattice::BCC>
        //{
        //    static const int num_cc_comp = 2;
        //};

        //template <>
        //struct LatticeTypeInfo<Lattice::FCC>
        //{
        //    static const int num_cc_comp = 4;
        //};


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
    }
}

#endif /* _CUDALATTICE_ENUM_TYPES_ */