// ---------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// cudalattice.cuh
// Include necessary headers.
//
// Author: Gergely Ferenc, Racz
// ---------------------------------------------------------------------------

#pragma once

#ifndef _CUDALATTICE_H_
#define _CUDALATTICE_H_


//////////////////////////////////////////////////
//  include necessary headers

// enumeration types
#include "enum_types.h"

// conversion between lattices
#include "lattice_converter.cuh"

// texture fetch functions
#include "texture_fetch.cuh"

//////////////////////////////////////////////////
//  declare using directives for simpler usage

#ifndef CUDALATTICE_NO_USING
using namespace CudaLattice::Texture;
#endif


#endif /* _CUDALATTICE_H_ */