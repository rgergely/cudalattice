// ---------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// macro.h
// Definitions of project macros. 
//
// Author: Gergely Ferenc, Racz
// ---------------------------------------------------------------------------

#pragma once

#ifndef _CUDALATTICE_MACRO_H_
#define _CUDALATTICE_MACRO_H_

////////////////////////////////////////
//  USER DEFINED MACROS

#ifndef HOST
#ifdef __CUDACC__
#define HOST __host__
#else
#define HOST
#endif
#endif

#ifndef DEVICE
#ifdef __CUDACC__
#define DEVICE __device__
#else
#define DEVICE
#endif
#endif

#ifndef HOST_DEVICE
#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif
#endif

#ifndef INLINE
#ifdef __CUDACC__
#define INLINE __inline__
#else
#define INLINE __inline
#endif
#endif

#ifndef FORCEINLINE
#ifdef __CUDACC__
#define FORCEINLINE __forceinline__
#else
#define FORCEINLINE __forceinline
#endif
#endif

#endif /* _CUDALATTICE_MACRO_H_ */