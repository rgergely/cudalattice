// ---------------------------------------------------------------------------
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
//
// utility.h
// Definition of utility functions, structures and classes. 
//
// Author: Gergely Ferenc, Racz
// ---------------------------------------------------------------------------

#pragma once

#ifndef _CUDALATTICE_UTILITY_H_
#define _CUDALATTICE_UTILITY_H_

namespace CudaLattice
{
	namespace Utility
	{
		/**
		 * Compile time check wether two types are the same.
		**/

		template <typename T1, typename T2>
		struct is_same_type
		{
			enum { value = false };
		};

		template <typename T>
		struct is_same_type<T, T>
		{
			enum { value = true };
		};


		/**
		 * Compile time assertion.
		**/

		template <bool, typename T = void>
		struct enable_if
		{
			// default definition is empty
		};

		template <typename T>
		struct enable_if<true, T>
		{
			typedef T type;
		};


		/**
		 * Compile time type selection.
		**/

		template <bool, typename T1, typename T2>
		struct select_type
		{
			// default definition is empty
		};

		template <typename T1, typename T2>
		struct select_type<true, T1, T2>
		{
			typedef T1 type;
		};

		template <typename T1, typename T2>
		struct select_type<false, T1, T2>
		{
			typedef T2 type;
		};
	}
}

#endif /* _CUDALATTICE_UTILITY_H_ */