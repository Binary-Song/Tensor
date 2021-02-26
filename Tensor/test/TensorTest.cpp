#define BOOST_TEST_MODULE TensorTest
#include <boost/test/unit_test.hpp>
#include "Tensor.hpp"  
using namespace Ten;

BOOST_AUTO_TEST_CASE(Trivial)
{
	Tensor<int> t1({ 1,2,3 });
	t1(0, 0, 0) = 1;
	t1(0, 0, 1) = 2;
	t1(0, 0, 2) = 3;
	t1(0, 1, 0) = 4;
	t1(0, 1, 1) = 5;
	t1(0, 1, 2) = 6;
	Tensor<int> const& t2 = t1;
	int const* p = t2.data();
	int* q = t1.data();
	t1.reshape({ 3,2,2,2,2 });
	auto s = t1.shape()[0];
}
BOOST_AUTO_TEST_CASE(DotProduct)
{
	/*
		1 2
		3 4
*
		4 3
		2 1
=
		8 5
		20 13
	*/
	Tensor<int> A({ 2,2 }, { 1,2,3,4 });
	Tensor<int> B({ 2,2 }, { 4,3,2,1 });
	Tensor<int> C({ 2,2 }, { 8,5,20,13 });
	Tensor<int> D({ 2,2 }, { 5,5,5,5 });

	BOOST_CHECK(C == A.dot(B));
	BOOST_CHECK(D == A + B);
}
BOOST_AUTO_TEST_CASE(Conv2d)
{
	/*
		1 2 0
		-1 3 1
		2 -1 1
		0 1 2
		1 2 0
conv
		1 0
		-1 1
=
		5 0
		-4 5
		3 0
		1 -1
	*/

	Tensor<int> X({ 5,3 }, { 1,2,0,-1,3,1,2,-1,1,0,1,2,1,2,0 });
	Tensor<int> Y({ 2,2 }, { 1,0,-1,1 });
	Tensor<int> Z({ 4,2 }, { 5,0,-4,5,3,0,1,-1 });
	BOOST_CHECK(Z == X.convolve2D(Y));
}

bool AllowCopy = true;
BOOST_AUTO_TEST_CASE(Moving)
{
	class NoCopy
	{
	public:
		NoCopy() {}

		NoCopy(const NoCopy& other)
		{
			BOOST_ASSERT(AllowCopy);
		}
		NoCopy& operator=(const NoCopy&)
		{
			BOOST_ASSERT(AllowCopy);
			return *this;
		}

		NoCopy(NoCopy&&) noexcept
		{}
		NoCopy& operator=(NoCopy&&) noexcept
		{
			return *this;
		}
		NoCopy operator+(const NoCopy& nocopy)const
		{
			return NoCopy();
		}
	};
	Tensor<NoCopy> X{ NoCopy() };
	Tensor<NoCopy> Y{ NoCopy() };
	AllowCopy = false;
	Tensor<NoCopy> Z = X + Y;
}

//BOOST_AUTO_TEST_CASE(Elemwise)
//{
//	Tensor<int> X({ 2,2 }, { 1,2,3,4 });
//	Tensor<int> X_power_2({ 2,2 }, { 1,4,9,16 });
//	Tensor<int> X_times_3({ 2,2 }, { 3,6,9,12 });
//	Tensor<int> X_div_2({ 2,2 }, { 0,1,1,2 });
//
//	BOOST_CHECK(X.elemwise([=](int elem) {return elem * elem; }) == X_power_2);
//	BOOST_CHECK(X * 3 == X_times_3);
//	BOOST_CHECK(X / 2 == X_div_2);
//	BOOST_CHECK(Tensor<int>::Ones({ 2,2 }) * 3 == Tensor<int>::Constants({ 2,2 }, 3));
//	BOOST_CHECK(Tensor<int>::Ones({ 2,2 }) * 3 == Tensor<int>::Constants({ 2,2 }, 3));
//	BOOST_CHECK(Tensor<int>::Zeros({ 2,2 }) - Tensor<int>::Ones({ 2,2 }) == -Tensor<int>::Ones({ 2,2 }));
//}
//
//BOOST_AUTO_TEST_CASE(Random)
//{
//	Tensor <double> X = RandomUniform({ 20,20 }, 0.0, 5.0);
//	Tensor <double> Y = RandomNormal({ 20,20 }, 5.0, 0.1);
//	Tensor <int> Z = RandomUniform({ 20,20 }, 0, 5);
//	auto W = Tensor<int>({ 2,2 }, {1,2,3,4}).elemwise([=](std::tuple<int, int> i, int e) {
//		auto&& [i0, i1](i);
//		return i0 + i1;
// 		});
//	BOOST_CHECK()
//}