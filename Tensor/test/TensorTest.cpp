#define BOOST_TEST_MODULE TensorTest
#include <boost/test/unit_test.hpp>
#include "Tensor.hpp"  
BOOST_AUTO_TEST_CASE(TensorTest) {
    using namespace Ten;
    Tensor<int> t1({ 1,2,3 });
    t1(0, 0, 0) = 1;
    t1(0, 0, 1) = 2;
    t1(0, 0, 2) = 3;
    t1(0, 1, 0) = 4;
    t1(0, 1, 1) = 5;
    t1(0, 1, 2) = 6;
    Tensor<int> const& t2 = t1;
    std::cout << t2(0, 0, 0);
    int const* p = t2.data();
    int* q = t1.data();
    t1.reshape({ 3,2,2,2,2 });
    auto s = t1.shape()[0];

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
