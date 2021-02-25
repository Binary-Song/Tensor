#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_
#include <vector>
#include <functional>

#ifdef NDEBUG 
// release mode
#define TENSOR_ASSERT(cond,msg)
#else
// debug mode
#include <iostream>
#define TENSOR_ASSERT(cond,msg) if(!(cond)) throw msg
#endif

namespace Ten
{
	template<typename Scalar>
	class Tensor
	{
	public:
		class Shape : public std::vector<int>
		{
		public:
			Shape()
			{}

			Shape(std::vector<int> const& shape)
				:std::vector<int>(shape) {}

			Shape(Tensor const& t)
			{
				for (size_t r = 0; r < t.rank(); ++r) {
					push_back(t._dims[r].elems);
				}
			}

			bool operator==(const Shape& other) const
			{
				if (size() != other.size()) return false;
				for (size_t r = 0; r < size(); ++r) {
					if ((*this)[r] != other[r])
						return false;
				}
				return true;
			}
		};
		friend class Shape;
	private:

		class Dimension
		{
		public:
			int elem_size = 0;
			int elems = 0;

			Dimension()
				:elem_size(0), elems(0)
			{}
			Dimension(int elem_size, int elems)
				:elem_size(elem_size), elems(elems)
			{}
		};
		std::vector<Dimension> _dims;
		std::vector<Scalar> _data;


	public:

		Tensor(std::vector<int>const& elem_numbers)
			:Tensor(elem_numbers, std::vector<Scalar>())
		{}

		Tensor(std::vector<int>const& elem_numbers, std::vector<Scalar>const& data)
		{
			reshape(elem_numbers);
			TENSOR_ASSERT(data.size() <= _data.size(), "too many initial values");
			for (size_t i = 0; i < data.size(); i++) {
				_data[i] = data[i];
			}
		}

		Tensor(const Scalar& number)
			:_dims{ Dimension(1,1) }, _data{ number }
		{}

		Tensor()
			:_dims{}, _data{}
		{}

		template<typename... Args>
		Scalar& operator()(Args... args)
		{
			assert_index_range(IndexNumber(0), args...);
			assert_count_index(args...);
			auto i = flat_index(IndexNumber(0), args...);
			return _data[i];
		}

		template<typename... Args>
		Scalar const& operator()(Args... args) const
		{
			return const_cast<Tensor&>(*this).operator()(args...);
		}

		Scalar& operator[](int flat_index)
		{
			return _data[flat_index];
		}

		Scalar const& operator[](int flat_index) const
		{
			return  _data[flat_index];
		}

		size_t size() const
		{
			return _data.size();
		}

		Scalar* data()
		{
			return _data.data();
		}

		Scalar const* data() const
		{
			return _data.data();
		}

		void reshape(std::vector<int> const& elem_numbers)
		{
			// init dimensions
			_dims.resize(elem_numbers.size());
			for (size_t i = 0; i < elem_numbers.size(); ++i) {
				_dims[i].elems = elem_numbers[i];
				int e = 1;
				for (size_t j = i + 1; j < elem_numbers.size(); ++j) {
					e *= elem_numbers[j];
				}
				_dims[i].elem_size = e;
			}
			// compute data size
			size_t data_size = 1;
			for (size_t i = 0; i < elem_numbers.size(); ++i) {
				data_size *= elem_numbers[i];
			}
			_data.resize(data_size);
		}

		Shape shape() const
		{
			return Shape(*this);
		}

		int shape(int i) const
		{
			return _dims[i].elems;
		}

		size_t rank() const
		{
			return _dims.size();
		}

		int rows() const
		{
			return _dims[0].elems;
		}

		int cols() const
		{
			return _dims[1].elems;
		}

		Tensor<Scalar> dot(Tensor<Scalar> const& B) const
		{
			Tensor<Scalar> const& A = *this;
			TENSOR_ASSERT(A.rank() == 2 && B.rank() == 2, "only matrices support dot product");
			TENSOR_ASSERT(A.cols() == B.rows(), "can't multiply with wrong shapes");
			Tensor<Scalar> C({ A.rows(), B.cols() });
			for (int r = 0; r < C.rows(); ++r) {
				for (int c = 0; c < C.cols(); ++c) {
					Scalar comb = 0;
					for (int k = 0; k < A.cols(); ++k) {
						comb += A(r, k) * B(k, c);
					}
					C(r, c) = comb;
				}
			}
			return C;
		}

		Tensor<Scalar> operator+(Tensor<Scalar> const& B) const
		{
			Tensor<Scalar> const& A = *this;
			TENSOR_ASSERT(A.shape() == B.shape(), "can't add with wrong shapes");
			Tensor<Scalar> C(A.shape());
			for (size_t i = 0; i < A.size(); ++i) {
				C._data[i] = A._data[i] + B._data[i];
			}
			return C;
		}

		Tensor<Scalar> operator-(Tensor<Scalar> const& B) const
		{
			Tensor<Scalar> const& A = *this;
			TENSOR_ASSERT(A.shape() == B.shape(), "can't add with wrong shapes");
			Tensor<Scalar> C(A.shape());
			for (size_t i = 0; i < A.size(); ++i) {
				C._data[i] = A._data[i] - B._data[i];
			}
			return C;
		}

		bool operator==(Tensor<Scalar> const& B) const
		{
			Tensor<Scalar> const& A = *this;
			TENSOR_ASSERT(A.shape() == B.shape(), "can't compare equality with wrong shapes");
			for (size_t i = 0; i < A.size(); ++i) {
				if (A._data[i] != B._data[i])
					return false;
			}
			return true;
		}

		Tensor<Scalar> convolve2D(Tensor<Scalar> const& B) const
		{
			Tensor<Scalar> const& A = *this;
			TENSOR_ASSERT(A.rank() == 2 && B.rank() == 2, "only matrices support convolution");
			Tensor<Scalar> Z({ A.rows() - B.rows() + 1 , A.cols() - B.cols() + 1 });
			for (int r = 0; r < Z.rows(); r++) {
				for (int c = 0; c < Z.cols(); c++) {
					Scalar z = 0;
					for (int i = 0; i < B.rows(); i++) {
						for (int j = 0; j < B.cols(); j++) {
							z += A(r + i, c + j) * B(i, j);
						}
					}
					Z(r, c) = z;
				}
			}
			return Z;
		}

		Tensor<Scalar> elemwise(std::function<Scalar(Scalar)> func) const
		{
			Tensor<Scalar> Z;
			Z._dims = this->_dims;
			for (auto&& elem : this->_data) {
				Scalar z = func(elem);
				Z._data.push_back(z);
			}
		}

#pragma region Details

	private:

		struct IndexNumber
		{
			int i = 0;
			explicit IndexNumber(int i) : i(i) {}
		};

		void assert_index_range(IndexNumber i)
		{}

		template<typename T, typename... Args>
		void assert_index_range(IndexNumber i, T t, Args... args)
		{
#ifndef NDEBUG
			TENSOR_ASSERT(t >= 0 && t < _dims[i.i].elems, "index out of range");
			assert_index_range(IndexNumber(i.i + 1), args...);
#endif
		}

		template<typename... T>
		void assert_count_index(T... t)
		{
#ifndef NDEBUG 
			TENSOR_ASSERT(sizeof...(T) == _dims.size(), "index count must equal tensor dimension");
#endif 
		}

		size_t flat_index(IndexNumber i)
		{
			return 0;
		}

		template<typename T, typename... Args>
		size_t flat_index(IndexNumber i, T t, Args... args)
		{
			size_t a = t * _dims[i.i].elem_size;
			return a + flat_index(IndexNumber(i.i + 1), args...);
		}

#pragma endregion 
	};
}
#endif