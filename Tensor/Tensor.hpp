#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_
#include <vector>
#include <functional>
#include <random>
#include <tuple>
#include <utility>
#ifdef NDEBUG 
// release mode
#define TENSOR_ASSERT(cond,msg)
#else
// debug mode
#include <iostream>
#define TENSOR_ASSERT(cond,msg) if(!(cond)) throw msg
#endif


namespace Ten {
	constexpr struct ReserveDataComputeDimsTag {} reserve_data_compute_dims;
	constexpr struct ReserveDataSetDimsTag {} reserve_data_set_dims;
	constexpr struct UseFlatIndexTag {} use_flat_index;

	template<typename Scalar>
	class Tensor
	{
	public:
		class Shape : public std::vector<int>
		{
		public:
			Shape() {}

			Shape(std::vector<int> const& shape)
				:std::vector<int>(shape) {}

			Shape(Tensor const& t) {
				for (size_t r = 0; r < t.rank(); ++r) {
					push_back(t._dims[r].elems);
				}
			}

			bool operator==(Shape const& other) const {
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
				:elem_size{ 0 }, elems{ 0 }
			{}
			Dimension(int elem_size, int elems)
				:elem_size{ elem_size }, elems{ elems }
			{}
		};
		std::vector<Dimension> _dims;
		std::vector<Scalar> _data;




		Tensor(std::vector<int> const& shape, ReserveDataComputeDimsTag) {
			init_dimensions(shape);
			_data.reserve(compute_data_size(_dims));
		}

		Tensor(std::vector<Dimension> const& dims, int total_size, ReserveDataSetDimsTag) {
			_dims = dims;
			_data.reserve(total_size);
		}

	public:

		/// Construct tensor with shape and data
		Tensor(std::vector<int> const& shape, std::vector<Scalar> const& data)
			:Tensor(shape, reserve_data_compute_dims) {
			int size = compute_data_size(_dims);
			TENSOR_ASSERT(data.size() <= size, "too many initial values");
			for (auto&& x : data) {
				_data.push_back(x);
			}
			_data.resize(size);
		}

		/// Construct tensor with given shape and default data
		explicit Tensor(std::vector<int> const& shape)
			:Tensor(shape, std::vector<Scalar>{}) {}

		/// Move constructor
		Tensor(Tensor&& other) noexcept {
			_dims = std::move(other._dims);
			_data = std::move(other._data);
		}

		/// Move assignment
		Tensor& operator=(Tensor&& other) noexcept {
			_dims = std::move(other._dims);
			_data = std::move(other._data);
			return *this;
		}

		/// Copy constructor
		Tensor(Tensor const& other)
			:_dims{ other._dims }, _data{ other._data }
		{}

		/// Copy assignment
		Tensor& operator=(Tensor const& other) {
			this->_data = other._data;
			this->_dims = other._dims;
			return *this;
		}

		/// Construct a scalar tensor
		Tensor(const Scalar& number)
			:_dims{ Dimension(1,1) }, _data{ number }
		{}

		/// Uninitialized tensor (rank 0)
		Tensor()
			:_dims{}, _data{}
		{}

		/// Construct a tensor with a function that returns values.
		Tensor(std::vector<int> const& shape, std::function<Scalar()> func)
			:Tensor(shape, reserve_data_compute_dims) {
			int size = compute_data_size(shape);
			while (size--) {
				_data.push_back(func());
			}
		}

		/// Construct a tensor with a function that returns a value for each index.
		Tensor(std::vector<int> const& shape, std::function<Scalar(int)> func)
			:Tensor(shape, reserve_data_compute_dims) {
			TENSOR_ASSERT(this->rank() == 1, "this tensor cannot be addressed by 1 index");
			for (int i = 0; i < shape[0]; ++i)
				_data.push_back(func(i));
		}


		/// Construct a tensor with a function that returns a value for each index group.
		Tensor(std::vector<int> const& shape, std::function<Scalar(int, int)> func)
			:Tensor(shape, reserve_data_compute_dims) {
			TENSOR_ASSERT(this->rank() == 2, "this tensor cannot be addressed by 2 indices");
			for (int i = 0; i < shape[0]; ++i)
				for (int j = 0; j < shape[1]; ++j)
					_data.push_back(func(i, j));
		}

		/// Construct a tensor with a function that returns a value for each index group.
		Tensor(std::vector<int> const& shape, std::function<Scalar(int, int, int)> func)
			:Tensor(shape, reserve_data_compute_dims) {
			TENSOR_ASSERT(this->rank() == 3, "this tensor cannot be addressed by 3 indices");
			for (int i = 0; i < shape[0]; ++i)
				for (int j = 0; j < shape[1]; ++j)
					for (int k = 0; k < shape[2]; ++k)
						_data.push_back(func(i, j, k));
		}

		/// Construct a tensor with a function that returns a value for each FLATTENED index.
		Tensor(std::vector<int> const& shape, std::function<Scalar(int)> func, UseFlatIndexTag use_flat_index)
			:Tensor(shape, reserve_data_compute_dims) {
			int size = compute_data_size(shape);
			for (int i = 0; i < size; ++i)
				_data.push_back(func(i));
		}

		void assign(std::function<Scalar(int)> func) {
			TENSOR_ASSERT(this->rank() == 1, "this tensor cannot be addressed by 1 index");
			for (int i = 0; i < shape(0); ++i)
				(*this)(i) = func(i);
		}

		void assign(std::function<Scalar(int, int)> func) {
			TENSOR_ASSERT(this->rank() == 2, "this tensor cannot be addressed by 2 indices");
			for (int i = 0; i < shape(0); ++i)
				for (int j = 0; j < shape(1); ++j)
					(*this)(i, j) = func(i, j);
		}

		void assign(std::function<Scalar(int, int, int)> func) {
			TENSOR_ASSERT(this->rank() == 3, "this tensor cannot be addressed by 3 indices");
			for (int i = 0; i < shape(0); ++i)
				for (int j = 0; j < shape(1); ++j)
					for (int k = 0; k < shape(2); ++k)
						(*this)(i, j, k) = func(i, j, k);
		}

		void assign(std::function<Scalar(int)> func, UseFlatIndexTag use_flat_index) {
			for (int i = 0; i < size(); ++i)
				(*this)[i] = func(i);
		}



		void for_each(std::function<void(int)> func) const {
			TENSOR_ASSERT(this->rank() == 1, "this tensor cannot be addressed by 1 index");
			for (int i = 0; i < shape(0); ++i)
				func(i);
		}

		void for_each(std::function<void(int, int)> func) const {
			TENSOR_ASSERT(this->rank() == 2, "this tensor cannot be addressed by 2 indices");
			for (int i = 0; i < shape(0); ++i)
				for (int j = 0; j < shape(1); ++j)
					func(i, j);
		}

		void for_each(std::function<void(int, int, int)> func) const {
			TENSOR_ASSERT(this->rank() == 3, "this tensor cannot be addressed by 3 indices");
			for (int i = 0; i < shape(0); ++i)
				for (int j = 0; j < shape(1); ++j)
					for (int k = 0; k < shape(2); ++k)
						func(i, j, k);
		}

		void for_each(std::function<void(int)> func, UseFlatIndexTag use_flat_index) const {
			for (int i = 0; i < size(); ++i)
				func(i);
		}

		Tensor<Scalar> elementwise(std::function<Scalar(Scalar)> func) const {
			Tensor t(_dims, this->size(), reserve_data_set_dims);
			for (int i = 0; i < size(); ++i)
				t._data.push_back(func((*this)[i]));
			return t;
		}

		static Tensor<Scalar> Zeros(std::vector<int> const& shape) {
			return Tensor(shape, []() {return 0; });
		}

		static Tensor<Scalar> Ones(std::vector<int> const& shape) {
			return Tensor(shape, []() {return 1; });
		}

		static Tensor<Scalar> Constants(std::vector<int> const& shape, Scalar c) {
			return Tensor(shape, [=]() {return c; });
		}


		template<typename... Args>
		Scalar& operator()(Args... args) {
#ifndef NDEBUG 
			assert_index_range<0>(args...);
			assert_count_index(args...);
#endif // !NDEBUG 
			auto i = flat_index<0>(args...);
			return _data[i];
		}

		template<typename... Args>
		Scalar const& operator()(Args... args) const {
			return const_cast<Tensor&>(*this).operator()(args...);
		}

		Scalar& operator[](int flat_index) {
			return _data[flat_index];
		}

		Scalar const& operator[](int flat_index) const {
			return _data[flat_index];
		}

		size_t size() const {
			return _data.size();
		}

		Scalar* data() {
			return _data.data();
		}

		Scalar const* data() const {
			return _data.data();
		}


		void reshape(std::vector<int> const& shape) {
			init_dimensions(shape);
			// compute data size & resize data
			_data.resize(compute_data_size(shape));
		}

		Shape shape() const {
			return Shape(*this);
		}

		int shape(int i) const {
			return _dims[i].elems;
		}

		size_t rank() const {
			return _dims.size();
		}

		int rows() const {
			return _dims[0].elems;
		}

		int cols() const {
			return _dims[1].elems;
		}


		Tensor<Scalar> operator+(Tensor<Scalar> const& B) const {
			Tensor<Scalar> const& A = *this;
			TENSOR_ASSERT(A.shape() == B.shape(), "can't add with wrong shapes");
			return Tensor<Scalar>(
				A.shape(),
				[&](int i) {  return A[i] + B[i]; },
				use_flat_index
				);
		}

		Tensor<Scalar> operator-(Tensor<Scalar> const& B) const {
			Tensor<Scalar> const& A = *this;
			TENSOR_ASSERT(A.shape() == B.shape(), "can't add with wrong shapes");
			return Tensor<Scalar>(
				A.shape(),
				[&](int index) {  return A[index] - B[index]; },
				use_flat_index
				);
		}

		bool operator==(Tensor<Scalar> const& B) const {
			Tensor<Scalar> const& A = *this;
			TENSOR_ASSERT(A.shape() == B.shape(), "can't compare equality with wrong shapes");
			for (size_t i = 0; i < A.size(); ++i)
				if (A[i] != B[i])
					return false;
			return true;
		}

		bool operator!=(Tensor<Scalar> const& B) const {
			return !((*this) == B);
		}

		Tensor<Scalar> dot(Tensor<Scalar> const& B) const {
			Tensor<Scalar> const& A = *this;
			TENSOR_ASSERT(A.rank() == 2 && B.rank() == 2, "only matrices support dot product");
			TENSOR_ASSERT(A.cols() == B.rows(), "can't multiply with wrong shapes");

			return Tensor<Scalar>(
				{ A.rows(), B.cols() },
				[&](int i, int j) {
					Scalar sum = 0;
					for (int k = 0; k < A.cols(); ++k)
						sum += A(i, k) * B(k, j);
					return sum;
				}
			);
		}

		Tensor<Scalar>& operator+=(Tensor<Scalar> const& other) {
			*this = *this + other;
			return *this;
		}

		Tensor<Scalar>& operator-=(Tensor<Scalar> const& other) {
			*this = *this - other;
			return *this;
		}

		Tensor<Scalar>& operator*=(Scalar other) {
			*this = *this * other;
			return *this;
		}

		Tensor<Scalar>& operator/=(Scalar other) {
			*this = *this / other;
			return *this;
		}

		Tensor<Scalar> convolve2D(Tensor<Scalar> const& B) const {
			Tensor<Scalar> const& A = *this;
			TENSOR_ASSERT(A.rank() == 2 && B.rank() == 2, "only matrices support convolution");
			return Tensor<Scalar>(
				{ A.rows() - B.rows() + 1, A.cols() - B.cols() + 1 },
				[&](int r, int c) {
					Scalar z = 0;
					for (int i = 0; i < B.rows(); i++)
						for (int j = 0; j < B.cols(); j++)
							z += A(r + i, c + j) * B(i, j);
					return z;
				}
			);
		}

		Tensor<Scalar> transpose() const {
			Tensor<Scalar> const& A = *this;
			TENSOR_ASSERT(A.rank() == 2, "only matrices support transposition");
			return Tensor<Scalar>(
				{ A.cols(), A.rows() },
				[&](int r, int c) {
					return A(c, r);
				}
			);
		}



		//Tensor<Scalar> elemwise(std::function<Scalar(Scalar elem)> func) const
		//{
		//	CREATE_TENSOR_Z;
		//	for (auto&& elem : this->_data) {
		//		Scalar z = func(elem);
		//		Z._data.push_back(z);
		//	}
		//	return Z;
		//}

		//Tensor<Scalar> elemwise(std::function<Scalar(int index, Scalar elem)> func) const
		//{
		//	TENSOR_ASSERT(this->rank() == 1, "this tensor cannot be addressed by 1 index");
		//	CREATE_TENSOR_Z;
		//	for (int i = 0; i < size(); i++) {
		//		Scalar z = func(i, _data[i]);
		//		Z._data.push_back(z);
		//	}
		//	return Z;
		//}

		//Tensor<Scalar> elemwise(std::function<Scalar(std::tuple<int, int> index, Scalar elem)> func) const
		//{
		//	TENSOR_ASSERT(this->rank() == 2, "this tensor cannot be addressed by 2 indices");
		//	CREATE_TENSOR_Z;
		//	for (int i0 = 0; i0 < Z.shape(0); ++i0) {
		//		for (int i1 = 0; i1 < Z.shape(1); ++i1) {
		//			for (auto&& elem : this->_data) {
		//				Scalar z = func({ i0,i1 }, elem);
		//				Z._data.push_back(z);
		//			}
		//		}
		//	}
		//	return Z;
		//}

		//Tensor<Scalar> elemwise(std::function<Scalar(std::tuple<int, int, int> index, Scalar elem)> func) const
		//{
		//	TENSOR_ASSERT(this->rank() == 3, "this tensor cannot be addressed by 3 indices");
		//	CREATE_TENSOR_Z;
		//	for (int i0 = 0; i0 < Z.shape(0); ++i0) {
		//		for (int i1 = 0; i1 < Z.shape(1); ++i1) {
		//			for (int i2 = 0; i1 < Z.shape(2); ++i2) {
		//				for (auto&& elem : this->_data) {
		//					Scalar z = func({ i0,i1,i2 }, elem);
		//					Z._data.push_back(z);
		//				}
		//			}
		//		}
		//	}
		//	return Z;
		//}

		//void elemwise_assign(std::function<Scalar(Scalar elem)> func)
		//{
		//	for (auto&& elem : this->_data) {
		//		Scalar z = func(elem);
		//		Z._data.push_back(z);
		//	}
		//	/// to do : finish this
		//}

		Tensor<Scalar> operator*(Scalar coeff) const {
			return this->elemwise([=](Scalar elem) {
				return elem * coeff;
				});
		}

		Tensor<Scalar> operator/(Scalar coeff) const {
			return this->elemwise([=](Scalar elem) {
				return elem / coeff;
				});
		}

		Tensor<Scalar> operator-() const {
			return this->elemwise([=](Scalar elem) {
				return -elem;
				});
		}

	private:

#pragma region Details: Accessing Index


#ifndef NDEBUG

		template<unsigned i>
		void assert_index_range() {}

		template<unsigned i, typename T, typename... Args>
		void assert_index_range(T t, Args... args) {
			TENSOR_ASSERT(t >= 0 && t < _dims[i].elems, "index out of range");
			assert_index_range<i + 1>(args...);
		}

		template<typename... T>
		void assert_count_index(T... t) {
			TENSOR_ASSERT(sizeof...(T) == _dims.size(), "index count must equal tensor dimension");
		}
#endif

		template<unsigned i = 0>
		size_t flat_index() {
			return 0;
		}

		template<unsigned i, typename T, typename... Args>
		size_t flat_index(T t, Args... args) {
			return t * _dims[i].elem_size + flat_index<i + 1>(args...);
		}

#pragma endregion 
#pragma region Details: Dimension and sizes

		void init_dimensions(std::vector<int> const& shape) {
			// init dimensions
			_dims.resize(shape.size());
			for (size_t i = 0; i < shape.size(); ++i) {
				_dims[i].elems = shape[i];
				int e = 1;
				for (size_t j = i + 1; j < shape.size(); ++j) {
					e *= shape[j];
				}
				_dims[i].elem_size = e;
			}
		}

		static int compute_data_size(const std::vector<Dimension>& d) {
			int data_size = 1;
			for (int i = 0; i < d.size(); ++i) {
				data_size *= d[i].elems;
			}
			return data_size;
		}

		static int compute_data_size(std::vector<int> const& shape) {
			int data_size = 1;
			for (int i = 0; i < shape.size(); ++i) {
				data_size *= shape[i];
			}
			return data_size;
		}
#pragma endregion


	};

	inline std::default_random_engine rand_eng;

	inline Tensor<double> RandomUniform(std::vector<int> const& shape, double lower_bound, double upper_bound) {
		std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
		return Tensor<double>(shape, [&]() {return unif(rand_eng); });
	}

	inline Tensor<int> RandomUniform(std::vector<int> const& shape, int lower_bound, int upper_bound) {
		std::uniform_int_distribution<int> unif(lower_bound, upper_bound);
		return Tensor<int>(shape, [&]() {return unif(rand_eng); });
	}

	inline Tensor<double> RandomNormal(std::vector<int> const& shape, double mean, double stddev) {
		std::normal_distribution<double> norm(mean, stddev);
		return Tensor<double>(shape, [&]() {return norm(rand_eng); });
	}
}
#endif