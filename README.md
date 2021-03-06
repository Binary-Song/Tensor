# 简单的C++张量库

## 特点

- 只有头文件，用起来简单。

- 没有使用表达式模板，不会为每个表达式构建一个类型，减少了传参和赋值的麻烦。

- 没有使用表达式模板，因此不支持懒求值，会给每个中间表达式创建临时对象。因此如果想要最大性能，可以将公式展开。

## 使用

创建一个2x2x2张量，指定值：

```
Tensor<double> t({3,3,3},{1,2,3,4,5,6,7,8});
```

创建一个2x2矩阵，指定值：
```
Tensor<int> t({2,2},
{
    1, 2,
    3, 4,
});
```

创建一个2x2零矩阵：
```
Tensor<int> t({2,2});
```
或
```
auto t = Tensor<int>::Zeros({2,2});
```

创建一个只含一个标量的1维张量：
```
Tensor<int> t(42);
```

用`()`读写值。张量是几维就要给几个下标：
```
Tensor<int> a({5,5}); // 2维张量
a(1,2) = 42;
Tensor<int> b({5,5,5}); // 3维张量
b(1,2,3) = 42;
```

张量的“维数”也称为“秩”，即`rank()`：
```
Tensor<int> a({5,5,5}); // a.rank() == 3
Tensor<int> b({5,5}); // a.rank() == 2
Tensor<int> c({5}); // a.rank() == 1
Tensor<int> d; // a.rank() == 0 
```

