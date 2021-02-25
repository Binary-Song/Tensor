AddModule(
   header_only     #[opt.]  whether the module is header-only or compiled. Must specify exactly one of them.
   module_name     Tensor#[sing.] name of the module
   dependencies    ""#[mult.] libraries the module depends on   src             #[mult.] source files, if not header-only
   tests           "./test/TensorTest.cpp"#[mult.] test sources, each file built into one executable. Tests will depend on TEST_DEFAULT_DEPENDENCY if set. Ususally, you set TEST_DEFAULT_DEPENDENCY to Boost::unit_test_framework
   test_dependencies "Boost::unit_test_framework" #[mult.] By default, tests depend on: 1) the module being tested, 2) TEST_DEFAULT_DEPENDENCY. If these are not enough, you specify test_dependencies to add others.
)