AddModule(
   header_only      
   module_name       Tensor 
   dependencies      "" 
   tests             "./test/TensorTest.cpp"  
   test_dependencies "Boost::unit_test_framework" 
)