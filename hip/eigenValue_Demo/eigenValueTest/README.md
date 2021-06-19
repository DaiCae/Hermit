# 先导杯-特征值求解 环境准备说明
Lapacke-3.9.1编译(注意cmake版本>3.2)：
对CMakeLists.txt修改如下：
351：option(LAPACKE "Build LAPACKE" OFF) 改为：option(LAPACKE "Build LAPACKE" ON) 
编译产生lapacke的库文件。
编译命令：
/public/home/zhaohp/rocBLAS_1228/bin/cmake-3.13.5/bin/cmake -DCMAKE_INSTALL_PREFIX=lapack-install ..
make -j
make install

Demo编译方式及参数解释：   
hipcc main.cpp  -o test.out  -I/public/home/zhaohongpeng/zhaohp/Lapack_test/lapack-install/include -L/public/home/zhaohongpeng/zhaohp/Lapack_test/lapack-install/lib  -llapacke -llapack -lblas -lgfortran  
运行方式   
./test.out 0 1 4 2   
第一个参数是控制输出结果的参数：0表示求解结果包含特征值和特征向量，1表示求解结果只包含特征值；
第二个参数是控制Hermite矩阵的初始化方式，0表示只需初始化下三角矩阵，1表示只需初始化上三角矩阵；
第三个参数是设置Hermite矩阵的行列数;
第四个参数是设置特征值求解函数的调用次数，为了便于计算性能。
