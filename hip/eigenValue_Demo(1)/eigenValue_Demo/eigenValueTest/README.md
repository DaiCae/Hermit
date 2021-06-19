# 先导杯-特征值求解 环境准备说明
/***************************************************************************************
										lapack-3.9.1安装
/***************************************************************************************
Lapacke-3.9.1编译(注意cmake版本>3.2)：
对CMakeLists.txt修改如下：
351：option(LAPACKE "Build LAPACKE" OFF) 改为：option(LAPACKE "Build LAPACKE" ON) 
编译产生lapacke的库文件。
编译命令：
cd lapack-3.9.1
mkdir build && cd build
module load module load compiler/cmake/3.15.6
cmake -DCMAKE_INSTALL_PREFIX=lapack-install ..
make -j
make install

/***************************************************************************************
										ROCm环境加载
/***************************************************************************************
ROCm环境：初步拟定使用rocm-3.9.1环境作为统一的环境进行赛题的评分
环境加载方法：
module rm compiler/rocm/2.9
module load compiler/rocm/3.9.1

/***************************************************************************************
										程序编译命令参考
/***************************************************************************************
Demo编译方式及参数解释：   
hipcc main.cpp  -o test.out  -I/public/home/zhaohongpeng/zhaohp/Lapack_test/lapack-install/include -L/public/home/zhaohongpeng/zhaohp/Lapack_test/lapack-install/lib  -llapacke -llapack -lblas -lgfortran  

hipcc main.cpp -o test.out  -I/public/home/zyx0616/eigenValue_Demo/lapack-3.9.1/build/include -L/public/home/zyx0616/eigenValue_Demo/lapack-3.9.1/build/lib  -llapacke -llapack -lblas -lgfortran  


hipcc main.cpp -o test.out  -I/public/home/achus9mjo4/eigenValue_Demo/lapack-3.9.1/build/include -L/public/home/achus9mjo4/eigenValue_Demo/lapack-3.9.1/build/lib  -llapacke -llapack -lblas -lgfortran  

srun -p PilotCup -N 1 -n 1 --gres=dcu:1 -o test.log -e test.err ./test.out 1 1 1280 1

运行方式   
./test.out 0 1 4 2   
第一个参数是控制输出结果的参数：0表示求解结果包含特征值和特征向量，1表示求解结果只包含特征值；
第二个参数是控制Hermite矩阵的初始化方式，0表示只需初始化下三角矩阵，1表示只需初始化上三角矩阵；
第三个参数是设置Hermite矩阵的行列数;
第四个参数是设置特征值求解函数的调用次数，为了便于计算性能。
