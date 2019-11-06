# Neutron Supermirror Library

Author and email:  
Wenbo Mo  
mwb19@mails.tsinghua.edu.cn  
Huarui Wu    
huarui92@163.com  
Xuewu Wang  
wangxuewu@tsinghua.edu.cn  
Department of Engineering Physics, Tsinghua University  
Key Laboratory of Particle & Radiation Imaging (Tsinghua University), Ministry of Education  
Beijing 100084, China  

------------------------------------------------------
## Update log  

### Version 1
2019.03.27  
version1.0.0  
* Add basic reflectivity calculation function  

### Version 2
2019.03.31  
version2.0.0  
* Add reflectivity calculation function under the introduction of roughness  
2019.04.03  
version2.1.0  
* Add reflectivity calculation function under absorbing conditions  
2019.04.09  
version2.2.0  
* Add average reflectivity calculation function from 1 to m  
2019.4.10  
version2.3.0  
* Add a function to calculate the height and width of the Bragg peaks generated by n film pairs  

### Version 3
2019.04.10  
version3.0.0  
* Added 7 implementation functions of super-mirror design algorithm  
2019.04.11  
version3.0.1  
修复RSD算法函数中的bug:在接近迭代终点处可能会锐减至负值  
2019.04.11  
version3.1.0  
新增Improved Mezei算法调参函数  
2019.04.12  
version3.2.0  
新增反射率曲线拟合函数  
2019.04.12  
version3.3.0  
新增参数优化系列函数  
新增SM算法调参函数  
新增ABC算法调参函数  
2019.04.13  
version3.4.0  
新增ABC算法实现函数（使用优化后的参数）  
新增EAYAO算法调参函数  
2019.04.14  
version3.5.0  
新增反射率曲线拟合函数2  
新增IC算法调参函数  
2019.04.18  
version3.6.0  
新增引入厚度制造误差后的膜对平均反射率、拟合m值的计算函数  
2019.04.19  
version3.6.5  
修复平均反射率计算函数中存在的bug：当输入m值非常接近于1时，分母为0  
新增计算全m范围的平均反射率的函数  
2019.04.23  
version3.7.0  
完善了引入厚度制造误差后的膜对平均反射率、拟合m值的计算函数的功能，加入了无误差情况的比较  
2019.04.25  
version3.8.0  
新增反射率曲线通用拟合函数（适用于引入粗糙度的情形）  
2019.4.30  
version3.8.1  
扩大了反射率曲线拟合区域，最大拟合m值可达10  

### Version 4
2019.04.30  
version4.0.0  
针对大m值、超多膜对的超镜进行优化  
发现问题：基于矩阵方法反射率计算函数不适合计算膜对数目超过1300层的超镜的反射率，会出现矩阵计算溢出的问题  
新增量子力学迭代计算反射系数的反射率计算函数（简称递推方法），原先的反射率计算函数称为矩阵方法  
新增基于递推方法的引入吸收条件下的反射率计算函数  
新增基于递推方法的引入吸收、同时考虑粗糙度为常数条件下的反射率计算函数  
新增基于递推方法的引入吸收、同时考虑粗糙度逐层增长条件下的反射率计算函数  
将反射率曲线拟合函数中使用的基于矩阵方法的反射率计算函数修改为基于递推方法的  
将计算n个膜对产生的布拉格峰的高度和宽度的函数中使用的基于矩阵方法的反射率计算函数修改为基于递推方法的  
2019.05.04  
version4.1.0  
删除反射率曲线通用拟合函数RefFitGen(x0,y)（因为其并不适用于引入粗糙度的情形）  
新增常数反射率曲线拟合函数  
新增粗糙度逐层增长的反射率曲线拟合函数  
以上两个拟合函数还在测试中，数据可以拟合上，但是拟合参数尚不能反映粗糙度水平的物理意义  
2019.05.07  
version4.2.0  
新增标定膜层厚度的精度函数  
2019.05.29  
version4.3.0  
新增粗糙度优化的GRB算法实现函数  
新增引入绝对厚度制造误差后的膜对平均反射率、拟合m值的计算函数  
2019.06.17  
version4.5.0  
新增可以设计满足任意形状反射率曲线的超镜的RSD算法实现函数  
2019.09.04  
version4.5.1  
对函数包中的注释文字进行完善  
2019.09.04  
version4.6.0  
修复version4.0。0中发现的基于矩阵方法反射率计算函数在计算膜对数目超过1300层的超镜的反射率时，会出现矩阵元素值溢出的问题  
每一对膜对迭代前，对矩阵元素进行归一化，防止膜层数目过多时矩阵元素数值过大溢出  
2019.09.05  
version4.7.0  
新增基于矩阵方法的各类粗糙度条件下的反射率计算函数  
将第一大类反射率计算函数分为1.1基于矩阵方法和1.2基于量子力学方法的两个小类  
两种方法方法的反射率计算函数的计算结果是一致的，  
基于量子力学方法的计算函数在基于矩阵方法的计算函数名称后加了一个“2”以示区分，  
使用函数时同名函数可以任意选择，输入参数和输出数据都相同  
