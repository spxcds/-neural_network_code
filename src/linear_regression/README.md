## 线性回归求导

首先, 要求的拟合函数为
$$
y = X * W + b
$$
其中

$\mathbf{X}=\begin{bmatrix}x_{1}\\ x_{2}\\ \vdots\\ x_{m} \end{bmatrix}$, $\mathbf{W}=\begin{bmatrix}w_{1} \end{bmatrix}$, $\mathbf{b}=\begin{bmatrix}b_{1} \end{bmatrix}$
为了方便, 可以把X增广一维, 变为

$\mathbf{X}=\begin{bmatrix}x_{1} & 1\\ x_{2}& 1\\ \vdots\\ x_{m}& 1\end{bmatrix}$, $\mathbf{W}=\begin{bmatrix}w_{1} \ b1 \end{bmatrix}$
那么, 拟合函数就变成了
$$
y = X * W
$$


## 需要用到的矩阵求导公式

首先给出一个m\*n的矩阵A和一个n\*1的列向量x
$$
A=\begin{bmatrix}
a_{11} & a_{12} & \cdots & x_{1n}\\
a_{21} & a_{22} & \cdots & x_{2n}\\
\vdots & \vdots & &  \vdots\\
a_{m1} & a_{m2} & \cdots& x_{mn}\\
\end{bmatrix} 

x=\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix} \\

Ax = \begin{bmatrix}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n\\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n\\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n\\
\end{bmatrix}
$$
Ax对x求偏导, 得
$$
\frac{\partial Ax}{\partial x} = \begin{bmatrix}
a_{11} & a_{21} & \cdots & a_{m1} \\
a_{12} & a_{22} & \cdots & a_{m2} \\
\vdots & \vdots & & \vdots \\
a_{1n} & a_{2n} & \cdots & a_{mn} \\    
\end{bmatrix} = A^T
$$
Ax对$x^T$求偏导, 得
$$
\frac{\partial Ax}{\partial x^T} =\begin{bmatrix}
a_{11} & a_{12} & \cdots & x_{1n}\\
a_{21} & a_{22} & \cdots & x_{2n}\\
\vdots & \vdots & &  \vdots\\
a_{m1} & a_{m2} & \cdots& x_{mn}\\
\end{bmatrix} = A \\

\frac{\partial x^T A}{\partial x} = \left[ \left(  \frac{\partial x^T A}{\partial x} \right)^T\right]^T\\ 

= \left[  \frac{(\partial x^T A)^T}{\partial x^T}\right]^T \\
= \left[ \frac{\partial A^T x}{\partial x^T}\right]^T \\
= (A^T)^T \\
= A
$$



## Loss函数求导

Loss函数为
$$
L(W) = \frac{1}{2m} (X * W - y)^2 = \frac{1}{2m} (X * W - y)^T * (X * W - y)
$$

设
$$
f(W) = X * W - y
$$

其导函数为
$$
\frac{\partial L(W)}{\partial W} =  \frac{\partial f(W)}{ \partial W} * \frac{\partial L(W)}{ \partial f(W)}
$$

其中
$$
\frac{\partial f(W) }{\partial W} = 
$$

