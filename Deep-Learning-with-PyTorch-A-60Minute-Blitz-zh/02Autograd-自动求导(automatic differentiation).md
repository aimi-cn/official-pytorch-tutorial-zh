# Autograd-自动求导(automatic differentiation)

> 搜索微信公众号:'AI-ming3526'或者'计算机视觉这件小事' 获取更多算法、机器学习干货  
> csdn：https://blog.csdn.net/baidu_31657889/  
> github：https://github.com/aimi-cn/AILearners

> 译者注：本教程在pytorch官方教程的基础上翻译修改得到,代码输出是在本人在自己笔记本上运行之后放上去的，可能会和官方的输出结果有所不同，一切输出结果按照官方教程为准,原教程请点击[pytorch-official-tutorial](https://pytorch.org/tutorials/index.html)

PyTorch中，所有神经网络的核心是autograd包。先简单介绍一下这个包，然后训练我们的第一个的神经网络。

autograd包为张量上的所有操作提供了自动求导机制。它是一个在运行时定义（define-by-run）的框架，这意味着反向传播是根据代码如何运行来决定的，并且每次迭代可以是不同的.

让我们用一些简单的例子来看看吧。

## 张量（Tensor）

torch.Tensor是这个包的核心类。如果设置它的属性 .requires_grad为True，那么它将会追踪对于该张量的所有操作。当完成计算后可以通过调用.backward()，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到.grad属性。

要阻止一个张量被跟踪历史，可以调用.detach()方法将其与计算历史分离，并阻止它未来的计算记录被跟踪。

为了防止跟踪历史记录（和使用内存），可以将代码块包装在with torch.no_grad():中。在评估模型时特别有用，因为模型可能具有requires_grad = True的可训练的参数，但是我们不需要在此过程中对他们进行梯度计算。

还有一个类对于autograd的实现非常重要：Function。

Tensor和Function是互联的，它们组成一个非循环图，记录了完整的计算过程。每个tensor都有一个.grad_fn属性指向了创建这个Tensor的Function（用户自己创建的Tensor类型除外，它们的grad_fn为None）。

如果想要计算Tensor的导数，可以调用.backward()。如果tensor是标量（即它包含一个元素），那么不需要为backward()指定任何参数，如果有多个元素，那么就需要指定梯度参数。

创建一个张量，并设置requires_grad=True来跟踪计算

```python
import torch
x = torch.ones(2, 2, requires_grad=True)
print(x)
```
输出：
```python
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
```

做一个张量运算:

```python
y = x + 2
print(y)
```
输出：
```python
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
```
y是计算的结果，所以它有grad_fn属性。
```python
print(y.grad_fn)
```
输出：
```python
<AddBackward0 object at 0x00000000023457B8>
```

对y进行更多操作:

```python
z = y * y * 3
out = z.mean()

print(z, out)
```
输出：
```python
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward1>)
```

.requires_grad_(...) 原地改变了现有张量的 requires_grad 标志。如果没有指定的话，默认输入的这个标志是False。
```python
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)
```
输出：
```python
False
True
<SumBackward0 object at 0x0000000002345F60>
```
## 梯度（Gradients）

现在我们进行反向传播计算梯度，因为out只包含一个标量，out.backward()与out.backward(torch.tensor(1.))等价。

```python
print(x)
print(out)
#计算反向传播
out.backward()
#现在计算d(out)/d(x)
print(x.grad)
```
输出：
```python
tensor([[1., 1.],
        [1., 1.]], requires_grad=True)

tensor(27., grad_fn=<MeanBackward1>)

tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```

那么4.5这个数值是如何得到的呢，我整理之后给出比较直观的计算过程，首先我们要先知道这个out是怎么来的，y = x + 2 --> z = y * y * 3 --> out = z.mean(), x是一个2*2 的矩阵，所以z去平均的时候是除以4的。那么就可以得到下面公式：

调用out张量 $“o”$。得到：

$$o = \frac{1}{4}\sum_i z_i$$

$$z_i = 3(x_i+2)^2$$

和：

$$z_i\bigr\rvert_{x_i=1} = 27$$

因此,

$$\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)$$

所以，

$$\frac{\partial o}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5$$

这样我们就可以非常直观的感受到这个4.5的来源了。

github加载不出来公式的同学可以安装**MathJax Plugin for Github**这个插件来看。

数学上，若有向量值函数 $\vec{y}=f(\vec{x})$，那么 $\vec{y}$ 相对于 $\vec{x}$ 的梯度是一个雅可比矩阵：

$$
J=\left(\begin{array}{ccc}{\frac{\partial y_{1}}{\partial x_{1}}} & {\cdots} & {\frac{\partial y_{1}}{\partial x_{n}}} \\ {\vdots} & {\ddots} & {\vdots} \\ {\frac{\partial y_{m}}{\partial x_{1}}} & {\cdots} & {\frac{\partial y_{m}}{\partial x_{n}}}\end{array}\right)
$$

通常来说，torch.autograd 是计算雅可比向量积的一个“引擎”。也就是说，给定任意向量 $v=\left(\begin{array}{cccc} v_{1} & v_{2} & \cdots & v_{m}\end{array}\right)^{T}$，计算乘积 $J\cdot v$。如果 $v$ 恰好是一个标量函数 $l=g\left(\vec{y}\right)$ 的导数，即 $v=\left(\begin{array}{ccc}\frac{\partial l}{\partial y_{1}} & \cdots & \frac{\partial l}{\partial y_{m}}\end{array}\right)^{T}$，那么根据链式法则，雅可比向量积应该是 $l$ 对 $\vec{x}$ 的导数：

$$
J^{T} \cdot v=\left(\begin{array}{ccc}{\frac{\partial y_{1}}{\partial x_{1}}} & {\cdots} & {\frac{\partial y_{m}}{\partial x_{1}}} \\ {\vdots} & {\ddots} & {\vdots} \\ {\frac{\partial y_{1}}{\partial x_{n}}} & {\cdots} & {\frac{\partial y_{m}}{\partial x_{n}}}\end{array}\right)\left(\begin{array}{c}{\frac{\partial l}{\partial y_{1}}} \\ {\vdots} \\ {\frac{\partial l}{\partial y_{m}}}\end{array}\right)=\left(\begin{array}{c}{\frac{\partial l}{\partial x_{1}}} \\ {\vdots} \\ {\frac{\partial l}{\partial x_{n}}}\end{array}\right)
$$

- 注意：$v^{T} \cdot J$得到的是一个行向量，我们可以通过计算$J^{T} \cdot v$来得到它的列向量。

雅可比向量积的这一特性使得将外部梯度输入到具有非标量输出的模型中变得非常方便。

现在让我们看一个向量雅可比矩阵乘积的例子：

```python
x = torch.randn(3, requires_grad=True)

y = x * 2
#求范数 y.data.norm()是向量y所有的元素平方和，再开根号。
while y.data.norm() < 1000:
    y = y * 2

print(y)
```
输出：
```python
tensor([-507.5154,  772.6517,  455.8939], grad_fn=<MulBackward0>)
```
在这种情况下，y不再是标量。torch.autograd不能直接计算出整个雅可比矩阵，但是如果我们只想要向量-雅可比矩阵的乘积，只需将向量作为backward参数传递：

```python
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
 
print(x.grad)
```
输出：
```python
tensor([5.1200e+01, 5.1200e+02, 5.1200e-02])
```
您还可以使用.requires_grad=True来停止autograd跟踪Tensor上的历史记录，方法是使用torch.no_grad()来封装代码块
```python
print(x.requires_grad)
print((x ** 2).requires_grad)
 
with torch.no_grad():
    print((x ** 2).requires_grad)
```
输出：
```python
True
True
False
```

> 后续阅读：  
> autograd和Function的文档见：https://pytorch.org/docs/autograd  
> 中文文档地址：https://github.com/apachecn/pytorch-doc-zh

笔记github地址~：[地址](https://github.com/aimi-cn/official-pytorch-tutorial)






