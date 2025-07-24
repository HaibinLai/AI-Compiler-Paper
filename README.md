
## Project

### TVM

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723170618532.png)

Transform your code to Relay (an IR) and try to optimize though TE 

比如
**拆分(tiling),** 比如大矩阵乘拆成小 tile（小方块）在 GPU 上做。
**向量化(vectorization),** 用 SIMD 指令
**并行化(parallelization),**  生成多线程的 parallel-for 循环
**循环展开(unrolling),** 小循环（比如 reduce over 4 个元素）常常直接完全展开
**融合(fusion)**

先 fusion 然后 Tiling → 再 Parallelization → 再 Unrolling → 再 Vectorization


[Quick Start — tvm 0.22.dev0 documentation](https://tvm.apache.org/docs/get_started/tutorials/quick_start.html)
```txt
mod, [param_spec](https://docs.python.org/3/library/stdtypes.html#list "builtins.list") = MLPModel().export_tvm(
    spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)
mod.show()

# from tvm.script import ir as I
# from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def forward(x: R.Tensor((1, 784), dtype="float32"), fc1_weight: R.Tensor((256, 784), dtype="float32"), fc1_bias: R.Tensor((256,), dtype="float32"), fc2_weight: R.Tensor((10, 256), dtype="float32"), fc2_bias: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            permute_dims: R.Tensor((784, 256), dtype="float32") = R.permute_dims(fc1_weight, axes=None)
            matmul: R.Tensor((1, 256), dtype="float32") = R.matmul(x, permute_dims, out_dtype="void")
            add: R.Tensor((1, 256), dtype="float32") = R.add(matmul, fc1_bias)
            relu: R.Tensor((1, 256), dtype="float32") = R.nn.relu(add)
            permute_dims1: R.Tensor((256, 10), dtype="float32") = R.permute_dims(fc2_weight, axes=None)
            matmul1: R.Tensor((1, 10), dtype="float32") = R.matmul(relu, permute_dims1, out_dtype="void")
            add1: R.Tensor((1, 10), dtype="float32") = R.add(matmul1, fc2_bias)
            gv: R.Tensor((1, 10), dtype="float32") = add1
            R.output(gv)
        return gv
```


### XLA

XLA (Accelerated Linear Algebra) is an open-source machine learning (ML) compiler for GPUs, CPUs, and ML accelerators.
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723172849549.png)
[TensorFlow XLA优化原理与示例 - 吴建明wujianming - 博客园](https://www.cnblogs.com/wujianming-110117/p/15333306.html)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723173126047.png)



![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723173150461.png)

补充，在做XlaOps时，基本上会做这三件事情：

- 算子融合 (Fusion)：把连续的逐元素运算合并为一个核函数，减少内存访问。
- 常量折叠 (Constant Folding)：计算时已知的值直接预计算。
- Layout 优化：调整张量内存布局，使后续 kernel 更高效。

| 优化                        | XLA 能不能做 | 做的层次 / 阶段                                                |
| ------------------------- | -------- | -------------------------------------------------------- |
| **拆分 (Tiling)**           | ✅ 可以     | 在后端 Lowering 和 GPU TPU 后端里做 block / warp / thread tiling |
| **向量化 (Vectorization)**   | ✅ 可以     | CPU 后端：HLO -> LLVM IR，再用 LLVM 的 Loop Vectorizer          |
| **并行化 (Parallelization)** | ✅ 可以     | GPU/TPU：kernel launch，CPU：OpenMP/线程                      |
| **循环展开 (Unrolling)**      | ✅ 可以     | HLO Lowering → LLVM → LLVM loop unroll pass              |

- 高层 → HLO IR 做 fusion、partition
- 低层 → LLVM / GPU backend 做 tiling/unroll/vectorize/parallel

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723173927799.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723173937743.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723173944871.png)


### TensorRT

非常难装
[TensorRT SDK | NVIDIA Developer](https://developer.nvidia.com/tensorrt)

[Writing custom operators with TensorRT Python plugins - NVIDIA TensorRT Standard Python API Documentation 10.13.0](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/pluginGuide.html)

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723175117813.png)


## OSDI 20

### RAMMER

上科大神作

[算子调度优化论文分享：Rammer - 知乎](https://zhuanlan.zhihu.com/p/616050345)

以往的算子优化是：算子内优化（如循环展开等等）+算子间优化（fusion）
RAMMER尝试用最小的单元：task来完成

“为了解决以上问题，Rammer通过一个称为rTask的新抽象统一了算子间和算子内的调度。rTask使调度能够打破算子边界，并允许将计算细粒度调度到设备上。RAMMER是一种统一的纯软件解决方案，它不依赖于底层硬件，因此可以被不同的加速器采用，而不是将调度分成由软件和硬件分别管理的两部分。“
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723185317787.png)

重新定义一个问题和定位一个工作并不是在用不同的写法来写“茴”。之前我们只是在做一个广义上的kernel fusion，也没有设立起rTask和vEU的抽象。而在弄明白本质的问题在于原本系统中两层调度的gap以后，新的抽象很快帮助我们探明了更大的优化空间：

首先是将原本的通过cost model来选择子图进行fusion的问题，转变为了以更细粒度下的调度和资源分配问题。而得益于绝大部分情况下，神经网络计算的特征（DFG, 算子和张量）在compile time是已知的，我们因此可以将调度的开销移交给编译器，这既提升了搜索的效率也简化了系统设计。

更重要的是，**让inter operator与intra operator parallelism相互影响这个问题走进我们的视野**。举一个具体的例子，如果对于同一个算子有两种kernel实现，其中一个相较另一个消耗三倍的资源（CUDA Cores, Shared Memory, etc.），但是只取得两倍的加速，这在并行计算中是很常见的一个现象。而在此前单个算子独占整个硬件的情况下，毫无疑问我们会选择更快的实现。而我们的实验表明，在inter和intra operator两种parallelism协同调度的情况下，选择资源“性价比”最高的实现而非“最快”往往是更优的选择。这其实挑战了之前许多生成高性能算子的工作如AutoTVM[[9]](https://zhuanlan.zhihu.com/p/275837455#ref_9)等的一个基本假设，单个算子独占整个硬件表现出的计算性能，是否真的是性能调优的金标准？那么显然的，subgraph substitution (TASO[[10]](https://zhuanlan.zhihu.com/p/275837455#ref_10)) + high performance kernel (TVM)两个“optimal”相结合，并没有带来真的optimal。而我们基于新的抽象，只是浅尝一下简单的policy，就在一些场景下获得了超过现有SOTA的性能。


Rammer 的实现用五万两千行 C++ 代码，其中编译模块和[调度函数](https://zhida.zhihu.com/search?content_id=225031606&content_type=Article&match_order=1&q=%E8%B0%83%E5%BA%A6%E5%87%BD%E6%95%B0&zhida_source=entity)占据3k行代码，它的输入是 TensorFlow frozen graph、TorchScript 或 ONNX 格式的 DNN 模型，Rammer 首先将输入模型转换为 rOperators 的DFG，在 DFG 基础上进行常见的图优化：[常量折叠](https://zhida.zhihu.com/search?content_id=225031606&content_type=Article&match_order=1&q=%E5%B8%B8%E9%87%8F%E6%8A%98%E5%8F%A0&zhida_source=entity)、[公共子表达式消除](https://zhida.zhihu.com/search?content_id=225031606&content_type=Article&match_order=1&q=%E5%85%AC%E5%85%B1%E5%AD%90%E8%A1%A8%E8%BE%BE%E5%BC%8F%E6%B6%88%E9%99%A4&zhida_source=entity)等。对于来自优化后的DFG中的每个rOperator，Rammer从不同的源（例如自动内核生成器、手动调整内核，或从其他框架中的现有算子转换而来）加载一个或多个版本的 rKernel 实现。Rammer 编译器然后将DFG划分为子图（例如基于[算法1](https://zhida.zhihu.com/search?content_id=225031606&content_type=Article&match_order=1&q=%E7%AE%97%E6%B3%951&zhida_source=entity)中的策略），并将每个子图编译为 rProgram。作为输出，每个 rProgram 被进一步生成为在加速器上运行的设备代码（例如GPU内核）。

Rammer 不是动态调度每个rOperator，而是将DFG编译成一个称为 rProgram（由rTask组成）的静态执行计划，并通过一个名为vDevice的软件设备抽象将其映射到硬件。


## MICRO 20

### Optimizing the Memory Hierarchy by Compositing Automatic Transformations on Computations and Data

Polyhedral
[53年来国内唯三！清华校友获芯片顶会最佳论文提名，华为昇腾芯片性能显著提升 - 知乎](https://zhuanlan.zhihu.com/p/268210680)

在Polyhedral模型中研究如何实现一种更好的[循环分块](https://zhida.zhihu.com/search?content_id=147733024&content_type=Article&match_order=2&q=%E5%BE%AA%E7%8E%AF%E5%88%86%E5%9D%97&zhida_source=entity)和合并新组合，来优化存储进而提升程序在不同芯片上的执行效率。  
**该团队是国内少有的从事Polyhedral模型研究的团队之一。**

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723190335662.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723190349228.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723190404733.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723190416708.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723190430955.png)



## PLDI 21
AKG

## LISA 21
the talk

## OSDI 22

2022 年操作系统设计与实现研讨会（OSDI）有哪些值得关注的文章？ - Hsword的回答 - 知乎
https://www.zhihu.com/question/522950146/answer/2404943067

## MLSys22

Apollow; Mind IR

##  OSDI 23

2023 年操作系统设计与实现研讨会（OSDI）有哪些值得关注的文章？ - Hsword的回答 - 知乎
https://www.zhihu.com/question/591516372/answer/2953251460

Welder

## OSDI 24

USHER
MonoNN
Enabling Tensor Language Model
LADDER
MScalar
Chameleon API

## NIPS 24

ASPEN

## ASPLOS 24

On the fly micro kernel

## OSDI 25
### Nanoflow
NanoFlow: Towards Optimal Large Language Model Serving Throughput

这篇文章首先抛出了一个惊人的结论：
Through a detailed analysis, we show that despite having memory-intensive components, **end-to-end LLM serving is compute bound for most common workloads and LLMs**.

Alas, most existing serving engines fall short from optimal compute utilization, because the heterogeneous operations that comprise LLM serving--compute, memory, networking--**are executed sequentially within a device**.

We propose NanoFlow, a novel serving framework that exploits intra-device parallelism, which overlaps the usage of heterogeneous resources within a single device. NanoFlow splits inputs into smaller nano-batches and duplicates operations to operate on each portion independently, enabling overlapping. NanoFlow automatically identifies the number, size, ordering, and GPU resource allocation of nano-batches to minimize the execution time, while considering the interference of concurrent operations. We evaluate NanoFlow's end-to-end serving throughput on several popular models such as LLaMA-2-70B, Mixtral 8x7B, LLaMA-3-8B, etc. With practical workloads, NanoFlow provides 1.91x throughput boost compared to state-of-the-art serving systems achieving 50% to 72% of optimal throughput across popular models.

#### Introduction

LLM需要很多的资源。
LLMs are more resource-intensive than earlier DNNs due to unique characteristics.

在本文里，我们会详细研究为什么是compute bound的。Profile 理由有3：
In this paper, we show that while the self-attention opera tion is indeed memory-bound, **LLM serving is often compute bound when considered as a whole**. We provide a detailed analysis (which we also validate empirically) of the common operations used in current LLMs and find that: (1) **Batching prefill and decode requests amortizes weight loading, making general matrix multiplications (GEMMs) compute-bound.**(2) **Although batched decode request still need to load a unique KV-cache per sequence, novel optimizations such as grouped query attention (GQA) can reduce the memory loads.** (3) **As model sizes grow, the compute operations from GEMMs tend to dominate compared to self-attention.** Consequently, for many common workloads and LLMs, the total compute operations dominate network and memory I/O.

As model sizes grow, the compute operations from GEMMs tend to dominate compared to self-attention. 呃？self-attention 难道没有GEMMs吗？

However, in practice, we find that LLM serving engines are far from optimal in terms of compute utilization when measured end-to-end and compared to the hardware FLOPs. 为什么？This is because LLMs use heterogeneous operations.

So how many heterogeneous operations? Here it draws a figure:
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250723154208635.png)

To address this gap, we present NanoFlow, a serving frame work that aims to maximize utilization of the workload’s resource bottleneck, when considered as a whole. **The key insight is to leverage intra-device parallelism for heteroge neous operations**. In particular, **NanoFlow splits each input batch into nano-batches and duplicates operations across nano-operations,with each nano-operation processing a single nano-batch.**

Since nano-operations operate on separate nano batches without dependencies, heterogeneous operations— **such as memory-bound and compute-bound operations—can execute simultaneously rather than sequentially**. This ap proach facilitates fine-grained pipelining across compute, memory, and network resources within a single device. As nano-operations duplicate each operation, this method in creases memory I/O for weight loading. However, when the workload is compute-bound as a whole, the additional mem ory I/O can be hidden via pipelining.

这里的计算重叠是尝试将机器的计算、内存、网络三者都压榨到极致。

### Mirage


### Pipeline Threader
### QiMeng X Compiler
### Bayesian Code Diffusion
### Neutrim
### Principle and methodology for serial performnace optimization

