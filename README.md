
LLVM, TVM
[BBuf/tvm_mlir_learn: compiler learning resources collect.](https://github.com/BBuf/tvm_mlir_learn)

## Project

### TVM

TVM最早是GW那边老师做的一个系统。他们不把这个东西成为AI编译器，认为它包括了更多的东西。这个东西的输入是python，输出是machine code。在中间有优化。

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
**知乎原文：**

重新定义一个问题和定位一个工作并不是在用不同的写法来写“茴”。之前我们只是在做一个广义上的kernel fusion，也没有设立起rTask和vEU的抽象。**而在弄明白本质的问题在于原本系统中两层调度的gap以后，新的抽象很快帮助我们探明了更大的优化空间**：

首先是将原本的通过cost model来选择子图进行fusion的问题，转变为了以更细粒度下的调度和资源分配问题。而得益于绝大部分情况下，神经网络计算的特征（DFG, 算子和张量）在compile time是已知的，我们因此可以将调度的开销移交给编译器，这既提升了搜索的效率也简化了系统设计。

更重要的是，**让inter operator与intra operator parallelism相互影响这个问题走进我们的视野**。举一个具体的例子，如果对于同一个算子有两种kernel实现，其中一个相较另一个消耗三倍的资源（CUDA Cores, Shared Memory, etc.），但是只取得两倍的加速，这在并行计算中是很常见的一个现象。而在此前单个算子独占整个硬件的情况下，毫无疑问我们会选择更快的实现。而我们的实验表明，在inter和intra operator两种parallelism协同调度的情况下，选择资源“性价比”最高的实现而非“最快”往往是更优的选择。这其实挑战了之前许多生成高性能算子的工作如AutoTVM[[9]](https://zhuanlan.zhihu.com/p/275837455#ref_9)等的一个基本假设，单个算子独占整个硬件表现出的计算性能，是否真的是性能调优的金标准？那么显然的，subgraph substitution (TASO[[10]](https://zhuanlan.zhihu.com/p/275837455#ref_10)) + high performance kernel (TVM)两个“optimal”相结合，并没有带来真的optimal。而我们基于新的抽象，只是浅尝一下简单的policy，就在一些场景下获得了超过现有SOTA的性能。


Rammer 的实现用五万两千行 C++ 代码，其中编译模块和[调度函数](https://zhida.zhihu.com/search?content_id=225031606&content_type=Article&match_order=1&q=%E8%B0%83%E5%BA%A6%E5%87%BD%E6%95%B0&zhida_source=entity)占据3k行代码，它的输入是 TensorFlow frozen graph、TorchScript 或 ONNX 格式的 DNN 模型，Rammer 首先将输入模型转换为 rOperators 的DFG，在 DFG 基础上进行常见的图优化：[常量折叠](https://zhida.zhihu.com/search?content_id=225031606&content_type=Article&match_order=1&q=%E5%B8%B8%E9%87%8F%E6%8A%98%E5%8F%A0&zhida_source=entity)、[公共子表达式消除](https://zhida.zhihu.com/search?content_id=225031606&content_type=Article&match_order=1&q=%E5%85%AC%E5%85%B1%E5%AD%90%E8%A1%A8%E8%BE%BE%E5%BC%8F%E6%B6%88%E9%99%A4&zhida_source=entity)等。对于来自优化后的DFG中的每个rOperator，Rammer从不同的源（例如自动内核生成器、手动调整内核，或从其他框架中的现有算子转换而来）加载一个或多个版本的 rKernel 实现。Rammer 编译器然后将DFG划分为子图（例如基于[算法1](https://zhida.zhihu.com/search?content_id=225031606&content_type=Article&match_order=1&q=%E7%AE%97%E6%B3%951&zhida_source=entity)中的策略），并将每个子图编译为 rProgram。作为输出，每个 rProgram 被进一步生成为在加速器上运行的设备代码（例如GPU内核）。

Rammer 不是动态调度每个rOperator，而是将DFG编译成一个称为 rProgram（由rTask组成）的静态执行计划，并通过一个名为vDevice的软件设备抽象将其映射到硬件。


### Ansor: Generating High-Performance Tensor Programs for Deep Learning

[tvm_mlir_learn/paper_reading/Ansor Generating High-Performance Tensor Programs for Deep Learning.md at main · BBuf/tvm_mlir_learn](https://github.com/BBuf/tvm_mlir_learn/blob/main/paper_reading/Ansor%20Generating%20High-Performance%20Tensor%20Programs%20for%20Deep%20Learning.md)

高性能的张量化程序对于保证深度神经网络的高效执行是至关重要的。然而，在各种硬件平台上为不同的算子都获得高效的张量化程序是一件充满挑战的事。目前深度学习系统依赖硬件厂商提供的内核库或者各种搜索策略来获得高性能的张量化程序。这些方法要么需要大量的工程工作来开发特定于平台的优化代码，要么由于搜索空间受限和无效的探索策略而无法找到高性能的程序。

我们提出了Ansor，一个用于深度学习应用的张量化程序生成框架。与现有的搜索策略相比，**Ansor通过从搜索空间的分层表示中采样程序来探索更多的优化组合**。然后Ansor使用进化搜索和一个可学习的代价模型对采样程序进行微调，以确定最佳程序。Ansor可以找到现有SOTA方法的搜索空间之外的高性能程序。此外，Ansor利用scheduler同时优化深度神经网络中的多个子图。实验表明，相对于Intel CPU，ARM CPU，NVIDIA GPU，Ansor分别将神经网络的执行性能提高了3.8倍，2.6倍，和1.7倍。

**Ansor是一个自动的张量化程序生成框架。** Figure 4展示了Ansor的整体架构。Ansor的输入是一组待优化的DNN。Ansor使用Relay[42]的算符融合算法将DNN从流行的模型格式（例如ONNX，TensorFlow PB）转换为小的子图，然后Ansor为这些子图产生张量化程序。Ansor具有三个重要的组件：（1）一个程序采样器，它构建一个大的搜索空间并从中采样不同的程序。（2）微调采样程序性能的性能调试器。（3）一个任务调度器，用于分配时间资源以优化 DNN 中的多个子图。

**Program sampler.** Ansor 必须解决的一个关键挑战是为给定的计算图生成一个大的搜索空间。为了覆盖具有各种high-level结构和low-level细节的各种张量化程序，Ansor利用具有两个级别的搜索空间的分层表示：草图和注释（第 4 节）。 Ansor 将程序的high-level结构定义为草图，并将数十亿个low-level选择（例如，tile size、parallel、unroll annotations）作为注释。 这种表示允许Ansor灵活地枚举high-level结构并有效地采样low-level细节。Ansor包含一个程序采样器，它从空间中随机采样程序以提供对搜索空间的全面覆盖。

**Performance tuner.** 随机采样程序的性能不一定好。下一个挑战是对它进行微调、Ansor使用进化搜索和一个可学习的代价模型迭代式微调（第5节）。在每次迭代中，Ansor使用重新采样的新程序以及来自先前迭代的性能还不错的程序作为初始种群执行进化搜索。进化搜索通过变异和交叉对程序进行微调，执行乱序重写并解决顺序构造的局限性。 查询学习到的代价模型比实际测试快几个数量级，因此我们可以在几秒钟内评估数千个程序。

**Task scheduler.** 使用程序采样器和性能调试器允许 Ansor 为计算图找到高性能的张量化程序。 直观地说，将整个DNN视为单个计算图并为其生成完整的张量化程序可能会实现最佳性能。 然而，这是低效的，因为它必须处理搜索空间不必要的指数爆炸。 通常，编译器将DNN的大型计算图划分为几个小的子图 [11, 42]。由于 DNN 的逐层构造特性，该划分对性能的影响可以忽略不计。这带来了Ansor最终的挑战：在为多个子图生成程序时如何分配时间资源。 Ansor中的任务调度器（第 6 节）使用基于梯度下降的调度算法将资源分配给更可能提高端到端DNN性能的子图。


Ansor 的主要贡献是做到了自动寻找高效的Schedule（循环展开、合并、分块、缓存使用、改变并行度等等），**不再需要开发者在[TVM](https://zhida.zhihu.com/search?content_id=209483109&content_type=Article&match_order=1&q=TVM&zhida_source=entity)中基于Tensor Expression手写Schedule模板，大大增强了算子编译器（Tensor Compiler）的易用性并且对一些典型的算子和模型效果也很不错，算是AutoTVM的升级版（因为AutoTVM还需要手动指定需要搜索的Schedule模板**

这里可以看到，Ansor是在尝试不同的优化组合，它是个程序生成框架。然后分成了好几个section。这和之前的TVM有点类似？但是它靠划分，来加快生成速度。如果RAMMER说的是对的话，那么这样其实是不好的。

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724111503338.png)



## MICRO 20

### Optimizing the Memory Hierarchy by Compositing Automatic Transformations on Computations and Data

这篇文章主要还是在用多面体，然后利用了memory hierarchy。进一步利用芯片上的高速缓存，从而提升程序在CPU、GPU以及**AI芯片华为昇腾910等不同芯片上的性能**。

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

### AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations

把TVM的那套迁移到NPU上。

AKG

开发一个面向NPU架构的编译器仍然面临着诸多挑战，因为NPU上通常需要更复杂融合策略设计，以期充分利用NPU上的快速存储部件。编译器需要解决不同程序对并行性和局部性的需求不同的挑战，并找到最优的schedule结果。另一方面，领域特定芯片的内存层级都是多层次、多方向设计的，编译器需要解决如何在软件层面上实现自动内存管理的挑战。解决面向领域特定硬件，自动生成运算密集型算子的挑战，例如卷积，矩阵乘等。

[PLDI 2021: AKG: Automatic Kernel Generation for Neural Processing Units using Polyhedral Transformations](https://www.pldi21.org/poster_pldi.760.html)

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724110827069.png)


## LISA 21
the talk

更多的CPU Core 的用处没有那么多了。从学术上讲通信成本，锁，复杂度都很难受，商业上GPU薄纱它

[Computing Performance: On the Horizon | USENIX](https://www.usenix.org/conference/lisa21/presentation/gregg-computing)
![02c5d1bc0bc068dd957f79a8dd682658.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724111035742.png)

![330b1bd1ed36bdc73cc3d05abf856d6e.png](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724111024201.png)

## OSDI 21

### PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections

[tvm_mlir_learn/paper_reading/PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections.md at main · BBuf/tvm_mlir_learn](https://github.com/BBuf/tvm_mlir_learn/blob/main/paper_reading/PET%3A%20Optimizing%20Tensor%20Programs%20with%20Partially%20Equivalent%20Transformations%20and%20Automated%20Corrections.md)

标题可以翻译为：**基于部分等价变换和自动校正来优化张量化程序**。作者团队来自清华，CMU和FaceBook等。这篇论文的一作王豪杰来自清华大学。后面会介绍到这篇论文在生成突变程序集合时，要维护K个效率最高的突变程序时，使用了ASO中的代价模型和评估方式，所以作者有贾志豪大神也不奇怪。

现有的框架在图层做优化一般都是基于等价变换，也就时说变换前后的程序是完全等价的。这里等价的意思是给定相同的输入，那么变换前后的程序一定可以得到相同的输出。而这篇论文挖了一个新坑，**即做了一个新的框架PET，在优化过程中允许出现部分等价的变换**，并且设计了一个高效的搜索算法去组合完全等价以及部分等价的变换以探索更大的搜索空间。并且最终结果也比较好。

PET不关心算子的Schedule，而是从部分等价变换的新角度出发去增加并行度或者改善缓存从而达到加速效果

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724112025354.png)


## OSDI 22

- [NNFusion](https://github.com/microsoft/nnfusion): Rammer (OSDI'20), Roller (OSDI'22), Welder (OSDI'23), Cocktailer (OSDI'23), T10 (SOSP'24)

2022 年操作系统设计与实现研讨会（OSDI）有哪些值得关注的文章？ - Hsword的回答 - 知乎
https://www.zhihu.com/question/522950146/answer/2404943067




### ROLLER: Fast and Efficient Tensor Compilation for Deep Learning

作者团队在TVM上实现并开源了roller

- 论文链接：[https://www.usenix.org/conference/osdi22/presentation/zhu](https://link.zhihu.com/?target=https%3A//www.usenix.org/conference/osdi22/presentation/zhu)
- 代码链接：[https://github.com/microsoft/nn](https://link.zhihu.com/?target=https%3A//github.com/microsoft/nnfusion/)

无论是Ansor，AutoTVM还是PET（一部分代码生成也是基于TVM AutoTVM/Ansor的）它们都面临了同样一个问题，那就是在对算子的Schedule进行搜索时需要耗费大量的时间，在特定硬件上对一个常见的视觉模型进行自动调优和生成代码kennel需要数小时。这严重阻碍了AI编译器应用于模型部署。基于这个痛点，Roller横空出世。

现代的张量编译器虽然取得了很多的进展，但通常这些编译器都需要小时计的时间去搜索和生成高效的Kernel，这是因为现有张量编译器通常指定的搜索空间很大。为了解决编译时间长的问题，本文提出了Roller，它的核心是rTile，这是一种新的Tile抽象，它封装了和底层加速器的关键特性一致的张量shape，从而通过限制shape的选择实现高效的运行。**Roller采用了基于rTile的递归构建算法来生成目标程序（rProgram）。最终，Roller可以在几秒内就生产高效的Kernel**，性能可以媲美目前主流加速器上的其它张量编译器，并且为IPU等新的加速器生产更好的Kernel。

## SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute

针对稀疏的神经网络，生成特定的优化算子。

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724112724077.png)


- 本篇文章主要探索深度神经网络的**稀疏性**，借助名为TeSA的新抽象为模型提供端到端稀疏性模式，并根据不同的稀疏性模式，产生高效的专用的operator，从而为稀疏的DNN模型提供更小的内存占用和更低的推理延迟。



### Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences

GPU调度的一篇文章

[hanosdi22.pdf](https://ipads.se.sjtu.edu.cn/_media/publications/hanosdi22.pdf)


### Alpa: Automating Inter- and Intra-Operator Parallelism for Distributed Deep Learning

生成LLM并行计划的一篇文章

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113043827.png)


### Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization

CMU的嘉志豪作品

试图进行联合优化

Unity 在 FlexFlow、TASO 和 MetaFlow 的基础上，提出在并行计算图（PCG）中代数变换和并行化的统一表示（OP，Operator）和共优化（图替代，Substitution）方法，可以同时考虑分布式训练中的计算、并行和通信过程。对于共优化，Unity 使用一个多级搜索算法来高效搜索性能最好的图替代组合以及相应的硬件放置策略。

Unity 基于先前工作，定义了 DNN 并行系统中常见的六类基本形式：

数据并行（Data Parallelism）：最基本的并行形式；
模型并行（Model Parallelism）：将 DNN 模型拆分为多个子模型，分别在特定的设备训练；
空间并行（Spatial Parallelism）：对 tensor 的空间维度（如图像的高/宽）进行划分；
规约并行（Reduction Parallelism）：利用 OP 的线性。将 tensor 划分为 n 份到 n 个设备上，分别进行 OP 操作。注意每个结果 tensor 的 shape 都和最终 tensor 相同，因此只需相加进行规约；
管道并行（Pipeline Parallelism）：不同训练代数之间的并行；
指定 OP 的并行（Operator-specific Parallelsim）：对 batch 内不同的输入 sample 有不同的 weight，不需要 tensor 备份或同步；
许多并行策略是不同 cost 指标的 trade-off，因此需要不同 OP 用不同的并行方式，以达到最优性能。DNN 结构表示的传统方法是计算图，节点是 OP，边是 tensor，代数变换被表示为迭代的图替代，通过给每个节点分配并行注释来并行化模型。然而，该方法存在如下局限：

**对代数变换和并行化使用独立表示阻碍了共优化。代数变换会增删节点，但并行化需要静态计算图；**
**计算图并没有显式考虑并行带来的通信开销。这导致代数变换难以预测最终模型的性能**

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113239546.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113303099.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113319680.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113359004.png)

![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113415932.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113430925.png)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113505064.png)

[Microsoft PowerPoint - OSDI 22 - Animations Expanded.pptx](https://www.usenix.org/system/files/osdi22_slides_unger.pdf)
![](https://blog-1327458544.cos.ap-guangzhou.myqcloud.com/N2025/20250724113540522.png)

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

