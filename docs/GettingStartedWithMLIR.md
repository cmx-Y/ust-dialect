[English](#en) | [中文](#cn)

<span id="en">MLIR Getting Started Tutorial</span>
===========================
The most important and comprehensive resource for getting started with MLIR is the official documentation: https://mlir.llvm.org/.

However, relying solely on the official documentation may not be sufficient and could lead to confusion. I recommend diving directly into coding by following the official toy tutorial: https://mlir.llvm.org/docs/Tutorials/Toy/.

Based on my personal experience, getting the toy tutorial up and running isn't necessarily straightforward. Due to the comprehensive nature of the official documentation, completing it might still leave you perplexed about MLIR without a clear conceptual understanding. Therefore, it might be beneficial to try following the tutorials in this repository. They take you step-by-step, which I believe will lead to significant insights.

Firstly, in simple terms, the most crucial concept in MLIR is the dialect. Concepts like passes and code generation (codegen) are all based on dialects. A dialect can be understood as an intermediate representation (IR) that contains various Operations. An Operation can be thought of as a node in a graph, where its inputs and outputs are edges in the graph, known as Values in MLIR. An MLIR program consists of multiple Operations and their corresponding Values, forming a graph. A Pass transforms this graph, while Codegen traverses this graph to produce backend code.

The official documentation provides several dialects, as detailed here: https://mlir.llvm.org/docs/Dialects/.

This tutorial first describes an MLIR program using the dialects provided by the official sources. It then completes backend Codegen and specific passes based on this program. By accomplishing these two tasks, I believe you'll gain a deeper understanding of IR Structure and common MLIR APIs. Finally, it uses the ODS framework to customize dialects and advance further work.

For Codegen-related tutorials, please refer to: docs/AddYourCodegen.md.

For Pass-related tutorials, please refer to: docs/AddNewPass.md.

<span id="cn">MLIR入门指北</span>
===========================
入门 MLIR 最重要最全的资料就是官方文档：https://mlir.llvm.org/。

但是光看官方文档肯定是不行的，很容易玉玉，推荐直接上手代码，可以跟着做官网的 toy tutorial：https://mlir.llvm.org/docs/Tutorials/Toy/。

根据我的个人经验，toy tutorial 跑起来貌似也没有那么简单，并且由于官方文档需要考虑全面性，跑完可能还是一头雾水，对 MLIR 没有一个比较直观的概念，所以不妨试试跟着本仓库的教程跑一跑，由点到面，相信会有比较大的收获。

首先简单讲一下个人理解，MLIR 中最重要的概念就是 dialect，pass、codegen 等等都是基于 dialect 展开的。一个 dialect 可以理解为一个 IR，里面包含各种 Operation。一个 Operation 可以理解为一个图节点，它的输入和输出是图的边，在 MLIR 中叫做 Value。一个 MLIR 程序就是由多个 Operation 和相应的 Value 组成的图。Pass 就是对这个图进行变换，Codegen 就是遍历这个图输出对应的后端代码。

官方提供了多个 dialect，详见 https://mlir.llvm.org/docs/Dialects/。

本教程首先用官方提供的 dialect 描述一个 mlir 程序，然后根据这个程序完成后端 Codegen 以及特殊的 pass，完成这两个任务后，相信对于 IR Structure 以及 mlir 常用 API 会有比较深刻的认识，最终再用 ODS 框架自定义 dialect，并基于此开展后续工作。

Codegen 相关教程详见：docs/AddYourCodegen.md。

Pass 相关教程详见：docs/AddNewPass.md。