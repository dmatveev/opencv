# Implementing a face beautification algorithm on G-API {#tutorial_gapi_face_beautification}

@prev_tutorial{tutorial_gapi_anisotropic_segmentation}

[TOC]

# Introduction {#gapi_fb_intro}

In this tutorial you are going to learn:
* How to use the Inference-API to infer networks inside a graph;
* How to create a custom kernel implementation (using OpenCV backend);
* How to process a videostream inside a graph using the Streaming-API.

You can find source code in the `modules/gapi/samples/face_beautification.cpp`
of the OpenCV source code library.

# Face Beautification algorithm {#gapi_fb_algorithm}

An algorithm of a face retouching has been chosen as a subject of this tutorial.
The main idea is to implement a real-time videostream processing that detects
faces and applies some filters to make them look beautiful (more or less). The
pipeline is the following:

![The Face Beautification algorithm](pics/algo.png)

The algorithm consists of two parts: an inference of two networks (including
data pre- and post-processing) and an image filtering pipeline that uses the
inference data (creating masks, applying filters and compositing the output
image). 

## Custom kernels implementation {#gapi_fb_custom_kernels}

A new kernel implementation has been already described clearly in OpenCV G-API
documentation (see [G-API Kernel API](@ref gapi_kernel_api)). Bare steps without
a lot of details will be mentioned here.

### 1) A new kernel interface definition{#gapi_fb_c_k_1}

There are a couple of macros to define a new kernel. The first is
G_TYPED_KERNEL():

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_decl

To get the full description, see [Defining a kernel](@ref gapi_defining_kernel).

A little but significant adjustment has to be made when a kernel with a multiple
return value is going to be defined. For this case, you should use
G_TYPED_KERNEL_M():

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_m_decl

As we can see, almost everything is unchanged (speaking about the highest
level, of course). The sign to use the modified macro is the `std::tuple`
appearance as a kernel signature output.

Note that outMeta() function should return a tuple too.

### 2) A specific backend kernel implementation.{#gapi_fb_c_k_2}

The second step also begins with a macro. If we want to implement a CPU kernel
version using OpenCV functions, we should apply GAPI_OCV_KERNEL():

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_impl

For more details, visit [Implementing a kernel](@ref gapi_kernel_implementing).

A note about a multiple return: all the output arguments of an "::on()" method
are just put after input ones.

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_m_impl

### 3) A custom kernels declaration for a graph{#gapi_fb_c_k_3}

To allow a specific pipeline to use our custom kernels, we have to pass them
to a graph by compile arguments:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp kern_pass_1
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp kern_pass_2

## Networks inference using the Inference-API {#gapi_fb_inference}

Two models from OMZ were used in this sample: the face detector
(https://github.com/opencv/open_model_zoo/tree/master/models/intel/face-detection-adas-0001)
and the facial landmarks detector
(https://github.com/opencv/open_model_zoo/blob/master/models/intel/facial-landmarks-35-adas-0002/description/facial-landmarks-35-adas-0002.md).

Let's prepare our chosen networks to be processed by G-API's Inference-API.
A new network support is quite similar to a new kernel implementation.
There are three main steps:

### 1) A network declaration

To declare a network we should use the special macro G_API_NET:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp net_decl
