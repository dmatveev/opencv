# Implementing a face beautification algorithm on G-API {#tutorial_gapi_face_beautification}

@prev_tutorial{tutorial_gapi_anisotropic_segmentation}

[TOC]

# Introduction {#gapi_fb_intro}

In this tutorial you are going to learn:
* How to use Inference-API to infer networks inside a graph;
* How to create a custom kernel implementation (using OpenCV backend);
* How to process a videostream inside a graph using Streaming-API.

You can find source code in the `modules/gapi/samples/face_beautification.cpp`
of the OpenCV source code library.

# Face Beautification algorithm {#gapi_fb_algorithm}

An algorithm of a face retouching has been chosen as a subject for this
tutorial to discover G-API usage convenience and efficiency in the context of
solving a real CV problem. The main idea is to implement a real-time videostream
processing that detects faces and applies some filters to make them look
beautiful (more or less). The pipeline is the following:

![The Face Beautification algorithm](pics/algo.png)

The algorithm consists of two parts: an inference of two networks (including
data pre- and post-processing) and an image filtering pipeline that uses the
inference data (creating masks, applying filters and composing the output
image). 

## Custom kernels implementation {#gapi_fb_custom_kernels}

A new kernel implementation has been already described clearly in OpenCV G-API
documentation (see [G-API Kernel API](@ref gapi_kernel_api)). Bare steps without
a lot of details will be mentioned here. In simple words, to implement a brand
new G-API kernel we have to take 3 main steps:

### 1) New kernel interface definition {#gapi_fb_c_k_1}

There are a couple of macros to define a new kernel. The first is
G_TYPED_KERNEL(), which declares a new operation properties:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_decl

To get the full description, see [Defining a kernel](@ref gapi_defining_kernel).

A little but significant adjustment has to be made when a kernel with a multiple
return value is going to be defined. For this case, we should use another
macro, G_TYPED_KERNEL_M():

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_m_decl

As we can see, almost everything is unchanged (speaking about the highest
level, of course). For a note: the sign to use the modified macro is the
`std::tuple` appearance as a kernel signature output.

The outMeta() function of such a kernel must return a tuple too.

### 2) Specific backend kernel implementation. {#gapi_fb_c_k_2}

The second step begins with a macro too. If we want to implement a CPU kernel
version using OpenCV functions, we should apply GAPI_OCV_KERNEL() macro, which
defines what the operation should do:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_impl

For more details (including arguments of `run()` function and their order),
visit [Implementing a kernel](@ref gapi_kernel_implementing).

A note about multiple returning: all the output arguments of an "::on()" method
are just put after input ones keeping order:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_m_impl

### 3) Custom kernels declaration for a graph {#gapi_fb_c_k_3}

To allow a specific pipeline to use our custom kernels, we have to pass them
to a graph by compile arguments:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp kern_pass_1
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp apply

Congratulations! Now we are able to use your new kernel in the pipeline. E.g.:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_usg

## Networks inference using Inference-API {#gapi_fb_inference}

Two models from OMZ have been used in this sample: the face detector
(https://github.com/opencv/open_model_zoo/tree/master/models/intel/face-detection-adas-0001)
and the facial landmarks detector
(https://github.com/opencv/open_model_zoo/blob/master/models/intel/facial-landmarks-35-adas-0002/description/facial-landmarks-35-adas-0002.md).

Let's prepare our chosen networks for being processed by G-API's Inference-API.
Providing a new network support is quite similar to a new kernel implementation.
There are three main steps:

### 1) Network declaration {#gapi_fb_i_1}

To declare a network we should use the special macro G_API_NET:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp net_decl

Similar to a kernel macro, it takes three arguments to register a new type. They
are:
1. Network interface name -- also serves as a name of new type defined
   with this macro;
2. Network signature -- an `std::function<>`-like signature which defines
   inputs and outputs;
3. Network's unique name -- used to identify it within the system when the type
   is stripped.

### 2) Network parametrization {#gapi_fb_i_2}

To set network's structure, weights and a backend we have to
generate `cv::gapi::ie::Params<>` object and then pass it to a graph by the
compile args we've already mentioned:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp net_prop
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp apply

### 3) Network inference inside a graph {#gapi_fb_i_3}

We are able to use the network in our pipline now! To do that, we should call
`cv::gapi::infer<>()` specifying the network we want to infer:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp net_usg

## Videosource processing via Streaming-API {#gapi_fb_streaming}

There are no interest in using a lot of computer vision pipelines to process
just a single picture -- a big group of them are created to be launched with a
videosource as an input. Therefore, it is essential for the G-API as an
interface for pipelines implementation to provide a possibility of a stream
processing without any external instruments to be applied by user.



