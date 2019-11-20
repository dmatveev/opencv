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

# New G-API features description {#gapi_fb_features}

Before we can discuss an implementation of any algorithm, we should
sort out new G-API features applied there. Obviously, paragraphs below are not
detailed documental articles, they are just brief guides how to use one feature
or another with examples of usage.

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

### 1) Network definition {#gapi_fb_i_1}

To define a network we should use the special macro G_API_NET:

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

To set network's parameters we have to generate `cv::gapi::ie::Params<>` object
and then give it to a graph by the compile args we've already mentioned:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp net_prop
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp apply

Every `cv::gapi::ie::Params<>` object is related to a network specified by its
template argument. We should pass there the network type we have described in
the previous step.
These objects are able to contain the paths to structure and weights files and
a desirable backend which can be one of backends supported by InferenceEngine
("CPU", "GPU", etc.).

### 3) Network inference inside a graph {#gapi_fb_i_3}

We are able to use the network in our pipline now! To do that, we should call
`cv::gapi::infer<>()` setting the network type as a template argument to
specify the network we want to infer:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp net_usg

## Videosource processing via Streaming-API {#gapi_fb_streaming}

There are no interest in using the most of computer vision pipelines to process
just a single picture -- a big group of them are created to be launched with a
videosource as an input. It is therefore essential for the G-API as an
interface for pipelines implementation to provide a possibility of a stream
processing without any external instruments to be applied by user.
So, that's why Streaming-API is needed.

The usage of this G-API feature makes codewriting easier, or, at least, more
independent. As usual in this tutorial, let's deal with it by several steps:

### 1) Pipeline graph definition {#gapi_fb_s_1}

To begin with, a `cv::GComputation` object (which contains all the
pipeline) used to be defined statically by graph's input and output described in
a code (almost all the graph's body is skipped in this code snippet except the
input and output nodes):

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp comp_old_1
...
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp comp_old_2
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp comp_old_3
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp comp_old_4

To apply Streaming-API to our problem we should use a different version of a
pipeline expression  with a lambda-based constructor to keep all temporary data
objects in a dedicated scope:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp comp_str_1
...
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp comp_str_2
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp comp_str_3
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp comp_str_4

### 2) Frames processing {#gapi_fb_s_2}

The next step used to be a videostream definition and a source setup:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp header
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp in_out_1

and then a compiled graph application to every single frame. This is an example
of a processing loop without Streaming:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp in_out_2

Now this can be done without explicit use of additional libraries. First of all,
define a stream variable and pass to it:
    - video properties (this unconvenience is temporary);
    - graph compile arguments (which will be passed to the graph inside
the stream):

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp str_comp

Then we should set a source:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp str_src

And for this moment our stream is ready to be launched! An example of the
Streaming loop:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp str_loop

There are some other useful possibilities provided, e.g. you can abort stream
processing via `cv::GStreamingCompiled::stop()` method (the next call of
`cv::GStreamingCompiled::start()` will start processing the stream from the very
beginning).

### 3) ??? {#gapi_fb_s_3}

### 4) You are breathtaking! {#gapi_fb_s_4}

Now, when all the new G-API features are described and understood, we can
discuss some algorithms.

# Face Beautification algorithm {#gapi_fb_algorithm}

The algorithm of a face retouching has been chosen as a subject for this
tutorial to discover G-API usage convenience and efficiency in the context of
solving a real CV problem. Its main idea is to implement a real-time videostream
processing that detects faces and applies some filters to make them look
beautiful (more or less). The pipeline is the following:

![The Face Beautification algorithm](pics/algo.png)

The algorithm consists of two parts: an inference of two networks (including
data pre- and post-processing) and an image filtering pipeline which uses the
inference data (creating masks, applying filters and composing the output
image).

The details of obvious operations (like command line parsing or .. ) are not
going to be described in this text; there will be only comments about debatable
features implementation.

Let's start at the begining of a pipline graph description.

## Face detector inference and post-processing {#gapi_fb_face_detect}

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp fd_inf

- After the inference, a post-processing of an output `cv::Mat` is required: the
face detector outputs a blob with the shape [1, 1, N, 7], where N is
the number of boxes bounding detected faces. Structure of an output for every
face is the following: [image_id, label, conf, x_min, y_min, x_max, y_max];
all the seven elements are floating point. For our further goals we need to take
only results with `image_id > 0` as a negative value in this field indicates the
end of detections; also we can cut detections by the `conf` field to avoid
mistakes; so the Mat parsing looks like this code:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp fd_pp

- There is `cv::Mat` data parsing by pointer on float used. 

- Float numbers we've got from the Mat are between 0 and 1 and denote normalized
coordinates; to get the real pixel coordinates we should multiply it with the
image sizes respectively to the directions. To be fully accurate, the received
numbers should be then rounded in the right way and casted to `int` to construct
integer Points; these operations have been wrapped by `toIntRounded()`:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp toInt

- Rectangles was constructed by two Points - the top-left and the bottom-right
corners.

- By far the most important thing here is solving an issue that sometimes
detector spits out coordinates out of image; if we pass such an ROI
to be processed, errors of landmarks detector will occure. To handle these cases
the frame's borders rectangle `borders` is created and then intersected with
the face rect (by `operator&()`) so the ROI saved in `outFaces` are for sure
inside the frame.

## Facial landmarks detector inference and post-processing {#gapi_fb_landm_detect}

snippet

for arrays of points which are naturally contours - Contour type:

snippet

the array with all landmarks is array of points; syntactically it is Contour
 though semantically - not a contour (ordered set) so I put another type
Landmarks which syntactically is equal to Contour type:

snippet




## Possible further improvements {#gapi_fb_to_improve}

There are some points in the algorithm to be improved.

### Correct ROI reshaping for meeting conditions required by fac landm detector {#gapi_fb_padding}

The input of f.l.d is square ROI, but f.d. gives non-square rectangles in general.
If let backend within Inf-API reshape rect to squre - lack of inf accuracy.
The solution: take not rect but a square describing it - improvement. Problem conducted -
some describing squares can go out of frame -> errors of landmarks detector.
Solution - count all squares needed; count the farthest outer coordinates; to pad
the source image by borders with counted size; take ROIs for f.l.d from padded img.
