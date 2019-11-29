# Implementing a face beautification algorithm on G-API {#tutorial_gapi_face_beautification}

@prev_tutorial{tutorial_gapi_anisotropic_segmentation}

[TOC]

# Introduction {#gapi_fb_intro}

In this tutorial you will learn:
* How to implement a custom kernel (using OpenCV backend);
* How to use Inference-API to infer networks inside the graph;
* How to process a videostream inside the graph using Streaming-API;
* The main ideas of the Face Beautification algorithm.

You can find source code in the `modules/gapi/samples/face_beautification.cpp`
of the OpenCV source code library.

# New G-API features description {#gapi_fb_features}

Before we can discuss an implementation of any algorithm, we should
sort out new G-API features applied there. Obviously, the paragraphs below are
not detailed documentary articles, they are just brief guides how to use one
feature or another with real, living examples given.

## Custom kernels implementation {#gapi_fb_custom_kernels}

A new kernel implementation has been already described clearly in OpenCV G-API
documentation (see [G-API Kernel API](@ref gapi_kernel_api)). Bare steps will
be mentioned here with few details. In simple words, to implement a brand
new G-API kernel we have to take 3 main steps:

### 1) New kernel interface definition {#gapi_fb_c_k_1}

There is a couple of macros to define a new kernel. The first one is
`G_TYPED_KERNEL()`, which declares a new operation properties:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_decl

To get the full description, see [Defining a kernel](@ref gapi_defining_kernel).

A small yet significant adjustment has to be made when a kernel with a multiple
return value is defined. For this case, we must use another macro,
`G_TYPED_KERNEL_M()`:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_m_decl

As you can see, almost everything is unchanged (speaking of the initial
look, of course). Note: the `std::tuple` appearance in a kernel signature could
be the sign to change the macro.

The `outMeta()` function for such kernels must return a tuple too.

### 2) Kernel implementationfor a specific backend. {#gapi_fb_c_k_2}

Similarly, the second step begins with a macro. If we want to implement a CPU
kernel version using OpenCV functions, we must use the `GAPI_OCV_KERNEL()`
macro, which defines what the operation should do:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_impl

For more details (including arguments of the function `run()` and their order),
visit [Implementing a kernel](@ref gapi_kernel_implementing).

Note: all the output arguments of a kernel signature
are just put after input ones keeping the order if they are multiple:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_m_impl

### 3) Custom kernels declaration of a graph {#gapi_fb_c_k_3}

To allow a certain pipeline to use our custom kernels, we have to pass them
to a graph by compile arguments:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp kern_pass_1
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp apply

Congratulations! Now we are able to use our new kernel in the pipeline. E.g.:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp kern_usg

## Networks inference using Inference-API {#gapi_fb_inference}

Two models from OMZ have been used in this sample: the
<a href="https://github.com/opencv/open_model_zoo/tree/master/models/intel
/face-detection-adas-0001">face detector</a>
and the
<a href="https://github.com/opencv/open_model_zoo/blob/master/models/intel
/facial-landmarks-35-adas-0002/description/facial-landmarks-35-adas-0002.md">
facial landmarks detector</a>.

Let's prepare our chosen networks for being processed by G-API's Inference-API.
Providing a new network support is quite similar to a new kernel implementation.
There are three main steps:

### 1) Network definition {#gapi_fb_i_1}

To define a network we should use the specific macro G_API_NET:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp net_decl

Similarly to a kernel macro, it takes three arguments to register a new type.
They are:
1. Network interface name --- also serves as a name of a new type defined
   with this macro;
2. Network signature      --- an `std::function<>`-like signature which defines
   inputs and outputs;
3. Network's unique name  --- used to identify it within the system when the
   type is stripped.

### 2) Network parametrization {#gapi_fb_i_2}

To set parameters of the network we have to generate a `cv::gapi::ie::Params<>`
object and then give it to the graph by the compile args we've already
mentioned:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp net_prop
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp apply

Every `cv::gapi::ie::Params<>` object is related to the network specified by its
template argument. We should pass there the network type we have described in
the previous step.
Those objects are able to contain paths to structure and weights files and
a desirable backend which can be one of the backends supported by
InferenceEngine ("CPU", "GPU", etc.).

### 3) Network inference inside a graph {#gapi_fb_i_3}

New we are able to use the network in our pipeline! To do that, we should call
`cv::gapi::infer<>()` setting the network type as a template argument to
specify the network we want to infer:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp net_usg

## Videosource processing via Streaming-API {#gapi_fb_streaming}

There is no interest in using the most of computer vision pipelines to process
just a single picture -- a big group of them are created to be launched with a
videosource as an input. Therefore, it is essential for G-API as an
interface for pipelines implementation to provide a possibility of a stream
processing without any external instruments to be applied by user.
So, that's why Streaming-API is needed.

The usage of this G-API feature makes codewriting easier, or, at least, more
independent. As usual in this tutorial, let's deal with it by several steps:

### 1) Pipeline graph definition {#gapi_fb_s_1}

To begin with, a `cv::GComputation` object (which contains the whole
pipeline) used to be defined statically by input of the graph and output
described in a code (almost all the graph's body is skipped in this code snippet
except for input and output nodes):

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp comp_old_1
...
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp comp_old_2
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp comp_old_3
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp comp_old_4

To apply Streaming-API we should use a different version of a
pipeline expression  with a lambda-based constructor to keep all the objects
with temporary data in a dedicated scope:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp comp_str_1
...
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp comp_str_2
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp comp_str_3
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp comp_str_4

### 2) Frames processing {#gapi_fb_s_2}

The next step used to be a videostream definition and a source setup:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp header
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp in_out_1

and then a compiled graph used to be applied to every single frame.
This is an example of a processing loop without Streaming:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp in_out_2

Now this can be done without explicit use of additional libraries. First of all,
let's define a stream variable and pass to it:
    - video properties (this unconvenience is temporary);
    - graph compile arguments (which will be passed to the graph inside
the stream):

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp str_comp

Then we should set a source:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp str_src

And for this moment our stream is ready to be launched! An example of a
Streaming loop:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp str_loop

There are some other useful possibilities provided, e.g. you can abort stream
processing via `cv::GStreamingCompiled::stop()` method (the next call of
`cv::GStreamingCompiled::start()` will start processing the stream from the very
beginning).

### 3) You are breathtaking! {#gapi_fb_s_3}

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

Details of obvious operations (such as command line parsing or a single-string
kernel implementation) are not going to be described in this text; there will be
only comments on debatable features implementation.

Let's start at the begining of the pipline graph definition.

## Face detector inference and post-processing {#gapi_fb_face_detect}

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp gar_ROI
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp fd_inf

The `config::smth` variables are constants defined in the `config` namespace:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp config

After the inference, post-processing for the output `cv::Mat` is required: the
face detector returns a blob with the shape [1, 1, N, 7], where N is
the number of bounding boxes containing detected faces. The structure of an
output for every face is the following:
`[image_id, label, conf, x_min, y_min, x_max, y_max]`;
all the seven elements are floating point. For our further goals we need to take
only results with `image_id > 0` because a negative value in this field
indicates the end of detections; also we can cut detections by the `conf` field
to avoid mistakes of the detector; so the `Mat` can be parsed this way:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp vec_ROI
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp fd_pp

Some points to be mentioned about this kernel:

- It takes a `cv::Mat` from the detector and a `cv::Mat` from the input; it
returns an array of ROI's where faces have been detected.

- There is `cv::Mat` data parsing by the pointer on a float used.

- Float numbers we've got from the Mat are between 0 and 1 and denote normalized
coordinates; to get the real pixel coordinates we should multiply it by the
image sizes respectively to the directions. To be fully accurate, the received
numbers should be then rounded the right way and casted to `int` to construct
integer Points; these operations have been wrapped in `toIntRounded()`:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp toInt

- Rectangles are constructed by two Points --- the top-left and the bottom-right
corners.

- By far the most important thing here is solving an issue that sometimes
detector returns coordinates located outside of the image; if we pass such an
ROI to be processed, errors in the landmarks detection will occur. The frame box
`borders` is created and then intersected with the face rect (by `operator&()`)
to handle such cases and save the ROI in `outFaces` which is for sure
inside the frame.

## Facial landmarks detector inference and post-processing {#gapi_fb_landm_detect}

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp ld_inf

In the first place, a few words about custom types: the `Contour` type is used
for arrays of points which are naturally contours (ordered sets of points):

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp cont

The array with all landmarks is an array of points; syntactically it is
`Contour` though semantically is not so I used another type `Landmarks` which
syntactically is equal to `Contour` type to underline semantical differences:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp landm

Post-processing of the Mat is also required here: there are 35 landmarks given by
the detector for each face in a frame; the first 18 of them are facial elements
(eyes, eyebrows, a nose, a mouth) and the last 17 --- a jaw contour. Landmarks
are floating point values of coordinates normalized relatively to an input ROI
(not the original frame). But other than that, we need contours of
eyes, mouths, faces, etc., not the landmarks, for the further goals.
So the process is split into two parts.

### Normalized points scaling {#gapi_fb_ld_scl}

The first step is to denormalize coordinates of the landmarks to the real pixel
coordinates of the source frame. Also, we are able to divide points into two
groups here --- face elements and a jaw contour. This is a kernel that does the
described actions:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp ld_pp_scl

Some points:

- It takes a `cv::Mat` from the detector and the array of the ROIs from the
previous step; an array of arrays which contain facial elements' points and
a jaw contours' array are what it returns.

- Data parsing is the same --- by pointer.

- We iterate by 2 in one step because for every point `Point.x` and `Point.y`
are contained successively.

- Since the coordinates are related to the input ROI, the coordinates of the
ROI's top-left corner should be added.

### Getting contours {#gapi_fb_ld_cnts}

The idea of this face beautification algorithm can be defined more certainly:
the thing is to smooth out the
skin and to make eyes, a nose and a mouth sharper. So we need contours of facial
elements based on landmarks and the whole face contour (not only a jaw) as they
are drawn in the picture:

![Contours](pics/contours.png)

To increase code readability, the `GGetContours` kernel is split up to pieces,
which are wrapped in separate functions, with the goal to describe just the
sequence of actions:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp ld_pp_cnts

The kernel takes two arrays from the previous step and returns an array of
elements' finished contours and an array of faces' full contours; in other
words, outputs are, the first, an array of contours to be sharped and, the
second, another one to be smoothed.

To understand the points numeration we should just look at the following
illustration:

![Landmarks order](pics/landmarks_illustration.png)

The separate discussion of every element drawing follows.

#### Getting an eye contour {#gapi_fb_ld_eye}

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp ld_pp_eye

The function used to get the bottom side of an eye contour should be discussed:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp ld_pp_eye

Briefly, this function restores the bottom side of an eye by a half-ellipse
based on two points in left and right eyecorners.

As you can see, to minimize the frequency of memory allocations the
`std::vector::reserve()` function is used; we can pass the real quantity of
points to be stored in `cntEyeBottom` through the `capacity` argument if we know
it apriory.

Why isn't memory allocated at the beginning of the `getEyeEllipse()` function?
The reason is the `cv::ellipse2Poly()` feature: instead of just filling out the
given array the function assigns to it an array of points created inside;
therefore, there is no sense to allocate memory before `cv::ellipse2Poly()`
call.

The rest of the `getEyeEllipse()` function is parameters preparation for the
`cv::ellipse2Poly()` function call.

![Ellipse2poly illustration](modules/imgproc/doc/pics/ellipse.svg)

What has to be defined:
- the ellipse center and the X half-axis calculated by two eye Points;
- the Y half-axis calculated according to the assumption that an average eye
width is 1/3 of its length;
- the start and the end angles which are 0 and 180 (see the picture above)
- the angle delta: how frequently (which causes by how much points) the ellipse
will be approximated;
- the inclination angle of axes calculated via the
`getLineInclinationAngleDegrees()` function:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp toDbl
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp ld_pp_incl

The use of the `atan2()` function instead of just `atan()` is essential as it
allows to return a negative value depending on the `x` and the `y` signs
so we can get the right angle even in case of upside-down face arrangement
(if we put the points in the right order, of course).

Back to the `getEyeEllipse()` function, it returns a constructed `Contour`
to use return value optimization (RVO).

After the bottom side of an eye is approximated, we just push the eyebrow points
to the contour in the right order - and the eye is finished!

#### Getting a mouth contour {#gapi_fb_ld_mth}

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp ld_pp_mth

We have four points of the mouth given: two of them are in corners and the other
two are in middle of upper and lower lip. So the mouth can be approximated by
two patched half-ellipses:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp ld_pp_mth

At this point Y half-axes are defined without any assumptions, but they are
different for the upper lip and the lower one.

#### Getting a forehead contour {#gapi_fb_ld_fhd}

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp ld_pp_fhd

This snippet describes two actions: the approximation of a forehead by
half-ellipse and pushing the jaw contour to the array in the appropriate order.

The function to approximate a forehead:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp ld_pp_fhd

As we have jaw points only, we have to get a half-ellipse based on three points
of a jaw: the leftmost, the rightmost and the lowest. Obviously,
the jaw width is equal to the forehead width and can be calculated using the
left and the right points. Speaking of the Y axis, we have no points to get it
directly; but (according to assumption) the forehead height is about 2/3 of
the jaw height, which can be received from the face center (the middle between
the left and right points) and the lowest jaw point.

We have all the counters needed! Let's draw masks.

## Drawing masks {#gapi_fb_masks_drw}

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp msk_ppline

Honestly speaking, there is no need to get the most accurate contours in the
world: we can always blur borders by gaussian filter to occupy a bit more area.
Another idea is to get three masks which are for bilateral-filtered image, for
sharped image and for the untouched one and have no intersections with each
other; this approach allows just to summarize images with the masks applied and
get the output without any other operations:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_old.cpp msk_appl

Thus, the steps to get the masks are:
* the "sharp" mask:
    * fill the contours that should be sharpened;
    * blur that to get the "sharp" mask;
* the "bilateral" mask:
    * fill all the face contours fully;
    * blur that;
    * subtract areas which intersect with the "sharp" mask --- and get the
      "bilateral" mask;
* the mask of all the area that shouldn't be touched by filters (background):
    * add two previous masks
    * set all non-zero pixels of the result as 255 (by `cv::gapi::threshold()`)
    * revert the output (by `cv::gapi::bitwise_not`) to get the "untouched"
      mask.

## UnsharpMask() algorithm {#gapi_fb_unsh}

The algorithm of `unsharpMask()` filter was implemented as described in this
<a href="https://www.idtools.com.au/unsharp-masking-python-opencv/">article</a>.

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification_stream.cpp unsh

## Possible further improvements {#gapi_fb_to_improve}

There are some points in the algorithm to be improved.

### Correct ROI reshaping for meeting conditions required by the facial landmarks detector {#gapi_fb_padding}

The input of the facial landmarks detector is a square ROI, but the face
detector gives non-square rectangles in general. If we let the backend within
Inference-API compress the rectangle to a square by itself, the lack of
inference accuracy can be noticed in some cases.
There is a solution: we can give a describing square ROI instead of the
rectangular one to the landmarks detector, so there will be no need to compress
the ROI, which will lead to accuracy improvement.
Unfortunately, another problem occurs if we do that:
if the rectangular ROI is near the border, a describing square will probably go
out of the frame --- that leads to errors of the landmarks detector.
To aviod such a mistake, we have to implement an algorithm that, firstly,
describes every rectangle by a square, then counts the farthest coordinates
turned up to be outside of the frame and, finally, pads the source image by
borders (e.g. single-colored) with the size counted. It will be safe to take
square ROIs for the facial landmarks detector after that frame adjustment.

### Research for the best parameters (used in GaussianBlur() or unsharpMask(), etc.) {#gapi_fb_res_params}

### Parameters autoscaling {#gapi_fb_auto}

# Conclusion {#gapi_fb_cncl}

The tutorial has two goals: to show the use of brand new features of OpenCV
G-API module appeared in 4.2 release on living examples and to describe some
ideas of implementing the certain CV algorithm.
