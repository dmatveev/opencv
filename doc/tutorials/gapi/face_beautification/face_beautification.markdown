# Implementing a face beautification algorithm on G-API {#tutorial_gapi_face_beautification}

@prev_tutorial{tutorial_gapi_anisotropic_segmentation}

[TOC]

# Introduction {#gapi_fb_intro}

In this tutorial you will learn:
* How to infer different networks inside the pipeline with G-API;
* How to process a videostream inside the pipeline with G-API;
* The main ideas of the Face Beautification algorithm.

You can find the source code in
`samples/cpp/tutorial_code/face_beautification.cpp`
of the OpenCV library source tree.

It is a well-known fact that geterogeneous algorithms based on Computer Vision
image processing with the backup of deep neural networks inference
(Deep Learning) are able to solve a huge variety of problems which CV-only
ones might experience difficulties with. Let's consider the certain
algorithm as an example.

# Face Beautification algorithm {#gapi_fb_algorithm}

The algorithm of retouching a face has been chosen as the subject for this
tutorial. Its main idea is to implement a real-time
videostream processing that detects faces and applies some filters to make them
look beautiful (more or less). The pipeline is the following:

![The Face Beautification algorithm](pics/algo.png)

The algorithm consists of two parts: an inference of two networks (including
data pre- and post-processing) and an image filtering pipeline which uses the
inference data to create masks, then applies filters and composes the output
retouched image. What is each of these parts for? To be more certain, the
"beautification" implemented here is smoothing out the skin (by the
`Bilateral filter`) and making eyes, a nose and a mouth sharper (by the
`Unsharp mask`). Thus, to know which filter should be applied to which area of
image, we need to calculate a contour of the face and contours of facial
elements --- we can get them from data received via deep learning methods.

Two topologies from OMZ have been used in this sample: the
<a href="https://github.com/opencv/open_model_zoo/tree/master/models/intel
/face-detection-adas-0001">face-detection-adas-0001</a>
and the
<a href="https://github.com/opencv/open_model_zoo/blob/master/models/intel
/facial-landmarks-35-adas-0002/description/facial-landmarks-35-adas-0002.md">
facial-landmarks-35-adas-0002</a>.

The face detector takes the input image and returns a blob with the shape
[1, 1, N, 7] after the inference, where `N = 200` is the maximum number of
faces which can be detected. Obviously, the output `cv::Mat` post-processing is
required. The structure of the output for every face is the following:
`[image_id, label, conf, x_min, y_min, x_max, y_max]`,
all the seven elements are floating point. For our further goals we need to take
only results with `image_id > 0` because a negative value in this field
indicates the end of detections; also we can cut detections by the `conf` field
to avoid mistakes of the detector. The last four float numbers we've got denote
normalized coordinates and are between 0 and 1; to get the real pixel
coordinates we should multiply it by the image sizes respectively to the
directions.

Then, to create masks we need to get landmarks of facial elements for every
detected face. The chosen detector expects to obtain square area with only one
face contained so we have to infer the model separately for every ROI we've
got from the previous detection. The output of the facial landmarks detector is
a blob with 35 landmarks: the first 18 of them are facial elements
(eyes, eyebrows, a nose, a mouth) and the last 17 --- a jaw contour. Landmarks
are floating point values of coordinates normalized relatively to an input ROI
(not the original frame). In addition, for the further goals we need contours of
eyes, mouths, faces, etc., not the landmarks. So, post-processing of the Mat is
also required here. The process is split into two parts --- landmarks'
coordinates denormalization to the real pixel coordinates of the source frame
and getting necessary closed contours based on these coordinates.

The last step of processing the inference data is drawing masks using the
received contours. Honestly speaking, there is no need to get the most accurate
contours in the world: we can always blur masks borders by gaussian filter to
occupy a bit more area. Another point that should be mentioned here is getting
three masks (for areas to be smoothed, for ones to be sharpened and for the
background) which have no intersections with each other; this approach allows to
apply the calculated masks to the corresponding images prepared beforehand and
then just to summarize them to get the output image without any other actions.

All the operations described above are supported fascinatingly by the G-API
functionality, so this algorithm is appropriate to illustrate G-API usage
convenience and efficiency in the context of solving a real CV/DL problem.

# Preparing networks to be inferenced with G-API {#gapi_fb_prep_nets}

Let's prepare our chosen networks for being processed inside the graph.
Providing a new network support is quite similar to a new operation
implementation.

To declare a deep learning topologies we should use the specific macro
`G_API_NET()`:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp net_decl

Similarly to `G_API_OP()` macro for a new operation, it takes three arguments to
register a new type.
They are:
1. Network interface name --- also serves as a name of a new type defined
   with this macro;
2. Network signature      --- an `std::function<>`-like signature which defines
   inputs and outputs;
3. Network's unique name  --- used to identify it within the system when the
   type is stripped.

# Graph compilation {#gapi_fb_comp}

For further convenience we should use the version of a pipeline expression
with a lambda-based constructor to keep all temporary objects in a dedicated
scope (almost all the graph's body is skipped in this code snippet
except for input and output nodes):

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp comp_str_1
...
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp comp_str_2
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp comp_str_3
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp comp_str_4

To apply a model inside the graph, we should call
`cv::gapi::infer<>()` setting a network type as a template argument to
specify the topology we want to infer:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp net_usg_fd

If we need to convey several ROIs for inference, we can use
the `cv::gapi::infer<>()` function's overload:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp net_usg_ld

The second argument of it is the image to be processed, and the first is a
`cv::GArray<>` of ROIs; the network will be automatically inferred for every single ROI.

# Custom operations implementation {#gapi_fb_proc}

A new kernel implementation has been already described clearly in OpenCV G-API
documentation (see [G-API Kernel API](@ref gapi_kernel_api)).

We usually use `G_TYPED_KERNEL()` macro to define a new kernel type; a small yet
significant adjustment has to be made when  defining a kernel with a
multiple return value. For this case, we must use `G_TYPED_KERNEL_M()` macro:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp kern_m_decl

Note: the `std::tuple` appearance in a kernel signature could
be the sign to change the macro. At the implementation step, all the output
arguments of a kernel signature are just put after input ones keeping the order
if they are multiple:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp kern_m_impl

## Face detector post-processing {#gapi_fb_face_detect}

As it was discussed before, the post-processing operation is required after
the inference. The output `Mat` can be parsed this way:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp vec_ROI
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp fd_pp

Some points to be mentioned about this kernel implementation:

- It takes a `cv::Mat` from the detector and a `cv::Mat` from the input; it
returns an array of ROI's where faces have been detected.

- There is `cv::Mat` data parsing by the pointer on a float used.

- By far the most important thing here is solving an issue that sometimes
detector returns coordinates located outside of the image; if we pass such an
ROI to be processed, errors in the landmarks detection will occur. The frame box
`borders` is created and then intersected with the face rect (by `operator&()`)
to handle such cases and save the ROI which is for sure inside the frame.

Data parsing after the facial landmarks detector happens according to the same
scheme with inconsiderable adjustments.

## Facial landmarks detector post-processing (getting contours) {#gapi_fb_landm_detect}

As it was said, we need to calculate closed contours of areas for different
filters to be applied. To increase code readability, the kernel is split up to
pieces, which are wrapped in separate functions, with the goal to describe just
the sequence of actions:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp ld_pp_cnts

The kernel takes two arrays of denormalized landmarks coordinates and returns an
array of elements' closed contours and an array of faces' closed contours; in
other words, outputs are, the first, an array of contours of image areas to be
sharpened and, the second, another one to be smoothed.

To understand the points numeration you should see the network documentation
(<a href="https://github.com/opencv/open_model_zoo/blob/master/models/intel
/facial-landmarks-35-adas-0002/description/facial-landmarks-35-adas-0002.md">
facial-landmarks-35-adas-0002</a>).

### Getting an eye contour {#gapi_fb_ld_eye}

The function to calculate the bottom side of an eye contour should be discussed:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp ld_pp_eye

Briefly, this function restores the bottom side of an eye by a half-ellipse
based on two points in left and right eyecorners. It returns a constructed
`Contour` on purpose to use return value optimization (RVO).

In the simplest words, the
function prepares parameters for the `cv::ellipse2Poly()` call.

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

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp toDbl
@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp ld_pp_incl

The use of the `atan2()` function instead of just `atan()` is essential as it
allows to return a negative value depending on the `x` and the `y` signs
so we can get the right angle even in case of upside-down face arrangement
(if we put the points in the right order, of course).

### Getting a forehead contour {#gapi_fb_ld_fhd}

The function to approximate a forehead:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp ld_pp_fhd

As we have only jaw points after the inference, we have to get a half-ellipse
based on three points of a jaw: the leftmost, the rightmost and the lowest.
Obviously, the jaw width is equal to the forehead width and can be calculated
using the left and the right points. Speaking of the Y axis, we have no points
to get it directly; but (according to assumption) the forehead height is about
2/3 of the jaw height, which can be received from the face center (the middle
between the left and right points) and the lowest jaw point.

## Drawing masks {#gapi_fb_masks_drw}

When we have all the contours needed, we are able to draw masks:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp msk_ppline

The steps to get the masks are:
* the "sharp" mask calculation:
    * fill the contours that should be sharpened;
    * blur that to get the "sharp" mask (`mskSharpG`);
* the "bilateral" mask calculation:
    * fill all the face contours fully;
    * blur that;
    * subtract areas which intersect with the "sharp" mask --- and get the
      "bilateral" mask (`mskBlurFinal`);
* the background mask calculation:
    * add two previous masks
    * set all non-zero pixels of the result as 255 (by `cv::gapi::threshold()`)
    * revert the output (by `cv::gapi::bitwise_not`) to get the background
      mask (`mskNoFaces`).

## UnsharpMask() algorithm {#gapi_fb_unsh}

The algorithm of `unsharpMask()` filter was implemented as described in this
<a href="https://www.idtools.com.au/unsharp-masking-python-opencv/">article</a>.

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp unsh

# Graph compile arguments {#gapi_fb_comp_args}

All the custom operations (including networks inference) must be defined for
every certain graph through `cv::compile_args`.

To set parameters of the network we have to generate a `cv::gapi::ie::Params<>`
object:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp net_param

Every `cv::gapi::ie::Params<>` object is related to the network specified by its
template argument. We should pass there the network type we have described in
the previous step.
Those objects are able to contain paths to structure and weights files and
a desirable backend which can be one of the backends supported by
InferenceEngine ("CPU", "GPU", etc.).

All the nets' parameter objects should be contained in an object returned by
`cv::gapi::networks()` function:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp netw

To allow a certain pipeline to use our custom kernels (as well as all other
non-standard kernels, e.g. `cv::gapi::core::fluid::kernels()`), we have to
describe them by `cv::gapi::kernels<>` object:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp kern_pass_1

After that, we must pass networks' parameters and kernels' descriptions to the
graph by compile arguments:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp apply

# Video stream processing with G-API {#gapi_fb_streaming}

There is no interest in using the most of computer vision pipelines to process
just a single picture --- a big group of them are created to be launched with a
videosource as an input. Therefore, it is essential for G-API as an
interface for pipelines implementation to provide a possibility of a stream
processing without any external instruments to be applied by user.
So, that's why the G-API feature providing such a possibility is needed.

## Videostream processing before OpenCV 4.2 release {#gapi_fb_str_before}

A videostream used to be captured by `cv::VideoCapture` object:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp bef_cap

and then a compiled graph used to be applied to every single frame:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp bef_loop

## Videostream processing after OpenCV 4.2 release {#gapi_fb_str_after}

Now this can be done without explicit use of additional libraries. First of all,
let's define a stream variable and pass to it graph compile arguments (which
the graph will use within the stream):

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp str_comp

Then we should set a source:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp str_src

And for this moment our stream is ready to be launched! An example of a
Streaming loop:

@snippet cpp/tutorial_code/gapi/face_beautification/face_beautification.cpp str_loop

There are some other useful possibilities provided, e.g. you can abort stream
processing via `cv::GStreamingCompiled::stop()` method (the next call of
`cv::GStreamingCompiled::start()` will start processing the stream from the very
beginning).

# Conclusion {#gapi_fb_cncl}

The tutorial has two goals: to show the use of brand new features of OpenCV
G-API module appeared in 4.2 release on living examples and to describe some
ideas of implementing the face beautification algorithm.

FIXME:another image

The example of the received result:

![Face Beautification example](pics/example.png)

Benchmarking has shown huge advantages of using new G-API features for
processing video streams: due to pipelining optimization, the performance
has been increased by .. times.

## Possible further improvements

There are some points in the algorithm to be improved.

### Correct ROI reshaping for meeting conditions required by the facial landmarks detector

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

### Research for the best parameters (used in GaussianBlur() or unsharpMask(), etc.)

### Parameters autoscaling
