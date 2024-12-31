# MAX-CV #

Brad Larson

[@bradlarson@hachyderm.io](https://hachyderm.io/@bradlarson)
[@bradlarson.bsky.social](https://bsky.app/profile/bradlarson.bsky.social)

## Overview ##

MAX-CV is an exploration of accelerated image processing using
[MAX](https://docs.modular.com/max/) and
[Mojo](https://docs.modular.com/mojo/manual/). Inspiration for this framework
has been drawn from the [GPUImage](https://github.com/BradLarson/GPUImage)
series of Objective-C and Swift frameworks based on OpenGL or Metal.

The objective of the framework is to make it as easy as possible to set up and
perform realtime video processing or machine vision against image or video
sources, running operations on a range of CPUs, GPUs, or more. Operations are
defined in a platform-independent manner using the Mojo language, with a
high-level Python API to ease integration into existing code. Graphs of image
processing operations are constructed and compiled to fuse and optimize
sequential operations.

## License ##

Apache License v2.0 with LLVM Exceptions.

## Technical requirements ##

MAX 24.6 or newer (nightly preferred). The command-line tool Magic will handle
all dependency setup for MAX, and can be installed via

```sh
curl -ssL https://magic.modular.com | bash
```

- OS: Ubuntu 22.04 or newer, macOS 14.0 or newer
- NVIDIA GPU: Ampere or newer architecture

## Examples ##

Currently, a single example is provided to demonstrate how to run an image
through a processing pipeline and obtain a resulting image. Assuming that you
have installed the Magic command-line tool referenced above, building and
running this example is as simple as calling the following in the current
directory:

```sh
magic run filter-single-image
```

## General architecture ##

[TODO]

## Using MAX-CV ##

[TODO]

## Performing common tasks ##

[TODO]

### Writing a custom image processing operation ###

[TODO]

## Built-in operations ##

Operations are currently being ported over from GPUImage 2. Here are the ones that are currently functional:

### Color adjustments ###

- **brightness**: Adjusts the brightness of the image.
  - *brightness*: The adjusted brightness (-1.0 - 1.0, with 0.0 as the default)

- **gamma**: Adjusts the gamma of an image.
  - *gamma*: The gamma adjustment to apply (0.0 - 3.0, with 1.0 as the default)
