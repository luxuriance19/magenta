# MusicVAE Deeplearn.js Implementation

This JavaScript implementation of [MusicVAE][https://g.co/magenta/music-vae] uses [Deeplearn.js](https://deeplearnjs.org) for GPU-accelerated inference.

## Usage

To use in your application, install the npm package [@magenta/music-vae](https://www.npmjs.com/package/@magenta/music-vae), or use the [pre-built library](/magenta/models/music-vae/js/dist/).

For a complete guide on how to build an app with MusicVAE, read the [Melody Mixer tutorial][mm-tutorial].

## Pre-trained Checkpoints

Several pre-trained MusicVAE checkpoints are hosted on GCS. While we do not plan to remove any of the current checkpoints, we will be adding more in the future, so your applications should reference the [checkpoints.js](https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_vae/dljs/checkpoints.json) file to see what checkpoints are available, which `DataConverter` they use, and what arguments to pass to the `DataConverter`.

If your application has a high QPS, you must mirror these files on your own server.

See the [Melody Mixer tutorial][mm-tutorial] for example usage.

## Test Usage

`yarn install` to get dependencies

`yarn run build` to produce a commonjs version with typescript definitions for MusicVAE in the dist/ folder that can then be consumed by others over NPM.

`yarn run build-demo` to build the demo.

`http-server demo/` to see it working.

[mm-tutorial]: http://TBD
