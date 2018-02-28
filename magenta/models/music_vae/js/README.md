# MusicVAE Deeplearn.js Implementation

COMING SOON!

## Pre-trained Checkpoints

Several pre-trained MusicVAE checkpoints are hosted on GCS. While we do not plan to remove any of the current checkpoints, we will be adding more in the future, so your applications should reference the [checkpoints.js](https://storage.googleapis.com/download.magenta.tensorflow.org/models/music_vae/dljs/checkpoints.json) file to see what checkpoints are available, which `DataConverter` they use, and what is the length of the sequences they encode (`numSteps`). 

If your application has a high QPS, you must mirror these files on your own server.

## Test Usage

`yarn install` to get dependencies

`yarn run build` to produce a commonjs version with typescript definitions for MusicVAE in the dist/ folder that can then be consumed by others over NPM.

`yarn run build-demo` to build the demo.

`http-server demo/` to see it working.
