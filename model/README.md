This directory stores versioned model weights through Git LFS. Each new release
uses a dated folder, for example:

`model/V8-RG-final-14p-20261005/model.weights.h5`

GitHub stores a small LFS pointer in the commit and the checkpoint binary in LFS.
Install Git LFS before cloning or run `git lfs pull` after cloning to fetch the
weights. The matching configuration and release manifest are ordinary small Git
files.

The legacy default model remains mirrored in S3 at:

s3://viewray-ai/Patient-vision/Pradeep/Region-aware-v8/model/

Expected files:
- checkpoint_best.weights.h5
- config.json
