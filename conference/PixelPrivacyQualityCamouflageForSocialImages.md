# Shield: Fast, Practical Defense and Vaccination for Deep Learning using JPEG Compression

- real time (quick)
- need retraining the model with cimpressed images
- on top of SHIELD, they add an additional layer of protection by employing randomization at test time that compresses different regions of an image using random Compression levels

## Proposed method: compression as Defense

- convert the given images from RGB to YCbCr color space
- perform spatial subsampling of the chrominance channels (human eye is less susceptible to these changes)
- divides images into blocks and applies a randomly selected JPEG compression quality (20,40,60 and 80) to each block to remove adversarial attacks.


