# An overlap-aware coarse-to-fine correspondence establishment method for indoor point cloud registration

### Prepare datasets

  ```
  sh scripts/download_data.sh
  ```
  
### Train

  ```
  sh scripts/train_3dmatch.sh
  ```
  
### Test

  + Point correspondences are first extracted by running:
  
  ```
  sh scripts/test_3dmatch.sh
  ```
  
  and stored on `snapshot/tdmatch_enc_dec_test/3DMatch/`. 
  
  
  + To evaluate on 3DLoMatch, please change the `benchmark` keyword in `configs/tdmatch/tdmatch_test.yaml` from `3DMatch` to  `3DLoMatch`.
  
  + The evaluation of extracted correspondences and relative poses estimated by RANSAC can be done by running:

  ```
  sh scripts/run_ransac.sh
  ```
  
  + The final results are stored in `est_traj/3DMatch/{number of correspondences}/result` and the results evaluated on our computer have been provided in `est_traj/`. 
  
  + To evaluate on 3DLoMatch, please change `3DMatch` in `scripts/run_ransac.sh` to `3DLoMatch`.
