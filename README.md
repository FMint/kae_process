# kae_process
process kae dataset, including screening valid actions and converting to the target format.

## prepare bvh data
extract excel & split bvh by frames tag

`process_bvh_by_excel.ipynb`

## bvh-smpl-data&joints

bvh-smpl: `convert_bvh2smpl.py`

```sh
python convert_bvh2smpl.py
```

smpl-data: `convert_smpl2data.py`

```sh
python convert_smpl2data.py
```

## calculate mean&variance

```sh
python cal_mean_variance.py
```

## divide the dataset

`divide_data.ipynb`

## generate text annotations and text word segmentation

`process_text.ipynb`

## Data Structure
```sh
<DATA-DIR>
./animations.rar        //Animations of all motion clips in mp4 format.
./new_joint_vecs.rar    //Extracted rotation invariant feature and rotation features vectors from 3d motion positions.
./new_joints.rar        //3d motion positions.
./texts.rar             //Descriptions of motion data.
./Mean.npy              //Mean for all data in new_joint_vecs
./Std.npy               //Standard deviation for all data in new_joint_vecs
./all.txt               //List of names of all data
./train.txt             //List of names of training data
./test.txt              //List of names of testing data
./train_val.txt         //List of names of training and validation data
./val.txt               //List of names of validation data
./all.txt               //List of names of all data
```