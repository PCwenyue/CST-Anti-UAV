# CST Anti-UAV Benchmark

CST Anti-UAV: A Thermal Infrared Benchmark for Tiny UAV Tracking in Complex Scenes

This toolkit is used to evaluate trackers on infrared UAV tracking benchmark called CST Anti—UAV. The benchmark comprises a total of 220 videos with over 240K manually annotated bounding boxes.


## Preparing the dataset
Download the CST Anti-UAV dataset([Baidu disk](https://pan.baidu.com/s/1mwGkvmQHoy5MhUCPvN26xA?pwd=nayi) Access code: nayi) or (Google Drive：https://drive.google.com/drive/folders/13DIRzt1pC2c4Dhyi6yFhHBeMFoKPQtxU?usp=sharing) to your disk, the organized directory should look 

    ```
    --CST Anti-UAV/
    	|--test
    	|--train
    	|--val
    ```


## Installation and testing
**Step 1.** Create a conda environment and activate it.

```shell
conda create -n CSTAntiUAV python=3.8
conda activate CSTAntiUAV
```

**Step 2.** Install the requirements.
```shell
pip install opencv-python, matplotlib, wget, shapely

pip install torch===1.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torchvision===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
Other versions of python, cuda and torch are also compatible.

**Step 3.** Preparing the tracking results for compartion.

Download the tracking results (Baidu disk: https://pan.baidu.com/s/1LSvgMyL7TTKUApWwRQfWMg?pwd=gmn2 提取码: gmn2) to your project directory, the organized directory should look like:The organized directory should look like:

    ```
    --project_dir/tracking_results/
    	|--Trained_with_CST
    ```

**Step 4.** Evaluating the trackers.

Change the dataset path and edit project_dir/utils/trackers.py to select the trackers to be evaluated.

Run
```shell
python Evaluation_for_ALL.py
python Evaluation_for_ALL_Frame_SA.py
```

The evaluation plots will be saved at project_dir/reports/CSTAntiUAV/.

## Citation

If you find this project useful in your research, please consider cite.
