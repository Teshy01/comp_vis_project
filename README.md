---------------------------------------------------------------
- SECTION 1 - HOW TO ACCESS DeTR DATASET USED FOR DeTR Program-
---------------------------------------------------------------
1. Use this URL: [https://www.kaggle.com/datasets/sainikhileshreddy/food-recognition-2022]
2. Download the dataset
3. Move To SECTION 2 to create the folder structure

-----------------------------------
-SECTION 2 - DeTR Folder Structure-
-----------------------------------
Your folder structure should look like below where:
'>' represents a directory
'-' represents the files that are inside the root directory of the project
'...' represents files and folders that either come with the downloaded dataset content or get generated by the program

-----------Root Directory ----------

> hub
...

> Modules
   - __init__.py
   - config.py
   - dataset.py
   - metrics.py
   - model_architecture.py
   - predict.py
   - process_data.py
   - train.py

> outputs (manually create this directory)
...

> raw_data
...

- evaluation.py
- main.py
- model_text.py
- requirements.txt
-------------------------------------

-------------------------------------
-SECTION 3 - How To Run DeTR Program-
-------------------------------------
1. Navigate to Root Directory that holds:
- evaluation.py
- main.py
- model_text.py
- requirements.txt

2. Run main.py to train
3. Run evaluation.py
4. Run model_test.py


------------------------------------------------------------------------------
- SECTION 4 - HOW TO ACCESS DATASET Used For YOLOv8 Program & TOLOv11 Program-
------------------------------------------------------------------------------
1. Use this URL: [http://foodcam.mobi/dataset100.html]
2. Download the dataset "UECFOOD-100 Dataset Ver.1.0 (945MB)"
3. Clone the ultralytics repository from github using this command: 'git clone https://github.com/ultralytics/ultralytics.git' WHILE IN desired root directory
4. Move To SECTION 5 to create the folder structure
   

-------------------------------------
-SECTION 5 - YOLOv8 Folder Structure-
-------------------------------------
Your folder structure should look like below where:
'>' represents directory
'-' represents files within the directory

-----------Root Directory ----------
- YOLOv8_Program.ipynb

- train_YOLOv8_model.py

> datasets (create this directory manually)

> UECFOOD100 downloaded dataset

> ultralytics

> runs (create this directory manually)
------------------------------------

---------------------------------------
-SECTION 6 - How To Run YOLOv8 Program-
---------------------------------------
1. After you have created the 'datasets' directory, you need to run the first 4 cell blocks of YOLOv8_Program.ipynb to preprocess the data. Do this only once.
2. Drag the preprocessed data (including its UECFOOD100 directory) into the datasets directory so that it fits the ultralytics YOLO format for reading and training models.
Your datasets directory should now look like this:

> datasets
-> UECFOOD100_YOLO -> images, -> labels, - data.yaml
   
4. run train_YOLOv8_model.py to train
5. After you've trained, in the cell block marked "Run Baseline Model" for YOLOv8_Program.ipynb, change the path of the model to your own path to trained model
6. Do the same for the next cell block (change the path to your desired path)
7. Run the rest of the blocks from YOLOv8_Program.ipynb to evaluate and predict
8. For Real Time Inference, uncomment the last cell block. Given that you have a camera on your device, you may run it.

----------------------------------------
-SECTION 6 - How To Run YOLOv11 Program-
----------------------------------------
1. Create a new directory and put YOLOv11_Program.ipynb in it
2. Drag the raw downloaded dataset from SECTION 5 into this new directory
3. Clone the ultralytics repository from github using this command: 'git clone https://github.com/ultralytics/ultralytics.git' WHILE IN desired root directory
4. Manually create a directory called datasets
5. After you have created the 'datasets' directory, you need to run the first 4 cell blocks of YOLOv11_Program.ipynb to preprocess the data. Do this only once.
6. Drag the preprocessed data (including its UECFOOD100 directory) into the datasets directory so that it fits the ultralytics YOLO format for reading and training models.Your datasets directory should now look like this:
   
> datasets
-> UECFOOD100_YOLO -> images, -> labels, - data.yaml

7. Run the rest of the blocks from YOLOv11_Program.ipynb to train
8. For Real Time Inference, uncomment the last cell block. Given that you have a camera on your device, you may run it.





