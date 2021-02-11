## Digger-Guider: High-frequency Factor Extraction on Stock Trend Prediction

0. ### **Dependencies**

   Install packages from `requirements.txt`.  


1. ### **Load Data using qlib**
	```linux
	$ cd ./load_data
	```

	#### Download daily data:

	```python
	$ python load_dataset.py
	```
	* Change parameter `market` to get data from different dataset: `csi300`, `NASDAQ` etc.

	  ##### Data Sample - SH600000 in CSI300

	  ![](https://ftp.bmp.ovh/imgs/2021/02/28e2e1b545cf8ffc.png)
	
	  features dimensions = 6 * 20 + 1 = 121

	#### Download high-frequency data:
	
	```python
	$ python high_freq_resample.py
	```
	
	* Change parameter `N` to get data from different frequencies: `15min`, `30min`, `120min` etc.
	
	  ##### Data Sample - SH600000 in CSI300
	
        ![](https://ftp.bmp.ovh/imgs/2021/02/21213511c92c4c44.png)
	
	  features dimensions = 16 * 6 * 20 + 1 = 1921

2. ### **Framework**
* Digger：`./framework/models/cnn_rnn_v2.py`
    * Min_Model( ) + Day_Model_2( )
* Rule-based Guider (RG): `./framework/models/cnn_rnn_v2.py`
    * Day_Model_1( )
* Parametric Guider (PG): `./framework/models/cnn_rnn_v2.py`
    * Mix_Model( ) + Day_Model_2( )
* Mutual Distillation: `./framework/models/main_cnn_rnn_v2.py`
    * Guider -> Digger: Mix_to_Min( ) 
    * Digger -> Guider: Min_to_Mix( )
3. ### **Run**
  ```linux
  $ cd ./framework
  ```

  #### Train `Digger-Guider` model:

  ```python
  $ python main_cnn_rnn_v2.py with config/main_model.json model_name=cnn_rnn_v2
  ```

  * Add `hyper-param` = {`values`} after `with` or change them in `config/main_model.json`
  * Prediction results of each model are saved as `pred_{model_name}.pkl` in `./out/`.

  #### Run `Market Trading Simulation`:
  * Prerequisites:   
  	* Server with qlib
  	* Prediction results 
  ```linux
  $ cd ./framework
  ```
  ```python
  $ python trade_sim.py
  ```
4. ### **Records**
	Records for each experiment are saved in `./framework/my_runs/`.  
	Each record file includes: 
	> config.json
	* contains the parameter settings and data path.

	> cout.txt
	* contains the name of dataset, detailed model output, and experiment results.

	> pred_{model_name}_{seed}.pkl
   * contains the  `score` (model prediction) and `label`
	
	> run.json
	
	* contains the hash ids of every script used in the experiment. And the source code can be found in `./framework/my_runs/source/`.
