# Requirements
- Tensorflow == 1.9.0
- NLTK
- joblib == 0.13.2

# Files
- run_model.py: This file has all the flags needed for training, it chooses the data module, models and starts training.
- utils.py: Utilities module containing DataGenerators, Parallel model etc.
  - DataGenerator: Creates batches for vanilla NSE.
  - DataGeneratorHier: Creates batches for hierarchical NSE.
- Memory: All the files below contain a variant of NSE.
  - NSE: This is a neural semantic encoder class.
  - HierNSE: This is the hier-NSE class.
- models: All the files below contain an encoder, decoder, loss, optimizer functions that use an NSE.
  - model.py: Model using vanilla NSE.
  - model_hier.py: Model using hier-NSE (use this).
  - model_hier_sc.py: Self-Critic model (use this). It carefully back-propagates through the same multinomial samples that are sampled while forward pass.
- rouge: Rouge scripts used.
  - rouge_batch: A NumPy implementation (faster than existing ones). Used outside the TensorFlow graph.
- Data
  - Create a folder named `data`
  - Download the following splits into data folder: 
        [train](https://github.com/abisee/cnn-dailymail/blob/master/url_lists/all_train.txt),
        [val](https://github.com/abisee/cnn-dailymail/blob/master/url_lists/all_val.txt),
        [test](https://github.com/abisee/cnn-dailymail/blob/master/url_lists/all_test.txt)
  - Download the CNN and Daily-Mail tokenized data:
        [CNN](https://drive.google.com/file/d/0BzQ6rtO2VN95cmNuc2xwUS1wdEE/view?usp=sharing),
        [DM](https://drive.google.com/file/d/0BzQ6rtO2VN95bndCZDdpdXJDV1U/view?usp=sharing)
  - Download [GloVe](http://nlp.stanford.edu/data/glove.840B.300d.zip)

# supervised model
  - Training

          python run_model.py --model="hier" --mode="train" --PathToCheckpoint=/path/to/checkpoint --PathToTB=/path/to/tensorboard/logs
  
  - Testing
      - Check the epoch number of the best supervised model from TensorBoard, let it be X
        
            python run_model.py --model="hier" --mode="test" --PathToCheckpoint=/path/to/checkpoint/model_epochX --PathToResults=/path/to/results
  
  - Evaluation
  
          python run_model.py --model="hier" --mode="eval" --PathToResults=/path/to/results
  
# self-critical model
  - Training
    - First copy the best supervised model to the rl checkpoint.
      
            python run_model.py --model="rlhier" --mode="train" --restore=True --PathToCheckpoint=/path/to/checkpoint --PathToTB=/path/to/tensorboard/logs
 
  - Testing
    - Check the epoch numner of the best supervised model from TensorBoard, let it be X.
      
            python run_model.py --model="rlhier" --mode="train" --restore=True --PathToCheckpoint=/path/to/checkpoint/model_epochX --PathToResults=/path/to/results
  
  - Evaluation
            
            python run_model.py --model="rlhier" --mode="eval" --PathToResults=/path/to/results
