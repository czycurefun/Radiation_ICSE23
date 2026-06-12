
# params for dataset and data loader
data_root = "data"
process_root = "data/processed"
batch_size = 16

# params for source dataset
src_encoder_restore = "result/alltoeasyquarter/RL-source-encoder-final.pt"
src_classifier_restore = "result/alltoeasyquarter/RL-source-classifier-final.pt"
src_model_trained = True

# params for target dataset
tgt_encoder_restore = "result/alltoeasyquarter/RL-target-encoder-final.pt"
tgt_model_trained = True

# params for setting up modelss
model_root = "result/alltoeasyquarter"
d_model_restore = "result/alltoeasyquarter/RL-critic-final.pt"

# params for training network
num_gpu = 1
manual_seed = None

# params for optimizing models
d_learning_rate = 3e-5
c_learning_rate = 3e-5
beta1 = 0.5
beta2 = 0.99
