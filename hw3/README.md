To run the code, please first to make sure that the following packages
are installed:
~~~~~~~~~~~~~~~~~~~~~~~~~
    pytorch >= 1.6.0
    allennlp >= 1.1.0
    spacy >= 2.3.2
    scispacy >= 0.3.0
    nltk >= 3.5
~~~~~~~~~~~~~~~~~~~~~~~~~
Next, download the file vec.npy from https://drive.google.com/drive/folders/1Ri3LIILKKBi3aBJjUVCOBpGX5PpONHRK
and put it in ./DocRED/code/prepro_data

, to train the models, run the following commands:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cd DocRed/code
    # For BiLSTM:
    # 1. Prepare the data
    python prepare_input_docred_format.py

    # 2. Training
    CUDA_VISIBLE_DIVICES=0 python train_biorel.py --train_prefix train --test_prefix dev --model_name BiLSTM --save_name checkpoint_bilstm 

    # For BERT:
    # 1. Prepare the data
    python prepare_input_docred_format_bert.py

    # 2. Training
    CUDA_VISIBLE_DIVICES=0 python train_biorel_bert.py --train_prefix train_bert --test_prefix dev_bert --model_name BERT --save_name checkpoint_bert     
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To evaluate the results, use:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   # For BiLSTM:
   python evaluate.py -t ../prepro_data/dev.json -path ../fig_result/dev_checkpoint_bilstm_index_converted.json

   # For BERT:
   python evaluate.py -t ../prepro_data/dev_bert.json -path ../fig_result/dev_bert_checkpoint_bert_index_converted.json
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
