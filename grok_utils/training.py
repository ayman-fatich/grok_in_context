from data import ArithmeticTokenizer, ArithmeticIterator, ArithmeticDataset, VALID_OPERATORS, EQ_TOKEN, EOS
import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from torch.nn import CrossEntropyLoss

def find_last_eq_pos(batch, eq_token_id):
    """Function to find the position of the last equal sign of the sequences in a batch"""
    eq_position = (batch[0]==eq_token_id).nonzero(as_tuple=True)[0]
    return eq_position[-1]

def train(
        train_pct: float,
        operator: str,
        data_dir: str,
        tr_in_context: int,
        val_in_context:int,
        lr: float,
        warmap_steps: int,
        epochs:int,
        
):

    #chose the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Create the datasets
    tr_ds, val_ds = ArithmeticDataset.splits(train_pct, operator, data_dir, tr_in_context, val_in_context)
    
    print(f"The device being used is: {device}")
    print(f"Training and validation DataSets were created successfully: Operation: {operator}, tr_in_context: {tr_in_context}, val_in_context: {val_in_context}")
    
    #Create the batch iterators 
    tr_iterator = ArithmeticIterator(tr_ds, device, True)
    val_iteraot = ArithmeticIterator(val_ds, device, True)
    
    #Create the tokenizer and get the "=" token id
    tokenizer = ArithmeticTokenizer(data_dir=data_dir)
    eq_token_id = tokenizer.stoi[EQ_TOKEN]
    

    #Initiate the model and move it to the propper device
    config = GPT2Config(
                vocab_size = len(tokenizer),
                n_positions = 49,
                n_emb = 128,
                n_layer = 2,
                n_head = 4,
                eos_token_id= tokenizer.stoi[EOS],
                bos_token_id= tokenizer.stoi[EOS],
            ) 
    model = GPT2LMHeadModel(config)
    model.to(device)
    
    #Set up the optimizer
    total_steps = epochs * len(tr_iterator)
    optimizer = AdamW(model.parameters(), lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmap_steps, num_training_steps=total_steps)
    
    #TODO: Implement the creationn of the plot to track the validation and training accuracy

    #Start the training loop
    global_step = 0
    save_path = os.path.join(data_dir, f"model_{VALID_OPERATORS[operator]}")
    
    for epoch in range(epochs):
        print(f"---------- Epoche {epoch} -------------")
        
        # set the model to training model 
        model.train()
        
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        num_train_batches = 0
        

        #Loop through batchs 
        for batch in tqdm(tr_iterator, ascii=True, desc=f"Epoch {epoch} progress: "):
            
            #Set the gradients to 0
            optimizer.zero_grad()

            #find the position of the last EQ_TOKEN, all sequences should be the same length and format
            last_eq_position = find_last_eq_pos(batch["text"], eq_token_id)

            #First approach to try: calculate the loss function based only on the token after the last eq sign.
            outputs = model(input_ids=batch["text"], attention_mask = torch.ones_like(batch["text"]))
            logits = outputs.logits

            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = torch.full_like(batch["target"], -100)
            for i in range(batch["text"].size(0)):
                labels[i, last_eq_position + 1] = batch["target"][i, last_eq_position + 1]

             

    return


train(0.5, "+", "data", 0, 0, 0.9, 10, 80)
