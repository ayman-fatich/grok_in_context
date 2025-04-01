from data import ArithmeticTokenizer, ArithmeticIterator, ArithmeticDataset, VALID_OPERATORS, EQ_TOKEN, EOS
import torch
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW



def train(
        train_pct: float,
        operator: str,
        data_dir: str,
        tr_in_context: int,
        val_in_context:int,
        lr: float,
        warmap_steps: int,
        epoches:int,
        
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
    total_steps = epoches * len(tr_iterator)
    optimizer = AdamW(model.parameters(), lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmap_steps, num_training_steps=total_steps)


    return


train(0.5, "+", "data", 0, 0, 0.9, 10, 80)
