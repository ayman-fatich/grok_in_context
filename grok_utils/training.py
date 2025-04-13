from grok_utils.data import ArithmeticTokenizer, ArithmeticIterator, ArithmeticDataset, VALID_OPERATORS, EQ_TOKEN, EOS
import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from grok_utils.visualization import log_metrics, plot_metrics



def find_last_eq_pos(batch, eq_token_id):
    """Function to find the position of the last equal sign of the sequences in a batch"""
    eq_positions = (batch[0]==eq_token_id).nonzero(as_tuple=True)[0]
    return eq_positions

def calculate_accuracy(logits, target, eq_positions):
    """calculates the accuracy of the model on a training batch"""
    batch_size = logits.size(0)
    pred = torch.argmax(logits, dim=-1)
    total_correct = 0

    for pos in eq_positions:
        total_correct += (pred[:, pos] == target[:, pos]).sum().item()
    return total_correct/(len(eq_positions)*batch_size)

def validation(model, val_iterator, eq_token_id):
    """returns the accuracy for the validation dataset"""
    model.eval()
    accuracy = 0

    with torch.no_grad():
        #Loop through batchs 
        for batch in val_iterator:
            #find the position of the last EQ_TOKEN, all sequences should be the same length and format
            eq_positions = find_last_eq_pos(batch["text"], eq_token_id)

            #First approach to try: calculate the loss function based only on the token after the last eq sign.
            outputs = model(input_ids=batch["text"], attention_mask = torch.ones_like(batch["text"]))
            logits = outputs.logits

            acc = calculate_accuracy(logits, batch["target"], eq_positions)
            accuracy += acc

        accuracy = accuracy /len(val_iterator) 
 
    return accuracy


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
    val_iterator = ArithmeticIterator(val_ds, device, True)
    
    #Create the tokenizer and get the "=" token id
    tokenizer = ArithmeticTokenizer(data_dir=data_dir)
    eq_token_id = tokenizer.stoi[EQ_TOKEN]
    

    #Initiate the model and move it to the propper device
    config = GPT2Config(
                vocab_size = len(tokenizer),
                n_positions = 49,
                n_embd = 128,
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
    

    #Start the training loop
    global_step = 0
    save_path = os.path.join(data_dir, f"model_{VALID_OPERATORS[operator]}")
    stop=20
    val_acc=0
    outerloop:
        for epoch in range(epochs):
            
            # set the model to training model 
            model.train()
            
            epoch_train_loss = 0
            epoch_train_accuracy = 0
            num_train_batchs = 0
            

            #Loop through batchs 
            for batch in tr_iterator:
                
                #Set the gradients to 0
                optimizer.zero_grad()

                #find the positions of the EQ_TOKEN, all sequences should be the same length and format
                eq_positions = find_last_eq_pos(batch["text"], eq_token_id)

                #First approach to try: calculate the loss function based only on the token after the last eq sign.
                outputs = model(input_ids=batch["text"], attention_mask = torch.ones_like(batch["text"]))
                logits = outputs.logits

                loss_fct = CrossEntropyLoss(ignore_index=-100)
                labels = torch.full_like(batch["target"], -100)
                for i in range(batch["text"].size(0)):
                    for pos in eq_positions:
                        labels[i, pos] = batch["target"][i, pos]

                logits = logits.contiguous()
                mask = labels.contiguous()

                loss = loss_fct(logits.view(-1, logits.size(-1)), mask.view(-1))
                epoch_train_loss += loss.item()
                
                tr_acc = calculate_accuracy(outputs.logits, batch["target"], eq_positions)
                epoch_train_accuracy += tr_acc
                #backprop and optim
                loss.backward()
                optimizer.step()
                scheduler.step()
                num_train_batchs += 1
                global_step += 1

                # Log and plot metrics
                if global_step % 10 == 0:
                    val_acc = validation(model, val_iterator, eq_token_id)

                    log_metrics(data_dir, VALID_OPERATORS[operator], train_pct, tr_in_context, val_in_context, lr, 
                           global_step, tr_acc, val_acc)
             
                if tr_acc >= 0.95 and tr_acheived_95 == 0:
                    tr_acheived_95 = global_step
                if val_acc >= 0.95:
                    val_acheived_95 = global_step
                    break outerloop

                    
            epoch_train_loss = epoch_train_loss / num_train_batchs
            epoch_train_accuracy = epoch_train_accuracy / num_train_batchs

            
            print(f"Epoch {epoch}: train loss: {round(epoch_train_loss,3)} Epoch train accuracy: {round(epoch_train_accuracy,3)} Epoch Validation accuracy: {round(val_acc, 3)}")

    plot_metrics(data_dir, VALID_OPERATORS[operator], train_pct, tr_in_context, val_in_context, lr, val_acheived_95 - tr_acheived_95)

    return model


