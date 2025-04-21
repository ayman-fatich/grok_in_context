from grok_utils.data import ArithmeticTokenizer, ArithmeticIterator, ArithmeticDataset, VALID_OPERATORS, EQ_TOKEN, EOS
import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from grok_utils.visualization import log_metrics, plot_metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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


def visualize_embeddings_tsne(model, dataset, tokenizer, output_dir, in_context_count, epoch):
    """Create t-SNE visualization of the embedding space"""
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Get all the tokens
    all_tokens = set()
    for i in range(len(dataset)):
        seq = dataset.data[i].cpu().numpy()
        for token_id in seq:
            all_tokens.add(token_id.item())
    
    # Create input tensor with all tokens
    token_ids = torch.tensor(list(all_tokens), dtype=torch.long).to(device)
    
    # Get embeddings for all tokens
    with torch.no_grad():
        embeddings = model.transformer.wte(token_ids).detach().cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
    tsne_results = tsne.fit_transform(embeddings)
    
    # Create categories for visualization
    categories = []
    token_texts = []
    
    for token_id in token_ids.cpu().numpy():
        token_text = tokenizer.itos[token_id]
        token_texts.append(token_text)
        
        # Categorize token
        if token_text == EQ_TOKEN:
            categories.append("equals")
        elif token_text == EOS:
            categories.append("eos")
        elif token_text in VALID_OPERATORS:
            categories.append("operator")
        else:
            categories.append("number")
    
    # Define colors for categories
    category_colors = {
        "equals": "red",
        "eos": "black",
        "operator": "blue",
        "number": "green"
    }
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot by category
    for category in category_colors:
        indices = [i for i, cat in enumerate(categories) if cat == category]
        if indices:
            plt.scatter(
                tsne_results[indices, 0],
                tsne_results[indices, 1],
                c=category_colors[category],
                label=category,
                alpha=0.7
            )
    
    # Add label annotations for important tokens
    for i, txt in enumerate(token_texts):
        if categories[i] in ["equals", "eos", "operator"]:
            plt.annotate(txt, (tsne_results[i, 0], tsne_results[i, 1]))
    
    plt.title(f"t-SNE of Token Embeddings (In-Context: {in_context_count}, Epoch: {epoch})")
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"tsne_ic{in_context_count}_epoch{epoch}.png"), dpi=300)
    plt.close()


def train(
        train_pct: float,
        operator: str,
        data_dir: str,
        tr_in_context: int,
        val_in_context: int,
        lr: float,
        warmap_steps: int,
        epochs: int,
        seed: int = 42,
        tsne_dir: str = None  # Directory for t-SNE visualizations
):
    # Choose the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the datasets
    tr_ds, val_ds = ArithmeticDataset.splits(train_pct, operator, data_dir, tr_in_context, val_in_context, seed=seed)
    
    print(f"The device being used is: {device}")
    print(f"Training and validation DataSets were created successfully: Operation: {operator}, tr_in_context: {tr_in_context}, val_in_context: {val_in_context}")
    
    # Create the batch iterators 
    tr_iterator = ArithmeticIterator(tr_ds, device, True)
    val_iterator = ArithmeticIterator(val_ds, device, True)
    
    # Create the tokenizer and get the "=" token id
    tokenizer = ArithmeticTokenizer(data_dir=data_dir)
    eq_token_id = tokenizer.stoi[EQ_TOKEN]
    
    # Initiate the model and move it to the proper device
    config = GPT2Config(
                vocab_size = len(tokenizer),
                n_positions = 256,
                n_embd = 128,
                n_layer = 2,
                n_head = 4,
                eos_token_id= tokenizer.stoi[EOS],
                bos_token_id= tokenizer.stoi[EOS],
            ) 
    model = GPT2LMHeadModel(config)
    model.to(device)
    
    # Set up the optimizer
    total_steps = epochs * len(tr_iterator)
    optimizer = AdamW(model.parameters(), lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmap_steps, num_training_steps=total_steps)
    
    # Start the training loop
    global_step = 0
    save_path = os.path.join(data_dir, f"model_{VALID_OPERATORS[operator]}")
    tr_achieved_95 = 0
    val_achieved_95 = 0
    
    # Variables to track if 95% has been achieved
    tr_reached_95 = False
    val_reached_95 = False
    
    for epoch in range(epochs):
        # Set the model to training model 
        model.train()
        
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        num_train_batches = 0
        
        # Loop through batches 
        for batch in tr_iterator:
            # Set the gradients to 0
            optimizer.zero_grad()

            # Find the positions of the EQ_TOKEN, all sequences should be the same length and format
            eq_positions = find_last_eq_pos(batch["text"], eq_token_id)

            # Calculate the loss function based only on the token after the last eq sign
            outputs = model(input_ids=batch["text"], attention_mask=torch.ones_like(batch["text"]))
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
            
            # Backprop and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()
            num_train_batches += 1
            global_step += 1

            # Perform validation and log metrics every 10 steps
            if global_step % 10 == 0:
                val_acc = validation(model, val_iterator, eq_token_id)

                log_metrics(data_dir, VALID_OPERATORS[operator], train_pct, tr_in_context, val_in_context, lr, 
                       global_step, tr_acc, val_acc)
                
                # Check if training accuracy reached 95% for the first time
                if tr_acc >= 0.95 and not tr_reached_95:
                    tr_achieved_95 = global_step
                    tr_reached_95 = True
                    if tsne_dir:
                        visualize_embeddings_tsne(model, tr_ds, tokenizer, tsne_dir, tr_in_context, epoch)
                    print(f"Training accuracy reached 95% at step {global_step}")
                
                # Check if validation accuracy reached 95% for the first time
                if val_acc >= 0.95 and not val_reached_95:
                    val_achieved_95 = global_step
                    val_reached_95 = True
                    if tsne_dir:
                        visualize_embeddings_tsne(model, tr_ds, tokenizer, tsne_dir, tr_in_context, epoch)
                    print(f"Validation accuracy reached 95% at step {global_step}")
                
            
            # Stop if both training and validation have achieved 95% accuracy
            if tr_reached_95 and val_reached_95:
                print(f"Both training and validation have achieved 95% accuracy. Stopping training.")
                break
                
        if tr_reached_95 and val_reached_95:
            break
            
        epoch_train_loss = epoch_train_loss / num_train_batches
        epoch_train_accuracy = epoch_train_accuracy / num_train_batches
        
        val_acc = validation(model, val_iterator, eq_token_id)
        print(f"Epoch {epoch}: train loss: {round(epoch_train_loss,3)} Epoch train accuracy: {round(epoch_train_accuracy,3)} Validation accuracy: {round(val_acc, 3)}")
        
            
    # Calculate the difference between when validation and training reached 95%
    difference = val_achieved_95 - tr_achieved_95
    
    
    print(f"Training finished! Difference between validation and training reaching 95%: {difference}")
    plot_metrics(data_dir, VALID_OPERATORS[operator], train_pct, tr_in_context, val_in_context, lr, difference)

    return difference, model
