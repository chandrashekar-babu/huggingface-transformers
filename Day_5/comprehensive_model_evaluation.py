def evaluate_model_comprehensive(model, tokenizer, test_dataset, task_type):
    """
    Comprehensive evaluation based on task type
    """
    from evaluate import load
    import numpy as np
    from tqdm.auto import tqdm
    
    results = {}
    
    if task_type == "classification":
        accuracy = load("accuracy")
        f1 = load("f1")
        precision = load("precision")
        recall = load("recall")
        
        all_preds, all_labels = [], []
        model.eval()
        
        for batch in tqdm(test_dataset, desc="Evaluating"):
            with torch.no_grad():
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        results["accuracy"] = accuracy.compute(
            predictions=all_preds, references=all_labels
        )["accuracy"]
        results["f1"] = f1.compute(
            predictions=all_preds, references=all_labels, average="weighted"
        )["f1"]
        results["precision"] = precision.compute(
            predictions=all_preds, references=all_labels, average="weighted"
        )["precision"]
        results["recall"] = recall.compute(
            predictions=all_preds, references=all_labels, average="weighted"
        )["recall"]
    
    elif task_type == "generation":
        rouge = load("rouge")
        bleu = load("bleu")
        
        all_preds, all_refs = [], []
        model.eval()
        
        for batch in tqdm(test_dataset, desc="Evaluating"):
            with torch.no_grad():
                outputs = model.generate(**batch['input_ids'])
                preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                refs = batch['references']
                all_preds.extend(preds)
                all_refs.extend(refs)
        
        results["rouge"] = rouge.compute(
            predictions=all_preds, references=all_refs
        )
        results["bleu"] = bleu.compute(
            predictions=all_preds, references=[[r] for r in all_refs]
        )
    
    return results, all_preds, all_labels if task_type == "classification" else all_refs