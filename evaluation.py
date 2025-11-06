import json, re
from typing import List, Dict
from difflib import SequenceMatcher
def load_eval_dataset(path=None):
    # default small QA set (10 QAs)
    dataset = [
        {"id":"q1","question":"What is the starting price of Product A?","answer":"$199"},
        {"id":"q2","question":"How long is the warranty for Product A?","answer":"1 year"},
        {"id":"q3","question":"Where to get firmware updates?","answer":"example.com/firmware"},
        {"id":"q4","question":"What should you do first when installing Product A?","answer":"Unbox the product"},
        {"id":"q5","question":"How to reset network issues?","answer":"Hold the button for 10s to reset network"},
        {"id":"q6","question":"Name one common issue with Product A.","answer":"network"},
        {"id":"q7","question":"Which feature categories does Product A support?","answer":"features X, Y, Z"},
        {"id":"q8","question":"Is Product A compact?","answer":"Yes"},
        {"id":"q9","question":"What is recommended for firmware?","answer":"visit example.com/firmware"},
        {"id":"q10","question":"What are core collaboration hours (sample)?","answer":"10 AM - 3 PM IST"}
    ]
    return dataset
def normalize_text(t):
    return re.sub(r'\s+',' ', t.strip().lower())
def exact_match(pred, gold):
    return 1 if normalize_text(pred) == normalize_text(gold) else 0
def f1_score(pred, gold):
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gold).split()
    if not p_tokens or not g_tokens:
        return 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0
    prec = len(common)/len(p_tokens)
    rec = len(common)/len(g_tokens)
    if prec+rec==0:
        return 0.0
    return 2*(prec*rec)/(prec+rec)
def faithfulness(pred, gold):
    # crude heuristic: measure overlap proportion
    p=set(normalize_text(pred).split()); g=set(normalize_text(gold).split())
    if not p: return 0.0
    return len(p & g)/len(p)
def run_evaluation(rag, dataset=None, top_k=3):
    if dataset is None:
        dataset = load_eval_dataset()
    results=[]
    for item in dataset:
        q=item['question']; gold=item['answer']
        # get answer
        q_emb = None
        try:
            if rag.hf_model:
                q_emb = rag.hf_model.encode(q, normalize_embeddings=True)
        except Exception:
            q_emb = None
        chunks = rag.retrieve(q_emb if q_emb is not None else q, top_k=top_k)
        pred = rag.generate_answer(q, chunks, llm_gen_fn=None)
        em = exact_match(pred, gold)
        f1 = f1_score(pred, gold)
        faith = faithfulness(pred, gold)
        results.append({'id':item['id'],'question':q,'gold':gold,'pred':pred,'EM':em,'F1':f1,'Faith':faith})
    # aggregate
    agg = {'EM': sum(r['EM'] for r in results)/len(results),
           'F1': sum(r['F1'] for r in results)/len(results),
           'Faith': sum(r['Faith'] for r in results)/len(results)}
    return {'aggregate':agg,'per_example':results}
