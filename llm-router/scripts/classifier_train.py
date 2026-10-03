#!/usr/bin/env python3
"""Generate the family split and deterministic hashed-linear classifier artifact."""
from __future__ import annotations
import hashlib, json, math, pathlib, random, re

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "classifier"
MODEL = ROOT / "assets" / "classifier" / "linear-v1.json"
BUCKETS = 8192
LABELS = ["code_generation", "code_understanding", "technical_design", "analytical_reasoning", "writing", "factual_lookup", "general"]

# Each tuple is one scenario family. Variants stay together in train or validation.
SEEDS = {
"code_generation": [
 ("Implement a {lang} function that parses CSV safely",3),("Fix the off-by-one bug in this {lang} loop",2),
 ("Refactor this API handler to return typed errors",3),("Add retries and exponential backoff to the client",3),
 ("Write tests for the authentication middleware",2),("Change this constant and rename the helper",1),
 ("Build a command line tool for batch image conversion",3),("Complete the missing method in this class",2),
 ("Optimize this database query and update the code",4),("Create a concurrent worker pool with graceful shutdown",4),
 ("Patch the null pointer crash shown in this stack trace",3),("Generate a migration script for these columns",2)],
"code_understanding": [
 ("Explain what this {lang} function does",1),("Review this code for security vulnerabilities",3),
 ("Find the race condition in this concurrent workflow",5),("Why does this query return duplicate rows",3),
 ("Trace how authentication flows through this repository",4),("Summarize the behavior of this class without changing it",2),
 ("Identify the bug in this stack trace",3),("What is the time complexity of this implementation",3),
 ("Audit this diff for regressions",4),("Explain why this test is flaky",4),
 ("Walk me through this unfamiliar module",2),("Review this SQL execution plan",4)],
"technical_design": [
 ("Design an API for multi-tenant file storage",4),("Propose an architecture for online feature serving",5),
 ("How should we migrate this monolith to services",4),("Compare queue based designs for reliable delivery",4),
 ("Plan a zero downtime database migration",5),("Design caching and invalidation for this workload",4),
 ("Choose a schema for versioned audit events",3),("Create a disaster recovery strategy across regions",5),
 ("How should these components communicate",3),("Design authorization boundaries for tenant isolation",5),
 ("Plan capacity and failover for the new service",5),("Recommend an observability architecture",4)],
"analytical_reasoning": [
 ("Compare transactional outbox and CDC for this failure model",4),("Solve this probability puzzle step by step",3),
 ("Determine why these metrics contradict each other",4),("Evaluate the tradeoffs between consistency and availability",4),
 ("Calculate the expected value of this decision",3),("Analyze which hypothesis best explains the evidence",4),
 ("Prove that this algorithm terminates",5),("Reconcile these invoices and identify the discrepancy",4),
 ("Rank these options under the stated constraints",3),("Derive the recurrence and solve it",4),
 ("Diagnose the root cause from these distributed traces",5),("Assess whether this experiment supports the conclusion",3)],
"writing": [
 ("Draft an email announcing the maintenance window",2),("Rewrite this paragraph for clarity and grammar",1),
 ("Summarize these release notes for customers",2),("Create a concise product description",2),
 ("Translate this message into professional English",1),("Write a blog post about remote collaboration",3),
 ("Turn these notes into meeting minutes",2),("Edit this report to use an executive tone",2),
 ("Compose a thank you letter",1),("Prepare a structured project proposal",3),
 ("Shorten this copy without losing key details",2),("Draft documentation for new users",3)],
"factual_lookup": [
 ("What is the capital of {place}",1),("Define idempotency",1),("When was the PostgreSQL project founded",1),
 ("Who wrote The Dispossessed",1),("What does HTTP status 429 mean",1),("List the planets in the solar system",1),
 ("Which port does HTTPS use by default",1),("What is the boiling point of water",1),
 ("Name the creator of the Python language",1),("What does ACID stand for",1),
 ("Give the current formula for compound interest",1),("Which country uses the yen",1)],
"general": [
 ("Hello how are you",1),("Thanks that was helpful",1),("Tell me a joke",1),("What can you help me with",1),
 ("Good morning",1),("I am bored",1),("That sounds interesting",1),("Can we chat for a minute",1),
 ("Recommend a fun weekend activity",1),("What is your favorite color",1),("Please continue",1),("Okay got it",1)],
}

VARIANTS = [
 lambda q: q,
 lambda q: "Please " + q[0].lower() + q[1:] + ".",
 lambda q: "Task: " + q.replace("this", "the supplied").replace(" a ", " one ", 1),
 lambda q: "Context: production system. " + q,
 lambda q: "plz " + q.lower().replace("please", "").replace("function", "fn").lstrip(),
]

def fnv(data: bytes) -> int:
    h=0xcbf29ce484222325
    for b in data: h=((h ^ b)*0x100000001b3)&0xffffffffffffffff
    return h
def add(out, name):
    b=fnv(name.encode()) & (BUCKETS-1); sign=1.0 if fnv(("sign:"+name).encode())&1==0 else -1.0
    out[b]=out.get(b,0.0)+sign
def add_text(out,prefix,text):
    low=text.lower(); words=re.findall(r"[^\W_]+",low,re.UNICODE)
    for w in words:add(out,f"{prefix}w:{w}")
    for a,b in zip(words,words[1:]):add(out,f"{prefix}b:{a}_{b}")
    chars=list(low)
    for n in range(3,6):
        for i in range(len(chars)-n+1):add(out,f"{prefix}c:{''.join(chars[i:i+n])}")
def features(row):
    out={}; q=row["query"]; add_text(out,"q:",q)
    if row.get("context"):add_text(out,"ctx:",row["context"])
    else:add(out,"shape:no_context")
    tests={"code_fence":"```" in q,"inline_code":"`" in q,"question":"?" in q,
           "json":"{" in q and ":" in q,"stack_trace":"Traceback" in q or " at " in q,
           "multiline":len(q.splitlines())>2,"many_constraints":q.count(",")>=3}
    for k,v in tests.items():
        if v:add(out,"shape:"+k)
    low=q.lower()
    groups={
      "intent_change":["implement","build","fix","patch","refactor","add","change","complete","generate"],
      "intent_understand":["explain","review","audit","trace","walk me","why does","identify"],
      "intent_design":["design","architecture","migrate","strategy","capacity","schema","components","queue based"],
      "intent_reason":["compare","analyze","evaluate","calculate","prove","derive","diagnose","reconcile","tradeoff","investigate","infer","invariant","event by event","root cause"],
      "intent_write":["draft","rewrite","summarize","compose","edit","translate","blog","minutes","description","copy","documentation","report","email","letter","notes"],
      "intent_fact":["what is","who ","when ","which ","define","what does","name the","list the","formula","stand for"],
      "intent_general":["hello","thanks","joke","chat","help me","continue","good morning","favorite","bored","got it"],
      "risk_high":["distributed","concurrent","failover","disaster","zero downtime","security","production","multi-tenant"]}
    for name,terms in groups.items():
        if name=="intent_design" and ("do not design" in low or "do not redesign" in low):continue
        if any(term in low for term in terms):add(out,"intent:"+name)
    if any(term in low for term in ["rewrite","grammar","tone","warmer","shorten","copy"]):add(out,"intent:text_rewrite")
    norm=max(1.0,math.sqrt(sum(abs(v) for v in out.values())))
    return {k:v/norm for k,v in out.items()}

def rows():
    train=[]; val=[]
    for label,seeds in SEEDS.items():
        for family,(base,complexity) in enumerate(seeds):
            for variant,transform in enumerate(VARIANTS):
                query=transform(base.format(lang=["Rust","Python","TypeScript"][family%3],place=["France","Japan","Kenya"][family%3]))
                context=None
                if variant==3: context="The request follows an existing production discussion with reliability constraints."
                row={"id":f"{label}-{family:02d}-{variant}","family":f"{label}-{family:02d}","query":query,"context":context,"request_type":label,"complexity":complexity}
                (val if family in (3,10) else train).append(row)
    return train,val

def softmax(z):
    m=max(z); e=[math.exp(v-m) for v in z]; s=sum(e); return [v/s for v in e]
def train_head(data, targets, classes, epochs=100, lr=.28, l2=2e-4):
    w=[[0.0]*BUCKETS for _ in range(classes)]; b=[0.0]*classes; rng=random.Random(20261003)
    for epoch in range(epochs):
        order=list(range(len(data))); rng.shuffle(order); rate=lr/(1+epoch*.025)
        for i in order:
            x=data[i]; y=targets[i]; p=softmax([b[c]+sum(w[c][j]*v for j,v in x.items()) for c in range(classes)])
            for c in range(classes):
                g=p[c]-(1.0 if c==y else 0.0); b[c]-=rate*g
                for j,v in x.items():w[c][j]-=rate*(g*v+l2*w[c][j])
    return w,b
def train_binary(data, targets, epochs=80, lr=.22, l2=2e-4):
    heads=[]; biases=[]
    for threshold in range(1,5):
        w=[0.0]*BUCKETS;b=0.0;rng=random.Random(20261003+threshold)
        for epoch in range(epochs):
            order=list(range(len(data)));rng.shuffle(order);rate=lr/(1+epoch*.03)
            for i in order:
                x=data[i]; y=1.0 if targets[i]>threshold else 0.0; z=b+sum(w[j]*v for j,v in x.items()); p=1/(1+math.exp(-max(-30,min(30,z))));g=p-y;b-=rate*g
                for j,v in x.items():w[j]-=rate*(g*v+l2*w[j])
        heads.append(w);biases.append(b)
    return heads,biases
def sparse(rows):return [{str(i):round(v,7) for i,v in enumerate(row) if abs(v)>=1e-6} for row in rows]
def accuracy(rows_, w,b,temp=1.0):
    ok=0; conf=[]
    for row in rows_:
        x=features(row);p=softmax([(b[c]+sum(w[c][j]*v for j,v in x.items()))/temp for c in range(7)]); pred=max(range(7),key=p.__getitem__);good=pred==LABELS.index(row["request_type"]);ok+=good;conf.append((max(p),good))
    ece=0
    for lo in [i/10 for i in range(10)]:
        bin_=[x for x in conf if lo<=x[0]<(lo+.1 if lo<.9 else 1.0001)]
        if bin_:ece+=len(bin_)/len(conf)*abs(sum(x[0] for x in bin_)/len(bin_)-sum(x[1] for x in bin_)/len(bin_))
    return ok/len(rows_),ece
def complexity_metrics(rows_, w, b):
    exact=within=0; error=0
    for row in rows_:
        x=features(row); predicted=1
        for head in range(4):
            z=b[head]+sum(w[head][j]*v for j,v in x.items())
            predicted += 1 if z >= 0 else 0
        distance=abs(predicted-row["complexity"]); exact += distance==0; within += distance<=1; error += distance
    n=len(rows_); return exact/n,within/n,error/n
def main():
    train,val=rows();DATA.mkdir(parents=True,exist_ok=True);MODEL.parent.mkdir(parents=True,exist_ok=True)
    train_families={row["family"] for row in train}; val_families={row["family"] for row in val}
    if train_families & val_families: raise RuntimeError("scenario-family leakage across splits")
    normalize=lambda text: " ".join(re.findall(r"[^\W_]+",text.lower(),re.UNICODE))
    train_queries={normalize(row["query"]) for row in train}; val_queries={normalize(row["query"]) for row in val}
    if len(train_queries)!=len(train) or len(val_queries)!=len(val): raise RuntimeError("normalized duplicate inside a split")
    if train_queries & val_queries: raise RuntimeError("normalized duplicate across splits")
    if set(LABELS)!={row["request_type"] for row in train} or set(range(1,6))!={row["complexity"] for row in train}: raise RuntimeError("label coverage is incomplete")
    for name,part in (("train.jsonl",train),("validation.jsonl",val)):
        (DATA/name).write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in part),encoding="utf-8")
    x=[features(r) for r in train]; y=[LABELS.index(r["request_type"]) for r in train]
    tw,tb=train_head(x,y,7);cw,cb=train_binary(x,[r["complexity"] for r in train])
    temperature=min([.7,.85,1.0,1.15,1.3,1.5,1.75,2.0],key=lambda t: accuracy(val,tw,tb,t)[1])
    artifact={"schema":"nasiko-request-classifier-linear-v1","buckets":BUCKETS,"labels":LABELS,"type_bias":[round(v,7) for v in tb],"type_weights":sparse(tw),"complexity_bias":[round(v,7) for v in cb],"complexity_weights":sparse(cw),"temperature":temperature,"provenance":{"seed":20261003,"train_rows":len(train),"validation_rows":len(val),"split":"scenario-family","train_sha256":hashlib.sha256((DATA/"train.jsonl").read_bytes()).hexdigest()}}
    MODEL.write_text(json.dumps(artifact,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8")
    acc,ece=accuracy(val,tw,tb,temperature);ce,cw1,cmae=complexity_metrics(val,cw,cb)
    print(json.dumps({"train":len(train),"validation":len(val),"accuracy":acc,"ece":ece,"complexity_exact":ce,"complexity_within_1":cw1,"complexity_mae":cmae,"temperature":temperature,"model_bytes":MODEL.stat().st_size},indent=2))
if __name__=="__main__":main()
