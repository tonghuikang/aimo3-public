Most notebooks get gpt-oss-120b to work on the problem independently, and implement some voting mechanism to decide which answer to submit.

My hypothesis is that I can beat the market if I enable multiple instances of gpt-oss-120b to collaborate with each other, by reading the results and findings of other solvers after submitting an answer.

The interaction is accessible at [aimo.huikang.dev/runs](http://localhost:4130/runs.html?datetime=2026-02-18-05-21-35&problem=86e8e5&copy=4&solution=0).
The visualization was built around Christmas last year. Just before publishing this writeup, I made it much easier to navigate.

---

Links

- Comment in submission thread: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/689703#3446458
- Notebook: https://www.kaggle.com/code/huikang/streaming-inference-private?scriptVersionId=298346744
- Model on Huggingface: TBC
- Model on Kaggle: https://www.kaggle.com/models/huikang/gpt-oss-120b-aimo3/Transformers/160a/9 as described in [Corpus Prize Submission](https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/discussion/672528)
- Writeup: https://www.kaggle.com/competitions/ai-mathematical-olympiad-progress-prize-3/writeups/attempts-to-get-gpt-oss-120b-to-collaborate

---

I assume that people are familiar with the popular [44-score notebook](https://www.kaggle.com/code/nihilisticneuralnet/44-50-let-me-over-cook). It has the following features.

- Use of gpt-oss-120b
- vLLM for inference
- The tool calling harness with the Jupyter client
- Early stopping once the majority vote is achieved

I now explain the additional features of my submission.

### Solver termination based on KV cache usage

When the KV cache is full, vLLM cannot generate any more tokens.
It picks one of the in-flight generations, pauses it, and frees up its KV cache. In the naive implementation, completions take turns to wait, and every time one exits the queue, the entire KV cache has to be prefilled with the O(n^2) computation. This becomes inefficient with more tool calls, as the prefix cache is frequently evicted.

I think it is more likely to get the answer if I have three solvers reaching the 60,000th token compared to eight solvers stuck at the 40,000th token. Therefore, I decided to terminate solvers when a threshold of the KV cache is reached.

I terminate the least-progressed solver first, with this priority:
- a solver that has not made any proposal yet
- a solver with no answer or proposal
- a solver with no answer
- the solver with the fewest tool calls


### Adaptive submission logic

In the widely-forked 44-score notebook, submission is made once there are 4 solvers agreeing on the answer.

I think I can implement a logic better than that popular notebook. Given that there are only 8 attempts, if three solvers agree on an answer, it is unlikely that after some time four of the other solvers will agree on a different answer. Therefore, I think we could make a submission as long as 3 solvers agree on the answer.

However, when I look at the submission logs, I do not think it is true that we can just submit when two solvers agree on the answer.
It depends on whether the answer is unanimous. It also depends on how much time we have left.

My submission entry has this answer-submission logic, where it will submit the answer if
- two solvers have submitted an answer and they agree on the answer
- there are three solvers remaining (due to the solver termination logic) and two solvers agree on the answer
- there are three solvers remaining (due to the solver termination logic) and only one solver has a submission

In addition, for the last question, I allow the agent to spend all the remaining time on the problem until all solvers submit the same answer.


### Information sharing between solvers

The main bet I am making is that collaboration between solvers will make it more likely to get the correct answer, as they can compare their answers with each other.

This is an example of a solver being [presented](http://localhost:4130/runs.html?datetime=2026-02-18-05-21-35&problem=86e8e5&copy=4&solution=0) with other answers after submitting its own.

The hope is that the solver will fix its answer based on the information provided by the other solvers.

```
assistant
final
\boxed{41754}
user
These are the currently submitted answers:
Solver 0 (you) is considering 41754
Solver 2 is submitting 8687
Solver 3 is submitting 8687
Solver 5 is submitting 41754

Another solver has provided this solution:

Insights: f(n)=min LCM(a,b,c) with a+b+c=n ⇒ f(n)/n = min_{d|n, d≥6} r(d), where r(d)=2/3−2/(3d) if d≡1(mod 3), r(d)=2/3−4/(3d) if d≡2(mod 3), r(d)=2/3 for d multiple of 3 (≥9).  

M=3^{2025!} ⇒ M≡1 (mod d) for d∤3.  

For each c: find smallest divisor ≥6 of N=M+c in each residue class; compute r.  

Results:  
g(0)=2/3 (d=9); g(4M)=5·2/3=10/3; g(1848374)=16/25 (d=25); g(10162574)=30/47 (d=47); g(265710644)=64/97 (d=97); g(44636594)=110/167 (d=167).  

Sum = 125561848/19033825 ⇒ (p+q) mod 99991 = **8687**.  

Pitfalls: ignore factor (1+c/M) for c=4M (g(4M)≠2/3) and ensure d≥6, distinct‑divisor condition.

Scrutinize your solution, using other solutions as a reference.
If you spot any critical mistake in your solution, work towards figuring out the correct answer.
Prioritize scrutinizing your solution so you can find any mistakes as soon as possible.
```

The summary from the other solvers is produced by gpt-oss-120b itself in a separate LLM call. When a solver proposes or updates an answer, I take a copy of its conversation tokens, append a prompt asking it to summarize its own current approach in under 250 characters (covering key insights, intermediate results, and pitfalls), and prefix the assistant turn with `Insights:`.
The resulting string is what other solvers see as "Another solver has provided this solution". The `Insights:` and `Pitfalls:` labels in the example above come from the model following this template.

If gpt-oss-120b arrived at the correct answer, presenting the summary of the wrong answer almost never causes gpt-oss-120b to adopt the wrong answer. If given 1 hour to arrive at a unanimous answer for Question 10, it seems that I can get 8687 [80% of the time](https://www.kaggle.com/code/huikang/streaming-inference-private/output?scriptVersionId=293581258&select=stats.csv).

I had previous iterations where the information was injected before the first answer was submitted.
However, this did not increase the probability of arriving at the correct answer in the same time budget.
I found that interrupting the chain of thought causes gpt-oss-120b to waste tokens. This is because on every interruption, gpt-oss-120b restarts its chain of thought from scratch, rather than continuing from where it left off. This makes it difficult to inject new information in real time. Therefore, I ended up injecting the information only after the first answer is submitted, to ensure that the performance is not worse than a setup with no collaboration.


### Answer deletion logic

I allow solvers to retract their submission if they find their submitted answer is wrong.

This makes unanimous agreement more likely, and hopefully the unanimously agreed answer is the correct answer.
If unanimous agreement is not achieved before time runs out, the retractions remove the wrong answers from participating in the voting process.

The retraction decision itself is made by gpt-oss-120b in a separate LLM call. I append the solver's conversation tokens with a prompt asking it to reply with either `I have confirmed that {answer} is wrong` or `I have yet to confirm that {answer} is wrong`, and string-match the response.

This prompt-based classifier seems to be unreliable.
Therefore, for an answer to be deleted, I require two consecutive retraction decisions.
The first decision marks the answer as pending retraction in its history. Only a second retraction decision (with no intervening confirmation) actually removes the answer from the voting pool.


### Backtracking logic

After a solver submits an answer, the conversation continues and the solver often re-examines its work in the next iteration.
The first time this happens, it is very likely that the solver will spend real effort checking the answer.

However, if the solver keeps arriving at the same answer, it may simply restate its previous conclusion and stop spending effort on verification.
This behavior is undesirable.

Therefore, I implemented a backtracking logic.
If the solver submits the same answer two iterations in a row, I rewind its conversation by popping tokens off the end of its token stream until two assistant turns have been removed — effectively returning it to the state just before the iteration that first produced that answer. Generation then resumes from that earlier point.

This makes it more likely that the solver will continue to spend effort checking the answer, hopefully figuring out that the answer it has submitted is wrong, and finding the correct answer instead.


---

### Results

Most of the changes were done by analyzing the errors and making changes to the code to address them.
I do not have the GPU budget to run ablation tests.

8 submissions were made for the notebook version.
The scores, in chronological order, are 42, 40, 40, 36, 39, 40, 40, and 41.
The scores have a mean of 39.75.

---

Initially, I did this so that I would have a good chance at the Longest Leader Prize. I did not win the prize. It seems that what was required is a clean implementation of gpt-oss-120b, as seen from the notebooks that first scored between 41 and 44.

I thought I could achieve a significantly better score than the 44-score notebook, but I could not prove this, even though my changes made the model perform much better on Question 9 and Question 10. I realized that Questions 9 and 10 from the reference problems probably did not represent the problems in the leaderboard. The two questions involve multiple parts, which makes it easy for gpt-oss-120b to be correct in one part but wrong in another. Collaboration is especially useful, as I can allow gpt-oss-120b to compare notes and figure out who is correct. Perhaps most problems in the public leaderboard do not benefit from comparing notes.
