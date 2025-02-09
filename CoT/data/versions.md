- v0.0.0 - First iteration, having a long expected answer. Adjacency graph representation.  
- v0.0.1 - Second iteration, having a shorter expected answer and the one line prompt for university Llama3-8B experiments. 
Adjacency graph representation.  
- v0.0.2 - Introduced Incident graph representation only for the step 5 (the final answer).  
- v0.0.3 - Unified all the steps to the incident representation (expected output is now much longer).  
- v0.0.4 - Extracted cases with only 4 variables, continue with the incident representation.  
- v0.0.5 - Refactored the answer from 'Step 5: ...' to 'Answer:', in order to run experiments without system prompt with 
step-by-step instructions.  
- v0.0.6 - Revert the changes from v0.0.5, therefore provide the same structure as in the v0.0.4. However, problems with
5 variables has been extracted.
- v0.0.7 - Extract problem with 6 (most possible) number of variables. Structure remains same.
- v0.0.8 - Extract problem with 4 (least possible) number of variables. Delete the step-by-step CoT reasoning in the 
examples, and instead only provides computed final answer (causal undirected graph).