# CORR2CAUSE dataset LLM fine-tuning

## General information

This is a note file. It contains general information about the project, learnings and other useful information collected
during the work.

## `causal-learn`'s package PC algorithm

The `causal-learn` is a Python package that implements the PC algorithm for causal discovery. 
The PC algorithm is designed to work with **actual data** and **statistical tests**, thus using it directly
with only specified independence tests from `CORR2CASUE` is not possible.

Why `PC` algorithm is not directly applicable to the `CORR2CAUSE` dataset:
* Algorithm cannot compute and perform statistical tests without data, thus it cannot compute p-values, determine significant etc.
* There is no built-in mechanism to input specified independencies directly.
* There might be option to create a custom CI test function which integrates into `PC` interface, but that might not be the easiest option to do.
* The algorithm's internal logic is tightly coupled with data-driven statistical testing procedures.

### `PC`'s algorithm data

The **actual data** refers to a dataset which contains observations (samples) of the variables of interest.
Data for variables A, B, C, D can be represented as follows:

| Sample |  A   |  B   |  C   |  D   |
|--------|------|------|------|------|
|   1    | 2.3  | 1.5  | 0.7  | 3.1  |
|   2    | 2.1  | 1.7  | 0.6  | 2.9  |
|   3    | 2.4  | 1.6  | 0.8  | 3.0  |
|  ...   | ...  | ...  | ...  | ...  |

### `PC`'s statistical tests

Statistical tests are procedures that use **actual data** to determine whether certain statistical relationships hold
between the variables. The `PC` algorithm uses statistical tests to determine the presence of edges between variables.
The algorithm requires data to compute values such as correlation coefficients and p-values that are used to decide
whether to accept or reject the null hypothesis of independence between variables.

In causal discovery, we often use the **conditional independence (CI) tests** to determine whether two variables 
are independent, possibly conditioning on other variables.

## Input from `CORR2CAUSE` dataset

The input from the dataset consists of specified **independence tests and variable dependencies** among variables, such as:
* "A is independent of B."
* "A and D are independent given C."

These are **qualitative statements** about the relationships between variables. They are outcomes of statistical tests
but not are derived from actual data in the given context.

Key outcomes:
* No actual data: there is no numerical data for all the variables.
* No statistical testing needed: the independencies are given, we do not need to perform tests to discover them.
* Specified relationships: the relationships are given in a qualitative form, there is again no need to to infer those relationships.

## Custom `PC` implementation

We will perform the sames sequence as used in the `PC` algorithm, but given the delivered input context.
The specified independencies will be used to construct the causal skeleton, replicating the logic of the `PC` algorithm
but without phase of executing statistical tests. The output will be detailed step-by-step reasoning based on the given 
independencies.

How it works:
* Work without data, using specified independencies.
* Uses provides independencies as absolute truth.
* Generally, we have a prior knowledge of theory, and there is no data to test these relationships, they are assumed to be true.
* Tailored for situation where only dependencies are specified.

### Step of execution

1. Edge initialization: create a complete graph with all possible edges between variables given correlation statements.
2. Edge removal: remove edges based on specified marginal and conditional independencies.
3. Output: return the causal skeleton based on the specified independencies, with provided detail reasoning
