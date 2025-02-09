import re
import pandas as pd

# The user prompt used for experiment without the system prompt, thus no CoT and provided expected structure
user_prompt="""
Given the provided premise, apply the PC (Peter-Clark) algorithm to compute the causal undirected skeleton.
Please present the computed causal undirected skeleton as an incident graph representation, following the format of the example below:
Answer:
  - Node A is connected to nodes B, C, D.
  - Node B is connected to nodes A, C.
  - Node C is connected to nodes A, B.
  - Node D is connected to node A.
"""

# The user prompt used for experiment without system prompt, but still with a few-shot examples with only the expected structure
user_prompt_with_structure="""
Given the provided premise, apply the PC (Peter-Clark) algorithm to compute the causal undirected skeleton.
Please present the computed causal undirected skeleton, following the format from the provided examples.
"""

def extract_premise(text):
    """
    Extracts the premise from the input text.

    Parameters:
    - text (str): The input text containing "Premise:" and possibly "Hypothesis:".

    Returns:
    - str: The extracted premise.
    """
    # Use regex to extract text between "Premise:" and "Hypothesis:"
    match = re.search(r"Premise:\s*(.*?)\s*Hypothesis:", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    else:
        # If "Hypothesis:" is not present, extract everything after "Premise:"
        match = re.search(r"Premise:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        else:
            # If "Premise:" is not found, return the original text
            return text.strip()


def generate_few_shot_prompt(df, num_examples=3, task_statement=None, random_state=None):
    """
    Generates a few-shot Chain of Thought prompt from a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - num_examples (int): Number of few-shot examples to include in the prompt.
    - task_statement (str): Task statement to include in the prompts.
    - random_state (int): Seed for reproducibility.

    Returns:
    - dict: Dictionary containing standard prompts, single-line prompts, new question details, and selected sample indices.
    """
    # Used for experiment where we are leveraging the CoT in the system prompt
    task_statement = "Given the provided premise, apply the PC (Peter-Clark) algorithm following the described step-by-step instructions to compute the causal undirected skeleton."

    # In case if no system prompt is provided, we can use the user prompt defined before
    # task_statement = user_prompt


    # Select the sample rows for a few-shot examples
    examples = df.sample(num_examples, random_state=random_state)

    standard_prompt = []
    # single_line_prompt = []
    question_counter = 1

    for index, row in examples.iterrows():
        # Print the actual index of the sample to console
        print(f"Selected sample index for Question {question_counter}: {index}")

        raw_input = row['input']
        premise = extract_premise(raw_input)
        answer = row['expected_answer']
        # single_line_answer = row.get('expected_answer_single_line', '')

        standard_prompt.append(f"Example {question_counter}:\n")
        standard_prompt.append(f"Task Description: {task_statement}")
        standard_prompt.append(f"Premise: {premise}")
        standard_prompt.append(f"Answer:")
        # standard_prompt.append(f"Answer {question_counter}:")
        standard_prompt.append(answer)
        standard_prompt.append("")  # Add an empty line for separation

        # Construct the single-line prompt with '\n' as newline indicators
        # single_line_part = (
        #     f"Question {question_counter}:\\n"
        #     f"Premise: {premise.replace('\\n', '\\\\n')}\\n"
        #     f"Task: {task_statement}\\n"
        #     f"Answer {question_counter}:\\n"
        #     f"{single_line_answer}"
        # )
        # single_line_prompt.append(single_line_part)

        question_counter += 1

    # Print the index of the new question to console
    new_index = df.sample(1, random_state=random_state).index[0]
    print(f"Selected sample index for new question: {new_index}")

    # Add a new question for the model to answer
    new_input = df.loc[new_index, 'input']
    new_premise = extract_premise(new_input)
    new_premise_escaped = new_premise.replace('\n', '\\n')

    # Standard prompt
    # standard_prompt.append(f"Example {question_counter}:\n")
    standard_prompt.append(f"Question:\n")
    standard_prompt.append(f"Task Description: {task_statement}")
    standard_prompt.append(f"Premise: {new_premise}")
    standard_prompt.append(f"Answer:")
    # standard_prompt.append(f"Answer {question_counter}:")
    # Leave the answer blank for the model to generate

    # Single-line prompt
    # single_line_new = (
    #     f"Question {question_counter}:\\n"
    #     f"Premise: {new_premise_escaped}\\n"
    #     f"Task: {task_statement}\\n"
    #     f"Answer {question_counter}:"
    # )
    # single_line_prompt.append(single_line_new)

    return {
        "standard_prompt": standard_prompt,
        # "single_line_prompt": single_line_prompt,
        "new_question_index": new_index,
        "selected_sample_indices": examples.index.tolist(),
    }


def save_prompts_to_files(standard_prompt, standard_output_path, single_line_prompt = None, single_line_output_path = None):
    """
    Saves the standard and single-line prompts to their respective files.

    Parameters:
    - standard_prompt (list): List of standard prompt strings.
    - single_line_prompt (list): List of single-line prompt strings.
    - standard_output_path (str): Path to save the standard prompt file.
    - single_line_output_path (str): Path to save the single-line prompt file.
    """
    with open(standard_output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(standard_prompt))
    print(f"Standard few-shot prompt saved to {standard_output_path}")

    # with open(single_line_output_path, 'w', encoding='utf-8') as f:
    #     f.write(" ".join(single_line_prompt))
    # print(f"Single-line few-shot prompt saved to {single_line_output_path}")


def main():
    csv_file_path = "data/v0.0.4/train.csv"
    standard_prompt_file_path = "few_shot_prompt.txt"
    # single_line_prompt_file_path = "few_shot_prompt_single_line.txt"

    df =  pd.read_csv(csv_file_path)

    result = generate_few_shot_prompt(df)
    standard_prompt = result["standard_prompt"]
    # single_line_prompt = result["single_line_prompt"]

    save_prompts_to_files(standard_prompt, standard_prompt_file_path)


if __name__ == "__main__":
    main()
