import os
import json
import random
from typing import Any
# import tiktoken
import argparse
import transformers
import yaml
import re
import torch
from datetime import datetime
from tqdm import tqdm
import hashlib
import csv
import uuid

from openai import OpenAI



class Evaluation:
    def __init__(self, args, cmd_args):
        self.args = args

        # Load API keys
        token_path = cmd_args.token_path

        if True or re.search(r'gpt', self.args['models']['llm_model']) is not None or re.search(r'o1', self.args['models']['llm_model']) or re.search(r'o3', self.args['models']['llm_model']):
            # with open(os.path.join(token_path, "openai_key.txt"), "r") as api_key_file:
            #     self.openai_key = api_key_file.read()
            self.openai_key = 'sk-123'
            self.client = OpenAI(api_key=self.openai_key, base_url=os.environ.get("OPENAI_API_BASE"))

        elif re.search(r'gemini', self.args['models']['llm_model']) is not None:
            with open(os.path.join(token_path, "gemini_key.txt"), "r") as genai_key_file:
                self.genai_key = genai_key_file.read()
            self.client = genai.Client(api_key=self.genai_key)

        elif re.search(r'claude', self.args['models']['llm_model']) is not None:
            with open(os.path.join(token_path, "claude_key.txt"), "r") as claude_key_file:
                self.claude_key = claude_key_file.read()
            self.client = anthropic.Client(api_key=self.claude_key)

        else:
            with open(os.path.join(token_path, "lambda_key.txt"), "r") as lambda_key_file:
                self.lambda_key = lambda_key_file.read()
            lambda_url = "https://api.lambdalabs.com/v1"
            self.client = OpenAI(api_key=self.lambda_key, base_url=lambda_url)


    def query_llm(self, question, all_options, context=None, instructions=None, verbose=False):
        assert context is None or isinstance(context, list), "Context must be a list of dictionaries"
        if instructions is None:
            instructions = "Find the most appropriate model response and give your final answer (a), (b), (c), or (d) after the special token <final_answer>."
        if context:
            messages = context + [{"role": "user", "content": question + '\n\n' + instructions + '\n\n' + all_options},]
        else:
            messages = [{"role": "user", "content": question + '\n\n' + instructions + '\n\n' + all_options},]

        # Call OpenAI API for GPT models by default
        if re.search(r'gpt', self.args['models']['llm_model']) is not None or re.search(r'o1', self.args['models']['llm_model']) or re.search(r'o3', self.args['models']['llm_model']):
            if 'o' in self.args['models']['llm_model']:
                messages = convert_role_system_to_user(messages)

            response = self.client.chat.completions.create(
                model=self.args['models']['llm_model'],
                messages=messages,
            )
            response = response.choices[0].message.content
            if verbose:
                print("model response: ", response)

        # Call Google Gemini API for Gemini models
        elif re.search(r'gemini', self.args['models']['llm_model']) is not None:
            messages = openai_to_gemini_history(messages)
            response = self.client.models.generate_content(
                model=self.args['models']['llm_model'], contents=messages
            )
            response = response.text

        # Call Claude API for Claude models
        elif re.search(r'claude', self.args['models']['llm_model']) is not None:
            messages = [
                {"role": "user", "content": question + '\n\n' + all_options + '\n\n' + instructions},
            ]
            response = self.client.messages.create(
                model=self.args['models']['llm_model'],
                max_tokens=1024,
                messages=messages,
            )
            response = response.content[0].text

        # Call lambda API for other models
        else:
            model = self.args['models']['llm_model']
            chat_completion = self.client.chat.completions.create(
                model=model,
                messages=messages,
            )
            response = chat_completion.choices[0].message.content

        return response


    def extract_answer(self, predicted_answer, correct_answer):
        def _extract_only_options(text):
            text = text.lower()
            in_parens = re.findall(r'\(([a-d])\)', text)
            if in_parens:
                return set(in_parens)
            else:
                return set(re.findall(r'\b([a-d])\b', text))

        correct = correct_answer.lower().strip("() ")

        # Clean predicted_answer
        full_response = predicted_answer
        predicted_answer = predicted_answer.strip()
        if "<final_answer>" in predicted_answer:
            predicted_answer = predicted_answer.split("<final_answer>")[-1].strip()
        if predicted_answer.endswith("</final_answer>"):
            predicted_answer = predicted_answer[:-len("</final_answer>")].strip()

        pred_options = _extract_only_options(predicted_answer)

        # First try the predicted_answer
        if pred_options == {correct}:
            return True, predicted_answer

        # Optionally fallback to model_response if provided
        response_options = _extract_only_options(full_response)
        if response_options == {correct}:
            return True, predicted_answer

        return False, predicted_answer


def openai_to_gemini_history(openai_messages):
    """
    Convert OpenAI-style messages to Gemini chat history format.

    Args:
        openai_messages (list of dict): Each dict has "role" and "content".

    Returns:
        list: Gemini-style history (UserContent and ModelContent instances).
    """
    gemini_history = []

    for msg in openai_messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if not content or role not in {"user", "assistant"}:
            continue  # Skip unsupported roles or empty content

        part = Part(text=content)

        if role == "user" or role == "system":
            gemini_history.append(UserContent(parts=[part]))
        elif role == "assistant":
            gemini_history.append(ModelContent(parts=[part]))

    return gemini_history


def count_tokens(all_strings, tokenizer, verbose=False):
    # all_strings = "\n\n".join(all_strings)
    tokens = tokenizer.encode(all_strings)
    return len(tokens)


def convert_role_system_to_user(messages_4o):
    """
    Convert OpenAI 4o-style messages (with 'system') to 4.0-style (no 'system').
    - System messages are merged into the next message (as a prefix).
    - Consecutive messages with the same role are merged into one.

    Args:
        messages_4o (list of dict): List of messages with roles like 'system', 'user', 'assistant'.

    Returns:
        list of dict: Cleaned message history for models without system role support.
    """
    messages_o1 = []
    system_buffer = ""

    for msg in messages_4o:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            system_buffer += f"[System]: {content}\n"
            continue

        # Prepend system message if buffered
        if system_buffer:
            content = system_buffer + content
            system_buffer = ""

        # Merge with previous message if role is the same
        if messages_o1 and messages_o1[-1]["role"] == role:
            messages_o1[-1]["content"] += "\n" + content
        else:
            messages_o1.append({"role": role, "content": content})

    return messages_o1


def generate_conversation_id(context):
    """Generates a unique, fixed-length conversation ID using a hash."""
    return hashlib.sha256(context.encode('utf-8')).hexdigest()[:16]  # First 16 characters of SHA-256 hash


def generate_shared_context_id(shared_context):
    """Returns a consistent ID for the same shared context using hashing."""
    if isinstance(shared_context, list):  # If it's a list, convert it to a string
        shared_context = " ".join(map(str, shared_context))  # Join elements with space

    return hashlib.sha256(shared_context.encode()).hexdigest()  # Generate hash


def question_type_mapping(old_name):
    mapping = {"recalling_facts_mentioned_by_the_user": "recall_user_shared_facts",
               "identifying_new_things_not_mentioned_by_the_user": "suggest_new_ideas",
               "recalling_the_latest_user_preferences": "acknowledge_latest_preferences",
               "tracking_the_full_sequence_of_preference_updates": "track_full_preference_evolution",
               "recalling_the_reasons_behind_previous_updates": "revisit_reasons_behind_preference_updates",
               "recommendation_aligned_with_users_latest_preferences": "provide_preference_aligned_recommendations",
               "generalizing_past_reasons_in_memory_to_new_scenarios": "generalize_to_new_scenarios"
               }
    return mapping[old_name]


def save_questions_to_csv(result, csv_file_path="data/questions.csv"):
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        # Write the header if the file is empty
        if os.stat(csv_file_path).st_size == 0:
            writer.writerow(["persona_id", "question_id", "question_type", "topic", "context_length_in_tokens", "context_length_in_letters",
                             "distance_to_ref_in_blocks", "distance_to_ref_in_tokens", "num_irrelevant_tokens", "distance_to_ref_proportion_in_context",
                             "user_question_or_message", "correct_answer", "all_options", "shared_context_id", "end_index_in_shared_context", "groundtruth_info"])

        percentage = f"{(result['distance_tokens'] / result['context_length_in_tokens']) * 100:.2f}%"
        question_type = question_type_mapping(result["question_type"])
        writer.writerow([
            result["idx_persona"],
            result["question_id"],
            question_type,
            result['topic'],
            result['context_length_in_tokens'],
            result['context_length_in_letters'],
            result['distance_blocks'],
            result['distance_tokens'],
            result["num_irrelevant_tokens"],
            percentage if result["context_length_in_tokens"] > 0 else "0%",
            result["question"],
            result["correct_answer"],
            result['all_options'],
            result["shared_context_id"],
            result["end_index_in_shared_context"],
            result["groundtruth_info"] if "groundtruth_info" in result else ""
        ])


def save_contexts_to_json(contexts_dict, json_file_path="data/contexts.jsonl"):
    """Appends JSON objects to a JSON Lines (NDJSON) file without loading the entire file."""
    with open(json_file_path, "a", encoding="utf-8") as file:  # 'a' mode for append
        file.write(json.dumps(contexts_dict, ensure_ascii=False) + "\n")  # Append as JSONL


def read_jsonl_file(json_file_path="data/contexts.jsonl"):
    """Reads a JSON Lines file line by line into a list of dictionaries."""
    data = []
    with open(json_file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line.strip()))  # Convert each line to a dictionary
    return data


def build_jsonl_index(jsonl_path):
    """
    Scan the JSONL file once to build a mapping: {key: file_offset}.
    Assumes each line is a JSON object with a single key-value pair.
    """
    index = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            key = next(iter(json.loads(line).keys()))
            index[key] = offset
    return index


def load_context_by_id(jsonl_path, offset):
    """
    Seek to a known offset in the JSONL and load exactly that line.
    Returns the value associated with the single key in the JSON object.
    """
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        f.seek(offset)
        item = json.loads(f.readline())
        return next(iter(item.values()))


def load_rows(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row_number, row in enumerate(reader, start=1):
            row_data = {}
            for column_name, value in row.items():
                row_data[column_name] = value
            yield row_data


def load_rows_with_context(csv_path, jsonl_path):
    jsonl_index = build_jsonl_index(jsonl_path)

    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        prev_sid = None
        prev_context = None

        for row_number, row in enumerate(reader, start=1):
            row_data = {}
            for column_name, value in row.items():
                row_data[column_name] = value

            sid = row_data["shared_context_id"]
            if sid != prev_sid:
                current_context = load_context_by_id(jsonl_path, jsonl_index[sid])

                prev_sid = sid
                prev_context = current_context
            else:
                current_context = prev_context

            yield row_data, current_context


def count_csv_rows(csv_path):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as f:
        return sum(1 for _ in f) - 1  # Subtract 1 for header row


def run_evaluation(args, cmd_args, llm, split, split_idx, verbose=False):
    question_path = cmd_args.question_path
    context_path = cmd_args.context_path
    result_path = cmd_args.result_path

    # if cmd_args.clean:
    #     user_input = input("The 'clean' flag is set. Do you really want remove existing eval_results.csv file? (y/n): ").strip().lower()
    #     if user_input == 'y':
    if os.path.exists(result_path):
        os.remove(result_path)
    #     else:
    #         print("Skipping cleanup.")

    all_errors = []
    total_rows = count_csv_rows(question_path)
    idx=-1
    assert split_idx < split, f"split_idx {split_idx} should be less than split {split}"
    for row_data, context in tqdm(load_rows_with_context(question_path, context_path), total=total_rows):
        idx+=1
        if idx%split != split_idx:
            continue
        try:
            # Extract relevant data from the row
            persona_id = row_data["persona_id"]
            question_id = row_data["question_id"]
            question_type = row_data["question_type"]
            topic = row_data["topic"]
            context_length_in_tokens = row_data["context_length_in_tokens"]
            context_length_in_letters = row_data["context_length_in_letters"]
            distance_to_ref_in_blocks = row_data["distance_to_ref_in_blocks"]
            distance_to_ref_in_tokens = row_data["distance_to_ref_in_tokens"]
            num_irrelevant_tokens = row_data["num_irrelevant_tokens"]
            distance_to_ref_proportion_in_context = row_data["distance_to_ref_proportion_in_context"]
            question = row_data["user_question_or_message"]
            correct_answer = row_data["correct_answer"]
            all_options = row_data["all_options"]
            shared_context_id = row_data["shared_context_id"]
            end_index_in_shared_context = row_data["end_index_in_shared_context"]

            # Prepare the context for the LLM query
            # print("context_length_in_tokens", context_length_in_tokens, "context_length_in_letters", context_length_in_letters)
            context = context[:int(end_index_in_shared_context)]  # Include up to the end index

            # Send the query to the LLM
            model_response = llm.query_llm(question, all_options, context)
            score, predicted_answer = llm.extract_answer(model_response, correct_answer)

            # Save the results back to a CSV file together with the question types
            if verbose:
                print(f"Question: {question}")
                print(f"Predicted Answer: {predicted_answer}")
                print(f"Correct Answer: {correct_answer}")
                print(f"Score: {score}")

            with open(result_path, mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)

                # Write the header if the file is empty
                if os.stat(result_path).st_size == 0:
                    writer.writerow(["score", "persona_id", "question_id", "user_question_or_message", "question_type", "topic", "context_length_in_tokens", "context_length_in_letters",
                                     "distance_to_ref_in_blocks", "distance_to_ref_in_tokens", "num_irrelevant_tokens", "distance_to_ref_proportion_in_context",
                                     "model_response", "len_of_model_response", "predicted_answer", "correct_answer"])
                writer.writerow([
                    score,
                    persona_id,
                    question_id,
                    question,
                    question_type,
                    topic,
                    context_length_in_tokens,
                    context_length_in_letters,
                    distance_to_ref_in_blocks,
                    distance_to_ref_in_tokens,
                    num_irrelevant_tokens,
                    distance_to_ref_proportion_in_context,
                    model_response,
                    len(model_response),
                    predicted_answer,
                    correct_answer,
                ])
        except Exception as e:
            print(f"Error: {e}")
            all_errors.append({
                "persona_id": row_data["persona_id"],
                "question_id": row_data["question_id"],
                "error": str(e)
            })
            continue

    if all_errors:
        for error in all_errors:
            print(f"Error for persona_id {error['persona_id']} and question_id {error['question_id']}: {error['error']}")


if __name__ == "__main__":
    # Load hyperparameters
    # try:
    #     with open('config.yaml', 'r') as file:
    #         args = yaml.safe_load(file)
    # except Exception as e:
    #     print('Error reading the config file')

    # torch.manual_seed(0)
    # world_size = torch.cuda.device_count()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # if world_size > 1:
    # #     assert world_size == 1
    # print('device', device)
    # print('torch.distributed.is_available', torch.distributed.is_available())
    # print('Using %d GPUs' % (torch.cuda.device_count()))

    # Command-line argument parsing
    parser = argparse.ArgumentParser(description='Command line arguments')

    """ General arguments """
    parser.add_argument('--model', type=str, default="gpt-4o", help='Set LLM model. Choose from o3-mini, o1, o1-mini, gpt-4o, gpt-4o-mini, '
                                                                    'Llama-3.3-70B-Instruct, Meta-Llama-3.1-70B-Instruct, Meta-Llama-3.1-8B-Instruct, '
                                                                    'claude-3-7-sonnet-20250219, DeepSeek-R1, DeepSeek-v3,'
                                                                    'gemini-2.0-flash, gemini-1.5-flash, gemini-2.0-flash-lite')
    parser.add_argument('--step', type=str, default='prepare', help='Step to run: prepare or evaluate')
    parser.add_argument('--token_path', type=str, default='api_tokens', help='Path to the API tokens')
    parser.add_argument('--clean', dest='clean', action='store_true', help='Remove existing csv and json files and start clean')
    parser.add_argument('--verbose', dest='verbose', action='store_true', help='Set verbose to True')

    """ Arguments for running the evaluation step """
    parser.add_argument('--question_path', type=str, default='data/questions_128k.csv', help='Path to the questions CSV file')
    parser.add_argument('--context_path', type=str, default='data/shared_contexts_128k.jsonl', help='Path to the contexts JSONL file')
    parser.add_argument('--result_path', type=str, default='data/eval_results.csv', help='Path to save the results CSV file')

    parser.add_argument('--split', type=int, default=10, help='Number of splits for evaluation')
    parser.add_argument('--split_idx', type=int, default=0, help='Index of the split to evaluate')
    
    cmd_args = parser.parse_args()
    from collections import defaultdict
    args = defaultdict(dict)
    args['models']['llm_model'] = cmd_args.model

    if re.search(r'gemini', cmd_args.model) is not None:
        from google import genai  # Gemini has conflicting requirements of the environment with OpenAI
        from google.genai.types import Part, UserContent, ModelContent

    base_dir = "./data/output"
    llm = Evaluation(args, cmd_args)
    # tokenizer = tiktoken.encoding_for_model("gpt-4o")

    if cmd_args.step == 'evaluate':
        run_evaluation(args, cmd_args, llm, cmd_args.split, cmd_args.split_idx, verbose=cmd_args.verbose)
    else:
        raise ValueError("Invalid step. Choose 'prepare' or 'evaluate'.")