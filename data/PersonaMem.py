
from re import L
from typing import List, Union, Tuple
from pathlib import Path
import pandas
import json

from dataclasses import dataclass

@dataclass
class QA:
    user_question_or_message: str
    correct_answer: str
    all_options: str

@dataclass
class Conversation:
    shared_context_id: str
    messages: List[str]

@dataclass
class PersonaMemSample:
    """A single sample from the PersonaMem dataset"""
    persona_id: int
    question_id: str
    topic: str
    shared_context_id: str
    end_index_in_shared_context: int
    qa: QA

def load_personamem_dataset(questions_path: Union[str, Path], conversations_path: Union[str, Path]) -> Tuple[List[PersonaMemSample], List[Conversation]]:
    """Load the PersonaMem dataset from a JSON file"""
    questions = pandas.read_csv(questions_path)
    samples = []
    for _, question in questions.iterrows():
        persona_id = question['persona_id']
        question_id = question['question_id']
        topic = question['topic']
        shared_context_id = question['shared_context_id']
        end_index_in_shared_context = question['end_index_in_shared_context']
        qa = QA(
            user_question_or_message=question['user_question_or_message'],
            correct_answer=question['correct_answer'],
            all_options=question['all_options'],
        )
        samples.append(PersonaMemSample(
            persona_id=persona_id,
            question_id=question_id,
            topic=topic,
            shared_context_id=shared_context_id,
            end_index_in_shared_context=end_index_in_shared_context,
            qa=qa,
        ))
    
    conversation = []
    with open(conversations_path, 'r') as f:
        conversations = [json.loads(line) for line in f]
    for c in conversations:
        conversation.append(Conversation(
            shared_context_id=list(c.keys())[0],
            messages=list(c.values())[0],
        ))
    return samples, conversation
        

if __name__ == '__main__':
    questions_path = 'data/PersonaMem/questions_32k.csv'
    conversations_path = 'data/PersonaMem/shared_contexts_32k.jsonl'
    samples, conversations = load_personamem_dataset(questions_path, conversations_path)
    print(samples[0])
    print(conversations[0])