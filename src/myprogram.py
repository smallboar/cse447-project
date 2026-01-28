#!/usr/bin/env python
import os
import string
import random
import sys
import heapq
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import pickle
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset # import from hugging face

class TrieNode:
    """
    This class is responsible for storing data for a trie.
    """

    def __init__(self) -> None:
        self.children: Dict[str, 'TrieNode'] = {}
        self.counts: Dict[str, int] = defaultdict(int)
        self.total_count: int = 0

class Predictor:
    """
    This class is responsible for training the model and predicting characters.
    """

    def __init__(self, max_context_length: int = 6) -> None:
        self.max_context_length: int = max_context_length
        self.trie_root: TrieNode = TrieNode()
        self.freq_by_char: Dict[str, int] = defaultdict(int)
        self.total_chars: int = 0
        self.context: List[str] = []
        
    def train(self, text: str) -> None:
        self.total_chars += len(text)

        for char in text:
            self.freq_by_char[char] += 1
            
            # Update trie
            node = self.trie_root
            for _, prev_char in enumerate(self.context[-self.max_context_length:]):
                if prev_char not in node.children:
                    node.children[prev_char] = TrieNode()
                node = node.children[prev_char]
                node.counts[char] += 1
                node.total_count += 1
            
            # Update context
            self.context.append(char)
            if len(self.context) > self.max_context_length:
                self.context.pop(0)
    
    def get_char_prob(self, char: str, context: List[str], context_length: int) -> float:
        """
        Calculate probability of character given context of specified context length.
        """

        # TODO: Implement smoothing as described in the doc (i think we said Kneser-Ney smoothing)

        # Unigram
        if context_length == 0:
            if self.total_chars == 0:
                return 0.0 # shouldnt happen but just in case
            return self.freq_by_char.get(char, 0) / self.total_chars
        
        # N-gram
        node = self.trie_root
        context_start = len(context) - context_length

        # Traverse trie
        for i in range(context_start, len(context)):
            if context[i] not in node.children:
                return 0.0
            node = node.children[context[i]]
        
        # Calculate prob
        char_count = node.counts.get(char, 0)
        total = node.total_count
        
        if total == 0:
            return 0.0
        
        return char_count / total
    
    def predict_next(self, num_predictions: int = 3) -> List[str]:
        """
        Predict the next N most likely characters. (in this case 3)
        """        
        possible_chars = set(self.freq_by_char.keys())
        if not possible_chars:
            raise Exception("No characters found")
        
        # Calculate probabilities for all possible characters
        probabilities: List[Tuple[float, str]] = []
        current_context_length = min(len(self.context), self.max_context_length)
        
        for char in possible_chars:
            prob = self.get_char_prob(char, self.context, current_context_length)
            probabilities.append((prob, char))
        
        # Get top N
        top_n = heapq.nlargest(num_predictions, probabilities, key=lambda x: x[0])
        return [char for _, char in top_n]
    
    def update_context(self, char: str) -> None:
        self.context.append(char)
        if len(self.context) > self.max_context_length:
            self.context.pop(0)
    
    def clear_context(self) -> None:
        self.context = []
    
    def save(self, filepath: str) -> None:
        """
        Save model data to the given filepath
        """
        model_data = {
            'root': self.trie_root,
            'vocab': dict(self.freq_by_char),
            'total_chars': self.total_chars,
            'max_context_length': self.max_context_length,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'Predictor':
        """
        Load model data from the given filepath
        """
        model_data = None

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        if model_data is None:
            raise Exception("Could not load model from filepath: " + filepath)
        
        engine = cls(max_context_length=model_data['max_context_length'])
        engine.trie_root = model_data['root']
        engine.freq_by_char = model_data['vocab']
        engine.total_chars = model_data['total_chars']
        return engine


class MyModel:
    """
    This class is responsible for training the model and predicting characters.
    """
    # Class constants
    NUM_PREDICTIONS = 3
    MODEL_FILENAME = 'ngram_model.pkl'

    def __init__(self, engine: Optional[Predictor] = None):
        """Initialize the model with a predictor"""
        if engine is None:
            self.engine = Predictor(max_context_length=6)
        else:
            self.engine = engine

    @classmethod
    def load_training_data(cls):
        """
        Load training data from the Wiki40B dataset
        """
        # Load the Wiki40B dataset
        # Currently only loading the English dataset for MVP purposes (will add the rest later on)
        # Looking to add langauges like Russian, Chinese, Spanish, etc... we listed them all in the doc
        # Stream to avoid loading entire dataset
        ds = load_dataset("google/wiki40b", "en", split="train", streaming=True)
        return ds

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir, dataset_fraction):
        """
        Train the model on training data. 
        Added a dataset fraction to limit the amount of data processed (for MVP purposes)
        """
        batch_size = 1000
        current_batch = []
        batch_num = 0
        items_processed = 0
        
        # Calculate training data len
        training_data_len = None
        if hasattr(data, '__len__'): # from the hugging facec dataset
            total_available = len(data)

            # We take a fraction of the dataset to train on (for now, MVP)
            training_data_len = int(total_available * dataset_fraction)
            print(f'Using {dataset_fraction*100:.1f}% of dataset ({training_data_len:,}/{total_available:,} items)')
        else:
            # Streaming dataset - use a target count based on dataset_fraction
            training_data_len = int(1_000_000 * dataset_fraction)
            print(f'Using streaming dataset and targeting {training_data_len:,} items)', file=sys.stderr)
        
        # Helper function to train on current batch
        def train_batch():
            nonlocal batch_num, items_processed
            if current_batch:
                training_text = '\n'.join(current_batch)
                self.engine.train(training_text)

                batch_num += 1
                items_processed += len(current_batch)
                
                # Added for QoL so I know roughly how long training is gonna take
                percentage = (items_processed / training_data_len) * 100
                print(f'Batch {batch_num} ({items_processed:,}/{training_data_len:,} items, {percentage:.1f}%)', file=sys.stderr)
                current_batch.clear()
        
        # Iterate through dataset and batch train
        try:
            # Making sure that it's iterable
            if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                items_seen = 0
                for item in data:
                    # TODO: POSSIBLY USE RANDOM SAMPLING? Right now just iterating through and choosing from the beginning... not sure if we can assume that dataset is random enough
                    # Check if we've reached the limit of training data
                    if training_data_len and items_seen >= training_data_len:
                        break
                    
                    # Hugging face dataset stores it under "text" key
                    text = item.get('text')

                    if text:
                        current_batch.append(text)
                        items_seen += 1
                        
                        # Train when hit the batch size
                        if len(current_batch) >= batch_size:
                            train_batch()

                            if training_data_len and items_processed >= training_data_len:
                                break
            elif isinstance(data, str):
                # shouldnt hit this but in case it's a string we can just use this
                current_batch.append(data)
                train_batch()
        except Exception as e:
            print(f'Error during training: {e}', file=sys.stderr)
            # Train on what we have so far
            if current_batch:
                train_batch()
            return
        
        # Train on remaining items
        if current_batch:
            train_batch()
        
        print(f'Finished training on {items_processed:,} items in {batch_num} batches.', file=sys.stderr)

    def run_pred(self, data):
        predictions = []

        for input_str in data:
            self.engine.clear_context()
            
            for c in input_str:
                self.engine.update_context(c)
            
            # Predict next chars
            next_predictions = self.engine.predict_next(MyModel.NUM_PREDICTIONS)

            # Make sure we have enough predictions (shouldn't happen)
            if len(next_predictions) < MyModel.NUM_PREDICTIONS:
                print(f"!!!!!RECEIVED LESS THAN {MyModel.NUM_PREDICTIONS} PREDICTIONS!!!!!")

                # just guess the most common chars instead
                common_chars = 'etaoinshrdlcumwfgypbvkjxqz'
                for c in common_chars:
                    if c not in next_predictions:
                        next_predictions.append(c)
                        if len(next_predictions) == MyModel.NUM_PREDICTIONS:
                            break
            
            # Join to one str
            predictions.append(''.join(next_predictions[:MyModel.NUM_PREDICTIONS]))
        
        return predictions

    def save(self, work_dir):
        """Save the trained model."""
        model_path = os.path.join(work_dir, MyModel.MODEL_FILENAME)
        self.engine.save(model_path)

    @classmethod
    def load(cls, work_dir, dataset_fraction=0.01):
        """Load a trained model."""
        model_path = os.path.join(work_dir, cls.MODEL_FILENAME)
        
        if os.path.exists(model_path):
            engine = Predictor.load(model_path)
            return cls(engine)
        else:
            # If no model exists, train one
            print('No trained model found, training new model.', file=sys.stderr)
            model = cls()
            train_data = cls.load_training_data()
            
            model.run_train(train_data, work_dir, dataset_fraction)
            model.save(work_dir)
            return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    parser.add_argument('--dataset_fraction', type=float, default=0.01, # TODO: Change default in future checkpoints (using 0.1% for now for MVP)
                       help='fraction of the dataset to train on')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir, dataset_fraction=args.dataset_fraction)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir, dataset_fraction=args.dataset_fraction)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
