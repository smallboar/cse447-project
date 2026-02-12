#!/usr/bin/env python3
"""
ISS 2046 Multilingual Character Predictor
N-Gram model with Kneser-Ney smoothing for character-level prediction
"""
import os
import sys
import pickle
import heapq
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


class NGramModel:
    """
    Character-level N-Gram model with Kneser-Ney smoothing.
    
    Uses fixed-order n-grams (1-5) with Kneser-Ney smoothing for
    character-level prediction with excellent latency performance.
    """
    
    def __init__(self, max_order: int = 5, discount: float = 0.75):
        """
        Initialize the N-Gram model.
        
        Args:
            max_order: Maximum n-gram order (default: 5)
            discount: Discount parameter for Kneser-Ney smoothing (default: 0.75)
        """
        self.max_order = max_order
        self.discount = discount
        
        # ngrams[order][context] = Counter({char: count})
        self.ngrams: Dict[int, Dict[str, Counter]] = {
            i: defaultdict(Counter) for i in range(1, max_order + 1)
        }
        
        # Continuation counts for Kneser-Ney: continuation[order][context] = set of following chars
        self.continuation: Dict[int, Dict[str, Set[str]]] = {
            i: defaultdict(set) for i in range(1, max_order)
        }
        
        # Vocabulary: all unique characters seen
        self.vocab: Set[str] = set()
        
        # Total counts per order for normalization
        self.total_counts: Dict[int, int] = defaultdict(int)
    
    def train(self, text: str) -> None:
        """
        Train the model on text data.
        
        Args:
            text: Training text (can be called multiple times)
        """
        if not text:
            return
        
        # Process text character by character
        for i in range(len(text)):
            char = text[i]
            self.vocab.add(char)
            
            # Build n-grams of all orders
            for order in range(1, self.max_order + 1):
                if i >= order - 1:
                    # Get context (preceding characters)
                    context_start = i - order + 1
                    context = text[context_start:i]
                    
                    # Increment n-gram count
                    self.ngrams[order][context][char] += 1
                    self.total_counts[order] += 1
                    
                    # Build continuation counts (for lower orders)
                    if order < self.max_order and i > 0:
                        prev_context = text[max(0, context_start - 1):i]
                        if prev_context:
                            self.continuation[order][prev_context].add(char)
    
    def _get_continuation_prob(self, char: str, order: int) -> float:
        """
        Calculate continuation probability for Kneser-Ney.
        Used for lower-order backoff.
        
        Args:
            char: Character to predict
            order: N-gram order
            
        Returns:
            Continuation probability
        """
        if order == 0:
            # Unigram: use frequency in vocabulary
            total_chars = sum(sum(counter.values()) for counter in self.ngrams[1].values())
            char_count = sum(self.ngrams[1][''].values()) if '' in self.ngrams[1] else 0
            if total_chars > 0:
                return char_count / total_chars
            return 1.0 / len(self.vocab) if self.vocab else 0.0
        
        # Count how many contexts this character follows
        # Sort contexts for deterministic iteration
        contexts_with_char = sum(1 for ctx_set in sorted(self.continuation[order].values(), key=str) if char in ctx_set)
        total_contexts = len(self.continuation[order])
        
        if total_contexts > 0:
            return contexts_with_char / total_contexts
        
        # Fallback to lower order
        return self._get_continuation_prob(char, order - 1)
    
    def _kneser_ney_prob(self, char: str, context: str, order: int) -> float:
        """
        Calculate Kneser-Ney smoothed probability.
        
        Args:
            char: Character to predict
            context: Context string (preceding characters)
            order: N-gram order
            
        Returns:
            Probability estimate
        """
        if order == 0:
            # Base case: use continuation probability
            return self._get_continuation_prob(char, 0)
        
        # Get n-gram counts for this order
        if context in self.ngrams[order]:
            char_count = self.ngrams[order][context].get(char, 0)
            total_count = sum(self.ngrams[order][context].values())
            
            if total_count > 0:
                # Discounted probability
                discounted = max(char_count - self.discount, 0.0) / total_count
                
                # Calculate lambda (normalization constant)
                unique_following = len(self.ngrams[order][context])
                lambda_weight = (self.discount * unique_following) / total_count
                
                # Backoff probability (shorter context)
                shorter_context = context[1:] if len(context) > 0 else ''
                backoff_prob = self._kneser_ney_prob(char, shorter_context, order - 1)
                
                return discounted + lambda_weight * backoff_prob
        
        # Context doesn't exist: backoff to lower order
        shorter_context = context[1:] if len(context) > 0 else ''
        return self._kneser_ney_prob(char, shorter_context, order - 1)
    
    def predict_next(self, context: str, num_predictions: int = 3) -> List[str]:
        """
        Predict the next N most likely characters given context.
        
        Args:
            context: Input string (context for prediction)
            num_predictions: Number of predictions to return (default: 3)
            
        Returns:
            List of predicted characters (most likely first)
        """
        # Use last max_order characters as context
        effective_context = context[-self.max_order:] if len(context) >= self.max_order else context
        
        # If no vocabulary, return common characters
        if not self.vocab:
            common_chars = 'etaoinshrdlcumwfgypbvkjxqz'
            return list(common_chars[:num_predictions])
        
        # Calculate probabilities for all characters in vocabulary
        # Sort vocab to ensure deterministic ordering
        sorted_vocab = sorted(self.vocab)
        probabilities: List[Tuple[float, str]] = []
        for char in sorted_vocab:
            prob = self._kneser_ney_prob(char, effective_context, self.max_order)
            probabilities.append((prob, char))
        
        # Get top N predictions, breaking ties deterministically by character value
        # Use negative probability for stable sort (higher prob first, then lexicographic order)
        top_n = heapq.nlargest(num_predictions, probabilities, key=lambda x: (x[0], x[1]))
        predictions = [char for _, char in top_n]
        
        # Ensure we have exactly num_predictions (pad with common chars if needed)
        common_chars = 'etaoinshrdlcumwfgypbvkjxqzETAOINSHRDLCUMWFGYPBVKJXQZ0123456789.,!? '
        while len(predictions) < num_predictions:
            for c in common_chars:
                if c not in predictions:
                    predictions.append(c)
                    break
            else:
                # If we've exhausted common chars, use any char from vocab (sorted for determinism)
                for c in sorted_vocab:
                    if c not in predictions:
                        predictions.append(c)
                        break
                else:
                    # Last resort: use space
                    predictions.append(' ')
                    break
        
        return predictions[:num_predictions]
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        # Convert Counter and Set to regular dicts/sets for serialization
        model_data = {
            'max_order': self.max_order,
            'discount': self.discount,
            'ngrams': {
                order: {ctx: dict(counter) for ctx, counter in ngrams.items()}
                for order, ngrams in self.ngrams.items()
            },
            'continuation': {
                order: {ctx: set(chars) for ctx, chars in cont.items()}
                for order, cont in self.continuation.items()
            },
            'vocab': list(self.vocab),
            'total_counts': dict(self.total_counts)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'NGramModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded NGramModel instance
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(max_order=model_data['max_order'], discount=model_data['discount'])
        
        # Restore n-grams
        for order, ngrams_dict in model_data['ngrams'].items():
            for ctx, char_counts in ngrams_dict.items():
                model.ngrams[order][ctx] = Counter(char_counts)
        
        # Restore continuation counts
        for order, cont_dict in model_data['continuation'].items():
            for ctx, chars in cont_dict.items():
                model.continuation[order][ctx] = set(chars)
        
        # Restore vocabulary
        model.vocab = set(model_data['vocab'])
        model.total_counts = defaultdict(int, model_data['total_counts'])
        
        return model


class MyModel:
    """
    N-Gram based character prediction model for ISS 2046.
    """

    def __init__(self, ngram_model: Optional[NGramModel] = None):
        """Initialize the model with an N-Gram model."""
        if ngram_model is None:
            self.ngram_model = NGramModel(max_order=5, discount=0.75)
        else:
            self.ngram_model = ngram_model

    @classmethod
    def load_training_data(cls):
        """Load training data from available sources."""
        # Default training: use common English patterns with some multilingual support
        # Focus on English to improve accuracy for common English text
        default_training = """
        The quick brown fox jumps over the lazy dog.
        Hello world! How are you today?
        Happy New Year! Welcome to 2046.
        That's one small step for man, one giant leap for mankind.
        To be or not to be, that is the question.
        In the beginning was the Word, and the Word was with God.
        It was the best of times, it was the worst of times.
        Call me Ishmael. Some years ago, never mind how long precisely.
        It is a truth universally acknowledged, that a single man in possession of a good fortune.
        All happy families are alike; each unhappy family is unhappy in its own way.
        It was a bright cold day in April, and the clocks were striking thirteen.
        The sun shone, having no alternative, on the nothing new.
        The past is a foreign country; they do things differently there.
        El rápido zorro marrón salta sobre el perro perezoso.
        Le renard brun rapide saute par-dessus le chien paresseux.
        Der schnelle braune Fuchs springt über den faulen Hund.
        Быстрая коричневая лиса прыгает через ленивую собаку.
        敏捷的棕色狐狸跳过懒狗。
        素早い茶色のキツネは怠け者の犬を飛び越えます。
        مرحبا بالعالم
        שלום עולם
        नमस्ते दुनिया
        """
        return [default_training]

    @classmethod
    def load_test_data(cls, fname):
        """Load test data from file."""
        data = []
        with open(fname, 'r', encoding='utf-8') as f:
            for line in f:
                inp = line.rstrip('\n\r')  # Remove trailing newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        """Write predictions to file."""
        with open(fname, 'wt', encoding='utf-8') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        """Train the model on training data."""
        # Combine all training data
        if isinstance(data, list):
            training_text = '\n'.join(data)
        else:
            training_text = data
        
        # Train the n-gram model
        self.ngram_model.train(training_text)
        
        # Also train on individual lines to capture line-end patterns
        if isinstance(data, list):
            for line in data:
                self.ngram_model.train(line)

    def run_pred(self, data):
        """Predict next character for each input string."""
        preds = []
        for inp in data:
            # Predict next 3 characters
            predictions = self.ngram_model.predict_next(inp, num_predictions=3)
            
            # Join predictions into a single string
            preds.append(''.join(predictions))
        
        return preds

    def save(self, work_dir):
        """Save the trained model."""
        model_path = os.path.join(work_dir, 'ngram_model.pkl')
        os.makedirs(work_dir, exist_ok=True)
        self.ngram_model.save(model_path)
        
        # Also save a checkpoint file for compatibility
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('ngram_model.pkl')

    @classmethod
    def load(cls, work_dir):
        """Load a trained model."""
        model_path = os.path.join(work_dir, 'ngram_model.pkl')
        
        if os.path.exists(model_path):
            ngram_model = NGramModel.load(model_path)
            return cls(ngram_model)
        else:
            # If no model exists, create a new one with default training
            print('No trained model found, creating new model with default training...', file=sys.stderr)
            model = cls()
            train_data = cls.load_training_data()
            model.run_train(train_data, work_dir)
            model.save(work_dir)
            return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
