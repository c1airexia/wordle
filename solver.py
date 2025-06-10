import nltk
import string
import re
import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
nltk.download('words', quiet=True)

class Game:
    def __init__(self, df_all_5l_words, df_all_guesses=None):
        self.possible_letters = list(string.ascii_uppercase)

        self.letter_info = {
            'correct': {},     # Position -> Letter
            'misplaced': {},   # Letter -> [Positions]
            'wrong': set()     # Letters that are not in the word
        }

        self.min_letter_counts = Counter()
        self.max_letter_counts = Counter()
        for letter in self.possible_letters:
            self.max_letter_counts[letter] = 5

        self.df_possible_5l_words = df_all_5l_words.copy()
        self.df_all_guesses = df_all_guesses if df_all_guesses is not None else df_all_5l_words.copy()
        self.all_words = self.df_possible_5l_words['word'].tolist()
        self.all_guess_words = self.df_all_guesses['word'].tolist()
        self.possible_words = set(self.all_words)
        self.pattern_cache = {}
        self.optimal_openers = ["SLATE", "CRANE", "SLANT", "TRACE", "CRATE", "ADIEU"]
        self.letter_freq = self._calculate_letter_frequencies()
        self.position_freq = self._calculate_position_frequencies()
        self.turn = 0

    def _calculate_letter_frequencies(self):
        freq = Counter()
        for word in self.all_words:
            for letter in set(word):  # Count each letter once per word
                freq[letter] += 1
        return freq

    def _calculate_position_frequencies(self):
        pos_freq = [{} for _ in range(5)]
        for word in self.all_words:
            for i, letter in enumerate(word):
                if letter not in pos_freq[i]:
                    pos_freq[i][letter] = 0
                pos_freq[i][letter] += 1
        return pos_freq

    def calculate_entropy(self, guess_word):
        if len(self.possible_words) <= 1:
            return 0

        pattern_buckets = defaultdict(int)
        total_words = len(self.possible_words)

        for target in self.possible_words:
            pattern = self._get_pattern(guess_word, target)
            pattern_buckets[pattern] += 1

        entropy = 0
        for count in pattern_buckets.values():
            prob = count / total_words
            entropy -= prob * math.log2(prob)

        return entropy

    def calculate_minimax_score(self, guess_word):
        if len(self.possible_words) <= 1:
            return 0

        pattern_buckets = defaultdict(int)
        for target in self.possible_words:
            pattern = self._get_pattern(guess_word, target)
            pattern_buckets[pattern] += 1
        max_remaining = max(pattern_buckets.values()) if pattern_buckets else 0
        return -max_remaining

    def calculate_expected_remaining(self, guess_word):
        if len(self.possible_words) <= 1:
            return 0

        pattern_buckets = defaultdict(int)
        total_words = len(self.possible_words)

        for target in self.possible_words:
            pattern = self._get_pattern(guess_word, target)
            pattern_buckets[pattern] += 1

        expected = 0
        for count in pattern_buckets.values():
            prob = count / total_words
            expected += prob * count

        return expected

    def _get_pattern(self, guess, target):
        cache_key = (guess, target)
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]

        guess = guess.upper()
        target = target.upper()
        result = [0] * 5
        target_letters = list(target)
        for i in range(5):
            if guess[i] == target_letters[i]:
                result[i] = 2
                target_letters[i] = None  # Mark as used

        for i in range(5):
            if result[i] == 0 and guess[i] in target_letters:
                result[i] = 1
                target_letters[target_letters.index(guess[i])] = None  # Mark as used

        pattern = tuple(result)
        self.pattern_cache[cache_key] = pattern
        return pattern

    def _letter_diversity_score(self, word):
        unique_letters = set(word)
        diversity = len(unique_letters)
        position_bonus = 0
        for i, letter in enumerate(word):
            if letter in self.position_freq[i]:
                position_bonus += self.position_freq[i][letter]

        frequency_bonus = sum(self.letter_freq[letter] for letter in unique_letters)
        return diversity * 100 + position_bonus * 0.1 + frequency_bonus * 0.05

    def _calculate_word_commonality(self, word):
        return sum(self.letter_freq.get(letter, 0) for letter in set(word))

    def guess(self):
        self.turn += 1
        if len(self.possible_words) == 0:
            print("Warning: No possible words left. Using a fallback word.")
            return pd.DataFrame([{
                'letter_1': 'S', 'letter_2': 'L', 'letter_3': 'A',
                'letter_4': 'T', 'letter_5': 'E', 'word': 'SLATE'
            }])

        if self.turn == 1:
            for opener in self.optimal_openers:
                opener_df = self.df_all_guesses[self.df_all_guesses['word'] == opener]
                if not opener_df.empty:
                    return opener_df
            return self.df_all_guesses.iloc[:1]

        if len(self.possible_words) == 1:
            word = list(self.possible_words)[0]
            return self.df_possible_5l_words[self.df_possible_5l_words['word'] == word]

        remaining = len(self.possible_words)
        if remaining == 2:
            word = list(self.possible_words)[0]
            return self.df_possible_5l_words[self.df_possible_5l_words['word'] == word]

        # Small number of possibilities use minimax
        elif remaining <= 6:
            return self._minimax_guess()

        # Medium number use entropy with answer preference
        elif remaining <= 25:
            return self._entropy_guess_with_answer_bias()

        # Large number use entropy with smart sampling
        else:
            return self._entropy_guess_with_sampling()

    def _minimax_guess(self):
        best_word = None
        best_score = float('-inf')
        candidates = set(self.possible_words)
        if len(candidates) <= 6:
            high_entropy_words = ['AUDIO', 'POUND', 'LIGHT', 'CHURN', 'MOIST']
            for word in high_entropy_words:
                if word in self.all_guess_words:
                    candidates.add(word)

        for word in candidates:
            minimax_score = self.calculate_minimax_score(word)
            entropy_score = self.calculate_entropy(word)
            score = minimax_score + entropy_score * 0.1
            if word in self.possible_words and len(self.possible_words) <= 4:
                score += 2.0
            elif word in self.possible_words:
                score += 0.5

            score += self._letter_diversity_score(word) * 0.001
            if score > best_score:
                best_score = score
                best_word = word

        return self._get_word_df(best_word)

    def _entropy_guess_with_answer_bias(self):
        best_word = None
        best_score = float('-inf')

        for word in self.possible_words:
            entropy = self.calculate_entropy(word)
            answer_bonus = 1.5 if len(self.possible_words) <= 15 else 1.2
            score = entropy * answer_bonus
            score += self._calculate_word_commonality(word) * 0.001
            if score > best_score:
                best_score = score
                best_word = word

        high_value_words = ['AUDIO', 'CHURN', 'MOIST', 'POUND', 'LIGHT']
        for word in high_value_words:
            if word in self.all_guess_words and word not in self.possible_words:
                entropy = self.calculate_entropy(word)
                threshold = best_score / 1.3
                if entropy > threshold:
                    if entropy > best_score:
                        best_score = entropy
                        best_word = word

        return self._get_word_df(best_word)

    def _entropy_guess_with_sampling(self):
        best_word = None
        best_entropy = -1
        answer_candidates = list(self.possible_words)
        other_candidates = []
        if len(self.all_guess_words) > len(answer_candidates):
            non_answers = [w for w in self.all_guess_words if w not in self.possible_words]

            scored_words = []
            for word in non_answers:
                diversity_score = self._letter_diversity_score(word)
                scored_words.append((word, diversity_score))

            scored_words.sort(key=lambda x: x[1], reverse=True)
            top_words = [w[0] for w in scored_words[:50]]
            remaining_words = [w for w in non_answers if w not in top_words]
            if remaining_words:
                random_sample = min(50, len(remaining_words))
                random_words = np.random.choice(remaining_words, size=random_sample, replace=False).tolist()
                other_candidates = top_words + random_words
            else:
                other_candidates = top_words

        all_candidates = answer_candidates + other_candidates

        for word in all_candidates:
            entropy = self.calculate_entropy(word)
            if word in self.possible_words:
                entropy *= 1.02

            if entropy > best_entropy:
                best_entropy = entropy
                best_word = word

        return self._get_word_df(best_word)

    def _get_word_df(self, word):
        if not word:
            word = list(self.possible_words)[0] if self.possible_words else "SLATE"

        result = self.df_all_guesses[self.df_all_guesses['word'] == word]
        if not result.empty:
            return result

        result = self.df_possible_5l_words[self.df_possible_5l_words['word'] == word]
        if not result.empty:
            return result

        return pd.DataFrame([{
            'letter_1': word[0], 'letter_2': word[1],
            'letter_3': word[2], 'letter_4': word[3],
            'letter_5': word[4], 'word': word
        }])

    def update(self, guess, results):
        guess = guess.upper()
        assert len(guess) == 5, 'Guess must be 5 characters long'
        assert len(results) == 5, 'Results list must contain 5 items'
        assert all([n in [0, 1, 2] for n in results]), 'Results list must only contain 0, 1, or 2'
        guess_letter_counts = Counter(guess)
        confirmed_letters = Counter()

        # handle correct letters
        for i, (letter, result) in enumerate(zip(guess, results)):
            if result == 2:
                self.letter_info['correct'][i] = letter
                confirmed_letters[letter] += 1

        # handle misplaced letters
        for i, (letter, result) in enumerate(zip(guess, results)):
            if result == 1:
                if letter not in self.letter_info['misplaced']:
                    self.letter_info['misplaced'][letter] = []
                self.letter_info['misplaced'][letter].append(i)
                confirmed_letters[letter] += 1

        # handle wrong letters and update counts
        for i, (letter, result) in enumerate(zip(guess, results)):
            if result == 0:
                if letter in confirmed_letters:
                    self.max_letter_counts[letter] = confirmed_letters[letter]
                else:
                    self.letter_info['wrong'].add(letter)
                    self.max_letter_counts[letter] = 0

        for letter, count in confirmed_letters.items():
            if count > self.min_letter_counts[letter]:
                self.min_letter_counts[letter] = count

        self._update_possible_words()

    def _update_possible_words(self):
        new_possible_words = set()
        for word in self.possible_words:
            if self._is_word_possible(word):
                new_possible_words.add(word)

        self.possible_words = new_possible_words
        mask = self.df_possible_5l_words['word'].isin(self.possible_words)
        self.df_possible_5l_words = self.df_possible_5l_words[mask].copy()

    def _is_word_possible(self, word):
        word = word.upper()

        # correct letters
        for pos, letter in self.letter_info['correct'].items():
            if word[pos] != letter:
                return False

        # misplaced letters
        for letter, positions in self.letter_info['misplaced'].items():
            if letter not in word:
                return False
            for pos in positions:
                if word[pos] == letter:
                    return False

        # wrong letters
        for letter in self.letter_info['wrong']:
            if letter in word:
                return False

        # letter count constraints
        word_letter_counts = Counter(word)
        for letter, min_count in self.min_letter_counts.items():
            if word_letter_counts[letter] < min_count:
                return False

        for letter, max_count in self.max_letter_counts.items():
            if word_letter_counts[letter] > max_count:
                return False

        return True


def play_game(target_word, df_possible_words, df_all_guesses=None, debug=False):
    target_word = re.sub(r'[^A-Z]', '', target_word.upper())
    assert len(target_word) == 5, 'target_word must be 5 characters long'
    game = Game(df_possible_words, df_all_guesses)

    for guess_turn in range(6):
        guess_df = game.guess()
        try:
            guess_word = guess_df.iloc[0]['word']
        except (IndexError, KeyError) as e:
            if debug:
                print(f"Error selecting word: {e}")
                print(f"DataFrame: {guess_df}")
            guess_word = "SLATE"

        results = generate_results(guess_word, target_word)
        if debug:
            print(f'Turn {guess_turn+1}, guess {guess_word}, results {results}')
            print(f'Possible words remaining: {len(game.possible_words)}')
            if len(game.possible_words) <= 10:
                print(f'Remaining words: {sorted(list(game.possible_words))}')

        if sum(results) == 10:
            if debug:
                print('Game won!')
            return (target_word, guess_turn+1)

        game.update(guess_word, results)
        if guess_turn == 5:
            if debug:
                print('Unsolved!')
                if len(game.possible_words) <= 10:
                    print(f'Remaining possible words: {game.possible_words}')
                else:
                    print(f'{len(game.possible_words)} words remain possible')
            return (target_word, 7)

    return (target_word, guess_turn+1)


def generate_results(guess, target):
    guess = guess.upper()
    target = target.upper()
    results = [0] * 5
    target_letters = list(target)

    # First pass marking correct letters
    for i in range(5):
        if guess[i] == target_letters[i]:
            results[i] = 2
            target_letters[i] = None

    # Second pass marking misplaced letters
    for i in range(5):
        if results[i] == 0 and guess[i] in target_letters:
            results[i] = 1
            for j in range(5):
                if target_letters[j] == guess[i]:
                    target_letters[j] = None
                    break

    return results


def analyze_performance(df_possible_words, df_all_guesses=None, num_samples=100, debug=False):
    all_words = df_possible_words['word'].tolist()
    sample_indices = np.random.choice(range(len(all_words)), size=num_samples, replace=False)
    sample_words = [all_words[i] for i in sample_indices]

    results = []
    for i, word in enumerate(tqdm(sample_words, desc="Analyzing performance")):
        try:
            result = play_game(word, df_possible_words, df_all_guesses, debug=debug)
            results.append(result)
        except Exception as e:
            print(f"\nError analyzing word {word} (index {i}): {e}")
            continue

    if not results:
        print("No valid results collected. Check for errors.")
        return 0, 0, Counter()

    guesses = [r[1] for r in results]
    avg_guesses = sum(guesses) / len(guesses)
    success_rate = sum(1 for g in guesses if g <= 6) / len(guesses) * 100
    guess_dist = Counter(guesses)

    print(f"Solver Results:")
    print(f"Average guesses: {avg_guesses:.2f}")
    print(f"Success rate: {success_rate:.2f}%")
    print("Guess distribution:")
    for i in range(1, 7):
        count = guess_dist[i]
        percentage = count/len(guesses)*100 if guesses else 0
        print(f"  {i} guesses: {count} games ({percentage:.2f}%)")

    count = guess_dist[7]
    percentage = count/len(guesses)*100 if guesses else 0
    print(f"  Failed: {count} games ({percentage:.2f}%)")
    return avg_guesses, success_rate, guess_dist


def main():
    with open('answers.txt') as file:
        possible_answers = file.readlines()

    list_possible_answers = sorted([
        re.sub(r'[^A-Z]', '', t.upper()) for t in possible_answers[0].split(',')
    ])
    print(f"Loaded {len(list_possible_answers)} possible answers")

    arr_words_5l = np.array([list(w) for w in list_possible_answers])
    df_words_5l = pd.DataFrame(data=arr_words_5l,
                              columns=[f'letter_{i+1}' for i in range(5)])
    df_words_5l['word'] = list_possible_answers
    df_all_guesses = df_words_5l.copy()

    print("\n--- Testing solver on individual words ---")
    if len(list_possible_answers) >= 3:
        test_words = np.random.choice(list_possible_answers, size=3)
    else:
        test_words = list_possible_answers[:min(3, len(list_possible_answers))]

    for i, word in enumerate(test_words):
        print(f'\nGAME {i+1}: target {word}')
        play_game(word, df_words_5l, df_all_guesses, debug=True)

    print("\n--- Analyzing solver performance ---")
    sample_size = min(50, len(list_possible_answers))
    print(f"Testing solver on {sample_size} random words...")
    avg_guesses, success_rate, guess_dist = analyze_performance(
        df_words_5l, df_all_guesses, num_samples=sample_size
    )


if __name__ == "__main__":
    main()
