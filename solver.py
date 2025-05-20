import nltk
import string
import re
import pandas as pd
import numpy as np
import math
from collections import defaultdict, Counter
from tqdm import tqdm
from matplotlib import pyplot as plt
nltk.download('words', quiet=True)

class Game:
    def __init__(self, df_all_5l_words):
        # Start with whole alphabet as list of possible letters in word
        self.possible_letters = list(string.ascii_uppercase)

        # Dictionary to store letter information by position
        self.letter_info = {
            'correct': {},     # Position -> Letter
            'misplaced': {},   # Letter -> [Positions]
            'wrong': set()     # Letters that are not in the word
        }

        # To store minimum counts of letters in word
        self.min_letter_counts = Counter()

        # To store maximum counts of letters in word
        self.max_letter_counts = Counter()
        for letter in self.possible_letters:
            self.max_letter_counts[letter] = 5

        # Possible answers
        self.df_possible_5l_words = df_all_5l_words.copy()

        # Track all words for entropy calculation
        self.all_words = self.df_possible_5l_words['word'].tolist()

        # Set of target words still possible
        self.possible_words = set(self.all_words)

        # Initialize the cache for pattern calculations
        self.pattern_cache = {}

        # First guess optimization: hard-code the best first guess
        # This is based on entropy calculation precomputed for the entire answer set
        self.first_guess = "SOARE"  # Statistically proven to be optimal first guess
        self.turn = 0

    def calculate_entropy(self, guess_word):
        """
        Calculate the information entropy of a guess word against the possible words.
        Higher entropy means the guess provides more information about the target word.
        """
        if len(self.possible_words) <= 1:
            return 0

        # Bucket words by their response pattern
        pattern_buckets = defaultdict(int)
        total_words = len(self.possible_words)

        for target in self.possible_words:
            pattern = self._get_pattern(guess_word, target)
            pattern_buckets[pattern] += 1

        # Calculate entropy
        entropy = 0
        for count in pattern_buckets.values():
            prob = count / total_words
            entropy -= prob * math.log2(prob)

        return entropy

    def _get_pattern(self, guess, target):
        """
        Get the response pattern (0=wrong, 1=misplaced, 2=correct) for a guess against a target word.
        Uses a cache for efficiency.
        """
        cache_key = (guess, target)
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]

        guess = guess.upper()
        target = target.upper()

        # First pass: mark correct letters
        result = [0] * 5
        target_letters = list(target)

        for i in range(5):
            if guess[i] == target_letters[i]:
                result[i] = 2
                target_letters[i] = None  # Mark as used

        # Second pass: mark misplaced letters
        for i in range(5):
            if result[i] == 0 and guess[i] in target_letters:
                result[i] = 1
                target_letters[target_letters.index(guess[i])] = None  # Mark as used

        pattern = tuple(result)
        self.pattern_cache[cache_key] = pattern
        return pattern

    def guess(self):
        """
        Return the best guess based on entropy calculation or the hard-coded first guess.
        """
        self.turn += 1

        # Fallback if we somehow have no possible words left
        if len(self.possible_words) == 0:
            print("Warning: No possible words left. Using a fallback word.")
            return pd.DataFrame([{
                'letter_1': 'S', 'letter_2': 'O', 'letter_3': 'A',
                'letter_4': 'R', 'letter_5': 'E', 'word': 'SOARE'
            }])

        # Use the optimal first guess if it's the first turn
        if self.turn == 1:
            first_guess_df = self.df_possible_5l_words[self.df_possible_5l_words['word'] == self.first_guess]
            if not first_guess_df.empty:
                return first_guess_df
            else:
                # If first guess not in dataframe, use first possible word
                word = list(self.possible_words)[0]
                return self.df_possible_5l_words[self.df_possible_5l_words['word'] == word]

        # If only one word remains, choose it
        if len(self.possible_words) == 1:
            word = list(self.possible_words)[0]
            return self.df_possible_5l_words[self.df_possible_5l_words['word'] == word]

        # If only a few words remain, prefer words from the possible answers
        if len(self.possible_words) <= 3:
            best_word = None
            best_entropy = -1

            for word in self.possible_words:
                entropy = self.calculate_entropy(word)
                if entropy > best_entropy:
                    best_entropy = entropy
                    best_word = word

            if best_word:
                return self.df_possible_5l_words[self.df_possible_5l_words['word'] == best_word]

        # For efficiency, use a sample of words for entropy calculation if many remain
        candidate_words = self.possible_words
        if len(self.possible_words) > 50:
            # Use a balanced approach: check all possible answers and a sample of other words
            possible_words_list = list(self.possible_words)
            sample_size = min(50, len(self.all_words) - len(self.possible_words))
            other_words = set(self.all_words) - self.possible_words
            if other_words:
                other_words_sample = np.random.choice(list(other_words),
                                                     size=min(sample_size, len(other_words)),
                                                     replace=False)
                candidate_words = set(possible_words_list) | set(other_words_sample)

        # Calculate entropy for each candidate and choose the best
        best_word = None
        best_entropy = -1

        for word in candidate_words:
            entropy = self.calculate_entropy(word)

            # Prefer words that are in the possible answers list
            if word in self.possible_words:
                entropy *= 1.01  # Small bonus for words that could be the answer

            if entropy > best_entropy:
                best_entropy = entropy
                best_word = word

        # Safety check
        if not best_word:
            best_word = list(self.possible_words)[0]

        result = self.df_possible_5l_words[self.df_possible_5l_words['word'] == best_word]
        if result.empty:
            # Fallback if the word isn't in the dataframe for some reason
            return pd.DataFrame([{
                'letter_1': best_word[0], 'letter_2': best_word[1],
                'letter_3': best_word[2], 'letter_4': best_word[3],
                'letter_5': best_word[4], 'word': best_word
            }])
        return result

    def update(self, guess, results):
        """
        Update the game state based on the guess and results.
        """
        guess = guess.upper()
        assert len(guess) == 5, 'Guess must be 5 characters long'
        assert len(results) == 5, 'Results list must contain 5 items'
        assert all([n in [0, 1, 2] for n in results]), 'Results list must only contain 0, 1, or 2'

        # Count letters in the guess
        guess_letter_counts = Counter(guess)

        # Track confirmed letter counts (from correct and misplaced positions)
        confirmed_letters = Counter()

        # Process correct letters (results = 2)
        for i, (letter, result) in enumerate(zip(guess, results)):
            if result == 2:
                # Add to correct letters
                self.letter_info['correct'][i] = letter
                confirmed_letters[letter] += 1

        # Process misplaced letters (results = 1)
        for i, (letter, result) in enumerate(zip(guess, results)):
            if result == 1:
                # Add to misplaced letters
                if letter not in self.letter_info['misplaced']:
                    self.letter_info['misplaced'][letter] = []
                self.letter_info['misplaced'][letter].append(i)
                confirmed_letters[letter] += 1

        # Process wrong letters (results = 0)
        for i, (letter, result) in enumerate(zip(guess, results)):
            if result == 0:
                # If letter appears elsewhere in the guess with result 1 or 2,
                # it means the word contains exactly the confirmed count of this letter
                if letter in confirmed_letters:
                    self.max_letter_counts[letter] = confirmed_letters[letter]
                else:
                    # Letter is not in the word at all
                    self.letter_info['wrong'].add(letter)
                    self.max_letter_counts[letter] = 0

        # Update minimum counts for confirmed letters
        for letter, count in confirmed_letters.items():
            if count > self.min_letter_counts[letter]:
                self.min_letter_counts[letter] = count

        # Filter possible words based on the new constraints
        self._update_possible_words()

    def _update_possible_words(self):
        """
        Filter the possible words based on all constraints.
        """
        new_possible_words = set()

        for word in self.possible_words:
            if self._is_word_possible(word):
                new_possible_words.add(word)

        self.possible_words = new_possible_words

        # Update the dataframe for compatibility with original code
        mask = self.df_possible_5l_words['word'].isin(self.possible_words)
        self.df_possible_5l_words = self.df_possible_5l_words[mask].copy()

    def _is_word_possible(self, word):
        """
        Check if a word satisfies all the constraints.
        """
        word = word.upper()

        # Check correct positions
        for pos, letter in self.letter_info['correct'].items():
            if word[pos] != letter:
                return False

        # Check misplaced positions
        for letter, positions in self.letter_info['misplaced'].items():
            for pos in positions:
                if word[pos] == letter:
                    return False
            if letter not in word:
                return False

        # Check wrong letters
        for letter in self.letter_info['wrong']:
            if letter in word:
                return False

        # Check letter counts
        word_letter_counts = Counter(word)
        for letter, min_count in self.min_letter_counts.items():
            if word_letter_counts[letter] < min_count:
                return False

        for letter, max_count in self.max_letter_counts.items():
            if word_letter_counts[letter] > max_count:
                return False

        return True


def play_game(target_word, df_possible_words, debug=False):
    """
    Play a game of Wordle with the given target word.
    """
    target_word = re.sub(r'[^A-Z]', '', target_word.upper())
    assert len(target_word) == 5, 'target_word must be 5 characters long'

    # Initialize game
    game = Game(df_possible_words)

    for guess_turn in range(6):
        # Get the best guess
        guess_df = game.guess()

        try:
            guess_word = guess_df.iloc[0]['word']
        except (IndexError, KeyError) as e:
            if debug:
                print(f"Error selecting word: {e}")
                print(f"DataFrame: {guess_df}")
            # Use a fallback word
            guess_word = "SOARE"

        # Generate results
        results = generate_results(guess_word, target_word)

        if debug:
            print(f'Turn {guess_turn+1}, guess {guess_word}, results {results}')
            print(f'Possible words remaining: {len(game.possible_words)}')

        # Check if we've won
        if sum(results) == 10:  # All 5 letters correct (5 * 2 = 10)
            if debug:
                print('Game won!')
            else:
                # Return the number of guesses needed
                return (target_word, guess_turn+1)
            break

        # Update game state
        game.update(guess_word, results)

        # If we've exhausted all guesses
        if guess_turn == 5:
            if debug:
                print('Unsolved!')
                if len(game.possible_words) <= 10:
                    print(f'Remaining possible words: {game.possible_words}')
                else:
                    print(f'{len(game.possible_words)} words remain possible')
            else:
                return (target_word, 7)  # 7 indicates failure after 6 guesses

    # Return the number of guesses needed
    return (target_word, guess_turn+1)


def generate_results(guess, target):
    """
    Generate the result pattern for a guess against a target word.
    """
    guess = guess.upper()
    target = target.upper()
    results = [0] * 5
    target_letters = list(target)

    # First pass: mark correct letters
    for i in range(5):
        if guess[i] == target_letters[i]:
            results[i] = 2
            target_letters[i] = None  # Mark as used

    # Second pass: mark misplaced letters
    for i in range(5):
        if results[i] == 0 and guess[i] in target_letters:
            results[i] = 1
            # Find and mark the position as used
            for j in range(5):
                if target_letters[j] == guess[i]:
                    target_letters[j] = None
                    break

    return results


def analyze_performance(df_possible_words, num_samples=100, debug=False):
    """
    Analyze the performance of the solver on random target words.
    """
    # Select random target words
    all_words = df_possible_words['word'].tolist()
    sample_indices = np.random.choice(range(len(all_words)), size=num_samples, replace=False)
    sample_words = [all_words[i] for i in sample_indices]

    # Play games with each word
    results = []
    for i, word in enumerate(tqdm(sample_words, desc="Analyzing performance")):
        try:
            result = play_game(word, df_possible_words, debug=debug)
            results.append(result)
        except Exception as e:
            print(f"\nError analyzing word {word} (index {i}): {e}")
            continue

    if not results:
        print("No valid results collected. Check for errors.")
        return 0, 0, Counter()

    # Compile statistics
    guesses = [r[1] for r in results]
    avg_guesses = sum(guesses) / len(guesses)
    success_rate = sum(1 for g in guesses if g <= 6) / len(guesses) * 100

    # Distribution of guesses
    guess_dist = Counter(guesses)

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
    # Load the word list

    with open('answers.txt') as file:
        possible_answers = file.readlines()

    list_possible_answers = sorted([
        re.sub(r'[^A-Z]', '', t.upper()) for t in possible_answers[0].split(',')
    ])
    print(f"Loaded {len(list_possible_answers)} possible answers")

    # Create dataframe
    arr_words_5l = np.array([list(w) for w in list_possible_answers])
    df_words_5l = pd.DataFrame(data=arr_words_5l,
                              columns=[f'letter_{i+1}' for i in range(5)])
    df_words_5l['word'] = list_possible_answers

    # Test with a few example words
    print("\n--- Testing individual words ---")
    if len(list_possible_answers) >= 3:
        test_words = np.random.choice(list_possible_answers, size=3)
    else:
        test_words = list_possible_answers[:min(3, len(list_possible_answers))]

    for i, word in enumerate(test_words):
        print(f'\n**GAME {i+1}: target {word}**')
        play_game(word, df_words_5l, debug=True)

    # Analyze performance on a sample
    print("\n--- Analyzing overall performance ---")
    sample_size = min(25, len(list_possible_answers))
    print(f"Testing on {sample_size} random words...")
    avg_guesses, success_rate, guess_dist = analyze_performance(df_words_5l, num_samples=sample_size)

    print("\nSolver performance summary:")
    print(f"- Average guesses needed: {avg_guesses:.2f}")
    print(f"- Success rate: {success_rate:.2f}%")
    if guess_dist:
        most_common = guess_dist.most_common(1)[0]
        print(f"- Most common guess count: {most_common[0]} ({most_common[1]} games)")
    else:
        print("- No performance data collected")

    print("\nTo run a full benchmark with more words, adjust the sample_size parameter.")


if __name__ == "__main__":
    main()
