import unittest
from dataclasses import dataclass, field
from typing import List, Dict, Any

import sys
import os
# Add parent directory to path to allow importing transcribe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transcribe import PostProcessor, ASRResult, AudioChunk

class TestPostProcessor(unittest.TestCase):
    def setUp(self):
        self.processor = PostProcessor()
        # Provide all required arguments for AudioChunk
        self.chunk = AudioChunk(
            original_file="test.wav",
            chunk_path="/tmp/test.wav",
            start_time=0.0,
            end_time=10.0,
            duration=10.0
        )

    def _run_clean(self, words, original_text="", language="zh"):
        """Helper method to run clean_word_alignments concisely."""
        res = ASRResult(
            chunk=self.chunk,
            raw_response="[]",
            segments=[{
                "text": original_text,
                "words": words,
                "start": words[0].get("start", 0.0) if words else 0.0,
                "end": words[-1].get("end", 0.0) if words else 0.0
            }],
            language=language
        )
        cleaned = self.processor.clean_word_alignments([res])
        seg = cleaned[0].segments[0]
        return seg.get("words", []), seg.get("text", "")

    def test_english_fragment_merge(self):
        """Test merging 'H', 'e', 'l', 'l', 'o' -> 'Hello'"""
        words = [
            {"word": "H", "start": 0.0, "end": 0.1},
            {"word": "e", "start": 0.1, "end": 0.2},
            {"word": "l", "start": 0.2, "end": 0.3},
            {"word": "l", "start": 0.3, "end": 0.4},
            {"word": "o", "start": 0.4, "end": 0.5},
        ]
        result_words, _ = self._run_clean(words, "Hello")
        expected_words = [
            {"word": "Hello", "start": 0.0, "end": 0.5}
        ]
        self.assertEqual(result_words, expected_words)

    def test_english_word_separation(self):
        """Test 'Hello', 'World' remain separate if space exists in original text"""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "World", "start": 0.6, "end": 1.0},
        ]
        result_words, _ = self._run_clean(words, "Hello World")
        expected_words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "World", "start": 0.6, "end": 1.0}
        ]
        self.assertEqual(result_words, expected_words)

    def test_multiple_english_words_fragment_merge(self):
        """Test merging fragments 'H','e','l','l','o','W','o','r','l','d' -> 'Hello', 'World' based on original text space"""
        words = [
            {"word": "H", "start": 0.0, "end": 0.1},
            {"word": "e", "start": 0.1, "end": 0.2},
            {"word": "l", "start": 0.2, "end": 0.3},
            {"word": "l", "start": 0.3, "end": 0.4},
            {"word": "o", "start": 0.4, "end": 0.5},
            {"word": "W", "start": 0.5, "end": 0.6},
            {"word": "o", "start": 0.6, "end": 0.7},
            {"word": "r", "start": 0.7, "end": 0.8},
            {"word": "l", "start": 0.8, "end": 0.9},
            {"word": "d", "start": 0.9, "end": 1.0},
        ]
        result_words, _ = self._run_clean(words, "Hello World")
        expected_words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "World", "start": 0.5, "end": 1.0}
        ]
        self.assertEqual(result_words, expected_words)

    def test_leading_punctuation(self):
        """Test leading punctuation '《' attaching to next word 'Title'"""
        words = [
            {"word": "《", "start": 0.0, "end": 0.1},
            {"word": "Title", "start": 0.2, "end": 0.6},
        ]
        result_words, _ = self._run_clean(words, "《Title")
        expected_words = [
            {"word": "《Title", "start": 0.0, "end": 0.6}
        ]
        self.assertEqual(result_words, expected_words)

    def test_trailing_punctuation(self):
        """Test trailing punctuation '。' attaching to previous word 'End'"""
        words = [
            {"word": "End", "start": 0.0, "end": 0.4},
            {"word": "。", "start": 0.4, "end": 0.5},
        ]
        result_words, _ = self._run_clean(words, "End。")
        expected_words = [
            {"word": "End。", "start": 0.0, "end": 0.5}
        ]
        self.assertEqual(result_words, expected_words)

    def test_complex_mixed_sequence(self):
        """Test mixed sequence: 'Prefix', '(', 'Inside', ')', 'Suffix', '.'"""
        words = [
            {"word": "Prefix", "start": 0.0, "end": 0.5},
            {"word": "(", "start": 0.6, "end": 0.7},
            {"word": "Inside", "start": 0.8, "end": 1.2},
            {"word": ")", "start": 1.3, "end": 1.4},
            {"word": "Suffix", "start": 1.5, "end": 2.0},
            {"word": ".", "start": 2.0, "end": 2.1},
        ]
        result_words, _ = self._run_clean(words, "Prefix (Inside) Suffix.")
        expected_words = [
            {"word": "Prefix", "start": 0.0, "end": 0.5},
            {"word": "(Inside)", "start": 0.6, "end": 1.4},
            {"word": "Suffix.", "start": 1.5, "end": 2.1}
        ]
        self.assertEqual(result_words, expected_words)

    def test_multiple_leading_punctuation(self):
        """Test multiple leading punctuation: '“', '《', 'Title' -> '“《Title'"""
        words = [
            {"word": "“", "start": 0.0, "end": 0.1},
            {"word": "《", "start": 0.1, "end": 0.2},
            {"word": "Title", "start": 0.3, "end": 0.8},
        ]
        result_words, _ = self._run_clean(words, "“《Title")
        expected_words = [
            {"word": "“《Title", "start": 0.0, "end": 0.8}
        ]
        self.assertEqual(result_words, expected_words)

    def test_punctuation_with_spaces_and_deduplication(self):
        """Test trailing punctuation with spaces and deduplication"""
        words = [
            {"word": "Hello ", "start": 0.0, "end": 0.5},
            {"word": " , ", "start": 0.5, "end": 0.6},
            {"word": "world", "start": 0.7, "end": 1.0},
            {"word": " . ", "start": 1.0, "end": 1.1},
        ]
        result_words, result_text = self._run_clean(words, "Hello , world .", language="en")
        expected_words = [
            {"word": "Hello,", "start": 0.0, "end": 0.6},
            {"word": "world.", "start": 0.7, "end": 1.1}
        ]
        self.assertEqual(result_words, expected_words)
        self.assertEqual(result_text, "Hello, world.")

    def test_leftover_leading_punctuation(self):
        """Test leftover leading punctuation at the end of the segment"""
        words = [
            {"word": "Hello", "start": 0.0, "end": 0.5},
            {"word": "(", "start": 0.5, "end": 0.6},
        ]
        result_words, _ = self._run_clean(words, "Hello (", language="en")
        expected_words = [
            {"word": "Hello(", "start": 0.0, "end": 0.5}
        ]
        self.assertEqual(result_words, expected_words)

    def test_mixed_chinese_english_seg_text(self):
        """Test segment text reconstruction for mixed Chinese and English"""
        words = [
            {"word": "我", "start": 0.0, "end": 0.1},
            {"word": "爱", "start": 0.1, "end": 0.2},
            {"word": "P", "start": 0.2, "end": 0.3},
            {"word": "y", "start": 0.3, "end": 0.4},
            {"word": "t", "start": 0.4, "end": 0.5},
            {"word": "h", "start": 0.5, "end": 0.6},
            {"word": "o", "start": 0.6, "end": 0.7},
            {"word": "n", "start": 0.7, "end": 0.8},
            {"word": "编", "start": 0.8, "end": 0.9},
            {"word": "程", "start": 0.9, "end": 1.0},
        ]
        result_words, result_text = self._run_clean(words, "我爱Python编程")
        expected_words = [
            {"word": "我", "start": 0.0, "end": 0.1},
            {"word": "爱", "start": 0.1, "end": 0.2},
            {"word": "Python", "start": 0.2, "end": 0.8},
            {"word": "编", "start": 0.8, "end": 0.9},
            {"word": "程", "start": 0.9, "end": 1.0}
        ]
        self.assertEqual(result_words, expected_words)
        self.assertEqual(result_text, "我爱Python编程")

    def test_only_punctuation_in_segment(self):
        """Test segment with only punctuation"""
        words = [
            {"word": "...", "start": 0.0, "end": 0.5},
        ]
        result_words, _ = self._run_clean(words, "...")
        expected_words = [
            {"word": "...", "start": 0.0, "end": 0.5}
        ]
        self.assertEqual(result_words, expected_words)

    def test_empty_segment(self):
        """Test segment with empty words array"""
        result_words, _ = self._run_clean([], "")
        self.assertEqual(result_words, [])

    def test_missing_word_or_end_key(self):
        """Test robustness when 'word' or 'end' keys are missing"""
        words = [
            {"start": 0.0, "end": 0.1},
            {"word": "Hello", "start": 0.1},
            {"word": "!", "start": 0.2},
        ]
        result_words, _ = self._run_clean(words, "Hello!", language="en")
        expected_words = [
            {"start": 0.0, "end": 0.1},
            {"word": "Hello!", "start": 0.1}
        ]
        self.assertEqual(result_words, expected_words)

    def test_duplicate_trailing_punctuation(self):
        """Test duplicate trailing punctuation is not appended repeatedly"""
        words = [
            {"word": "Hello,", "start": 0.0, "end": 0.5},
            {"word": ",", "start": 0.5, "end": 0.6},
        ]
        result_words, _ = self._run_clean(words, "Hello,", language="en")
        expected_words = [
            {"word": "Hello,", "start": 0.0, "end": 0.6}
        ]
        self.assertEqual(result_words, expected_words)

    def test_multiple_trailing_punctuations(self):
        """Test multiple distinct trailing punctuations (e.g. 'Word', '.', '"')"""
        words = [
            {"word": "Word", "start": 0.0, "end": 0.5},
            {"word": ".", "start": 0.5, "end": 0.6},
            {"word": '"', "start": 0.6, "end": 0.7},
        ]
        result_words, _ = self._run_clean(words, 'Word."', language="en")
        expected_words = [
            {"word": 'Word."', "start": 0.0, "end": 0.6}
        ]
        self.assertEqual(result_words, expected_words)

    def test_trailing_punctuation_at_beginning(self):
        """Test trailing punctuation appearing at the very beginning of the segment"""
        words = [
            {"word": "，", "start": 0.0, "end": 0.1},
            {"word": "你好", "start": 0.1, "end": 0.5},
        ]
        result_words, _ = self._run_clean(words, "，你好")
        expected_words = [
            {"word": "，", "start": 0.0, "end": 0.1},
            {"word": "你好", "start": 0.1, "end": 0.5}
        ]
        self.assertEqual(result_words, expected_words)

    def test_number_fragment_merge(self):
        """Test merging number fragments '1', '2', '3' -> '123' in zh context"""
        words = [
            {"word": "1", "start": 0.0, "end": 0.1},
            {"word": "2", "start": 0.1, "end": 0.2},
            {"word": "3", "start": 0.2, "end": 0.3},
        ]
        result_words, result_text = self._run_clean(words, "123")
        expected_words = [
            {"word": "123", "start": 0.0, "end": 0.3}
        ]
        self.assertEqual(result_words, expected_words)
        self.assertEqual(result_text, "123")

    def test_mixed_english_number_merge(self):
        """Test merging mixed english and number 'A', '1', 'B' -> 'A1B' in zh context"""
        words = [
            {"word": "A", "start": 0.0, "end": 0.1},
            {"word": "1", "start": 0.1, "end": 0.2},
            {"word": "B", "start": 0.2, "end": 0.3},
        ]
        result_words, result_text = self._run_clean(words, "A1B")
        expected_words = [
            {"word": "A1B", "start": 0.0, "end": 0.3}
        ]
        self.assertEqual(result_words, expected_words)
        self.assertEqual(result_text, "A1B")

    def test_english_words_no_space_in_original_zh(self):
        """Test english words without space in original text in zh context (should merge)"""
        words = [
            {"word": "Apple", "start": 0.0, "end": 0.5},
            {"word": "Pie", "start": 0.5, "end": 1.0},
        ]
        result_words, result_text = self._run_clean(words, "ApplePie")
        expected_words = [
            {"word": "ApplePie", "start": 0.0, "end": 1.0}
        ]
        self.assertEqual(result_words, expected_words)
        self.assertEqual(result_text, "ApplePie")

    def test_english_words_with_space_in_original_zh(self):
        """Test english words with space in original text in zh context (should NOT merge, and space in seg text)"""
        words = [
            {"word": "Apple", "start": 0.0, "end": 0.5},
            {"word": "Pie", "start": 0.5, "end": 1.0},
        ]
        result_words, result_text = self._run_clean(words, "Apple Pie")
        expected_words = [
            {"word": "Apple", "start": 0.0, "end": 0.5},
            {"word": "Pie", "start": 0.5, "end": 1.0}
        ]
        self.assertEqual(result_words, expected_words)
        self.assertEqual(result_text, "Apple Pie")

    def test_mixed_language_spacing(self):
        """Test spacing rules when English and Chinese words are adjacent"""
        words = [
            {"word": "我", "start": 0.0, "end": 0.1},
            {"word": "用", "start": 0.1, "end": 0.2},
            {"word": "Mac", "start": 0.2, "end": 0.5},
            {"word": "Book", "start": 0.5, "end": 0.8},
            {"word": "编", "start": 0.8, "end": 0.9},
            {"word": "程", "start": 0.9, "end": 1.0},
        ]
        result_words, result_text = self._run_clean(words, "我用Mac Book编程")
        expected_words = [
            {"word": "我", "start": 0.0, "end": 0.1},
            {"word": "用", "start": 0.1, "end": 0.2},
            {"word": "Mac", "start": 0.2, "end": 0.5},
            {"word": "Book", "start": 0.5, "end": 0.8},
            {"word": "编", "start": 0.8, "end": 0.9},
            {"word": "程", "start": 0.9, "end": 1.0}
        ]
        self.assertEqual(result_words, expected_words)
        self.assertEqual(result_text, "我用Mac Book编程")

    def test_punctuation_without_timestamps(self):
        """Test that punctuation missing start/end fields correctly attaches to adjacent words"""
        # Leading punctuation without timestamps
        words_leading = [
            {"word": "《"},  # No start/end
            {"word": "Title", "start": 0.1, "end": 0.5},
        ]
        result_leading, _ = self._run_clean(words_leading, "《Title")
        expected_leading = [
            {"word": "《Title", "start": 0.1, "end": 0.5}
        ]
        self.assertEqual(result_leading, expected_leading)

        # Trailing punctuation without timestamps
        words_trailing = [
            {"word": "End", "start": 0.0, "end": 0.4},
            {"word": "。"},  # No start/end
        ]
        result_trailing, _ = self._run_clean(words_trailing, "End。")
        expected_trailing = [
            {"word": "End。", "start": 0.0, "end": 0.4}
        ]
        self.assertEqual(result_trailing, expected_trailing)

        # Middle punctuation without timestamps
        words_middle = [
            {"word": "Hello", "start": 0.0, "end": 0.3},
            {"word": ","},  # No start/end
            {"word": "world", "start": 0.4, "end": 0.8},
        ]
        result_middle, _ = self._run_clean(words_middle, "Hello, world", language="en")
        expected_middle = [
            {"word": "Hello,", "start": 0.0, "end": 0.3},
            {"word": "world", "start": 0.4, "end": 0.8}
        ]
        self.assertEqual(result_middle, expected_middle)

if __name__ == "__main__":
    unittest.main()
