
import unittest
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path to allow importing transcribe
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import from the main script
from transcribe import ResultMerger, ASRResult, AudioChunk, MergedResult

class TestResultMergerLogic(unittest.TestCase):
    
    def setUp(self):
        self.merger = ResultMerger()

    def create_mock_asr_result(self, chunk_start: float, chunk_end: float, words: List[Dict[str, Any]]) -> ASRResult:
        chunk = AudioChunk(
            original_file="test_audio",
            chunk_path="dummy_path",
            start_time=chunk_start,
            end_time=chunk_end,
            duration=chunk_end - chunk_start
        )
        # Add 'score' implicitly by position relative to center? 
        # The merger calculates it. We just provide raw words.
        return ASRResult(
            chunk=chunk,
            raw_response="[]",
            segments=[{"words": words}],  # Populate segments as ResultMerger reads from here
            word_segments=words
        )
    
    def _assert_words(self, result_words: List[Dict[str, Any]], expected_words: List[Dict[str, Any]]):
        """Helper to compare result words with expected words using dictionary comparison."""
        self.assertEqual(len(result_words), len(expected_words), 
                         f"Length mismatch: {len(result_words)} != {len(expected_words)}. \nGot: {[w['word'] for w in result_words]}\nExp: {[w['word'] for w in expected_words]}")
        
        for i, (res, exp) in enumerate(zip(result_words, expected_words)):
            self.assertEqual(res["word"], exp["word"], f"Word mismatch at index {i}")
            if "start" in exp:
                self.assertAlmostEqual(res["start"], exp["start"], places=2, msg=f"Start time mismatch at index {i}")
            if "end" in exp:
                self.assertAlmostEqual(res["end"], exp["end"], places=2, msg=f"End time mismatch at index {i}")

    def test_basic_deduplication(self):
        """Test that identical overlapping words are merged (deduplicated)."""
        # Chunk 1: 0-10s. Word "test" at 8.0-9.0.
        words1 = [{"word": "test", "start": 8.0, "end": 9.0}]
        res1 = self.create_mock_asr_result(0.0, 10.0, words1)
        
        # Chunk 2: 8-18s. Word "test" at 8.0-9.0 (global) -> 0.0-1.0 (relative).
        words2 = [{"word": "test", "start": 0.0, "end": 1.0}]
        res2 = self.create_mock_asr_result(8.0, 18.0, words2)
        
        merged_list = self.merger.process([res1, res2])
        merged = merged_list[0]
        
        expected = [{"word": "test", "start": 8.0, "end": 9.0}]
        self._assert_words(merged.word_segments, expected)

    def test_conflict_resolution_center_bias(self):
        """
        Test that conflicts (different words in same spot) are resolved by 
        preferring the word closer to its chunk's center.
        """
        # Chunk 1: [0, 10]. Center 5.
        # Word A at 8.0-9.0. Midpoint 8.5. Dist to Center = |8.5 - 5| = 3.5.
        words1 = [{"word": "better_c1", "start": 8.0, "end": 9.0}]
        res1 = self.create_mock_asr_result(0.0, 10.0, words1)
        
        # Chunk 2: [8, 18]. Center 13.
        # Word B at 8.0-9.0 (global) -> 0.0-1.0 (relative).
        # Global Midpoint 8.5. Global Center 13. Dist to Center = |8.5 - 13| = 4.5.
        # Dist 3.5 < 4.5, so Chunk 1 is closer/better.
        words2 = [{"word": "worse_c2", "start": 0.0, "end": 1.0}]
        res2 = self.create_mock_asr_result(8.0, 18.0, words2)
        
        merged_list = self.merger.process([res1, res2])
        merged = merged_list[0]
        
        # Should pick "better_c1"
        expected = [{"word": "better_c1", "start": 8.0, "end": 9.0}]
        self._assert_words(merged.word_segments, expected)
        
    def test_conflict_resolution_center_bias_switch(self):
        """Test the switchover point where Chunk 2 becomes better."""
        # Chunk 1: [0, 10]. Center 5.
        # Word at 9.0-10.0. Mid 9.5. Dist |9.5-5| = 4.5.
        words1 = [{"word": "worse_c1", "start": 9.0, "end": 10.0}]
        res1 = self.create_mock_asr_result(0.0, 10.0, words1)
        
        # Chunk 2: [8, 18]. Center 13.
        # Word at 9.0-10.0 (global) -> 1.0-2.0 (relative).
        # Global Mid 9.5. Dist |9.5-13| = 3.5.
        # Dist 3.5 < 4.5, so Chunk 2 is closer/better.
        words2 = [{"word": "better_c2", "start": 1.0, "end": 2.0}]
        res2 = self.create_mock_asr_result(8.0, 18.0, words2)
        
        merged_list = self.merger.process([res1, res2])
        merged = merged_list[0]
        
        # Should pick "better_c2"
        expected = [{"word": "better_c2", "start": 9.0, "end": 10.0}]
        self._assert_words(merged.word_segments, expected)

    def test_repro_missing_word_resolved(self):
        """
        Verify the missing 'a' case still works with new logic.
        """
        # Chunk 1: [0, 10]. 'this'(8.0), 'is'(8.6)
        words1 = [
            {"word": "this", "start": 8.0, "end": 8.5},
            {"word": "is", "start": 8.6, "end": 8.9}
        ]
        res1 = self.create_mock_asr_result(0.0, 10.0, words1)
        
        # Chunk 2: [8, 18]. 'is'(8.6 global -> 0.6 relative), 'a'(9.0 global -> 1.0 relative), 'test'(9.3 global -> 1.3 relative)
        words2 = [
            {"word": "is", "start": 0.6, "end": 0.9},
            {"word": "a", "start": 1.0, "end": 1.2},
            {"word": "test", "start": 1.3, "end": 1.8}
        ]
        res2 = self.create_mock_asr_result(8.0, 18.0, words2)
        
        merged_list = self.merger.process([res1, res2])
        merged = merged_list[0]
        
        expected = [
            {"word": "this", "start": 8.0, "end": 8.5},
            {"word": "is", "start": 8.6, "end": 8.9},
            {"word": "a", "start": 9.0, "end": 9.2},
            {"word": "test", "start": 9.3, "end": 9.8}
        ]
        self._assert_words(merged.word_segments, expected)

    def test_chinese_overlap_basic(self):
        """
        Test merging overlapping Chinese characters.
        Chunk 1: [0, 10]. '你'(8.0), '好'(8.5)
        Chunk 2: [8, 18]. '好'(8.5), '世'(9.0), '界'(9.5)
        Expected: '你', '好', '世', '界'
        """
        words1 = [
            {"word": "你", "start": 8.0, "end": 8.5},
            {"word": "好", "start": 8.5, "end": 9.0}
        ]
        res1 = self.create_mock_asr_result(0.0, 10.0, words1)
        
        # Chunk 2: '好' at 8.5 (global) -> 0.5 (relative)
        words2 = [
            {"word": "好", "start": 0.5, "end": 1.0},
            {"word": "世", "start": 1.0, "end": 1.5},
            {"word": "界", "start": 1.5, "end": 2.0}
        ]
        res2 = self.create_mock_asr_result(8.0, 18.0, words2)
        
        merged_list = self.merger.process([res1, res2])
        merged = merged_list[0]
        
        expected = [
            {"word": "你", "start": 8.0, "end": 8.5},
            {"word": "好", "start": 8.5, "end": 9.0},
            {"word": "世", "start": 9.0, "end": 9.5},
            {"word": "界", "start": 9.5, "end": 10.0}
        ]
        self._assert_words(merged.word_segments, expected)

    def test_chinese_missing_word_in_overlap(self):
        """
        Test recovery when one chunk misses a character in the overlap region.
        Chunk 1: [0, 10]. '我'(8.0), '喜'(8.5), '欢'(9.0). (Misses '吃')
        Chunk 2: [8, 18]. '喜'(8.5), '欢'(9.0), '吃'(9.5), '苹'(10.0)
        Expected: '我', '喜', '欢', '吃', '苹'
        """
        words1 = [
            {"word": "我", "start": 8.0, "end": 8.5},
            {"word": "喜", "start": 8.5, "end": 9.0},
            {"word": "欢", "start": 9.0, "end": 9.5}
        ]
        res1 = self.create_mock_asr_result(0.0, 10.0, words1)
        
        # Chunk 2: '喜' at 8.5 global -> 0.5 relative
        words2 = [
            {"word": "喜", "start": 0.5, "end": 1.0},
            {"word": "欢", "start": 1.0, "end": 1.5},
            {"word": "吃", "start": 1.5, "end": 2.0},
            {"word": "苹", "start": 2.0, "end": 2.5}
        ]
        res2 = self.create_mock_asr_result(8.0, 18.0, words2)
        
        merged_list = self.merger.process([res1, res2])
        merged = merged_list[0]
        
        expected = [
            {"word": "我", "start": 8.0, "end": 8.5},
            {"word": "喜", "start": 8.5, "end": 9.0},
            {"word": "欢", "start": 9.0, "end": 9.5},
            {"word": "吃", "start": 9.5, "end": 10.0},
            {"word": "苹", "start": 10.0, "end": 10.5}
        ]
        self._assert_words(merged.word_segments, expected)

    def test_timestamp_offset_deduplication(self):
        """
        Test that words are merged even if timestamps are slightly offset.
        Chunk 1: 'hello' at 8.0-9.0
        Chunk 2: 'hello' at 8.05-9.05
        Expected: Merged 'hello' (union timestamps 8.0-9.05)
        """
        words1 = [{"word": "hello", "start": 8.0, "end": 9.0}]
        res1 = self.create_mock_asr_result(0.0, 10.0, words1)
        
        # Chunk 2: 'hello' at 8.05 global -> 0.05 relative
        words2 = [{"word": "hello", "start": 0.05, "end": 1.05}]
        res2 = self.create_mock_asr_result(8.0, 18.0, words2)
        
        merged_list = self.merger.process([res1, res2])
        merged = merged_list[0]
        
        # Expect union of timestamps
        expected = [{"word": "hello", "start": 8.0, "end": 9.05}]
        self._assert_words(merged.word_segments, expected)

    def test_garbage_edge_hallucination_filtered(self):
        """
        Test that garbage at the end of Chunk 1 is filtered out in favor of clean start of Chunk 2.
        Chunk 1 (Center 5): 'good'(8.0), 'garbage'(9.8-10.0)
        Chunk 2 (Center 13): 'good'(8.0), 'clean'(9.8-10.3)
        Conflict at 9.8: 'garbage' vs 'clean'.
        'garbage' mid=9.9. Dist to 5 = 4.9.
        'clean' mid=10.05. Dist to 13 = 2.95.
        2.95 < 4.9, so 'clean' should win.
        """
        words1 = [
            {"word": "good", "start": 8.0, "end": 9.0},
            {"word": "garbage", "start": 9.8, "end": 10.0}
        ]
        res1 = self.create_mock_asr_result(0.0, 10.0, words1)
        
        # Chunk 2: 'good' at 8.0 global -> 0.0 relative. 'clean' at 9.8 global -> 1.8 relative
        words2 = [
            {"word": "good", "start": 0.0, "end": 1.0},
            {"word": "clean", "start": 1.8, "end": 2.3}
        ]
        res2 = self.create_mock_asr_result(8.0, 18.0, words2)
        
        merged_list = self.merger.process([res1, res2])
        merged = merged_list[0]
        
        expected = [
            {"word": "good", "start": 8.0, "end": 9.0},
            {"word": "clean", "start": 9.8, "end": 10.3}
        ]
        self._assert_words(merged.word_segments, expected)

    def test_chinese_punctuation_diff(self):
        """
        Test conflict resolution when one chunk has punctuation and the other doesn't.
        Chunk 1: '好'(8.0), '吗'(8.5), '？'(9.0) -> Punctuation often attached to word '吗？' by cleaner
        But here we assume raw words are '吗' and '？' or '吗？'
        Let's assume cleaner produced attached punct: '吗？' vs '吗'
        Chunk 1 (Center 5): '吗？'(9.0-9.5). Mid 9.25. Dist to 5 = 4.25.
        Chunk 2 (Center 13): '吗'(9.0-9.5). Mid 9.25. Dist to 13 = 3.75.
        3.75 < 4.25, so Chunk 2 ('吗') wins.
        This demonstrates center bias prefers the start of the next chunk over the end of the previous one.
        """
        words1 = [
            {"word": "你", "start": 8.0, "end": 8.5},
            {"word": "好", "start": 8.5, "end": 9.0},
            {"word": "吗？", "start": 9.0, "end": 9.5}
        ]
        res1 = self.create_mock_asr_result(0.0, 10.0, words1)
        
        # Chunk 2: '好' at 8.5 global -> 0.5 relative. '吗' at 9.0 global -> 1.0 relative
        words2 = [
            {"word": "好", "start": 0.5, "end": 1.0},
            {"word": "吗", "start": 1.0, "end": 1.5}
        ]
        res2 = self.create_mock_asr_result(8.0, 18.0, words2)
        
        merged_list = self.merger.process([res1, res2])
        merged = merged_list[0]
        
        expected = [
            {"word": "你", "start": 8.0, "end": 8.5},
            {"word": "好", "start": 8.5, "end": 9.0},
            {"word": "吗", "start": 9.0, "end": 9.5}
        ]
        self._assert_words(merged.word_segments, expected)

    def test_long_overlap_mixed_errors(self):
        """
        Test a longer overlap sequence with errors in different places.
        Chunk 1: [0, 10]. Overlap [8, 10]. Words A(8.0), B(8.5), C(9.0), D(9.5).
        Chunk 2: [8, 18]. Overlap [8, 10]. Words A(8.0), B(8.5), C(9.0), D(9.5).
        
        Scenario:
        - 8.0: Both 'A'.
        - 8.5: Chunk 1 'B_bad' vs Chunk 2 'B_good'. (Dist to C1=3.5, C2=4.5 -> C1 wins 'B_bad')
        - 9.0: Chunk 1 'C_good' vs Chunk 2 'C_bad'. (Dist to C1=4.25, C2=3.75 -> C2 wins 'C_bad')
        - 9.5: Chunk 1 'D_bad' vs Chunk 2 'D_good'. (Dist to C1=4.75, C2=3.25 -> C2 wins 'D_good')
        
        This shows that Center Bias isn't perfect (it prefers C1 at 8.5 even if C1 is bad), 
        but it correctly switches to C2 at the end (9.5) where C1 is most likely to be bad (edge).
        It also switches at 9.0 (midpoint slightly favors C2).
        """
        words1 = [
            {"word": "A", "start": 8.0, "end": 8.5},
            {"word": "B_bad", "start": 8.5, "end": 9.0},
            {"word": "C_good", "start": 9.0, "end": 9.5},
            {"word": "D_bad", "start": 9.5, "end": 10.0}
        ]
        res1 = self.create_mock_asr_result(0.0, 10.0, words1)
        
        # Chunk 2 relative times:
        # 8.0->0.0, 8.5->0.5, 9.0->1.0, 9.5->1.5
        words2 = [
            {"word": "A", "start": 0.0, "end": 0.5},
            {"word": "B_good", "start": 0.5, "end": 1.0},
            {"word": "C_bad", "start": 1.0, "end": 1.5},
            {"word": "D_good", "start": 1.5, "end": 2.0}
        ]
        res2 = self.create_mock_asr_result(8.0, 18.0, words2)
        
        merged_list = self.merger.process([res1, res2])
        merged = merged_list[0]
        
        expected = [
            {"word": "A", "start": 8.0, "end": 8.5},
            {"word": "B_bad", "start": 8.5, "end": 9.0},   # C1 wins (closer to center)
            {"word": "C_bad", "start": 9.0, "end": 9.5},   # C2 wins (closer to center)
            {"word": "D_good", "start": 9.5, "end": 10.0}  # C2 wins (closer to center)
        ]
        self._assert_words(merged.word_segments, expected)

if __name__ == "__main__":
    unittest.main()
