import unittest
import sys
import re
from unittest.mock import MagicMock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from subtitle.handler import (
    LanguageHandler, EnglishHandler, ChineseHandler, LanguageContext, language_context,
    remove_punctuation, split_text_by_word, add_spaces_around_english,
    is_all_english, count_chinese_word, add_punctuation
)
from subtitle.subtitle import Element, Elements

class TestHandler(unittest.TestCase):
    def test_add_punctuation(self):
        mock_model = MagicMock()
        mock_model.restore_punctuation.return_value = "Hello, world!"
        text = "Hello world"
        result = add_punctuation(text, mock_model)
        mock_model.restore_punctuation.assert_called_once_with(text)
        self.assertEqual(result, "Hello, world!")

    def test_remove_punctuation(self):
        text = "Hello, world! This is a test. 100% $100."
        self.assertEqual(remove_punctuation(text), "Hello world This is a test 100% \\$100")
        self.assertEqual(remove_punctuation("a -- b — c - d"), "a b c d") # double hyphen logic

    def test_split_text_by_word(self):
        # English
        self.assertEqual(split_text_by_word("Hello world", lang="en"), ["Hello", "world"])
        # Chinese (fallback if jieba is not fully tested here, but we can test list split)
        zh_res = split_text_by_word("你好世界", lang="zh")
        self.assertTrue(isinstance(zh_res, list))

    def test_add_spaces_around_english(self):
        text = "这是一个test测试。"
        self.assertEqual(add_spaces_around_english(text), "这是一个 test 测试。")
        text2 = "你好Hello world世界"
        self.assertEqual(add_spaces_around_english(text2), "你好 Hello world 世界")

    def test_language_context(self):
        LanguageContext.set_language("en")
        self.assertEqual(LanguageContext.get_language(), "en")
        with language_context("zh"):
            self.assertEqual(LanguageContext.get_language(), "zh")
        self.assertEqual(LanguageContext.get_language(), "en")

    def test_guess_language(self):
        self.assertEqual(LanguageHandler.guess_language("Hello world"), "en")
        self.assertEqual(LanguageHandler.guess_language("你好，世界"), "zh")

    def test_guess_handler(self):
        self.assertIsInstance(LanguageHandler.guess_handler("Hello world"), EnglishHandler)
        self.assertIsInstance(LanguageHandler.guess_handler("你好，世界"), ChineseHandler)

class TestEnglishHandler(unittest.TestCase):
    def setUp(self):
        self.handler = EnglishHandler()

    def test_is_end(self):
        self.assertTrue(self.handler.is_end("Hello.", True))
        self.assertTrue(self.handler.is_end("Hello?", True))
        self.assertTrue(self.handler.is_end("Hello!", True))
        self.assertFalse(self.handler.is_end("Hello", True))
        self.assertFalse(self.handler.is_end("Mr.", True))
        
        self.assertTrue(self.handler.is_end("Hello,", False))
        self.assertTrue(self.handler.is_end("Hello:", False))

    def test_clean_punctuation(self):
        self.assertEqual(self.handler.clean_punctuation('"Hello!"'), "Hello")
        self.assertEqual(self.handler.clean_punctuation('Hello,'), "Hello")

    def test_length(self):
        self.assertEqual(self.handler.length("Hello"), 5)

    def test_split_and_join(self):
        text = "Hello world from EnglishHandler"
        words = self.handler.split(text)
        self.assertEqual(words, ["Hello", "world", "from", "EnglishHandler"])
        self.assertEqual(self.handler.join(words), text)

    def test_correct_incorrect_restore_punctuation(self):
        before = "hello.world this is a test"
        after = "hello.world, this is a test."
        self.assertEqual(self.handler.correct_incorrect_restore_punctuation(before, after), "hello.world this is a test.")

    def test_find_half_join_balance(self):
        # Two elements should return [1]
        self.assertEqual(self.handler.find_half_join_balance(["A", "B"]), [1])
        # Three elements, balance depends on lengths
        res = self.handler.find_half_join_balance(["This is long", "short", "short"])
        self.assertTrue(isinstance(res, list))
        self.assertTrue(len(res) > 0)

class TestChineseHandler(unittest.TestCase):
    def setUp(self):
        self.handler = ChineseHandler()

    def test_is_end(self):
        self.assertTrue(self.handler.is_end("你好。", True))
        self.assertTrue(self.handler.is_end("你好？", True))
        self.assertFalse(self.handler.is_end("你好", True))
        self.assertTrue(self.handler.is_end("你好，", False))

    def test_clean_punctuation(self):
        self.assertEqual(self.handler.clean_punctuation("《你好》！"), "你好")

    def test_length(self):
        self.assertEqual(self.handler.length("你好"), 2)
        # ChineseHandler calculates length of ascii words differently: (len + 1) // 2
        self.assertEqual(self.handler.length("Hello"), 3)

    def test_split_and_join(self):
        text = "你好，世界！This is test。"
        words = self.handler.split(text)
        self.assertTrue(isinstance(words, list))
        joined = self.handler.join(words)
        # Should join back properly
        self.assertTrue(isinstance(joined, str))

    def test_norm_and_beauty(self):
        text = "Hello世界"
        # It should add spaces between English and Chinese
        self.assertEqual(self.handler.norm_and_beauty(text), "Hello 世界")

    def test_isascii(self):
        self.assertTrue(self.handler.isascii("a"))
        self.assertTrue(self.handler.isascii("1"))
        self.assertFalse(self.handler.isascii("你"))
        self.assertFalse(self.handler.isascii("。"))

class TestLanguageHandlerBaseMethods(unittest.TestCase):
    def setUp(self):
        self.en_handler = EnglishHandler()
        self.zh_handler = ChineseHandler()

    def test_split_by_max_length(self):
        text = "This is a very long sentence that needs to be split into multiple lines."
        # max_length of 20
        lines = self.en_handler.split_by_max_length(text, 20)
        self.assertTrue(len(lines) > 1)
        
    def test_split_by_max_length_order(self):
        text = "This is a very long sentence that needs to be split into multiple lines."
        lines = self.en_handler.split_by_max_length_order(text, 20)
        self.assertTrue(len(lines) > 1)

    def test_merge_by_max_length(self):
        texts = ["Hello", "world", "this", "is", "a", "test"]
        merged = self.en_handler.merge_by_max_length(texts, 12)
        # "Hello world" len 11, "this is a" len 9, "test" len 4
        self.assertTrue(len(merged) > 1)
        self.assertEqual(merged[0], "Hello world")
        self.assertEqual(merged[1], "this is a")
        self.assertEqual(merged[2], "test")

    def test_split_by_punctuation(self):
        text = "Hello world. This is test. How are you?"
        segments = self.en_handler.split_by_punctuation(text)
        self.assertEqual(len(segments), 3)

    def test_calculate_phrase_timestamps(self):
        timestamps = Elements([
            Element(0.0, 0.5, "Hello"),
            Element(0.5, 1.0, "world"),
            Element(1.0, 1.5, "test")
        ])
        start, end = self.en_handler.calculate_phrase_timestamps("world test", timestamps)
        self.assertEqual(start, 0.5)
        self.assertEqual(end, 1.5)

    def test_calculate_word_timestamps(self):
        sentence = Elements([Element(0.0, 2.0, "Hello world")])
        word_timestamps = self.en_handler.calculate_word_timestamps(sentence)
        self.assertEqual(len(word_timestamps), 2)
        self.assertEqual(word_timestamps[0].text, "Hello")
        self.assertEqual(word_timestamps[1].text, "world")
        self.assertTrue(word_timestamps[1].end <= 2.0)

    def test_check_and_correct_split_sentence(self):
        sentence = "Hello world this is test."
        split_sentence = ["Hello world", "this is test."]
        good, fixed = self.en_handler.check_and_correct_split_sentence(split_sentence, sentence)
        self.assertTrue(good)

if __name__ == '__main__':
    unittest.main()
