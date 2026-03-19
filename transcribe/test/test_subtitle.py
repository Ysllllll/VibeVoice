import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from subtitle.subtitle import Subtitle, Elements, Element
from subtitle.handler import LanguageContext, EnglishHandler, ChineseHandler

class TestSubtitle(unittest.TestCase):
    def setUp(self):
        # Setup basic elements
        self.elements = Elements([
            Element(0.0, 1.0, "Hello"),
            Element(1.0, 2.0, "world,"),
            Element(2.0, 3.0, "this is a test."),
            Element(3.0, 4.0, "Another sentence!")
        ])

    def test_element_creation(self):
        e = Element(1.5, 2.5, "test")
        self.assertEqual(e.start, 1.5)
        self.assertEqual(e.end, 2.5)
        self.assertEqual(e.text, "test")
        self.assertTrue(bool(e))

    def test_elements_creation(self):
        self.assertEqual(len(self.elements), 4)
        self.assertEqual(self.elements.start, 0.0)
        self.assertEqual(self.elements.end, 4.0)
        self.assertEqual(self.elements.text, "Hello world, this is a test. Another sentence!")

    def test_subtitle_creation(self):
        sub = Subtitle("en", self.elements, "test.srt")
        self.assertEqual(sub.lang, "en")
        self.assertEqual(len(sub), 4)
        self.assertEqual(sub.text, "Hello world, this is a test. Another sentence!")

    def test_elements_split_by_punctuation(self):
        LanguageContext.set_language("en")
        # Ensure elements are handled by EnglishHandler
        self.elements.handler = EnglishHandler()
        split_elements = self.elements.split_by_punctuation(use_end_punctuation=True)
        # Should split by . and !
        self.assertTrue(len(split_elements) > 0)
        
    def test_subtitle_iter_sentence(self):
        sub = Subtitle("en", self.elements, "test.srt")
        sentences = list(sub.iter_sentence())
        self.assertTrue(len(sentences) > 0)
        self.assertEqual(sentences[0].text, "Hello world, this is a test.")

    def test_to_dict_from_dict(self):
        sub = Subtitle("en", self.elements, "test.srt")
        sub_dict = sub.to_dict()
        self.assertEqual(sub_dict["language"], "en")
        self.assertEqual(sub_dict["filename"], "test.srt")
        self.assertEqual(len(sub_dict["elements"]), 4)

        sub2 = Subtitle.from_dict(sub_dict)
        self.assertEqual(sub2.lang, "en")
        self.assertEqual(len(sub2), 4)

    def test_remove_overlap_element(self):
        overlap_elements = Elements([
            Element(0.0, 1.0, "A"),
            Element(0.5, 1.5, "B"),
            Element(1.5, 2.0, "C")
        ])
        cleaned = overlap_elements.remove_overlap_element()
        self.assertTrue(len(cleaned) > 0)

if __name__ == '__main__':
    unittest.main()
