import json
import os
import re
from typing import List, Dict, Union, Optional
from pathlib import Path

def format_timestamp(seconds: float) -> str:
    """Format seconds into SRT timestamp format (HH:MM:SS,mmm)."""
    if seconds < 0:
        seconds = 0.0
    ms = round(seconds * 1000)
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{int(ms):03d}"

class Subtitle:
    """
    A lightweight Subtitle class for generating customized subtitle files 
    from high-precision word-level timestamp data.
    """
    def __init__(self, words: List[Dict[str, Union[str, float]]]):
        """
        Initialize with word-level timestamps.
        
        :param words: List of dicts containing 'word', 'start', and 'end'.
                      Example: [{"word": "Hello", "start": 0.1, "end": 0.5}, ...]
        """
        self.words = []
        for w in words:
            # Handle cases where some words might lack valid timestamps
            if 'word' in w and 'start' in w and 'end' in w:
                self.words.append(w)
            elif 'word' in w and self.words:
                # Basic fallback if a word is missing timestamps
                fallback_w = w.copy()
                fallback_w['start'] = self.words[-1]['end']
                fallback_w['end'] = fallback_w['start'] + 0.1
                self.words.append(fallback_w)

        self.segments = []

    @classmethod
    def from_whisperx_json(cls, file_path: Union[str, Path]) -> "Subtitle":
        """
        Load words data from a WhisperX alignment output JSON file.
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        words = []
        # Prefer the global 'word_segments' if it exists
        if 'word_segments' in data:
            words = data['word_segments']
        # Fallback to extracting words from 'segments'
        elif 'segments' in data:
            for seg in data['segments']:
                if 'words' in seg:
                    words.extend(seg['words'])
                    
        return cls(words)
        
    def group_by_sentence(self, punctuations: tuple = ('.', '?', '!', '。', '？', '！', '；', ';')):
        """
        Group words into segments by sentence-ending punctuation.
        """
        self.segments = []
        if not self.words:
            return self
            
        current_segment = []
        for word in self.words:
            current_segment.append(word)
            text = word['word'].strip()
            
            # If the word ends with a sentence punctuation, we break the segment
            if any(text.endswith(p) for p in punctuations):
                self._add_segment(current_segment)
                current_segment = []
                
        if current_segment:
            self._add_segment(current_segment)
            
        return self
        
    @staticmethod
    def _calculate_visual_length(text: str) -> float:
        """
        Calculate visual length of text where:
        - Chinese/Full-width characters count as 1.0
        - English/Half-width characters/Numbers count as 0.5
        """
        length = 0.0
        for char in text:
            # Basic check for CJK/Full-width vs English/Half-width
            if '\u4e00' <= char <= '\u9fff' or '\u3000' <= char <= '\u303f' or '\uff00' <= char <= '\uffef':
                length += 1.0
            else:
                length += 0.5
        return length

    def group_by_length(self, max_chars: int = 30):
        """
        Group words into segments based on maximum visual length.
        """
        self.segments = []
        if not self.words:
            return self
            
        current_segment = []
        current_len = 0.0
        
        for word in self.words:
            w_text = word['word'].strip()
            w_len = self._calculate_visual_length(w_text)
            
            # If adding this word exceeds max_chars and we already have words in segment
            if current_segment and current_len + w_len > max_chars:
                self._add_segment(current_segment)
                current_segment = [word]
                current_len = w_len
            else:
                current_segment.append(word)
                current_len += w_len
                
        if current_segment:
            self._add_segment(current_segment)
            
        return self
        
    def group_custom(self, max_chars: int = 30, max_duration: float = 5.0, punctuations: tuple = ('.', '?', '!', '。', '？', '！')):
        """
        Advanced grouping: split by punctuation, max visual length, or max duration.
        """
        self.segments = []
        if not self.words:
            return self
            
        current_segment = []
        current_len = 0.0
        
        for word in self.words:
            w_text = word['word'].strip()
            w_len = self._calculate_visual_length(w_text)
            
            # Check duration condition
            duration_exceeded = False
            if current_segment:
                duration = word['end'] - current_segment[0]['start']
                if duration > max_duration:
                    duration_exceeded = True
                    
            if current_segment and (current_len + w_len > max_chars or duration_exceeded):
                self._add_segment(current_segment)
                current_segment = [word]
                current_len = w_len
            else:
                current_segment.append(word)
                current_len += w_len
                
            # Check punctuation condition
            if any(w_text.endswith(p) for p in punctuations):
                self._add_segment(current_segment)
                current_segment = []
                current_len = 0.0
                
        if current_segment:
            self._add_segment(current_segment)
            
        return self

    def _add_segment(self, words: List[Dict]):
        if not words:
            return
            
        start_time = words[0]['start']
        end_time = words[-1]['end']
        
        # Smart concatenation: add space between English/numeric words
        text = ""
        for i, w in enumerate(words):
            w_text = w['word']
            if i > 0:
                prev_w = words[i-1]['word'].strip()
                curr_w = w_text.strip()
                # If both previous word ends with English/Number and current starts with English/Number
                if prev_w and curr_w and re.match(r'[A-Za-z0-9]', prev_w[-1]) and re.match(r'[A-Za-z0-9]', curr_w[0]):
                    text += " " + w_text
                else:
                    text += w_text
            else:
                text += w_text
                
        self.segments.append({
            "start": start_time,
            "end": end_time,
            "text": text.strip(),
            "words": words
        })

    def group_smart(self, 
                    max_chars: int = 30, 
                    end_punctuations: tuple = ('.', '?', '!', '。', '？', '！'),
                    mid_punctuations: tuple = (',', ';', '，', '；', '、', '：', ':')):
        """
        Smart cascaded grouping:
        1. Try to group by sentence-ending punctuations.
        2. If a sentence exceeds max_chars, split it by mid-sentence punctuations (comma, etc.).
        3. If a segment still exceeds max_chars, forcibly split by max_chars length.
        """
        self.segments = []
        if not self.words:
            return self

        # Helper to split a list of words by max_chars
        def split_by_length(word_list: List[Dict], max_len: int) -> List[List[Dict]]:
            chunks = []
            curr_chunk = []
            curr_len = 0
            for w in word_list:
                w_len = len(w['word'].strip())
                if curr_chunk and curr_len + w_len > max_len:
                    chunks.append(curr_chunk)
                    curr_chunk = [w]
                    curr_len = w_len
                else:
                    curr_chunk.append(w)
                    curr_len += w_len
            if curr_chunk:
                chunks.append(curr_chunk)
            return chunks

        # Helper to split a list of words by specific punctuations
        def split_by_punc(word_list: List[Dict], puncs: tuple) -> List[List[Dict]]:
            chunks = []
            curr_chunk = []
            for w in word_list:
                curr_chunk.append(w)
                if any(w['word'].strip().endswith(p) for p in puncs):
                    chunks.append(curr_chunk)
                    curr_chunk = []
            if curr_chunk:
                chunks.append(curr_chunk)
            return chunks

        # 1. First, split the entire words stream by end punctuations
        sentences = split_by_punc(self.words, end_punctuations)

        for sentence in sentences:
            # Calculate visual length of the sentence
            sentence_len = sum(self._calculate_visual_length(w['word'].strip()) for w in sentence)

            if sentence_len <= max_chars:
                # Good to go, add as one segment
                self._add_segment(sentence)
            else:
                # 2. Too long, try splitting by mid punctuations
                mid_chunks = split_by_punc(sentence, mid_punctuations)
                
                for chunk in mid_chunks:
                    chunk_len = sum(self._calculate_visual_length(w['word'].strip()) for w in chunk)
                    if chunk_len <= max_chars:
                        self._add_segment(chunk)
                    else:
                        # 3. Still too long, forcibly split by max_chars using visual length
                        # Re-implement split_by_length to use visual length
                        curr_chunk = []
                        curr_len = 0.0
                        for w in chunk:
                            w_len = self._calculate_visual_length(w['word'].strip())
                            if curr_chunk and curr_len + w_len > max_chars:
                                self._add_segment(curr_chunk)
                                curr_chunk = [w]
                                curr_len = w_len
                            else:
                                curr_chunk.append(w)
                                curr_len += w_len
                        if curr_chunk:
                            self._add_segment(curr_chunk)

        return self

    def to_srt(self, output_path: Union[str, Path]):
        """Export the current segments to an SRT file with strictly non-overlapping timestamps."""
        if not self.segments:
            # If no segments grouped yet, default to a sensible grouping
            self.group_custom()
            
        # Ensure strict monotonicity: next segment's start must be >= previous segment's end
        for i in range(1, len(self.segments)):
            prev_end = self.segments[i-1]['end']
            curr_start = self.segments[i]['start']
            
            # If overlap detected, clamp the current start to the previous end
            if curr_start < prev_end:
                self.segments[i]['start'] = prev_end
                
                # Also ensure the current segment's end doesn't become smaller than its new start
                if self.segments[i]['end'] < self.segments[i]['start']:
                    self.segments[i]['end'] = self.segments[i]['start'] + 0.001
            
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, seg in enumerate(self.segments, 1):
                start_str = format_timestamp(seg['start'])
                end_str = format_timestamp(seg['end'])
                f.write(f"{i}\n")
                f.write(f"{start_str} --> {end_str}\n")
                f.write(f"{seg['text']}\n\n")
                
    def to_json(self, output_path: Union[str, Path]):
        """Export the current segments to a JSON file."""
        if not self.segments:
            self.group_custom()
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.segments, f, ensure_ascii=False, indent=2)

    def get_segments(self) -> List[Dict]:
        """Return the grouped segments."""
        if not self.segments:
            self.group_custom()
        return self.segments
