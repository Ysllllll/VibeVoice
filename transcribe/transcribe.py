#!/usr/bin/env python3
"""
Test VibeVoice vLLM API with Chunking, Batch Requesting, and WhisperX Alignment.

This script demonstrates an engineering-focused pipeline:
1. AudioProcessor: Splits large audio files into smaller chunks.
2. ASRClient: Concurrently sends chunks to the vLLM API.
3. ASRCleaner: Cleans the raw JSON response into a standardized format.
4. ResultMerger: Merges the chunk-level ASR results back into file-level results with global timestamps.
5. WhisperXAligner: Uses WhisperX to perform word-level alignment on the full audio files.
6. PostProcessor: Formats the final word-level aligned results.
"""

import json
import base64
import time
import argparse
import asyncio
import aiohttp
import subprocess
import tempfile
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Import WhisperX alignment tools
from whisperx_utils.alignment import align, load_align_model
from whisperx_utils.schema import AlignedTranscriptionResult
from whisperx_utils.audio import load_audio

from subtitle_formatter import Subtitle

def is_eng_or_num(s: str) -> bool:
    """Matches English letters, numbers, and basic word symbols like apostrophe."""
    # Matches English letters, numbers, and basic word symbols like apostrophe
    return bool(re.match(r"^[A-Za-z0-9\'-]+$", s.strip()))

# ============================================================================
# Data Structures (数据流转结构)
# ============================================================================

@dataclass
class AudioChunk:
    """Represents a segment of an original audio file."""
    original_file: str
    chunk_path: str
    start_time: float
    end_time: float
    duration: float

@dataclass
class ASRResult:
    """Represents the raw ASR output for a single chunk."""
    chunk: AudioChunk
    raw_response: str
    segments: List[Dict[str, Any]] = field(default_factory=list)
    word_segments: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "zh"  # Defaulting to Chinese for this test
    
    def __str__(self):
        print_str = ""
        for seg in self.segments:
            print_str += f"{seg['start']:.2f} -> {seg['end']:.2f}: {seg['text']}\n"
        for word_seg in self.word_segments:
            if word_seg.get("start"):
                print_str += f"{word_seg['start']:.2f} -> {word_seg['end']:.2f}: {word_seg['word']}\n"
            else:
                print_str += f"   ->   : {word_seg['word']}\n"
        return print_str

@dataclass
class MergedResult:
    """Represents the merged ASR results for a complete original file."""
    original_file: str
    segments: List[Dict[str, Any]] = field(default_factory=list)
    word_segments: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "zh"

@dataclass
class AlignedResult:
    """Represents the word-level aligned result for a complete original file."""
    original_file: str
    segments: List[Dict[str, Any]] = field(default_factory=list)
    word_segments: List[Dict[str, Any]] = field(default_factory=list)
    language: str = "en"

# ============================================================================
# Modules (功能模块)
# ============================================================================

class AudioProcessor:
    """Splits audio files into smaller chunks using ffmpeg with overlap support."""
    def __init__(self, chunk_duration: int = 720, overlap: int = 20, temp_dir: Optional[str] = None):
        self.chunk_duration = chunk_duration
        self.overlap = overlap
        self.temp_dir = temp_dir or tempfile.mkdtemp(prefix="vibevoice_chunks_")
        
    def _get_duration(self, path: str) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            path,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8").strip()
        return float(out)

    def process(self, file_paths: List[str]) -> List[AudioChunk]:
        print(f"🎵 [AudioProcessor] Splitting {len(file_paths)} files into ~{self.chunk_duration}s chunks (overlap={self.overlap}s)...")
        chunks = []
        for file_path in file_paths:
            duration = self._get_duration(file_path)
            base_name = Path(file_path).stem
            
            current_time = 0.0
            idx = 0
            
            step = self.chunk_duration - self.overlap
            if step <= 0:
                step = self.chunk_duration # fallback if overlap is too large
                
            while current_time < duration:
                chunk_file = f"{base_name}_{idx:04d}.wav"
                chunk_path = str(Path(self.temp_dir) / chunk_file)
                
                end_time = min(current_time + self.chunk_duration, duration)
                if duration - end_time < self.chunk_duration / 2:
                    end_time = duration
                actual_duration = end_time - current_time
                
                cmd = [
                    "ffmpeg", "-y", 
                    "-ss", str(current_time),
                    "-i", file_path,
                    "-t", str(actual_duration),
                    "-c:a", "pcm_s16le", "-ar", "24000", "-ac", "1", # VibeVoice optimized format (24000Hz)
                    chunk_path
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                chunks.append(AudioChunk(
                    original_file=file_path,
                    chunk_path=chunk_path,
                    start_time=current_time,
                    end_time=end_time,
                    duration=actual_duration
                ))
                print(f"🔊 [AudioProcessor] Created chunk: {chunk_file} ({actual_duration:.2f}s)")
                
                idx += 1
                current_time += step
                
                # To prevent creating tiny useless chunks at the end
                if duration - end_time < 0.1:
                    break
                
        print(f"✅ [AudioProcessor] Generated {len(chunks)} chunks in total.")
        return chunks

    def cleanup(self):
        """Clean up the temporary directory containing audio chunks."""
        import shutil
        if self.temp_dir and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
            print(f"🧹 [AudioProcessor] Cleaned up temporary directory: {self.temp_dir}")


class ASRClient:
    """Sends chunks to vLLM API concurrently using ThreadPoolExecutor and OpenAI client."""
    def __init__(self, base_url: str = "http://localhost:8000", max_concurrent: int = 5):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.client = OpenAI(
            api_key="EMPTY",  # vLLM doesn't strictly require a real key
            base_url=f"{base_url}/v1"
        )
        
    def _guess_mime_type(self, path: str) -> str:
        """Guess MIME type from file extension."""
        ext = Path(path).suffix.lower()
        mime_map = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".mp4": "video/mp4",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".opus": "audio/ogg",
        }
        return mime_map.get(ext, "application/octet-stream")

    def process(self, chunks: List[AudioChunk], print_time: bool = True) -> List[ASRResult]:
        total_chunks = len(chunks)
        completed = 0
        lock = threading.Lock()
        def _process_chunk(chunk: AudioChunk) -> ASRResult:
            nonlocal completed
            with open(chunk.chunk_path, "rb") as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            mime = self._guess_mime_type(chunk.chunk_path)
            data_url = f"data:{mime};base64,{audio_b64}"
            
            show_keys = ["Start time", "End time", "Content", "Language"]
            prompt_text = (
                f"This is a {chunk.duration:.2f} seconds audio, with extra info: 简体中文\n\nPlease transcribe it with these keys: "
                + ", ".join(show_keys)
            )

            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that transcribes audio input into text output in JSON format."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": data_url}},
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
            
            max_retries = 4
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="vibevoice",
                        messages=messages,
                        max_tokens=4096,     
                        temperature=0.0 if attempt == 0 else (attempt * 0.1), # 失败重试时稍微提高一点temperature增加随机性
                        stream=False,
                        top_p=1.0 if attempt == 0 else 0.95,
                    )
                    content = response.choices[0].message.content
                        
                    
                    # Simple validation to check if the content looks like a valid JSON list
                    # Since the prompt asks for JSON format and specific keys
                    cleaned_content = content.strip()
                    if cleaned_content.startswith("```json"):
                        cleaned_content = cleaned_content[7:]
                    if cleaned_content.startswith("```"):
                        cleaned_content = cleaned_content[3:]
                    cleaned_content = cleaned_content.strip()
                    
                    json.loads(cleaned_content)
                    
                    with lock:
                        completed += 1
                    return ASRResult(chunk=chunk, raw_response=content)
                except Exception as e:
                    print(f"❌ [ASRClient] Request failed for chunk {Path(chunk.chunk_path).name} (Attempt {attempt+1}/{max_retries}): {e}")
                finally:
                    with lock:
                        width = len(str(total_chunks))
                        if response and hasattr(response, 'usage'):
                            print(f"🔍 [ASRClient] {completed:-{width}}/{total_chunks} ({(completed/total_chunks)*100:.1f}%). Token usage: {response.usage}")
                        else:
                            print(f"🔄 [ASRClient] Progress: {completed:-{width}}/{total_chunks} ({(completed/total_chunks)*100:.1f}%)")
                    
            # If all retries fail
            print(f"🚨 [ASRClient] All {max_retries} attempts failed for chunk {Path(chunk.chunk_path).name}.")
            return ASRResult(chunk=chunk, raw_response="[]")

        if print_time:
            start_time = time.time()
            
        print(f"🚀 [ASRClient] Starting concurrent API requests for {len(chunks)} chunks (max_concurrent={self.max_concurrent})...")
        
        # 可能存在一个问题，就是如果重试塞满了队列，耗时会很久
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            # executor.map maintains the order of the original chunks list
            results = list(executor.map(_process_chunk, chunks))

        if print_time:
            end_time = time.time()
            print(f"⏱️ [ASRClient] Total processing time: {end_time - start_time:.2f} seconds")
        return results


class ASRCleaner:
    """Cleans and standardizes ASR results."""
    def process(self, results: List[ASRResult]) -> List[ASRResult]:
        print(f"🧹 [ASRCleaner] Cleaning raw API responses...")
        for res in results:
            raw_text = res.raw_response.strip()
            
            # Extract JSON block if it's wrapped in markdown
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            try:
                data = json.loads(raw_text.strip())
                if not isinstance(data, list):
                    data = [data]
            except json.JSONDecodeError:
                print(f"⚠️ [ASRCleaner] Failed to parse JSON for chunk {res.chunk.chunk_path}")
                data = []

            cleaned_segments = []
            detected_languages = []
            for item in data:
                # Convert keys to WhisperX expected format
                start = float(item.get("Start time", item.get("Start", item.get("start", 0.0))))
                end = float(item.get("End time", item.get("End", item.get("end", 0.0))))
                text = item.get("Content", item.get("text", "")).strip()
                lang = item.get("Language", item.get("language", "zh")).lower()
                speaker = item.get("Speaker ID", item.get("Speaker", item.get("speaker", 0)))
                
                # Remove empty texts or pure silence tags
                if not text or text.lower() == "[silence]" or text.lower() == "[music]" or text.lower() == "[human sounds]" or text.lower() == "[environmental sounds]":
                    continue
                    
                cleaned_segments.append({
                    "start": start,
                    "end": end,
                    "text": text,
                    "speaker": speaker
                })
                detected_languages.append(lang)
            
            res.segments = cleaned_segments
            if detected_languages:
                # Majority vote for chunk language
                res.language = max(set(detected_languages), key=detected_languages.count)
                
        print(f"✅ [ASRCleaner] Cleaned {len(results)} chunks.")
        return results


class ResultMerger:
    """Merges chunk results back into original file results with global timestamps and deduplication for overlaps."""
    
    def _merge_words(self, words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Merge overlapping words.
        If words overlap significantly and have same text, merge them.
        """
        if not words:
            return []
            
        merged = []
        for w in words:
            if not merged:
                merged.append(w)
                continue
                
            last_w = merged[-1]
            
            # Calculate overlap
            try:
                start1, end1 = last_w["start"], last_w["end"]
                start2, end2 = w["start"], w["end"]
            except Exception as e:
                import pdb;pdb.set_trace()
                print("test")
            
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            overlap_dur = max(0, overlap_end - overlap_start)
            
            # Check for text match
            text1 = last_w["word"]
            text2 = w["word"]
            
            # If same text and overlap exists (or very close), merge
            # Relaxed condition: if they are very close (< 0.1s gap) and same text, merge
            # Because sometimes overlap is negative (small gap) but it's the same word split?
            # No, if split, they are different parts.
            # Here we are merging *duplicate* words from different chunks.
            
            is_duplicate = False
            
            # 1. Exact text match
            if text1 == text2:
                if overlap_dur > 0:
                    is_duplicate = True
            
            # 2. Conflict (Different text, significant overlap)
            # Use overlap ratio to avoid merging distinct adjacent words that slightly overlap
            elif overlap_dur > 0:
                dur1 = end1 - start1
                dur2 = end2 - start2
                min_dur = min(dur1, dur2)
                if min_dur > 0 and (overlap_dur / min_dur) > 0.5:
                    # Significant overlap ( > 50% of shorter word) -> Treat as conflict
                    is_duplicate = True
                
            if is_duplicate:
                # Resolve conflict: prefer word closer to chunk center
                dist1 = last_w.get("dist_to_center", float('inf'))
                dist2 = w.get("dist_to_center", float('inf'))
                
                if dist2 < dist1:
                    # New word is better. Replace last_w with w.
                    merged[-1] = w
                else:
                    # Existing word is better (or equal). Keep last_w.
                    # If text is same, merge timestamps to cover full range
                    if text1 == text2:
                        last_w["start"] = min(start1, start2)
                        last_w["end"] = max(end1, end2)
                        # Keep better dist
                        last_w["dist_to_center"] = min(dist1, dist2)
            else:
                merged.append(w)
                
        return merged

    def process(self, asr_results: List[ASRResult]) -> List[MergedResult]:
        print(f"🧩 [ResultMerger] Merging chunk segments back to original files (handling overlaps)...")
        
        # Group by original file
        file_groups: Dict[str, List[ASRResult]] = {}
        for res in asr_results:
            orig = res.chunk.original_file
            if orig not in file_groups:
                file_groups[orig] = []
            file_groups[orig].append(res)
            
        merged_results = []
        for orig_file, results in file_groups.items():
            # Sort by chunk start time
            results.sort(key=lambda x: x.chunk.start_time)
            
            all_words = []
            languages = []
            
            # Collect all words with global timestamps
            for res in results:
                offset = res.chunk.start_time
                chunk_center = (res.chunk.start_time + res.chunk.end_time) / 2
                languages.append(res.language)
                
                # Get words from segments (more reliable structure usually)
                chunk_words = []
                for seg in res.segments:
                    if "words" in seg:
                        chunk_words.extend(seg["words"])
                
                # Adjust timestamps
                for w in chunk_words:
                    w_global = w.copy()
                    if "start" in w_global and "end" in w_global:
                        w_global["start"] += offset
                        w_global["end"] += offset
                    # Calculate distance to center for conflict resolution
                    if "start" in w_global and "end" in w_global:
                        w_center = (w_global["start"] + w_global["end"]) / 2
                        w_global["dist_to_center"] = abs(w_center - chunk_center)
                    else:
                        w_global["dist_to_center"] = float('inf')
                        
                    all_words.append(w_global)
            
            # Sort all words by start time
            all_words.sort(key=lambda x: x.get("start", 0))
            
            # Merge/Deduplicate
            merged_words = self._merge_words(all_words)
            
            # Temporary single segment fallback (Resegmentation logic will be added later)
            if merged_words:
                final_segments = [{
                    "text": "".join(w.get("word", "") for w in merged_words),
                    "start": merged_words[0].get("start", 0.0),
                    "end": merged_words[-1].get("end", 0.0),
                    "words": merged_words
                }]
            else:
                final_segments = []
            
            main_language = max(set(languages), key=languages.count) if languages else "zh"
            # Normalize language to 2-letter code for whisperx
            if "zh" in main_language: main_language = "zh"
            elif "en" in main_language: main_language = "en"
                
            merged_results.append(MergedResult(
                original_file=orig_file,
                segments=final_segments,
                word_segments=merged_words,
                language=main_language
            ))
            
        print(f"✅ [ResultMerger] Merged into {len(merged_results)} complete files.")
        return merged_results


class WhisperXAligner:
    """Performs word-level alignment using WhisperX on audio chunks."""
    def __init__(self, device: str = "cpu", interpolate_method: str = "nearest"):
        self.device = device
        self.interpolate_method = interpolate_method
        self.align_model = None
        self.current_language = None
        self.align_metadata = None

    def _load_model(self, language: str):
        if self.current_language != language or self.align_model is None:
            print(f"🔄 [WhisperXAligner] Loading alignment model for language: {language} on {self.device}...")
            self.align_model, self.align_metadata = load_align_model(
                language, self.device
            )
            self.current_language = language

    def process(self, asr_results: List[ASRResult]) -> List[ASRResult]:
        print(f"🎯 [WhisperXAligner] Starting word-level alignment for {len(asr_results)} chunks...")
        
        for res in asr_results:
            # Skip if no segments
            if not res.segments:
                continue

            # Load chunk audio
            try:
                audio = load_audio(res.chunk.chunk_path)
            except Exception as e:
                print(f"  ❌ Failed to load audio for chunk {res.chunk.chunk_path}: {e}")
                continue

            # Load the correct model for the language
            try:
                self._load_model(res.language)
            except Exception as e:
                print(f"  ❌ Failed to load align model for language {res.language}: {e}")
                continue
            
            # Perform alignment
            try:
                aligned_data: AlignedTranscriptionResult = align(
                    res.segments,
                    self.align_model,
                    self.align_metadata,
                    audio,
                    self.device,
                    interpolate_method=self.interpolate_method,
                    return_char_alignments=False,
                    print_progress=False,
                )
                
                # Update the result with aligned segments and words in-place
                res.segments = aligned_data["segments"]
                res.word_segments = aligned_data.get("word_segments", [])
                
            except Exception as e:
                print(f"  ❌ Alignment failed for chunk {res.chunk.chunk_path}: {e}")
                # Keep original segments, word_segments remains empty
                
        print(f"✅ [WhisperXAligner] Alignment complete for all chunks.")
        return asr_results


class PostProcessor:
    """Handles various post-processing tasks (cleaning, formatting, saving)."""
    
    def _format_srt_time(self, seconds: float) -> str:
        """Converts float seconds to SRT timestamp format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        # Handle millisecond overflow
        if millis >= 1000:
            secs += millis // 1000
            millis = millis % 1000
            if secs >= 60:
                minutes += secs // 60
                secs = secs % 60
                if minutes >= 60:
                    hours += minutes // 60
                    minutes = minutes % 60
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
    def clean_word_alignments(self, asr_results: List[ASRResult]) -> List[ASRResult]:
        """Merges fragmented English characters into words and attaches punctuation to adjacent words."""
        print(f"🧹 [PostProcessor] Cleaning up word segments (merging chars & punctuation)...")
        import string
        import re
        
        # Punctuation characters to match
        punct_chars = set(string.punctuation + "，。？！；：“”‘’（）【】《》〈〉、·")
        
        # Leading punctuation (attach to NEXT word): ( [ { < " ' “ ‘ 《 【 （ 〈
        leading_puncts = set("([{<\"'“‘《【（〈")
        
        def is_punct(s: str) -> bool:
            s = s.strip()
            return bool(s) and all(c in punct_chars for c in s)
            
        def is_eng_or_num(s: str) -> bool:
            # Matches English letters, numbers, and basic word symbols like apostrophe
            return bool(re.match(r"^[A-Za-z0-9\'-]+$", s.strip()))

        for res in asr_results:
            for seg in res.segments:
                if "words" not in seg or not seg["words"]:
                    continue
                    
                words = seg["words"]
                original_text = seg.get("text", "")
                
                # --- Step 1: Merge fragmented English words (e.g. D, D, o, S -> DDoS) ---
                step1_words = []
                for w in words:
                    text = w.get("word", "")
                    
                    if not step1_words:
                        step1_words.append(w)
                        continue
                        
                    prev_w = step1_words[-1]
                    prev_text = prev_w.get("word", "")
                    
                    # Check if both are English/Number fragments
                    if res.language == "zh" and is_eng_or_num(text) and is_eng_or_num(prev_text):
                         # If original text has them separated by space, they are distinct words
                         if prev_text.strip() + " " + text.strip() in original_text:
                             step1_words.append(w)
                         else:
                             # Merge into previous word
                             prev_w["word"] = prev_text.strip() + text.strip()
                             if "end" in w:
                                 prev_w["end"] = max(prev_w.get("end", 0), w["end"])
                         continue
                    
                    step1_words.append(w)
                
                # --- Step 2: Attach Punctuation (Leading -> Next, Trailing -> Prev) ---
                final_words = []
                pending_prefix = ""
                pending_start = None
                
                for w in step1_words:
                    text = w.get("word", "")
                    
                    if is_punct(text):
                        stripped_text = text.strip()
                        if stripped_text in leading_puncts:
                            # Leading punctuation: buffer it for the next word
                            pending_prefix += stripped_text
                            if pending_start is None:
                                pending_start = w.get("start")
                        else:
                            # Trailing punctuation: attach to previous word
                            if final_words:
                                if not final_words[-1]["word"].rstrip().endswith(stripped_text):
                                    final_words[-1]["word"] = final_words[-1]["word"].rstrip() + stripped_text
                                if "end" in w:
                                    final_words[-1]["end"] = max(final_words[-1].get("end", 0), w["end"])
                            else:
                                # Edge case: starts with trailing punct (e.g. "...Hello")
                                if pending_prefix:
                                    if not pending_prefix.endswith(stripped_text):
                                        pending_prefix += stripped_text
                                else:
                                    final_words.append(w)
                        continue
                        
                    # Normal word processing
                    if pending_prefix:
                        if not w["word"].lstrip().startswith(pending_prefix):
                            w["word"] = pending_prefix + w["word"].lstrip()
                        if pending_start is not None:
                            w["start"] = min(w.get("start", float('inf')), pending_start)
                        pending_prefix = ""
                        pending_start = None
                        
                    final_words.append(w)
                
                # Handle leftover pending prefix (rare, e.g. sentence ends with opening bracket)
                if pending_prefix:
                    if final_words:
                        if not final_words[-1]["word"].rstrip().endswith(pending_prefix):
                            final_words[-1]["word"] = final_words[-1]["word"].rstrip() + pending_prefix
                    else:
                        # Only punctuation in segment?
                        pass

                seg["words"] = final_words
                
                # Reconstruct segment text to reflect merged words and attached punctuation
                if final_words:
                    new_text = ""
                    for i, fw in enumerate(final_words):
                        w_text = fw.get("word", "").strip()
                        if not new_text:
                            new_text = w_text
                        else:
                            if res.language == "en":
                                new_text += " " + w_text
                            else:
                                prev_w_text = final_words[i-1].get("word", "").strip()
                                # Add space between two English/Number words in Chinese context
                                if is_eng_or_num(w_text) and is_eng_or_num(prev_w_text):
                                    new_text += " " + w_text
                                else:
                                    new_text += w_text
                    seg["text"] = new_text
                
        print(f"✅ [PostProcessor] Cleaned word segments for {len(asr_results)} chunks.")

        # 刷新word_segments
        # 处理没有时间戳的word
        for res in asr_results:
            for seg in res.segments:
                words = seg.get("words", [])
                prev_word = {"end": 0.0}
                char_time = 0
                char_leng = 1e-7
                for idx, word in enumerate(words):
                    if word.get("start") == None:
                        next_idx = idx + 1
                        while (
                            next_idx < len(words)
                            and words[next_idx].get("start") == None
                        ):
                            next_idx += 1
                        next_idx = min(next_idx, len(words) - 1)
                        next_time = words[next_idx].get("start", seg.get("end", 1e16))
                        if prev_word["end"] == next_time:
                            prev_word["end"] = prev_word["end"] - (
                                min(1, prev_word["end"] - prev_word["start"]) * 0.99
                            )

                        unit_time = char_time / char_leng
                        word["start"] = prev_word["end"]
                        word["end"] = min(
                            next_time, word["start"] + unit_time * len(word["word"])
                        )
                        word["score"] = 0.49
                    char_time += word["end"] - word["start"]
                    char_leng += len(word["word"])
                    prev_word = word

        for res in asr_results:
            res.word_segments = []
            for seg in res.segments:
                seg["words"] = seg.get("words", [])
                res.word_segments.extend(seg["words"])
            res.raw_response = ""
        return asr_results

    def save_json(self, aligned_results: List[MergedResult], output_dir: str) -> List[str]:
        """Formats to WhisperX schema and saves to JSON."""
        print(f"📝 [PostProcessor] Formatting final results (WhisperX compatible) & saving to JSON...")
        json_paths = []
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True)
        for res in aligned_results:
            # Clean up missing timestamps, etc.
            cleaned_segments = []
            word_segments = []  # Flat list of all words, similar to whisperx
            
            for seg in res.segments:
                words = seg.get("words", [])
                
                # Filter out empty words
                valid_words = []
                for w in words:
                    if 'start' in w and 'end' in w:
                        valid_words.append(w)
                    elif 'word' in w and valid_words:
                        # Fallback: assign previous end time if missing (very basic nearest interpolation)
                        w['start'] = valid_words[-1]['end']
                        w['end'] = w['start'] + 0.1
                        valid_words.append(w)
                
                # Update segment with valid words
                seg["words"] = valid_words
                cleaned_segments.append(seg)
                word_segments.extend(valid_words)
                
            # Construct output matching WhisperX schema
            output = {
                "segments": cleaned_segments,
                "language": res.language,
                "word_segments": word_segments
            }
            
            # Save to JSON
            out_path = Path(output_dir) / f"{Path(res.original_file).stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"  -> Saved final result to {out_path}")
            json_paths.append(out_path)
            
        return json_paths

    def save_srt(self, json_paths: List[str], output_dir: str):
        """Reads JSON files and exports them to SRT format."""
        print(f"📝 [PostProcessor] Generating SRT files from JSON...")
        if not Path(output_dir).exists():
            Path(output_dir).mkdir(parents=True)
        for json_path in json_paths:
            # Remove .aligned.json and replace with .srt, or just replace .json
            srt_path = Path(output_dir) / Path(json_path).with_suffix(".srt").name
            
            # Initialize subtitle formatter with JSON file
            sub_formatter = Subtitle.from_whisperx_json(json_path)
            # Use smart grouping (max 30 chars per line, cascade splitting)
            sub_formatter.group_smart(max_chars=30)
            # Export to SRT
            sub_formatter.to_srt(srt_path)
            
            print(f"  -> Saved smart SRT subtitle to {srt_path}")

    def process(self, data: Any, stage: str = "save_json", output_dir: str = "json_output") -> Any:
        """Dispatcher for different post-processing stages."""
        if stage == "clean_words":
            return self.clean_word_alignments(data)
        elif stage == "save_json":
            return self.save_json(data, output_dir)
        elif stage == "save_srt":
            return self.save_srt(data, output_dir)
        else:
            raise ValueError(f"Unknown post-processing stage: {stage}")

# ============================================================================
# Pipeline Runner (组装器)
# ============================================================================

def process_input_paths(file_paths: List[str]) -> List[str]:
    """Process input paths: handle directories, filter valid files, and extract audio from videos."""
    valid_audio_exts = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
    valid_video_exts = {'.mp4', '.mkv', '.avi', '.mov', '.webm'}
    
    processed_files = []
    
    # Flatten the input list if it contains directories
    expanded_paths = []
    for path_str in file_paths:
        p = Path(path_str)
        if p.is_dir():
            print(f"📂 [InputProcessor] Scanning directory: {p}")
            # Collect all files recursively
            for f in p.rglob("*"):
                if f.is_file() and f.suffix.lower() in (valid_audio_exts | valid_video_exts):
                    expanded_paths.append(f)
        elif p.is_file():
            expanded_paths.append(p)
        else:
            print(f"⚠️ [InputProcessor] Path not found: {p}")

    # Process each file
    for p in expanded_paths:
        ext = p.suffix.lower()
        if ext in valid_audio_exts:
            processed_files.append(str(p))
        elif ext in valid_video_exts:
            print(f"🎬 [InputProcessor] Extracting audio from video: {p.name}")
            # Output path for extracted audio
            out_audio = p.with_suffix('.wav')
            
            # If the wav file already exists, we skip extraction
            if not out_audio.exists():
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(p),
                    "-vn",  # No video
                    "-acodec", "pcm_s16le",
                    "-ar", "24000",
                    "-ac", "1",
                    str(out_audio)
                ]
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    print(f"   -> Extracted to: {out_audio.name}")
                except subprocess.CalledProcessError as e:
                    print(f"❌ [InputProcessor] Failed to extract audio from {p.name}: {e}")
                    continue
            else:
                print(f"   -> Audio already extracted: {out_audio.name}")
                
            processed_files.append(str(out_audio))
        else:
            print(f"⚠️ [InputProcessor] Unsupported file type: {p.name}")
    # 这里需要对文件做一次去重，可能会出现重复文件名的文件。
    # 按 stem 去重：保留第一个出现的文件（含路径），忽略后续同名不同扩展名的文件
    seen_stems = set()
    deduped = []
    for f in processed_files:
        stem = Path(f).stem
        if stem not in seen_stems:
            seen_stems.add(stem)
            deduped.append(f)
        else:
            print(f"⚠️ [InputProcessor] 跳过重复 stem 的文件: {f}")
    return deduped

def run_pipeline(
    file_paths: List[str],
    api_url: str = "http://localhost:8000",
    chunk_duration: int = 720,
    overlap: int = 20,
    max_concurrent: int = 5,
    device: str = "cpu",
    json_output: str = "json_output",
    srt_output: str = "srt_output"
):
    print(f"{'='*60}\n🚀 Starting Engineering VibeVoice Pipeline\n{'='*60}")
    
    pipeline_start_time = time.time()
    timings = {}

    def record_time(stage_name, start_time):
        elapsed = time.time() - start_time
        timings[stage_name] = elapsed
        print(f"⏱️ [Timer] {stage_name} completed in {elapsed:.2f} seconds.\n")

    # 0. Input Preprocessing
    t0 = time.time()
    file_paths = process_input_paths(file_paths)
    if not file_paths:
        print("❌ [Error] No valid audio or video files found. Exiting.")
        return []
    else:
        print(f"✅ [InputProcessor] Found {len(file_paths)} valid files for processing: " + "\n".join(f"  - {idx+1}: {i}" for idx, i in enumerate(file_paths)))
    record_time("0. Input Preprocessing", t0)

    # 1. Chunking
    t0 = time.time()
    processor = AudioProcessor(chunk_duration=chunk_duration, overlap=overlap)
    chunks = processor.process(file_paths)
    record_time("1. Audio Chunking", t0)
    
    # 2. ASR API Requests
    t0 = time.time()
    client = ASRClient(base_url=api_url, max_concurrent=max_concurrent)
    raw_results = client.process(chunks)
    record_time("2. ASR API Requests", t0)
    # 3. Clean API Outputs
    t0 = time.time()
    cleaner = ASRCleaner()
    cleaned_results = cleaner.process(raw_results)
    record_time("3. Clean API Outputs", t0)
    
    # 这里可以做标点符号校准
    
    # 4. WhisperX Alignment (on Chunks)
    t0 = time.time()
    aligner = WhisperXAligner(device=device)
    aligned_chunks = aligner.process(cleaned_results)
    record_time("4. WhisperX Alignment", t0)
    
    # 4.5 Clean Word Alignments (Merge chars & punctuation) via PostProcessor
    t0 = time.time()
    post_processor = PostProcessor()
    aligned_clean_chunks = post_processor.process(aligned_chunks, stage="clean_words")
    record_time("4.5 Clean Word Alignments", t0)
    
    # 5. Merge Results
    t0 = time.time()
    merger = ResultMerger()
    merged_results = merger.process(aligned_clean_chunks)
    record_time("5. Merge Results", t0)
    
    # 6. Post Processing & Output (Format and Save)
    t0 = time.time()
    # Save to JSON first
    json_paths = post_processor.process(merged_results, stage="save_json", output_dir=json_output)
    # Read from JSON and save to SRT
    if srt_output:
        post_processor.process(json_paths, stage="save_srt", output_dir=srt_output)
    record_time("6. Post Processing & Output", t0)
    
    # 7. Cleanup
    t0 = time.time()
    processor.cleanup()
    record_time("7. Cleanup", t0)
    
    total_time = time.time() - pipeline_start_time
    
    print(f"{'='*60}\n📊 Pipeline Timing Summary\n{'-'*60}")
    for stage, elapsed in timings.items():
        print(f"  - {stage:<30}: {elapsed:>8.2f}s")
    print(f"{'-'*60}")
    print(f"  - {'Total Time':<30}: {total_time:>8.2f}s")
    
    print(f"\n🎉 Pipeline Finished Successfully!\n{'='*60}")
    return json_paths

def main():
    parser = argparse.ArgumentParser(description="Test VibeVoice API + WhisperX Alignment Pipeline")
    parser.add_argument("audio_paths", nargs="+", help="Paths to audio files to process")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--chunk_duration", type=int, default=300, help="Audio chunk size in seconds")
    parser.add_argument("--overlap", type=int, default=20, help="Overlap between chunks in seconds")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent API requests")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4"], help="Device for WhisperX alignment")
    parser.add_argument("--json_output", default="json_output", help="Directory to save JSON files")
    parser.add_argument("--srt_output", default="srt_output", help="Directory to save SRT files")
    
    args = parser.parse_args()
    
    run_pipeline(
        file_paths=args.audio_paths,
        api_url=args.url,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap,
        max_concurrent=args.concurrency,
        device=args.device,
        json_output=args.json_output,
        srt_output=args.srt_output
    )

if __name__ == "__main__":
    main()
