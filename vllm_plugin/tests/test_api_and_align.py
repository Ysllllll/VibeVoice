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

import os
import json
import base64
import time
import argparse
import asyncio
import aiohttp
import subprocess
import tempfile
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Import WhisperX alignment tools
from whisperx_utils.alignment import align, load_align_model
from whisperx_utils.schema import AlignedTranscriptionResult
from whisperx_utils.audio import load_audio

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
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            
            current_time = 0.0
            idx = 0
            
            step = self.chunk_duration - self.overlap
            if step <= 0:
                step = self.chunk_duration # fallback if overlap is too large
                
            while current_time < duration:
                chunk_file = f"{base_name}_{idx:04d}.wav"
                chunk_path = os.path.join(self.temp_dir, chunk_file)
                
                end_time = min(current_time + self.chunk_duration, duration)
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
                
                idx += 1
                current_time += step
                
                # To prevent creating tiny useless chunks at the end
                if duration - current_time < 0.1:
                    break
                
        print(f"✅ [AudioProcessor] Generated {len(chunks)} chunks in total.")
        return chunks

    def cleanup(self):
        """Clean up the temporary directory containing audio chunks."""
        import shutil
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 [AudioProcessor] Cleaned up temporary directory: {self.temp_dir}")


class ASRClient:
    """Sends chunks to vLLM API concurrently."""
    def __init__(self, base_url: str = "http://localhost:8000", max_concurrent: int = 10):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        
    def _guess_mime_type(self, path: str) -> str:
        return "audio/wav"

    async def _process_chunk(self, session: aiohttp.ClientSession, chunk: AudioChunk, semaphore: asyncio.Semaphore) -> ASRResult:
        async with semaphore:
            with open(chunk.chunk_path, "rb") as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            
            mime = self._guess_mime_type(chunk.chunk_path)
            data_url = f"data:{mime};base64,{audio_b64}"
            
            show_keys = ["Start time", "End time", "Speaker ID", "Content", "Language"]
            prompt_text = (
                f"This is a {chunk.duration:.2f} seconds audio, please transcribe it with these keys: "
                + ", ".join(show_keys)
            )

            payload = {
                "model": "vibevoice",
                "messages": [
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
                ],
                "max_tokens": 32768-1024,     
                "temperature": 0.0,      
                "stream": False,
                "top_p": 1.0,
            }
            
            url = f"{self.base_url}/v1/chat/completions"
            try:
                async with session.post(url, json=payload, timeout=600) as response:
                    if response.status == 200:
                        data = await response.json()
                        content = data['choices'][0]['message']['content']
                        return ASRResult(chunk=chunk, raw_response=content)
                    else:
                        text = await response.text()
                        print(f"❌ [ASRClient] Error {response.status}: {text}")
                        return ASRResult(chunk=chunk, raw_response="[]")
            except Exception as e:
                print(f"❌ [ASRClient] Request failed: {e}")
                return ASRResult(chunk=chunk, raw_response="[]")

    async def process_async(self, chunks: List[AudioChunk]) -> List[ASRResult]:
        print(f"🚀 [ASRClient] Starting concurrent API requests for {len(chunks)} chunks (max_concurrent={self.max_concurrent})...")
        semaphore = asyncio.Semaphore(self.max_concurrent)
        # Using larger timeout for heavy models
        timeout = aiohttp.ClientTimeout(total=1800)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            tasks = [self._process_chunk(session, chunk, semaphore) for chunk in chunks]
            results = await asyncio.gather(*tasks)
        print(results)
        print(f"✅ [ASRClient] Completed {len(results)} API requests.")
        return results

    def process(self, chunks: List[AudioChunk]) -> List[ASRResult]:
        return asyncio.run(self.process_async(chunks))


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
                if not text or text.lower() == "[silence]":
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
            
            global_segments = []
            global_word_segments = []
            languages = []
            
            for i, res in enumerate(results):
                offset = res.chunk.start_time
                
                # Determine valid range to avoid duplicates in overlapping regions
                valid_start = 0.0
                if i > 0:
                    prev_chunk = results[i-1].chunk
                    overlap_start = res.chunk.start_time
                    overlap_end = prev_chunk.end_time
                    midpoint = (overlap_start + overlap_end) / 2.0
                    valid_start = midpoint
                    
                valid_end = float('inf')
                if i < len(results) - 1:
                    next_chunk = results[i+1].chunk
                    overlap_start = next_chunk.start_time
                    overlap_end = res.chunk.end_time
                    midpoint = (overlap_start + overlap_end) / 2.0
                    valid_end = midpoint
                    
                # Process segments
                for seg in res.segments:
                    # Apply offset to local timestamps
                    global_start = seg["start"] + offset
                    global_end = seg["end"] + offset
                    seg_mid = (global_start + global_end) / 2.0
                    
                    # Only keep segment if its midpoint falls within the valid non-overlapping range
                    if valid_start <= seg_mid < valid_end:
                        new_seg = seg.copy()
                        new_seg["start"] = global_start
                        new_seg["end"] = global_end
                        
                        # If segment has words, update their timestamps too
                        if "words" in new_seg:
                            new_words = []
                            for w in new_seg["words"]:
                                w_copy = w.copy()
                                if "start" in w_copy: w_copy["start"] += offset
                                if "end" in w_copy: w_copy["end"] += offset
                                new_words.append(w_copy)
                            new_seg["words"] = new_words
                            
                        global_segments.append(new_seg)
                        
                        # Add valid words to global word list
                        if "words" in new_seg:
                             global_word_segments.extend(new_seg["words"])

                # If word_segments exists at chunk level but not in segments (rare but possible in some WhisperX outputs),
                # we might miss them if we only rely on seg['words']. 
                # However, WhisperX usually nests words in segments. 
                # Let's rely on segments['words'] for consistency with the filtering logic above.
                
                languages.append(res.language)
                
            main_language = max(set(languages), key=languages.count) if languages else "zh"
            # Normalize language to 2-letter code for whisperx
            if "zh" in main_language: main_language = "zh"
            elif "en" in main_language: main_language = "en"
                
            merged_results.append(MergedResult(
                original_file=orig_file,
                segments=global_segments,
                word_segments=global_word_segments,
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
        return asr_results

    def format_and_save(self, aligned_results: List[MergedResult]) -> List[Dict]:
        """Formats to WhisperX schema and saves to JSON and SRT."""
        print(f"📝 [PostProcessor] Formatting final results (WhisperX compatible) & saving...")
        final_outputs = []
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
            final_outputs.append(output)
            
            # Save to JSON
            out_path = f"{res.original_file}.aligned.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)
            print(f"  -> Saved final result to {out_path}")
            
            # Save to SRT
            srt_path = f"{os.path.splitext(res.original_file)[0]}.srt"
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(cleaned_segments, 1):
                    start_str = self._format_srt_time(seg.get("start", 0.0))
                    end_str = self._format_srt_time(seg.get("end", 0.0))
                    text = seg.get("text", "").strip()
                    f.write(f"{i}\n{start_str} --> {end_str}\n{text}\n\n")
            print(f"  -> Saved SRT subtitle to {srt_path}")
            
        return final_outputs

    def process(self, data: Any, stage: str = "format_and_save") -> Any:
        """Dispatcher for different post-processing stages."""
        if stage == "clean_words":
            return self.clean_word_alignments(data)
        elif stage == "format_and_save":
            return self.format_and_save(data)
        else:
            raise ValueError(f"Unknown post-processing stage: {stage}")

# ============================================================================
# Pipeline Runner (组装器)
# ============================================================================

def run_pipeline(
    file_paths: List[str],
    api_url: str = "http://localhost:8000",
    chunk_duration: int = 720,
    overlap: int = 20,
    max_concurrent: int = 10,
    device: str = "cpu"
):
    print(f"{'='*60}\n🚀 Starting Engineering VibeVoice Pipeline\n{'='*60}")
    
    # 1. Chunking
    processor = AudioProcessor(chunk_duration=chunk_duration, overlap=overlap)
    chunks = processor.process(file_paths)
    
    # 2. ASR API Requests
    client = ASRClient(base_url=api_url, max_concurrent=max_concurrent)
    raw_results = client.process(chunks)
    
    # 3. Clean API Outputs
    cleaner = ASRCleaner()
    cleaned_results = cleaner.process(raw_results)
    
    # 4. WhisperX Alignment (on Chunks)
    aligner = WhisperXAligner(device=device)
    aligned_chunks = aligner.process(cleaned_results)
    
    # 4.5 Clean Word Alignments (Merge chars & punctuation) via PostProcessor
    post_processor = PostProcessor()
    aligned_chunks = post_processor.process(aligned_chunks, stage="clean_words")
    
    # 5. Merge Results
    merger = ResultMerger()
    merged_results = merger.process(aligned_chunks)
    
    # 6. Post Processing & Output (Format and Save)
    # Now MergedResult acts as AlignedResult
    final_outputs = post_processor.process(merged_results, stage="format_and_save")
    
    # 7. Cleanup
    processor.cleanup()
    
    print(f"\n🎉 Pipeline Finished Successfully!\n{'='*60}")
    return final_outputs

def main():
    parser = argparse.ArgumentParser(description="Test VibeVoice API + WhisperX Alignment Pipeline")
    parser.add_argument("audio_paths", nargs="+", help="Paths to audio files to process")
    parser.add_argument("--url", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--chunk_duration", type=int, default=720, help="Audio chunk size in seconds")
    parser.add_argument("--overlap", type=int, default=20, help="Overlap between chunks in seconds")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for WhisperX alignment")
    
    args = parser.parse_args()
    
    run_pipeline(
        file_paths=args.audio_paths,
        api_url=args.url,
        chunk_duration=args.chunk_duration,
        overlap=args.overlap,
        max_concurrent=args.concurrency,
        device=args.device
    )

if __name__ == "__main__":
    main()
