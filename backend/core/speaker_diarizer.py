from typing import Dict, List, Any
import os
from ..config import settings


class SpeakerDiarizer:
    def __init__(self):
        self.pipeline = None
        self.hf_token = settings.HUGGINGFACE_TOKEN

    def _load_pipeline(self):
        if self.pipeline is None:
            from pyannote.audio import Pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )

    def diarize(self, audio_path: str, segments: List[Dict] = None) -> Dict[str, Any]:
        try:
            self._load_pipeline()
            
            diarization = self.pipeline(audio_path)
            
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
            
            return {
                "speaker_segments": speaker_segments,
                "num_speakers": len(set([s["speaker"] for s in speaker_segments]))
            }
        except Exception as e:
            return self._mock_diarize(segments)

    def _mock_diarize(self, segments: List[Dict]) -> Dict[str, Any]:
        speaker_segments = []
        for i, seg in enumerate(segments):
            speaker = "SPEAKER_00" if i % 2 == 0 else "SPEAKER_01"
            speaker_segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": speaker
            })
        
        return {
            "speaker_segments": speaker_segments,
            "num_speakers": 2
        }

    def classify_speaker_roles(self, 
                               speaker_segments: List[Dict], 
                               transcription_segments: List[Dict]) -> Dict[str, Any]:
        storage_keywords = ["货架", "库存", "仓储", "入库", "出库", "盘点", "货架号", "仓位", "存储", "库存管理"]
        logistics_keywords = ["运输", "配送", "物流", "车辆", "路线", "时效", "调度", "车队", "配送中心", "运输路线"]

        speaker_texts = {}
        for spk_seg in speaker_segments:
            spk = spk_seg["speaker"]
            if spk not in speaker_texts:
                speaker_texts[spk] = ""
            
            for trans_seg in transcription_segments:
                if abs(spk_seg["start"] - trans_seg["start"]) < 1.0:
                    speaker_texts[spk] += " " + trans_seg["text"]
                    break

        speaker_roles = {}
        for speaker, text in speaker_texts.items():
            storage_count = sum(1 for kw in storage_keywords if kw in text)
            logistics_count = sum(1 for kw in logistics_keywords if kw in text)
            
            if storage_count > logistics_count:
                role = "仓储方"
            elif logistics_count > storage_count:
                role = "物流方"
            else:
                role = "其他参会方"
            
            speaker_roles[speaker] = {
                "role": role,
                "storage_keyword_count": storage_count,
                "logistics_keyword_count": logistics_count
            }

        role_segments = []
        for spk_seg in speaker_segments:
            spk = spk_seg["speaker"]
            role_info = speaker_roles.get(spk, {"role": "未知"})
            role_segments.append({
                **spk_seg,
                "role": role_info["role"]
            })

        return {
            "speaker_roles": speaker_roles,
            "role_segments": role_segments
        }

    def merge_transcription_with_roles(self,
                                         transcription_segments: List[Dict],
                                         role_segments: List[Dict]) -> List[Dict]:
        merged = []
        for trans_seg in transcription_segments:
            matched_role = "未知"
            for role_seg in role_segments:
                if abs(trans_seg["start"] - role_seg["start"]) < 2.0:
                    matched_role = role_seg["role"]
                    break
            
            merged.append({
                **trans_seg,
                "role": matched_role
            })
        
        return merged
