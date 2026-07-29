from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class AvatarChunk:
    index: int
    generation_start_frame: int
    generation_frames: int
    overlap_frames: int
    emitted_start_frame: int
    emitted_frames: int
    frame_rate: float

    @property
    def audio_start_seconds(self) -> float:
        return self.generation_start_frame / self.frame_rate

    @property
    def generation_duration_seconds(self) -> float:
        return self.generation_frames / self.frame_rate

    @property
    def emitted_duration_seconds(self) -> float:
        return self.emitted_frames / self.frame_rate


def plan_avatar_chunks(
    audio_duration_seconds: float,
    frame_rate: float,
    generation_frames: int,
    overlap_frames: int,
    max_chunks: int | None = None,
) -> list[AvatarChunk]:
    if audio_duration_seconds <= 0:
        raise ValueError("audio_duration_seconds must be positive")
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive")
    if generation_frames <= 0:
        raise ValueError("generation_frames must be positive")
    if not 0 <= overlap_frames < generation_frames:
        raise ValueError("overlap_frames must be in [0, generation_frames)")

    total_frames = max(1, math.ceil(audio_duration_seconds * frame_rate))
    chunks: list[AvatarChunk] = []
    emitted = 0
    index = 0
    while emitted < total_frames and (max_chunks is None or index < max_chunks):
        overlap = 0 if index == 0 else min(overlap_frames, emitted)
        capacity = generation_frames - overlap
        emitted_frames = min(capacity, total_frames - emitted)
        chunks.append(
            AvatarChunk(
                index=index,
                generation_start_frame=emitted - overlap,
                generation_frames=generation_frames,
                overlap_frames=overlap,
                emitted_start_frame=emitted,
                emitted_frames=emitted_frames,
                frame_rate=frame_rate,
            )
        )
        emitted += emitted_frames
        index += 1
    return chunks
