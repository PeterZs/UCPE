from pathlib import Path
from tqdm.auto import tqdm
import json
import yt_dlp


panflow_root = Path("data/360-1M")
panshot_root = Path("data/UCPE")
pf_clip_root = panshot_root / "PanFlow" / "match_clips"
pf_video_root = panshot_root / "PanFlow" / "videos"
pf_video_root.mkdir(parents=True, exist_ok=True)

clip_metas = list(pf_clip_root.glob("*.json"))
clip_metas.sort()
print(f"Found {len(clip_metas)} PanFlow clip files.")

videos = set()
for clip_meta in clip_metas:
    with open(clip_meta, "r") as f:
        meta = json.load(f)
    videos.add(meta["video_id"])
print(f"Found {len(videos)} unique PanFlow videos.")

downloaded_videos = pf_video_root.glob("*.mp4")
downloaded_videos = set([p.stem for p in downloaded_videos])
print(f"Found {len(downloaded_videos)} already downloaded videos.")

videos = videos - downloaded_videos
print(f"{len(videos)} videos to download.")


def download_video(video_id, output_path):
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "outtmpl": str(output_path),
        "format": "bestvideo[height<=2500][height>1500]",
        "quiet": False,
        "no_warnings": True,
        "simulate": False,
        "cookiefile": "~/.config/cookies.txt",
        "print": [
            "before_dl:Format: %(format_id)s | Res: %(resolution)s | FPS: %(fps)s",
            "after_move:Size: %(filesize:.2fMB)s"
        ],
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

for video_id in tqdm(videos, desc="Downloading videos"):
    video_path = Path(pf_video_root) / f"{video_id}.mp4"
    try:
        download_video(video_id, video_path)
    except Exception as e:
        tqdm.write(f"Failed to download {video_id}: {e}")
        continue
