class User:
    def __init__(self, user_id: int, name: str):
        self.user_id = user_id
        self.name = name
        self.watched_videos = []

    def watch_video(self, video_id: int):
        self.watched_videos.append(video_id)
