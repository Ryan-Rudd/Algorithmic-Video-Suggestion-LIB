from video_suggestion.models.user import User
from video_suggestion.models.video import Video


def test_user():
    # Create a test user
    user = User(user_id=1, name='Alice')

    # Test initial properties
    assert user.user_id == 1
    assert user.name == 'Alice'
    assert user.watched_videos == []

    # Test video watching functionality
    user.watch_video(10)
    user.watch_video(5)
    assert user.watched_videos == [10, 5]

def test_video():
    # Create a test video
    video = Video(video_id=1, title='Introduction to Python', description='Learn the basics of Python programming.', duration=3600)

    # Test initial properties
    assert video.video_id == 1
    assert video.title == 'Introduction to Python'
    assert video.description == 'Learn the basics of Python programming.'
    assert video.duration == 3600
