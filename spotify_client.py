# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials

# # Spotify 개발자 페이지에서 받은 키
# client_id = 'YOUR_CLIENT_ID'
# client_secret = 'YOUR_CLIENT_SECRET'

# # 인증
# auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
# sp = spotipy.Spotify(auth_manager=auth_manager)

# # 예시: forest_snowy_day → 분위기 키워드로 검색
# mood_keywords = {
#     "forest_snowy_day": "calm piano forest",
#     "city_night": "urban chill night",
#     "beach_sunny_day": "summer beach pop"
# }

# def get_tracks_by_mood(label):
#     query = mood_keywords.get(label, "relaxing music")
#     results = sp.search(q=query, type='track', limit=5)
#     tracks = []
#     for item in results['tracks']['items']:
#         tracks.append({
#             'name': item['name'],
#             'artist': item['artists'][0]['name'],
#             'url': item['external_urls']['spotify']
#         })
#     return tracks