# Music Recommendation System

## Prerequisites
1. Clean Ubuntu, tested on 26.04 LTS x86-64
2. Python 3

## Installation
1. curl -fsSL https://raw.githubusercontent.com/Areso/music-recommendation-system/main/install.sh | bash
2. on the http://<ipaddress> would be opened the index page of the recommendation 

## Recommenders 
1. Find artists by a tag
2. Find similar artists
3. Find users with similar taste
4. Recommend artists from a user's listen-weighted tag profile
5. Recommend artists from collaborative-filtering neighbours

## Command-line client
Run `python cli_client.py` to open the interactive client. It connects to
`http://127.0.0.1:8000/api` by default; use `/connect <IP-or-DNS>` to select a
remote server. Plain addresses use HTTP port 8000, while complete URLs are also
accepted. The selected address is saved in `cli_client.config` and reused the
next time the client starts. The client checks that server and prints its status
at startup. For a remote installation done by install.sh, use port 80.

Commands:
- `/connect <host|host:port|URL>` - select the API server and check its status
- `/check` - check whether the selected server is up
- `/tag_search` - find artists by tag
- `/similar_artists` - find artists with similar tag profiles
- `/find_similar_user` - find users with similar listening histories
- `/recommend_content` - recommend unlistened artists from a user's tag profile
- `/recommend_cf` - recommend unlistened artists from similar users
- `/help`
- `/quit`, and its `/exit` synonym

Tag and artist prompts show live suggestions, including when typing an argument
directly after `/tag_search` or `/similar_artists`. Use the arrow keys to choose
one and Enter to accept it.

## Jyputer noterbooks
1. bi130_module_0_data_cleaning.ipynb - data processing and data cleaning, common part for 1,2,3,4 modules
2. bi130_module_1_2.ipynb - module 1: graph centrality and artist prestige; module 2: community detection and profiling
3. bi130_module_3.ipynb - module 3: tag-based artist modelling with TF-IDF
4. bi130_module_4.ipynb - module 4: content-based and collaborative-filtering recommenders
  
all passing nbformat 5.10.4 and nbconvert 7.17.1 lints  
  
## Detailed description modules 1 and 2
[README_old](README_old.txt)

## Neo4j
It has very small, PoC demo scope: it shows user friendship connections.
1. neo4j.txt has instructions how to load data
2. neo4j.dump is a binary dump of a local instance of neo4j with loaded data
3. neo4j_dump_and_restore.txt has instructions how to create a dump and how to recover from one.
