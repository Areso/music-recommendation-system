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

## Jyputer noterbooks
1. bi130_module_0_data_cleaning.ipynb - data processing and data cleaning, common part for 1,2,3,4 modules
2. bi130_module_1_2.ipynb - module 1: graph centrality and artist prestige; module 2: community detection and profiling
3. bi130_module_3.ipynb - module 3: tag-based artist modelling with TF-IDF
4. bi130_module_4.ipynb - module 4: content-based and collaborative-filtering recommenders

## Detailed description modules 1 and 2
[README_old](README_old.txt)

## Neo4j
It has very small, PoC demo scope: it shows user friendship connections.
1. neo4j.txt has instructions how to load data
2. neo4j.dump is a binary dump of a local instance of neo4j with loaded data
3. neo4j_dump_and_restore.txt has instructions how to create a dump and how to recover from one.
