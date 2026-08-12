===============================================================================
BI130 PROJECT: MODULES 1 AND 2
Graph centrality, artist prestige, and community detection on the
HetRec 2011 Last.fm 2K social graph.
===============================================================================

This README covers BI130 - Modules - 1 and 2 Complete.ipynb, which produces
every table and figure in the Module 1 and Module 2 sections of the report.
Modules 3 and 4 are covered by their own notebook and README.
VodkaBalalaikaKlyukva

-------------------------------------------------------------------------------
1. WHAT THE NOTEBOOK DOES
-------------------------------------------------------------------------------

MODULE 1, GRAPH CENTRALITY AND PRESTIGE

  * Builds the user friendship graph (1,892 nodes, 12,717 undirected edges).
  * Builds the artist co-listener graph (17,619 nodes, 1,319,800 weighted
    edges, edge weight = number of distinct shared listeners).
  * Produces the basic graph report for both graphs: node and edge counts,
    connected components, isolated nodes, average clustering coefficient, and
    a degree-distribution plot.
  * Computes degree, betweenness, closeness and PageRank on the user graph,
    and ranks the top 15 users under each.
  * Compares the four rankings by Spearman correlation, top-15 overlap, and
    per-user rank divergence.
  * Computes weighted PageRank on the artist graph and contrasts it with raw
    listen-count popularity.

MODULE 2, COMMUNITY DETECTION AND PROFILING

  * Runs Louvain on the friendship graph across 26 seeds and selects the
    highest modularity partition.
  * Reports modularity, community count, and the community-size distribution.
  * Profiles the five largest communities using artists and tags, each
    measured by reach and by lift.
  * Assesses whether community membership aligns with musical taste.


-------------------------------------------------------------------------------
2. REQUIREMENTS
-------------------------------------------------------------------------------

Python 3.10 or later, with:

    pandas
    numpy
    scipy
    matplotlib
    networkx==3.4.2
    ipython

The NetworkX version is pinned because Louvain results depend on it. The
published modularity of 0.4640 at seed 11 was produced with NetworkX 3.4.2,
and the notebook records the running version in the Louvain configuration
table (cell 58). A different version may return a different partition even
with the same seed.

    pip install pandas numpy scipy matplotlib "networkx==3.4.2" ipython jupyter


-------------------------------------------------------------------------------
3. DATA DEPENDENCY
-------------------------------------------------------------------------------

THIS NOTEBOOK DOES NOT READ THE RAW HETREC FILES. It reads the cleaned exports
produced by the Module 0 preparation notebook
(BI130_Data_Cleaning_Template.ipynb). Run that notebook first.

The following six files must exist in the clean folder before any cell here
will run. The notebook asserts each row count on load and stops if any of them
is wrong.

    File                                    Expected rows
    ------------------------------------    -------------
    users_clean.csv                                 1,892
    user_friends_undirected_clean.csv              12,717
    user_artists_clean.csv                         92,829
    artists_clean.csv                              17,619
    tags_clean.csv                                  9,489
    user_taggedartists_clean.csv                  184,146

The load cell also asserts that the total listening weight equals 69,183,975,
that all IDs are unique, that no friendship is duplicated or self-referential,
and that every user and artist reference resolves.


-------------------------------------------------------------------------------
4. FOLDER LAYOUT AND CONFIGURATION
-------------------------------------------------------------------------------

    <PROJECT_DIR>/
        clean/           input, written by the Module 0 notebook
        outputs_m12/     output, created automatically by this notebook

PROJECT_DIR defaults to ~/Downloads/BIDA. Override it without editing any code
by setting an environment variable before starting Jupyter:

    macOS / Linux
        export BIDA_PROJECT_DIR=/path/to/BIDA
        jupyter notebook

    Windows PowerShell
        $env:BIDA_PROJECT_DIR = "C:\path\to\BIDA"
        jupyter notebook

The output folder is created if it does not exist. No input file is modified.


-------------------------------------------------------------------------------
5. HOW TO RUN
-------------------------------------------------------------------------------

CELL NUMBERS IN THIS README COUNT FROM 1, matching the execution counts
Jupyter shows in the left margin after a clean top-to-bottom run.

Run all cells top to bottom. Cells are not independent: later cells depend on
graph objects and dataframes built earlier, so a partial run will fail.

Every stage ends with assertion checks. If an assertion fires the notebook
stops rather than continuing on bad data, and the message names the specific
check that failed.

RUNTIME. Expect several minutes. The notebook prints its own timings; the
measured costs on the reference run were:

    Step                                        Cell      Time
    ----------------------------------------    ----    -------
    Artist clustering coefficient (exact)         17    189.7 s
    Community layout for the network plot         67     22.4 s
    Betweenness centrality                        27     15.4 s
    Artist PageRank (weighted)                    45      4.1 s
    Closeness centrality                          27      3.6 s

Shared-listener pair generation (cell 11) and the 26 Louvain runs (cell 59)
are the other substantial steps and print their own runtimes.

IF THE MACHINE IS SLOW OR MEMORY-CONSTRAINED, two settings can be reduced,
both at the cost of changing published values:

  * EXACT_ARTIST_CLUSTERING = True (cell 17) switches the artist clustering
    coefficient to an approximation if set to False. The reported value of
    0.7950 is exact and requires True.
  * The friendship network sample in cell 9 is capped at 120 nodes. Raising it
    makes the figure slower to lay out and harder to read.


-------------------------------------------------------------------------------
6. REPRODUCIBILITY
-------------------------------------------------------------------------------
7. REPRODUCING EACH TABLE
-------------------------------------------------------------------------------

Every table in the report is written to outputs_m12/ as a CSV by the cell
listed below. No table is hand-edited.

MODULE 1

    graph_basic_report.csv                                  (cell 17)
        Node, edge, component, clustering statistics for both graphs

    user_degree_distribution.csv                            (cell 20)
        Degree counts and fractions, user graph

    artist_degree_distribution.csv                          (cell 21)
        Degree counts and fractions, artist graph

    user_library_size_summary.csv                           (cell 22)
        Artists per user: maximum, minimum, median, mean, and the
        number of users sitting at the maximum

    user_centrality_all.csv                                 (cell 29)
        All four centrality scores and ranks for all 1,892 users

    top15_users_by_centrality.csv                           (cell 30)
        Top 15 users under each measure, with scores

    centrality_spearman_correlation.csv                     (cell 34)
        4x4 Spearman matrix

    centrality_top15_overlap.csv                            (cell 36)
        4x4 top-15 overlap counts

    centrality_consensus_users.csv                          (cell 39)
        Users appearing in all four top-15 lists

    centrality_rank_divergence.csv                          (cell 40)
        Per-user best, worst and range of ranks

    centrality_top15_side_by_side.csv                       (cell 41)
        Top 15 by position under each measure

    artist_prestige_popularity_all.csv                      (cell 46)
        Popularity, PageRank, both ranks, rank gap, all artists

    top15_artists_by_popularity.csv                         (cell 47)
        Top 15 by total listening weight

    top15_artists_by_pagerank.csv                           (cell 47)
        Top 15 by weighted PageRank

    artist_prestige_popularity_divergence.csv               (cell 49)
        Every artist in either top 15, sorted by rank gap

    artist_top15_side_by_side.csv                           (cell 51)
        Popularity and PageRank top 15 aligned by position

    module_1_summary.csv                                    (cell 55)
        Headline Module 1 results

MODULE 2

    user_community_membership.csv                           (cell 63)
        One community ID per user

    selected_community_sizes.csv                            (cell 64)
        Size, share and cumulative share per community

    community_listening_summary.csv                         (cell 71)
        Listening activity per profiled community

    community_artist_statistics.csv                         (cell 75)
        Community reach per artist, all combinations

    top15_artists_by_community_reach.csv                    (cell 75)
        Top 15 artists by reach, per community

    community_artist_lift_all.csv                           (cell 81)
        Artist lift, all community-artist combinations

    top15_distinctive_artists_by_lift.csv                   (cell 81)
        Top 15 artists by lift, per community

    artist_reach_lift_overlap.csv                           (cell 81)
        Overlap between the reach and lift artist lists

    community_tag_preparation_summary.csv                   (cell 87)
        Tag coverage and eligibility per community

    community_tag_reach_all.csv                             (cell 93)
        Tag reach, all community-tag combinations

    top15_common_tags_by_community.csv                      (cell 93)
        Top 15 tags by reach, per community

    common_tag_top15_overlap.csv                            (cell 93)
        Pairwise overlap between communities' common tags

    community_tag_lift_all.csv                              (cell 100)
        Tag lift, all community-tag combinations

    top15_distinctive_tags_by_lift.csv                      (cell 100)
        Top 15 tags by lift, per community

    tag_common_lift_overlap.csv                             (cell 100)
        Overlap between the common and distinctive tag lists

    integrated_community_profiles.csv                       (cell 105)
        Combined artist and tag profile per community

    module_2_limitations.csv                                (cell 113)
        The 13 documented limitation categories


-------------------------------------------------------------------------------
8. REPRODUCING EACH FIGURE
-------------------------------------------------------------------------------

All figures are written to outputs_m12/ at 160 dpi (180 dpi for the community
network) with tight bounding boxes.

    user_friendship_network_top10_coloured.png              (cell 9)
        Friendship network sample, top 10 users coloured

    user_degree_distribution.png                            (cell 20)
        User degree distribution, log-log

    artist_degree_distribution.png                          (cell 21)
        Artist degree distribution, log-log

    artist_degree_distribution_focused.png                  (cell 23)
        Artist degree 0 to 200, degree 49 highlighted

    artist_degree_49_two_panel.png                          (cell 24)
        Degree 49 against all others, and the same range with 49
        removed from display

    centrality_spearman_heatmap.png                         (cell 35)
        Spearman correlation between the four measures

    centrality_top15_overlap_heatmap.png                    (cell 37)
        Top-15 overlap between the four measures

    centrality_comparison_combined.png                      (cell 38)
        Both centrality comparisons in one two-panel figure

    artist_prestige_versus_popularity.png                   (cell 52)
        Left: all artists by both ranks. Right: slope chart for the
        artists appearing in either top 15

    community_size_distribution.png                         (cell 65)
        Louvain community sizes, log scale

    community_size_cumulative_share.png                     (cell 66)
        Cumulative share of users by community

    louvain_community_network_nodes_edges.png               (cell 67)
        Friendship network coloured by community

    community_top_distinctive_artists.png                   (cell 82)
        Top 5 artists by lift in each community, shared scale

    community_tag_lift_heatmap.png                          (cell 101)
        Tag lift grid across all five communities

Cells 20 and 21 both call the shared plot_degree_distribution helper defined
in cell 19, which takes the output filename as an argument.

centrality_comparison_combined.png (cell 38) merges the two heatmaps from
cells 35 and 37 into a single exhibit. All three are produced; use whichever
suits the layout.


-------------------------------------------------------------------------------
9. KNOWN NOTES
-------------------------------------------------------------------------------

  * Degree-zero nodes are excluded from the two logarithmic degree-distribution
    plots, because a logarithmic axis cannot display zero. The count of
    excluded nodes is printed on each figure. No node is excluded from any
    calculation.

  * Cell 24's lower panel removes degree 49 from the display only. The 10,350
    artists at degree 49 remain in the dataset and in every calculation.

  * Cell 36 contains one commented assertion,
    assert (np.diag(top15_overlap) == TOP_K).all()
    It was left commented deliberately; the same property is verified by the
    top-15 validation in cell 31.

  * The artist co-listener graph's degree and clustering values are strongly
    shaped by the top-50 library structure of the source data. Cell 43
    explains why artist prestige is measured with weighted PageRank rather
    than any degree-based measure, and no degree-based conclusion is drawn
    from that graph anywhere in the report.


-------------------------------------------------------------------------------
10. TEAM AND CONTRIBUTION STATEMENT
-------------------------------------------------------------------------------

RICK SYMONDS was responsible for Modules 1 and 2: construction of the user
friendship graph and the artist co-listening graph, the centrality and
prestige analysis, the Louvain community detection, and the artist and tag
profiling of the detected communities, together with the data preparation
used by those two modules.

ANTON GLADYSHEV was responsible for Modules 3 and 4: the tag preprocessing
and TF-IDF artist-tag matrix, the free-text retrieval function, and the
content-based and collaborative filtering recommenders, together with the
data preparation used by those two modules.

Each member prepared the cleaned data independently for their own modules, so
that cleaning decisions made for the graph analysis could not influence the
retrieval and recommendation results.

-------------------------------------------------------------------------------
11. DATASET
-------------------------------------------------------------------------------

HetRec 2011 Last.fm 2K. Cantador, Brusilovsky and Kuflik, RecSys 2011.
Available from grouplens.org/datasets/hetrec-2011

===============================================================================
