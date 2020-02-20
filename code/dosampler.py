"""
Direct replication from - https://gist.github.com/akelleh/2a741a57f0a6f75262146ab17b2a6ef3
"""

import numpy as np
import pandas as pd

import dowhy.api

# Skill of an author drives both the length of the title of the article and the 
# CTR generated on the article. Disregarding this would lead to the wrong 
# conclusion that title length is the cause of CTR

N = 100000
n = 30000

# Ground truth: CTR is independent of title length

title_length = np.random.choice(range(25), size=N) + 1
click_through_rate = np.random.beta(5, 100, size=N)

# except for the expert who write good articles and specific titles

title_length_2 = np.random.normal(13, 3, size=n).astype(int)
click_through_rate_2 = np.random.beta(10, 100, size=n)

title_lengths = np.array(list(title_length) + list(title_length_2))
click_through_rates = np.array(list(click_through_rate) + list(click_through_rate_2))

df = pd.DataFrame({"click_through_rate": click_through_rates,
                   "title_length": title_lengths,
                   "author": [1]*N + [0]*n})

subset_df = df.query("title_length > 0 & title_length < 25")

causal_df = (subset_df.causal
                      .do("title_length",
                          method="weighting",
                          variable_types={"title_length": 'd',
                                          "click_through_rate": 'c', 
                                          "author": 'd'},
                          outcome="click_through_rate",
                          common_causes=["author"],
                          proceed_when_unidentifiable=True))

causal_df.click_through_rate.mean()
subset_df.click_through_rate.mean()