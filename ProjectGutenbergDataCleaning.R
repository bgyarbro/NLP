library(gutenbergr)
library(dplyr)
library(tidytext)
library(stringr)

data(stop_words)
stop_words

?gutenberg_download

gutenberg_authors

test <- gutenberg_download(c(1,5,7,37,40))
test
tail(test)

df <- gutenberg_download(c(1:100))
df
tail(df)
df[1000000:1000020,]
df[10000:10010,]

testGrouped <- test %>% group_by(gutenberg_id) %>%
  mutate(paragraphs = paste0(text, collapse = ' ')) %>%
  select(gutenberg_id, paragraphs) %>%
  distinct(gutenberg_id, paragraphs)

testGrouped
testGrouped[1,2]  

work1 <- testGrouped[1,2]

as.character(work1)

