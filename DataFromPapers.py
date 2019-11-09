import tarfile

tar = tarfile.open("data/1_europarl.tgz")
tar.extractall()
tar.close()

tar = tarfile.open("data/2_news_commentary.tgz")
tar.extractall()
tar.close()

tar = tarfile.open("data/3_UN_corpus.tgz")
tar.extractall()
tar.close()

