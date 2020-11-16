import nest_asyncio
import time
from search_engine_parser.core.engines.google import Search as GoogleSearch
from search_engine_parser.core.engines.bing import Search as BingSearch
from search_engine_parser.core.engines.yahoo import Search as YahooSearch
from search_engine_parser.core.engines.github import Search as GithubSearch
from search_engine_parser.core.engines.duckduckgo import Search as DuckDuckGoSearch
import pandas as pd

def snippetsParser(query,search_engine_nbr=0):
    gsearch=GoogleSearch()
    bsearch=BingSearch()
    ysearch=YahooSearch()
    gitsearch=GithubSearch()
    dsearch=DuckDuckGoSearch()
    search_engines=[gsearch,bsearch,ysearch,gitsearch,dsearch]
    #nest_asyncio.apply() #allow the following event to be nested
    search_args=(query,3)
    results=search_engines[search_engine_nbr].search(*search_args)
    snippets_frame=pd.DataFrame([i for i in results] )
    return snippets_frame

deb=time.time()   
results=snippetsParser('Google history',0)
print(results)
print(time.time()-deb)


















