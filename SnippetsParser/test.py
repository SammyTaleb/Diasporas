import nest_asyncio
from search_engine_parser.core.engines.google import Search as GoogleSearch
from search_engine_parser.core.engines.bing import Search as BingSearch
from search_engine_parser.core.engines.yahoo import Search as YahooSearch
from search_engine_parser.core.engines.github import Search as GithubSearch
from search_engine_parser.core.engines.duckduckgo import Search as DuckDuckGoSearch
import pandas as pd



def snippetsParser(person,_query,search_engine_nbr=0):
    gsearch=GoogleSearch()
    bsearch=BingSearch()
    ysearch=YahooSearch()
    gitsearch=GithubSearch()
    dsearch=DuckDuckGoSearch()
    search_engines=[gsearch,bsearch,ysearch,gitsearch,dsearch]
    query=person+' '+_query
    nest_asyncio.apply() #allow the following event to be nested
    search_args=(query,1)
    results=search_engines[search_engine_nbr].search(*search_args)
    snippets_frame=[]
    try:
        for i in range(len(results)):
            snippets_frame.append({'site':results[i]['links'],'engine_search':search_engine_nbr,
                                   'id_person':person,'search':query,'title':results[i]['titles'],'text':results[i]['descriptions']})
    except:
        pass
    return snippets_frame

def query(personnes,_query,search_engine_nbr=0):
    prev_results=[]
    try:
        df=pd.read_csv('../DiasporaEnv/DiasporaGym/data/big.csv')
        for i in range(df.shape[0]):
            prev_results.append({'site':df.iloc[i]['site'],'engine_search':df.iloc[i]['engine_search'],
                                       'id_person':df.iloc[i]['id_person'],'search':df.iloc[i]['search'],'title':df.iloc[i]['title'],'text':df.iloc[i]['text']})
    except:
        pass
    results=[]
    results.extend(prev_results)
    for person in personnes:
        res=snippetsParser(person,_query,search_engine_nbr)
        for ele in res:
            results.append(ele)
    return pd.DataFrame(results)

if __name__=='__main__':
    personnes=['Sammy Taleb','Pegah Alizadeh','Tristan Darrigol','Donald Trump','Joe Biden','Barack Obama','Emmanuel Macron']
    results=query(personnes,'education',0)
    for i in range(len(results['site'])):
        if int(results['engine_search'][i])==0 and 'google.' in results['site'][i]:
            results['site'][i]=results['site'][i][29:]
        
    results.to_csv('../DiasporaEnv/DiasporaGym/data/big.csv')
    print(results)

















