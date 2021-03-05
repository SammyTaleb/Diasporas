
# -*- coding: utf-8 -*-
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
import json
import re
import numpy as np
import itertools
from collections import Counter
from keras_preprocessing.sequence import pad_sequences
from .snipparse import snippetsParser, snippetsParser2
import string

class DiasporaEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.actions_grid=['query_return','query_query_step','record_return','record_step','stop']
    self.actions_kb=["add_to_db_field","delete_db_field","stop"]
    self.kd_gold_standard=0
    #Change here the web_data_frame to be dynamic.
    self.web_data_frame = pd.read_csv("DiasporaGym/data/big.csv").drop_duplicates()
    self.person_ind=0
    self.querys_step=[]
    self.episodeDF=[]
    self.queryDF=[]
    self.queryDF_indexes=[]
    self.row_index_query=[]
    self.query_seen=[]
    self.agent_db=[]
    self.gold_standard=[]
    self.episode_continues=False
    self.num_steps_per_episode=0
    self.gold_json=json.load(open('DiasporaGym/data/gold_std.json','r'))
    self.persons=list(set(i for i in self.gold_json.keys()))
    self.num_persons=len(self.persons)-1
    self.words_organizations=open('DiasporaGym/data/jrc-organizations.txt','r',encoding='utf').read().lower().splitlines()
    self.env=self
    self.actionHash={key:selection for key,selection
                      in enumerate(
                        itertools.product(
                          range(len(self.actions_grid)-1),
                          range(len(self.actions_kb)-1)
                        )
                      )
                    }
    #preprocess variables
    self.word_dict={'PAD':0,'UNK':1}
    self.vocab_len=0
    self.let_dict={}
    self.let_len=0
    self._configParams=json.load(open('DiasporaGym/config/preprocess.json','r'))
    self._perprocessURL()
    


    #Space variables
    high = np.array([
           200,
           200])
    low = np.array([
           0,
           0])

    self.action_space = spaces.Discrete(len(self.actionHash.keys()))
    self.observation_space = spaces.Box(low, high, dtype=np.int32)
    self.firstQuery(self.isQueryPossible())
    self._preprocessText()
    self.person_ind=-1
  def _preprocessText(self):
    
    self.word_dict = self._preprocessGoldStd(self._configParams)
    self.word_dict = self._getWordDictFromPreprocessed(self._configParams)
    self.word_dict = self._getWordDictFromInstitutions(self._configParams)
    self.vocab_len = len(self.word_dict.keys())
    
  def _perprocessURL(self):
    self.let_dict={letter:str(index+1) for index,letter in enumerate(string.ascii_lowercase+string.digits)}
    self.let_dict['PAD']="0"
    self.let_dict['UNK']="9999"
    return self.let_dict
    
  def _preprocessGoldStd(self,configParams):
    currentMax=max(self.word_dict.values())
    for item in self.gold_json.keys():
      try:
        if self.gold_json[item]['institution'] is not None:
            for word in self.gold_json[item]['institution'].lower().split(" "):
                if not(word in self.word_dict.keys()) and len(word) > 3:
                  currentMax+=1
                  self.word_dict[word]=currentMax
      except Exception as E:
        print(E)
        print("couldn't preprocess key: "+str(item))
      year =self.gold_json[item]['year_finish']
      year_finish=str(year)
      if not(year_finish in self.word_dict.keys()):
          currentMax+=1
          self.word_dict[year_finish]=currentMax 
    return self.word_dict


  def _getWordDictFromPreprocessed(self,configParams):
    currentMax=max(self.word_dict.values())
    all_sentences=[]
    all_sentences=all_sentences+(self.web_data_frame['search'].tolist())
    all_sentences=all_sentences+(self.web_data_frame['title'].tolist())
    all_sentences=all_sentences+(self.web_data_frame['text'].tolist())
    all_sentences=[str(sentence) for sentence in all_sentences]
    all_sentences=list(set(all_sentences))
    all_sentences=" ".join(all_sentences)
    words = [word for word in all_sentences.split(" ") if len(word) > 3]
    cnt=Counter(words)
    for word,_ in cnt.most_common(configParams['config']['vocab_length']):
      if not(word in self.word_dict.keys()):
        currentMax+=1
        self.word_dict[word]=currentMax 

    return self.word_dict
  
  def _getWordDictFromInstitutions(self,configParams):
    #TODO
    return self.word_dict

  def _pathSeq(self,sequence,max_len,vocab_dict):
    try:
        seq = [int(vocab_dict.get(element,vocab_dict['UNK'])) for element in sequence if vocab_dict.get(element,vocab_dict['UNK'])]
    except:
        if vocab_dict.get(sequence,vocab_dict['UNK']):
            seq=[int(vocab_dict.get(sequence,vocab_dict['UNK']))]
    pad_sequence=pad_sequences([seq], maxlen=max_len, dtype='int32', padding='post', truncating='post', value=int(vocab_dict["PAD"]))
    return pad_sequence

  def _processOutput(self,nonProcessedOutput):
    output=[]
    if self._configParams['outputs']['cite']==True:
      output.append(
        self._pathSeq(
          nonProcessedOutput[0][0][1],
          self._configParams['config']['cite_length'],
          self.let_dict))
    if self._configParams['outputs']['engine_search']==True:
      output.append(nonProcessedOutput[0][0][2])
    if self._configParams['outputs']['id_person']==True:
      output.append(nonProcessedOutput[0][0][3])
    if self._configParams['outputs']['number_snippet']==True:
      output.append(nonProcessedOutput[0][0][0])
    if self._configParams['outputs']['search']==True:
      output.append(
        self._pathSeq(
          nonProcessedOutput[0][0][4].split(" "),
          self._configParams['config']['search_length'],
          self.word_dict))
    if self._configParams['outputs']['title']==True:
      output.append(
        self._pathSeq(
          nonProcessedOutput[0][0][5].split(" "),
          self._configParams['config']['title_length'],
          self.word_dict))
    if self._configParams['outputs']['text']==True:
      output.append(
        self._pathSeq(
          nonProcessedOutput[0][0][6].split(" "),
          self._configParams['config']['text_length'],
          self.word_dict))
    if self._configParams['outputs']['steps']==True:
      output.append(nonProcessedOutput[0][1])
    if self._configParams['outputs']['query_seen']==True:
      output.append(nonProcessedOutput[0][2])
    if self._configParams['outputs']['agent_db']==True:
      output.append(len(nonProcessedOutput[0][3][0]))
      output.append(len(nonProcessedOutput[0][3][1]))
    return output

  def processState(self,state,action=None):
    final_state=[]
    for ele in state:
        try:
            a=ele.tolist()[0]
            final_state=final_state+a
        except:
            final_state=final_state+[ele]
        
    final_state=np.array([final_state])
    actions=[0]*len(self.actionHash)
    if action:
        actions[action]=1
    actions=np.array([actions])
    f_state=np.concatenate([np.array(final_state),np.array(actions)],axis=1)
    return f_state
    
  def _expandAction(self, action):
    action_grid,action_kb=self.actionHash[action]
    return action_grid,action_kb

  def step(self, action,agent_db=None):
    """ eexecute the next step and return the reward """
    """we must return an arrangement of the type:
    response:["state","reward","boolean(if it finished or not)"]"""
    action_grid,action_kb=self._expandAction(action)#Get the actions to perform
    print('Action_grid : {}, Action_kb : {}'.format(action_grid,action_kb))
    self.num_steps_per_episode+=1
    agent_db=self._action_db_selector(action_kb)
    query=self._action_grid_selector(action_grid)
    reward=self._get_reward()
    self.query_seen[self.query_indexes[self.query_ind]]+=1
    query_seen=self.query_seen[self.query_indexes[self.query_ind]]
    nonProcessedOutput = [(query,self.num_steps_per_episode,query_seen,agent_db),reward,self.episode_continues]
    #print(nonProcessedOutput)
    state=self._processOutput(nonProcessedOutput)
    return [self.processState(state,action),reward,self.episode_continues,'info']
    #return [np.array([self.num_steps_per_episode,query_seen],dtype=np.float64),reward,self.episode_continues,{}]

  def reset(self):
    """ aquí avanzamos al siguiente episodio"""
    self.num_steps_per_episode=0
    if self.person_ind==self.num_persons:
      self.person_ind=0
    elif self.person_ind==-1:
      self.person_ind+=1
    else:
      self.person_ind+=1
      self.newQuery(self.isQueryPossible())
    self.agent_db=[[],[]]
    

    #set stepepisodeDFDF
    
    self.episodeDF = self.web_data_frame.loc[self.web_data_frame['id_person'] == self.persons[self.person_ind]].sort_values(['engine_search','number_snippet'])
    
    #set Query lists and indexes for queryDFs
    self.querys_step=list(set(self.episodeDF.search.tolist())) #Los querys
    self.query_indexes=list(range(len(self.querys_step))) #Los índices de la lista de querys
    self.row_index_query=[0]*len(self.querys_step) #En qué registro está cada query
    self.query_seen=[0]*len(self.querys_step) #Cuantos pasos he dado en ese query
    self.query_ind=0 #En qué query estoy
    self.querydf=self.episodeDF.loc[self.episodeDF["search"] == self.querys_step[self.query_indexes[self.query_ind]]]
    self.episode_continues=False

    #obtenemos el gold standard
    self.gold_std=self._get_gold_std()
    return self.render()

    
  def render(self, mode='human'):

    #return self.querydf.iloc[self.row_index_query[self.query_indexes[self.query_ind]]].tolist(),self.agent_db
    try:
        query_seen=self.query_seen[self.query_indexes[self.query_ind]]
    except IndexError:
        query_seen=0
    #return [(query,self.num_steps_per_episode,query_seen,agent_db),reward,self.episode_continues]
    
    query= self.querydf.iloc[self.row_index_query[self.query_indexes[self.query_ind]]].tolist()
    nonProcessedOutput = [(query,self.num_steps_per_episode,query_seen,[[],[]]),0,self.episode_continues]
    state=self._processOutput(nonProcessedOutput)
    return self.processState(state)

  def _action_grid_selector(self,action_grid):
      #print(action_grid)
      if action_grid==0:
      #query_return is 0
        self._query_return()
      if action_grid==1:
      #query_step is 1
        self._query_query_step()
      if action_grid==2:
      #record_return is 2
        self._record_return()
      if action_grid==3:
      #record_return is 3  
        self._record_step()
      if action_grid==4:
      #stop is 4
        query_ind=self.query_ind
        query_indexes=self.query_indexes[query_ind]
        row_index=self.row_index_query[query_indexes]
        return self.querydf.iloc[row_index].tolist()
      query_ind=self.query_ind
      query_indexes=self.query_indexes[query_ind]
      row_index=self.row_index_query[query_indexes]
      return self.querydf.iloc[row_index].tolist()
      

  def _action_db_selector(self,action_selector):
    #print(action_selector)
    if action_selector==0:
    #action 0 is push to db
      self._push_to_db()
    if action_selector==1:
    #action 1 is pop from db
      self._pop_from_db()
    if action_selector==2:
    #action 2 is do nothing on db
      return self.agent_db
    return self.agent_db
    #action 3 is replace entities with top on the data base

  def _push_to_db(self):
    text=(str(self.querydf.iloc[self.row_index_query[self.query_indexes[self.query_ind]]].tolist()[-2:]))
    #print(text)
    entities=self._get_entities(text)
    if len(entities[0])>0 and entities[0] not in  self.agent_db[0]:
      self.agent_db[0].append(entities[0])
    if len(entities[1])>0 and int(entities[1]) not in  self.agent_db[1]:
      self.agent_db[1].append(int(entities[1]))
    
  def _pop_from_db(self):
    try:
      self.agent_db[0].pop()
    except:
      pass
    try:
      self.agent_db[1].pop()
    except:
      pass
  
  def _get_entities(self,text):
    institutions=[]
    for word in self.words_organizations:
        idx=text.lower().find(word)
        if idx >= 0:
            word_finded=text[idx:idx+len(word)]
            institutions.append(word_finded.strip())
        else:
            initials=""
            try:
                for ele in word.split(' '):
                    initials+=ele[0]
                    if len(initials)>3:
                        idx=text.lower().find(initials)
                        if idx>=0:
                            institutions.append(initials.strip())
            except:
                pass
    years=re.findall("[12][901][0-9]{2}",text)
    if len(institutions)>0:
      institutions=institutions[0].lower()
    if len(years)>0:
      years=years[0]
    return (institutions,years)
  def _get_gold_std(self):
    institution=self.gold_json[str(\
       self.persons[self.person_ind])]['institution']
    if institution is not None:
        institution=institution.lower()
    year=self.gold_json[str(\
       self.persons[self.person_ind])][u'year_finish']
    return [institution,year]
  def _loadInstituteRE(self):
    with open("../data/JRC-Organizations.normal.txt",'r') as fd:
      self.words_organizations=[]
      for line in fd.readlines():
        self.words_organizations.append(" "+line.strip().lower()+" ")
        self.words_organizations.append(" "+line.strip().lower())
        self.words_organizations.append(line.strip().lower()+" ")
  def isQueryPossible(self):
    name_person=str(self.gold_json[self.persons[self.person_ind]]['name'])
    queries=[name_person+i for i in ['',' doctorate',' institute',' master',' PhD',' undergraduate',' university']]
    for query in queries:
      if query not in self.querys_step:
          return query
    return None
  def firstQuery(self,query):
      #search_engine=[0,1,2,3]
      search_engine=[0]
      #results=snippetsParser(query,self.persons[self.person_ind])
      
      for engine in search_engine:
          results=snippetsParser2(self.persons[self.person_ind],query,engine)
          
      self.web_data_frame=pd.DataFrame(results)
  def newQuery(self,query):
      #search_engine=[0,1,2,3]
      search_engine=[0]
      #results=snippetsParser(query,self.persons[self.person_ind])
      for engine in search_engine:
          results=snippetsParser2(self.persons[self.person_ind],query,engine)
      
      start=self.web_data_frame.shape[0]
      for i in range(len(results)):
          self.web_data_frame.loc[start+i]=results[i]
      self.querys_step.append(query)
      self.episodeDF = self.web_data_frame.loc[self.web_data_frame['id_person'] == self.persons[self.person_ind]].sort_values(['engine_search','number_snippet'])
      self.query_indexes.append(self.query_indexes[-1]+1)
      self.row_index_query.append(0)
      self.query_seen.append(0)
      self.word_dict = self._getWordDictFromPreprocessed(self._configParams)

      
      
  def _query_query_step(self):
    if self.query_ind==len(self.querys_step)-1:
      query=self.isQueryPossible()
      if  query is not None:
          self.newQuery(query)
          self.query_ind=self.query_ind+1
          
    else:
      self.query_ind+=1
    self.querydf=self.episodeDF.loc[self.episodeDF["search"] == self.querys_step[self.query_indexes[self.query_ind]]]
  def _query_return(self):
    if self.query_ind==0:
      self.query_ind=self.query_indexes[-1]
    else:
      self.query_ind-=1
    self.querydf=self.episodeDF.loc[self.episodeDF["search"] == self.querys_step[self.query_indexes[self.query_ind]]]
  def _record_return(self):
    
    if self.row_index_query[self.query_indexes[self.query_ind]]==0:
      self.row_index_query[self.query_indexes[self.query_ind]]=len(self.querydf.index)-1
    else:
      self.row_index_query[self.query_indexes[self.query_ind]]-=1

  def _record_step(self):
    
    if self.row_index_query[self.query_indexes[self.query_ind]]==len(self.querydf.index)-1:
      if self.query_ind==len(self.querys_step)-1:
        self.episode_continues=True
      else:
        self.row_index_query[self.query_indexes[self.query_ind]]=0
    else:
      self.row_index_query[self.query_indexes[self.query_ind]]+=1
    
  def _get_reward(self):
    if self.gold_std[0] in self.agent_db[0]:
      reward_institutes=1/len(self.agent_db[0])
    elif self.gold_std[0] is None:
        reward_institutes=0 if len(self.agent_db[0])==0 else 1/len(self.agent_db[0])
    else:
      initials=""
      for ele in self.gold_std[0].split(" "):
          initials+=ele[0]
      if initials in self.agent_db[0]:
          reward_institutes=1/len(self.agent_db[0])
      else:
          reward_institutes=0
    
    if self.gold_std[1] in self.agent_db[1]:
      reward_years=1/len(self.agent_db[1])
    else:
      reward_years=0
    return reward_institutes+reward_years-0.1

  def _get_reward_multiple(self):
    """ calcular con agent_db & gold_std """
    if len(self.gold_std[0])>1:
        if len(self.agent_db[0])>0:
            good_pred=0
            for ele in self.gold_std[0]:
                if ele.lower() in self.agent_db[0]:
                    good_pred+=1
            reward_institutes=((good_pred/len(self.agent_db[0]))+(good_pred/len(self.gold_std[0])))/2
        else:
            reward_institutes=0
    else:
        if self.gold_std[0][0].lower() in self.agent_db[0]:
          reward_institutes=1/len(self.agent_db[0])
        else:
          reward_institutes=0
    if len(self.gold_std[1])>1:
        if len(self.agent_db[1])>0:
            good_pred=0
            for ele in self.gold_std[1]:
                if int(ele) in self.agent_db[1]:
                    good_pred+=1
            reward_years=((good_pred/len(self.agent_db[1]))+(good_pred/len(self.gold_std[1])))/2
        else:
            reward_years=0
    else:
        if int(self.gold_std[1][0]) in self.agent_db[1]:
          reward_years=1/len(self.agent_db[1])
        else:
          reward_years=0
    return reward_institutes+reward_years-0.1
  """
  @gin.configurable
  def general_agent(step=None,reward=None):
    assert(step!=None)
  """  

