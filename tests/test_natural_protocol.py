import importlib.util
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/'benchmarks'))
from natural_reference_qa import mine as mine_qa
from natural_coreference import mine as mine_turns


def test_original_questions_and_upstream_evidence_preserved_without_outcome_selection():
 sample={'sample_id':'real-id','conversation':{'session_1':[{'dia_id':'D1:1','speaker':'Mira','text':'I went to the gallery yesterday.'},
                    {'dia_id':'D1:2','speaker':'Alex','text':'I like the gallery too.'}]},
         'qa':[{'question':'Where did Mira go?','evidence':['D1:1'],'category':1},
               {'question':'Where did Mira go?','evidence':['D1:2'],'category':1},
               {'question':'Where did Mira and Alex go?','evidence':['D1:1'],'category':1}]}
 rows,total=mine_qa([sample,sample,sample,sample])
 assert total==1 and rows[0]['gold']==['D1:1'] and rows[0]['question']=='Where did Mira go?'
 assert rows[0]['references']==['D1:1'] and not rows[0]['multi_turn_alias_validated']


def test_heuristic_screen_cannot_silently_label_coreference_as_verified():
 sample={'sample_id':'real-id','conversation':{'session_1':[{'dia_id':'D1:1','speaker':'Mira','text':'The gallery is across the street beside the big park.'},
       {'dia_id':'D1:2','speaker':'Alex','text':"It was great to hear from you yesterday and I hope we meet again soon."}]}}
 rows,total=mine_turns([sample])
 assert total==1 and rows[0]['query']==sample['conversation']['session_1'][0]['text']
 assert rows[0]['annotation']['valid_chain'] is None and rows[0]['annotation']['status']=='unreviewed'
