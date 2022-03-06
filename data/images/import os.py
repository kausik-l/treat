import os
import json
from os import listdir

recipe_obj = json.load(open("/Users/vedantkhandelwal/Desktop/recp_repn.json","r"))
for i in recipe_obj['recipe-ids']:
    if(i not in ['1','2']):
        for idx, val in enumerate(recipe_obj['recipe-ids'][i]['ingredients']):
            if((idx+1)/10<1):
                arr = listdir("/Users/vedantkhandelwal/Google Drive/My Drive/SyncWork/MultimodalQA/Demo/demo_RecipeQA/demoRecipeQA/data/images/" + recipe_obj['recipe-ids'][i]['recipe_name']+"/ingredients/")
                path = "./images/" + recipe_obj['recipe-ids'][i]['recipe_name']+"/ingredients/"+[i for i in arr if i.count("0"+str(idx+1))==1][0]
                #print(path)
                #raise KeyboardInterrupt
            #path = "./images/" + recipe_obj['recipe-ids'][i]['recipe_name']+"/ingredients/"+
                recipe_obj['recipe-ids'][i]['ingredients'][idx]['image'] = path
f = open("/Users/vedantkhandelwal/Desktop/recp_repn.json", "w")
json.dump(recipe_obj, f)
f.close()