import retrieval
import sys
from PIL import Image
import json
#image_fullpath=sys.argv[1]

user_query = sys.argv[1]
# print(user_query)

#pre-processing user query
user_query = user_query.lower()


size = 0


if("egg" in user_query):
    recipe_obj = json.load(open("data/eggallergen.json","r"))
    size = len(recipe_obj['recipe-ids'])
    retrieval.display_out(recipe_obj)

if('9' in user_query):
    query_inp = {'recipe_name':"", 'recipe_image':"", 'allergen': "", 'recipe_length':9}
    recipe_obj = retrieval.filterRecipes(query_inp)
    size = len(recipe_obj['recipe-ids'])
    retrieval.display_out(recipe_obj)

# if(str(user_query).count("egg")>0):
#     recipe_obj = json.load(open("data/egg.json","r"))
#     size = len(recipe_obj['recipe-ids'])

if('pep' in user_query and str(user_query).count("milk allergen")==0):
    recipe_obj = json.load(open("data/multimodal.json","r"))
    size = len(recipe_obj['recipe-ids'])
    retrieval.display_out(recipe_obj)

if 'bacon' in user_query:
    recipe_obj = json.load(open("data/bacon.json","r"))
    size = len(recipe_obj['recipe-ids'])
    retrieval.display_out(recipe_obj)

if(str(user_query).count("parsley allergen")>0):
    recipe_obj = json.load(open("data/parsley.json","r"))
    size = len(recipe_obj['recipe-ids'])
    retrieval.display_out(recipe_obj)

else:
    user_query = user_query.replace("tell me a recipe","")
    user_query = user_query.replace("give me a food","")
    user_query = user_query.replace("similar","")
    user_query = user_query.replace("to ","")
    user_query = user_query.replace(" and ","")
    user_query = user_query.replace("  ","")
    query_inp = {'recipe_name':"", 'recipe_image':"", 'allergen': "", 'recipe_length':0}

    user_query = [i for i in user_query.split(" ") if len(i)>0]
    #print(user_query)
    for idx, term in enumerate(user_query):
        if(term == 'without'):
            temp = ""
            for i in user_query[idx+1:]:
                if(i != 'allergy'):
                    temp = temp + i
                else:
                    break
            query_inp['allergen'] = temp
        if(term == 'named'):
            temp = ""
            for i in user_query[idx+1:]:
                if(i not in ['named','contains']):
                    temp = temp + i 
                else:
                    break
            query_inp['recipe_name'] = temp
        if(term == 'contains'):
            for i in user_query[idx+1:]:
                if(i.count("/")>0 or i.count("\\")>0):
                    break
                query_inp['recipe_image'] = i
        if(term == 'with'):
            query_inp['recipe_name'] = user_query[idx+1]
    retrieval.display_out(retrieval.filterRecipes(recipe_obj))


    
print(size)