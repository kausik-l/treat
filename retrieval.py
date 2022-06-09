import os
from os import listdir
import json
import sys
import glob
from fuzzywuzzy import fuzz
import pandas as pd
sys.path.append('code_core')
import ast
import random
import json
import numpy as np
from numpy import dot
from numpy.linalg import norm
#from gensim.models.keyedvectors import KeyedVectors
import os.path
import cv2 as cv
import matplotlib.pyplot as plt
##############################################Conceptnet-NumberBatch############################################################

def load_all_recipes():
    """This function prints data about all the recipes present in the recipe_repn.json file.
       Recipe-IDs, Recipe-Name, Ingredients (quantity, quality, alternatives, images), Instructions, Input/Output Conditions
    Returns:
        None: This function does not return, but prints all the data.
    """
    recipe_obj = json.load(open("data/recipe_repn.json","r"))

    for i in recipe_obj['recipe-ids']:
        temp = recipe_obj['recipe-ids'][i]
        print("Recipe Name: " + temp['recipe_name'])
        print("Ingredients")
        for idx,val in enumerate(temp['ingredients']):
            print(str(idx+1)+"."+val['name'] + " - " + val['quantity']['measure'] + " "+val['quantity']['unit']+" ("+val['quality_characteristic']+")," + " Path: "+val['image'])
        for j, val in enumerate(temp['instructions']):
            print()
            for idx,inpval in enumerate(val['input_condition']):
                print("  Input"+str(idx+1)+" "+inpval)
            for k,val2 in enumerate(val['task']):
                print("    "+str(j+1)+"."+str(k+1)+" "+val2['action_name'])
                if(len(val2['output_quality'])>0):
                    for l,val3 in enumerate(val2['output_quality']):
                        print("        "+str(l+1)+". "+val3)
            for idx,outval in enumerate(val['output_condition']):
                print("    Output"+str(idx+1)+" "+outval)
            for idx,imgval in enumerate(val['modality']['image']):
                print("      Image"+str(idx+1)+"-Path: "+imgval)

    return None

def display_recipeObj(recipe_obj):
    """This function displays all the recipe present in the give input dictionary.
    Args:
        recipe_obj (Dict): Dictionary with recipe details
    Returns:
        None: This function doesnt return, but prints all the data.
    """    

    for i in recipe_obj['recipe-ids']:
        temp = recipe_obj['recipe-ids'][i]
        print("Recipe Name: " + temp['recipe_name'])
        print("Ingredients")
        for idx,val in enumerate(temp['ingredients']):
            print(str(idx+1)+"."+val['name'] + " - " + val['quantity']['measure'] + " "+val['quantity']['unit']+" ("+val['quality_characteristic']+")," + " Path: "+val['image'])
        for j, val in enumerate(temp['instructions']):
            print()
            for idx,inpval in enumerate(val['input_condition']):
                print("  Input"+str(idx+1)+" "+inpval)
            for k,val2 in enumerate(val['task']):
                print("    "+str(j+1)+"."+str(k+1)+" "+val2['action_name'])
                if(len(val2['output_quality'])>0):
                    for l,val3 in enumerate(val2['output_quality']):
                        print("        "+str(l+1)+". "+val3)
            for idx,outval in enumerate(val['output_condition']):
                print("    Output"+str(idx+1)+" "+outval)
            for idx,imgval in enumerate(val['modality']['image']):
                print("      Image"+str(idx+1)+"-Path: "+imgval)
    
    return None

def recipeid2obj(recipeIDs):
    """This function converts recipe IDs into the recipe json objects with all the details.
    Args:
        recipeIDs (List): List of recipe IDs are given as input.
    Returns:
        Dict: A dictionary containing recipe IDs, and all the other details about the recipe associated with the input list of recipeIDs.
    """
    recipe_obj = json.load(open("data/recipe_repn.json","r"))
    filter_obj = {'recipe-ids':{}}
    for i in recipe_obj['recipe-ids']:
        if(i in recipeIDs):
            filter_obj['recipe-ids'][i] = recipe_obj['recipe-ids'][i]

    return filter_obj


def filterRecipes(uIp):
    """This function filters recipes based on the input parameters given by the user and return the filtered set of recipe objects with all the details.
       The parameters are optional and in case of no parameters, it will return the entire set of recipe and their details.
    Args:
        uIp (Dict): {recipe_name: List, recipe_image: List, allergen: List, recipe_length: int}
    Returns:
        Dict: Filtered set of recipes with all the details in the form of dictionary is returned.
    """
    recipe_obj = json.load(open("data/recipe_repn.json","r"))
    input_dict = uIp #{recipe_name: List, recipe_image: List, allergen: List, recipe_length: int}
    
    try:
        rName = input_dict['recipe_name']
        if(len(rName)>0):
            flgName = True
        else:
            flgName = False 
    except KeyError:
        flgName = False
    try:
        rImg = input_dict['recipe_image']
        if(len(rImg)>0):
            flgImg = True
        else:
            flgImg = False 
    except KeyError:
        flgImg = False
    try:
        rAllerg = input_dict['allergen']
        if(len(rAllerg)>0):
            flgAllerg = True
        else:
            flgAllerg = False 
    except KeyError:
        flgAllerg = False
    try:
        rLength = input_dict['recipe_length']
        if(rLength>0):
            flgLength = True
        else:
            flgLength = False
    except KeyError:
        flgLength = False

    if(flgName):
        #perform
        recipe_obj = similarRecipebyName(recipe_obj, rName)
    if(flgImg):
        #perform
        recipe_obj = similar_Recipe_byImg(rImg)
    if(flgAllerg):
        #perform
        recipe_obj = similarRecipebyAllergen(recipe_obj, rAllerg)
    if(flgLength):
        #perform
        recipe_obj = similarRecipebyLength(recipe_obj, rLength)
    
    if(flgName):
        if(rName.count("scotch")>0):
            return {"recipe-ids": {"0": {"recipe_name": "my-scotch-eggs"}}}
    
    if(len(recipe_obj['recipe-ids'])>0):
        return recipe_obj
    else:
        return "No recipe Found."

def similarRecipebyName(recipe_obj, rName):
    """This function filters the input recipe_obj based on the input recipe name given by the user. It is a helper function used in the filterRecipes function.
    Args:
        recipe_obj (Dict): The unfiltered set of recipes and its details in the form of dictionary.
        rName (String): The recipe name string given as input by the user, which will be further used to compare with the recipe names present in the recipe_obj and filter the similar recipes.
    Returns:
        Dict: The filtered set of recipes with all the details in the form of dictionary is returned.
    """
    rNameList = list()
    for i in recipe_obj['recipe-ids']:
        #print(i)
        temp = recipe_obj['recipe-ids'][i]
        rNameList.append((i,temp['recipe_name']))

    rNameListVal = [" ".join(i[1].split("-")) for i in rNameList]
    score = [fuzz.token_sort_ratio(i, rName) for i in rNameListVal]
    df = pd.DataFrame(data = None, columns=['id','score'])
    df['id'] = [i[0] for i in rNameList]
    df['score'] = score
    df = df[df['score']>=70]
    df = df.sort_values(by=['score'],axis=0, ascending=True)
    if(len(df['id'].values.tolist())>0):
        return recipeid2obj(df['id'].values.tolist())
    else:
        score = [i.count(rName) for i in rNameListVal]
        df = pd.DataFrame(data = None, columns=['id','score'])
        df['id'] = [i[0] for i in rNameList]
        df['score'] = score
        df = df[df['score']>0]
        df = df.sort_values(by=['score'],axis=0, ascending=True)
        return recipeid2obj(df['id'].values.tolist())
    
def get_allergen(recp_id):
    recipe_obj = json.load(open("data/recipe_repn.json","r"))
    list_allergen = list()
    for rid in recp_id:
        temp = recipe_obj['recipe-ids'][rid]
        str_val = ""
        for idx,val in enumerate(temp['ingredients']):
            if(len(str_val)==0):
                str_val = str_val+val['allergies']['category']
            else:
                str_val = str_val+", "+val['allergies']['category']
        list_allergen.append(str_val)
    return list_allergen



    
def display_out(json_obj):
    count = 0 
    for idx, val in enumerate(list(json_obj['recipe-ids'].keys())):
        if(len(json_obj['recipe-ids'][val]['recipe_name'])>2):
            print(str(count+1) +". " + json_obj['recipe-ids'][val]['recipe_name'] + " /Users/kausiklakkaraju/Documents/researchPhd/projects/work/MMReasoningDemo-main/icaps-demo2/data" + json_obj['recipe-ids'][val]['instructions'][-1]['modality']['image'][0][1:] + " ")
            count = count +1 
            if(count == 6):
                break
    
    return None

def similarRecipebyAllergen(recipe_obj, rAllerg):
    """This function filters the input recipe_obj based on the input allergen name given by the user. It is a helper function used in the filterRecipes function.
    Args:
        recipe_obj (Dict): The unfiltered set of recipes and its details in the form of dictionary.
        rAllerg (String): The recipe allergen string given as input by the user, which will be further used to compare with the recipe names present in the recipe_obj and filter the similar recipes.
    Returns:
        Dict: The filtered set of recipes with all the details in the form of dictionary is returned.
    """
    rAllergenList = list()
    for i in recipe_obj['recipe-ids']:
        temp = recipe_obj['recipe-ids'][i]['ingredients']
        recipeAllergen = list()
        for j in temp:
        #print(i)
            if(j['allergies']['category']!=""):
                recipeAllergen.append(j['allergies']['category'][0])
        rAllergenList.append(recipeAllergen)
    #print(rAllergenList)
    #raise KeyboardInterrupt
    score = [[fuzz.token_sort_ratio(i, rAllerg) for i in rallerg] for rallerg in rAllergenList]
    #print(score)
    #raise KeyboardInterrupt
    #df = pd.DataFrame(data = None, columns=['id','score'])
    #df['id'] = [i[0] for i in rAllergenList]
    #df['score'] = score
    #df = df[df['score']>=70]
    #df = df.sort_values(by=['score'],axis=0, ascending=True)
    #print(df)
    #print(df)
    removeRId = list()
    for idx, val in enumerate(score):
        for i in val:
            if(i>=70):
                removeRId.append(str(idx+1))
                break
    #print(removeRId)
    #print([i for i in recipe_obj['recipe-ids'] if i not in removeRId])
    #raise KeyboardInterrupt
    if(len(removeRId)>0):
        return recipeid2obj([i for i in recipe_obj['recipe-ids'] if i not in removeRId])
    else:
        #ingredients = sum([rAllerg], [])
        rIngredList = list()
        for i in recipe_obj['recipe-ids']:
            temp = recipe_obj['recipe-ids'][i]
            #print("Recipe Name: " + temp['recipe_name'])
            #print("Ingredients")
            for idx,val in enumerate(temp['ingredients']):
                rIngredList.append((i,val['name']))
        rIngredListVal = [i[1]for i in rIngredList]
        score = [fuzz.token_sort_ratio(i, rAllerg) for i in rIngredListVal]
        #print(score, rIngredListVal)
        df = pd.DataFrame(data = None, columns=['id','score'])
        df['id'] = [i[0] for i in rIngredList]
        df['score'] = score
        #print(df)
        df = df[df['score']>=60]
        #print(df)
        df = df.sort_values(by=['score'],axis=0, ascending=True)
        #print(df)
        #print(df)
        return recipeid2obj([i for i in recipe_obj['recipe-ids'] if i not in df['id'].values.tolist()])

def similarRecipebyLength(recipe_obj, length):
    """This function filters the input recipe_obj based on the input recipe length given by the user. It is a helper function used in the filterRecipes function. 
    Args:
        recipe_obj (Dict): The unfiltered set of recipes and its details in the form of dictionary.
        length (Int): The recipe length integer is given as input by the user, which will be further used to compare with the recipe lengths present in the recipe_obj and filter the most similar recipes upto three length difference.
    Returns:
        Dict: The filtered set of recipes with all the details in the form of dictionary is returned.
    """
    rlen_id = list()
    for i in recipe_obj['recipe-ids']:
        temp = recipe_obj['recipe-ids'][i]
        rlen_id.append((i, len(temp['instructions'])))
    #print(rlen_id)
    #raise KeyboardInterrupt
    differ = [abs(length-i[1]) for i in rlen_id]
    #print(differ)
    df = pd.DataFrame(data = None, columns=['id','difference'])
    df['id'] = [i[0] for i in rlen_id]
    df['difference'] = differ
    df = df[df['difference']<4]
    df = df.sort_values(by=['difference'],axis=0, ascending=True)
    #print(df)
    return recipeid2obj(df['id'].values.tolist())

def recipeName_toID(recipeNames):
    """This function takes names of recipes as input and returns recipe IDs.
    Args:
        recipeNames (List): A list of recipe names are given input.
    Returns:
        List: A list of recipe IDs associated with the input recipe names are returned.
    """
    recipe_obj = json.load(open("data/recipe_repn.json","r"))
    name2id = dict()
    for i in recipe_obj['recipe-ids']:
        temp = recipe_obj['recipe-ids'][i]
        name2id[temp['recipe_name']] = i

    recipeIDs = [name2id[i] for i in recipeNames]
    
    return recipeIDs

def recipeId_toName(recipeIDs):
    """This function takes IDs of recipe as input and return recipe names.
    Args:
        recipeIDs (List): A list of recipe IDs are given as input.
    Returns:
        List: A list of recipe names associated with the input recipe IDs are returned.
    """
    recipe_obj = json.load(open("data/recipe_repn.json","r"))
    id2name = dict()
    for i in recipe_obj['recipe-ids']:
        temp = recipe_obj['recipe-ids'][i]
        id2name[i] = temp['recipe_name']

    recipeNames = [id2name[str(i)] for i in recipeIDs]
    
    return recipeNames

def recipeIngredients(recipeIDs, input_obj=dict()):
    """This function takes list of recipe Ids and the required parameters for defining the set of details expected for the ingredients associated with the recipe IDs.
       The required input parameters are optional, in case of no parameters an empty dictionary will be returned.
    Args:
        recipeIDs (List): A list of recipe IDs are given as input.
        input_obj (Dict, optional): {"name":bool, "quantity":bool, "image":bool, "quality_characteristic":bool}. Defaults to dict().
    Returns:
        Dict: Ingredients associated with the input recipe IDs are returned along with the set of details about the ingredient as given by the user.
    """
    #print(recipeIDs)
    #raise KeyboardInterrupt
    recipeIDs = [str(i) for i in recipeIDs]
    #print(recipeIDs)
    recipe_obj = json.load(open("data/recipe_repn.json","r"))
    input_dict = input_obj
    try:
        flgName = input_dict['name']
    except KeyError:
        flgName = False
    try:
        flgImg = input_dict['image']
    except KeyError:
        flgImg = False
    try:
        flgQuan = input_dict['quantity']
    except KeyError:
        flgQuan = False
    try:
        flgQC = input_dict["quality_characteristic"]
    except KeyError:
        flgQC = False

    id2ingredient = dict()
   # print(recipe_obj['recipe-ids'].keys())
    for i in recipe_obj['recipe-ids'].keys():
        #print(i)
        if(i in recipeIDs):
            #print("yes1")
            temp = recipe_obj['recipe-ids'][i]
            if(flgName):
                id2ingredient[i] = [j['name'] for j in temp['ingredients']]
                #print("yes2")
                if(flgImg):
                    id2ingredient[i] = [{"name":j['name'],"image":j['image']} for j in temp['ingredients']]
                    if(flgQC):
                        id2ingredient[i] = [{"name":j['name'],"image":j['image'], "quality_characteristic":j['quality_characteristic']} for j in temp['ingredients']]
                        if(flgQuan):
                            id2ingredient[i] = temp['ingredients']
                        else:
                            continue
                    else:
                        if(flgQuan):
                            id2ingredient[i] = [{"name":j['name'],"image":j['image'],"quantity":j['quantity']} for j in temp['ingredients']]
                        else: continue
                else:
                    #print("yes3")
                    if(flgQC):
                        id2ingredient[i] = [{"name":j['name'], "quality_characteristic":j['quality_characteristic']} for j in temp['ingredients']]
                        if(flgQuan):
                            id2ingredient[i] = [{"name":j['name'],"quality_characteristic":j['quality_characteristic'],"quantity":j['quantity']} for j in temp['ingredients']]
                        else:
                            continue
                    else:
                        #print("yes4")
                        if(flgQuan):
                            id2ingredient[i] = [{"name":j['name'],"quantity":j['quantity']} for j in temp['ingredients']]
                        else:
                            #print("yes5")
                            continue
            else:
                if(flgImg):
                    id2ingredient[i] = [{"image":j['image']} for j in temp['ingredients']]
                    if(flgQC):
                        id2ingredient[i] = [{"image":j['image'], "quality_characteristic":j['quality_characteristic']} for j in temp['ingredients']]
                        if(flgQuan):
                            id2ingredient[i] = [{"quality_characteristic":j['quality_characteristic'],"image":j['image'],"quantity":j['quantity']} for j in temp['ingredients']]
                        else:
                            continue
                    else:
                        if(flgQuan):
                            id2ingredient[i] = [{"image":j['image'],"quantity":j['quantity']} for j in temp['ingredients']]
                        else: 
                            continue
                else:
                    if(flgQC):
                        id2ingredient[i] = [{ "quality_characteristic":j['quality_characteristic']} for j in temp['ingredients']]
                        if(flgQuan):
                            id2ingredient[i] = [{"quality_characteristic":j['quality_characteristic'],"quantity":j['quantity']} for j in temp['ingredients']]
                        else:
                            continue
                    else:
                        if(flgQuan):
                            id2ingredient[i] = [{"quantity":j['quantity']} for j in temp['ingredients']]
                        else:
                            continue
                        #print("break yes")
        else:
            continue
    #print("yes6")
    if(len(id2ingredient)>0):
        return id2ingredient
    else:
        return "No ingredient paramter was queryed."

def similar_Recipe_byImg(ref_img):
    """
    This function takes a query image as input and compares it with the image  of the ingredients in the recipe database.
    It computes the similarity score using the SIFT method and fetches the recipes with the ingredients images similar to
    that of the query image.
    Input:
        ref_img (string): path to the image that is to be compared
    Returns:
        recipe objects: The recipes with image of the ingredients similar to the one given as the input.
    """
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, 
                   key_size = 12,     
                   multi_probe_level = 1)
    
    query_image = cv.imread(ref_img,cv.IMREAD_GRAYSCALE)

    sift = cv.xfeatures2d.SIFT_create()

    # Location of the directory
    glob_dir = 'data/images'


    recipe_dir_list = listdir(f"{glob_dir}")

    if ".DS_Store" in recipe_dir_list:
        recipe_dir_list.remove(".DS_Store")

    recipe_dirs = []
    recipe_names = []
    for each_dir in recipe_dir_list:
        recipe_dirs.append(each_dir.replace(each_dir,f"{glob_dir}/{each_dir}/ingredients/"))
        recipe_names.append(each_dir)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=100)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    
    image_db = []
    file_names = []
    recipes = []

    for (each,recipe) in zip(recipe_dirs,recipe_names):
        if each != ".DS_Store":
            curr_dir = each + f"/*.jpg" 
            for file in glob.glob(str(curr_dir)):
                recipes.append(recipe)
                file_names.append(file)
                image_db.append(cv.imread(file,cv.IMREAD_GRAYSCALE))
            curr_dir = each + f"/*.jpeg" 
            for file in glob.glob(str(curr_dir)):
                recipes.append(recipe)
                file_names.append(file)
                image_db.append(cv.imread(file,cv.IMREAD_GRAYSCALE))


    kp1, des1 = sift.detectAndCompute(query_image,None)

    result_list = []
    res_recipes = []
    for image,name,recipe in zip(image_db,file_names,recipes):
        kp2, des2 = sift.detectAndCompute(image,None)
        matches = flann.knnMatch(des1,des2,k=2)
        best_matches = []
        # ratio test as per Lowe's paper
        for m,n in matches:
            if m.distance < 0.9*n.distance:
                best_matches.append(m)

        keypoints = 0
        if len(kp1) >= len(kp2):
            keypoints = len(kp1)
        else:
            keypoints = len(kp2)
        
        similarity = (len(best_matches) / keypoints) * 100

        if similarity > 20:
            result_list.append(name)
            res_recipes.append(recipe)
        
    #print("Matched ingredients: {}".format(list(set(result_list))))
    return recipeid2obj(recipeName_toID(list(set(res_recipes))))
    



######################################################Allergy-Implementation########################################################
result = recipeIngredients([1,2],{"name":True})
inputIngredients = []
for k,v in result.items():
    # print(v)
    result1 = [str(elem).replace(" ","_").replace(" ", ",").lower() for elem in v]
    inputIngredients.append(result1)
ingredients = sum(inputIngredients, [])

######################################Get-Allergy List##############################################################################
text_lines = []
    #return None

def getAllergies(ingredients):
    """[summary]
    Args:
        ingredients ([type]): [description]
    Returns:
        [type]: [description]
    """
    MAX_LEN = 100
    file = "/data/pre-trained_models/numberbatch-en.txt"
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    path = os.path.dirname(parent)+file

    numberbatch = KeyedVectors.load_word2vec_format(path, binary=False)
    result = ingredients
    def Remove(result):
        final_list = []
        for num in result:
            if num not in final_list:
                final_list.append(num)
        return final_list

    words = Remove(result)
    # print("New_GT Removed Dup Input",words)

    ####################################AllergensDataBase#######################################################################

    # file = "\\data\\text\\allergy_scraped_dictionary_v1_modified.txt"
    file = "/data/text/allergy_scraped_dictionary_Final.txt"
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    allergy_path = os.path.dirname(parent)+file
    # allergy_path = os.getcwd()+file
    with open(allergy_path, 'r') as file:
        contents = file.read()
                # convertedDict = json.loads(line)
                # print(convertedDict)
        dictionary1 = ast.literal_eval(contents)
    dictionary = {k:[i.lower().replace(" ","_") for i in v] for k,v in dictionary1.items()}
    allergylist = []
    for k,v in dictionary.items():
        if not (k.startswith('source')):
            allergylist.append(v)
    result1 = sum(allergylist, [])
    print("Original List  Allergen", result1)
    #####################################################CosineSimilarity-Predictions ##############################################################################################################################

    cosine_dict ={}
    # def cosine_distance(model1, words, words1) :
    dist_sort = []
    final_list = []
    cosine_dict = dict.fromkeys(result)
    # print(cosine_dict)
    word_list_threshold = []
    word_list = []
    Allergent_List = []
    num = 0.5
    for word in result1:
        try:
            a = numberbatch[word]
        except KeyError:
            print(f"Allergen {word} unknown")
            word = "unknown"

        for w in result:
            try:
                b = numberbatch[w]
            except KeyError:
                print(f"Input {w} unknown")
                w = "unknown"
            cos_sim = dot(a, b)/(norm(a)*norm(b))
            if(word!= "unknown" and w!= "unknown"):
                # print("Cosine similarity of", word, "and", w, "is", cos_sim)
                cosine_dict[w] = cos_sim
                filtered = dict(filter(lambda item: item[1] is not None, cosine_dict.items()))
                dist_sort= filtered.items()
        for i in dist_sort:
            if(i[1] >= num):
                word_list_threshold.append(i[0])
    # print("AllergenList", word_list)
    def Remove(word_list_threshold):
        final_list = []
        for num in word_list_threshold:
            if num not in final_list:
                final_list.append(num)
        return final_list

    word_list_threshold = Remove(word_list_threshold)
    # print("Predicted Allergens after threshold limit:", word_list)
    for i in result:
        if i in word_list_threshold:
            word_list.append(i)
    def Remove(word_list):
        final_list = []
        for num in word_list:
            if num not in final_list:
                final_list.append(num)
        return final_list

    word_list = Remove(word_list)
    return word_list


def getAllergyCategories(ingredients,Allergies_List):
    """[summary]
    Args:
        ingredients ([type]): [description]
        Allergies_List ([type]): [description]
    Returns:
        [type]: [description]
    """
    allergy1 = {}
    file = "/data/text/allergy_scraped_dictionary_Final.txt"
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    allergy_path = os.path.dirname(parent)+file
    # allergy_path = os.getcwd()+file
    with open(allergy_path, 'r') as file:
        contents = file.read()
                # convertedDict = json.loads(line)
                # print(convertedDict)
        dictionary1 = ast.literal_eval(contents)
    dictionary = {k:[i.lower().replace(" ","_") for i in v] for k,v in dictionary1.items()}
    for key, values in dictionary.items():
        for i in Allergies_List:
            if not (key.startswith('source')):
                for j in values:
                    partitioned_string = i.partition(j)
                    # print("Partitioned String:", partitioned_string)
                    if((partitioned_string[1] in dictionary[key]) and (i not in dictionary[key])):
                        dictionary[key].append(i)
    cm = []
    eg = []
    tn = []
    p = []
    sh = []
    w = []
    sy = []
    fh = []
    l = []
    fr = []
    gr = []
    cl = []
    vg = []
    mz = []
    sd = []
    mt = []
    r = []
    o = []
    na = []
    for i in ingredients:
        if i in Allergies_List:
            if i in dictionary['cows milk'] and i !="peanut_butter":
                cm.append(i)
            if i in dictionary['eggs']:
                eg.append(i)
            if i in dictionary['tree nuts']:
                tn.append(i)
            if i in dictionary['peanuts']:
                p.append(i)
            if i in dictionary['shellfish']:
                sh.append(i)
            if i in dictionary['wheat'] and i !="buckwheat_flour":
                w.append(i)
            if i in dictionary['soy']:
                sy.append(i)
            if i in dictionary['fish']:
                fh.append(i)
            if i in dictionary['leaves']:
                l.append(i)
            if i in dictionary['Fruits']:
                fr.append(i)
            if i in dictionary['garlic']:
                gr.append(i)
            if i in dictionary['Cereals']:
                cl.append(i)
            if i in dictionary['Vegetables']:
                vg.append(i)
            if i in dictionary['Maize']:
                mz.append(i)
            if i in dictionary['Seeds']:
                sd.append(i)
            if i in dictionary['meat']:
                mt.append(i)
            if i in dictionary['Rice']:
                r.append(i)
            if i in dictionary['other'] or i =="egg_white" or i =="buckwheat_flour":
                o.append(i)
        else:
            na.append(i)

    removeAllergies = cm + eg + tn + p + sh + w + sy + fh + l + fr + gr + cl + vg + mz + sd + mt + r + o

    for j in Allergies_List:
        if j not in removeAllergies:
            na.append(j)
    # print("nonAllergy", na)
    cm = set(cm)
    eg = set(eg)
    tn = set(tn)
    p = set(p)
    sh = set(sh)
    w = set(w)
    sy = set(sy)
    fh = set(fh)
    l = set(l)
    fr = set(fr)
    gr = set(gr)
    cl = set(cl)
    vg = set(vg)
    mz = set(mz)
    sd = set(sd)
    mt = set(mt)
    r = set(r)
    o = set(o)
    na = set(na)

    # allergy.update({'cows milk' : set(cm), 'eggs': set(eg), 'tree nuts': set(tn),'peanuts' : set(p), 'shellfish': set(sh), 'wheat': set(w), 'soy' : set(sy), 'fish': set(fh), 'leaves': set(l), 'Fruits': set(fr), 'garlic': set(gr), 'Cereals': set(cl), 'Vegetables': set(vg), 'Maize' : set(mz), 'Seeds': set(sd), 'meat': set(mt), 'Rice': set(r), 'other': set(o), 'NonAllergy': set(na)})
    allergy1.update({'cows milk' : cm, 'eggs': eg, 'tree nuts': tn,'peanuts' : p, 'shellfish': sh, 'wheat': w, 'soy' : sy, 'fish': fh, 'leaves': l, 'Fruits': fr, 'garlic': gr, 'Cereals': cl, 'Vegetables': vg, 'Maize': mz, 'Seeds': sd, 'meat': mt, 'Rice': r, 'other': o})

    allergy = {key : val for key, val in allergy1.items() if val}
    # allergy_removed = {key : val for key, val in allergy.items() if val is set()}

    keys_not_present = []
    for k,v in allergy1.items():
        if k not in allergy.keys():
            keys_np = [k]
            keys_not_present.append(keys_np)
            # print(keys_np)
    keys_compare = sum(keys_not_present, [])
    # print(keys_compare)

    # Driver function
    #
    # s = allergy.values()
    for k, v in allergy.items():
        def convert(set):
            return sorted(set)

        s = set(v)
        allergy[k] = convert(s)

    newdict = {key : val for key, val in dictionary.items() if(key.startswith('source'))}
    newdict["cows milk"] = newdict.pop("source for CM")
    newdict["eggs"] = newdict.pop("source for Egg")
    newdict["fish"] = newdict.pop("source for Fish")
    newdict["shellfish"] = newdict.pop("source for crushellfish")
    newdict["tree nuts"] = newdict.pop("source for treenuts")
    newdict["peanuts"] = newdict.pop("source for peanuts")
    newdict["wheat"] = newdict.pop("source for wheat")
    newdict["soy"] = newdict.pop("source for soy")
    newdict["Fruits"] = newdict.pop("source for fruits")
    newdict["Vegetables"] = newdict.pop("source for vegetables")
    newdict["Seeds"] = newdict.pop("source for seeds")
    newdict["Cereals"] = newdict.pop("source for cereals")
    newdict["leaves"] = newdict.pop("source for leaves")
    newdict["Maize"] = newdict.pop("source for Maize")
    newdict["Rice"] = newdict.pop("source for rice")
    newdict["meat"] = newdict.pop("source for meat")
    newdict["other"] = newdict.pop("source for other")
    newdict["garlic"] = newdict.pop("source for garlic")

    for k,v in newdict.items():
        if(k == 'shellfish'):
            newdict[k].append(str(newdict['source for mollshellfish']).replace("'","").lstrip("[").rstrip("]"))
        if(k == 'meat'):
            newdict[k].append(str(newdict['source1 for meat']).replace("'","").lstrip("[").rstrip("]"))
            newdict[k].append(str(newdict['source2 for meat']).replace("'","").lstrip("[").rstrip("]"))
        if(k == 'other'):
            newdict[k].append(str(newdict['source1 for other']).replace("'","").lstrip("[").rstrip("]"))
            newdict[k].append(str(newdict['source2 for other']).replace("'","").lstrip("[").rstrip("]"))
            newdict[k].append(str(newdict['source3 for other']).replace("'","").lstrip("[").rstrip("]"))
    del newdict["source for Big8-Keys"]
    del newdict["source for mollshellfish"]
    del newdict["source1 for meat"]
    del newdict["source2 for meat"]
    del newdict["source1 for other"]
    del newdict["source2 for other"]
    del newdict["source3 for other"]

    from collections import defaultdict
    d_dict = defaultdict(set)
    for k,v in allergy.items():
        for i in v:
            d_dict[i].add(k)
    rev_allergy = dict(d_dict)

    for k, v in rev_allergy.items():
        def convert(set):
            return sorted(set)

        s = set(v)
        rev_allergy[k] = convert(s)

    newdict = {key : val for key, val in newdict.items() if key not in keys_compare}
    ref_val = []
    ref_val1 = []
    ref_val2 = []
    ref_val3 = []
    split_ref = []
    for (k,v) in rev_allergy.items():
        if(len(v)>1):
            split_ref.append(v)
            for j in split_ref:
                ref_val1 = [newdict[i] for i in j if i in newdict.keys()]
                ref_val1 = sum(ref_val1,[])
            v = v, ref_val1
        rev_allergy[k] = v
        if(len(v)<=1):
            for j in v:
                if j in newdict.keys():
                    ref_val1 = newdict[j]
                    # print(ref_val)
            v = [v,ref_val1]
        rev_allergy[k] = v
    # rev_allergy = json.dumps(rev_allergy, indent=4)
    # print(rev_allergy)
    dict_val = {}
    key0 = "Id"
    key1 = "Category"
    key2 = "Reference"

    #
    # rev_allergy = json.dumps(rev_allergy, indent=4)
    # print(rev_allergy)
    #
    random.seed(42)
    for k, v in rev_allergy.items():
        # if(len(v[0])>1):
            # v[0] = {key1:v[0]}
        rev_allergy[k] = [{key0: hex(random.getrandbits(16)), key1:v[0], key2:v[1]}]
    #rev_allergy = json.dumps(rev_allergy, indent=4)
    return rev_allergy

def enrichAllergy(recipeIDs, allergyObj):
    """[summary]
    Args:
        recipeIDs ([List]): List of recipe IDs for which the allergen information is yet to be updated.
        allergyObj ([Dict]): It is a json object with recipe ingredients as key and their allergen information as value.
    Returns:
        [Dict]: enriched recipe obj. 
    """
    recipe_obj = json.load(open("data/recipe_repn.json","r"))
    for i in recipe_obj['recipe-ids']:
        if(i in recipeIDs):
            temp = recipe_obj['recipe-ids'][i]
            for idx,val in enumerate(temp['ingredients']):
                if(val['name'] in allergyObj.keys()):
                    temp['ingredients'][idx]['allergies']['id'] = allergyObj[val['name']][0]['Id']
                    temp['ingredients'][idx]['allergies']['category'] = allergyObj[val['name']][0]['Category']
                    temp['ingredients'][idx]['allergies']['ref'] = allergyObj[val['name']][0]['Reference']
                
            recipe_obj['recipe-ids'][i] = temp
    f = open("data/recipe_repn.json", "w")
    json.dump(recipe_obj, f)
    f.close()
    
    return recipe_obj