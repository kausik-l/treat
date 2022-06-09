from re import L
from django.shortcuts import render
import requests
import sys
import subprocess
from subprocess import run,PIPE
from django.core.files.storage import FileSystemStorage
import json
import random
# import querysys
# from querysys import size
from pathlib import Path

obj_list = []

def button(request):
      return render(request,'home.html')

def output(request):
      data=requests.get("https://www.google.com/")
      print(data.text)
      data=data.text
      return render(request,'home.html',{'data':data})

def get_allergen(recp_id):
    recipe_obj = json.load(open("data/recipe_repn.json","r"))
    list_allergen = list()
    for rid in recp_id:
        temp = recipe_obj['recipe-ids'][rid]
        str_val = ""
        for idx,val in enumerate(temp['ingredients']):
            if(len(str_val)==0):
                try:
                    str_val = str_val+val['allergies']['category'][0]
                except IndexError:
                    continue
            else:
                try:
                    str_val = str_val+", "+val['allergies']['category'][0]
                except IndexError:
                    continue
        list_allergen.append(str_val)
    return list_allergen

# def obj(request):
#       data=requests.get("https://www.google.com/")
#       print(data.text)
#       data=data.text
#       return render(request,'obj.html',{'data':data})

def external(request):
      inp= request.POST.get('param')
      image=request.FILES.get('image',False)
      print("image is ",image)
      fs=FileSystemStorage()
      
      if image != False:
          filename=fs.save(image.name,image)
          fileurl=fs.open(filename)
          templateurl=fs.url(filename)
          print("file raw url",filename)
          print("file full url", fileurl)
          print("template url",templateurl)
          import os
          filedir = Path(os.path.dirname(__file__)).parent.absolute()
          filepath = os.path.join(str(filedir), 'querysys.py')
        #   print("Path is:   ", filepath)
        #   out= subprocess.Popen([sys.executable,'/Users/kausiklakkaraju/Documents/researchPhd/projects/work/MMReasoningDemo-main/icaps-demo2/querysys.py',str(fileurl)],shell=False,stdout=subprocess.PIPE)
          out= subprocess.Popen([sys.executable,filepath,str(fileurl)],shell=False,stdout=subprocess.PIPE)
          text = out.communicate()[0].decode('utf-8').split("\r\n")
          a = []
          names = []
          images = []
      #     print(text)
          for each in text:
                a.append(each[3:].split(" "))

          for each in a[0][0::3]:
                names.append(each)      
          print(names)

                
          for each in a[0][1::3]:
                images.append(each.replace(str(filedir),""))

          tup = []
          number = a[0][-1]
          ids = []
          
          f = open('data/recipe_repn.json')
          data = json.load(f)
          for match in names:
              for id in data['recipe-ids']:
                  var = data['recipe-ids'][id]
                  if match == var['recipe_name']:
                      obj_list.append(json.dumps(var, indent=8))
                      ids.append(id) 
          f.close()
          
          alls = get_allergen(ids)
          
          res = []                  
          [res.append(x) for x in obj_list if x not in res]
          for (name,image,obj,all) in zip(names,images,res,alls):
                tup.append((name,image,obj,all))


          return render(request,'home.html',{'UploadedImage':templateurl,'mainData':a,'recipes': tup,'length':number})

       
      else:
          import os
          filedir = Path(os.path.dirname(__file__)).parent.absolute()
          filepath = os.path.join(str(filedir), 'querysys.py')
        #   print("Path is:   ", filepath)
          out= subprocess.Popen([sys.executable,filepath,inp],shell=False,stdout=subprocess.PIPE)
        #   out= subprocess.Popen([sys.executable,'/Users/kausiklakkaraju/Documents/researchPhd/projects/work/MMReasoningDemo-main/icaps-demo2/querysys.py',inp],stdout=subprocess.PIPE)
          text = out.communicate()[0].decode('utf-8').split("\r\n")
          print(text)
          
          a = []
          names = []
          images = []
          for each in text:
                a.append(each[3:].split(" "))
                
          for each in a[0]:
                if each == '':
                      a[0].remove(each)
      
          for each in a[0][1::3]:
                images.append(each.replace(str(filedir),""))
                
          for each in a[0][0::3]:
                names.append(each)      
                
          tup = []
          number = a[0][-1]
          print(number)
          ids = []
          


          f = open('data/recipe_repn.json')
          data = json.load(f)
          for match in names:
              for id in data['recipe-ids']:
                  var = data['recipe-ids'][id]
                  if match == var['recipe_name']:
                      obj_list.append(json.dumps(var, indent=8))
                      ids.append(id) 
          f.close()
          
          alls = get_allergen(ids)
                
                
                
          res = []                  
          [res.append(x) for x in obj_list if x not in res]
      #     print(matched_names)
          
          for (name,image,obj,alls) in zip(names,images,res,alls):
                tup.append((name,image,obj,alls))

          return render(request, 'home.html',{'text':inp,'mainData':a[0],'recipes':tup, 'length':number})
    
      

def obj(request):
      
      return render(request, 'obj.html', {'obj':obj_list} )