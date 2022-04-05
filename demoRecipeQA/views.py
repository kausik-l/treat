from django.shortcuts import render
import requests
import sys
import subprocess
from subprocess import run,PIPE
from django.core.files.storage import FileSystemStorage

def button(request):
      return render(request,'home.html')

def output(request):
      data=requests.get("https://www.google.com/")
      print(data.text)
      data=data.text
      return render(request,'home.html',{'data':data})

def external(request):
      inp= request.POST.get('param')
      image=request.FILES.get('image',False)
      print("image is ",image)
      fs=FileSystemStorage()
      
      if image!= False :
          filename=fs.save(image.name,image)
          fileurl=fs.open(filename)
          templateurl=fs.url(filename)
          print("file raw url",filename)
          print("file full url", fileurl)
          print("template url",templateurl)
          out= subprocess.Popen([sys.executable,'/Users/kausiklakkaraju/Documents/researchPhd/projects/work/MMReasoningDemo-main/demo/demoRecipeQA/querysys.py',str(fileurl)],shell=False,stdout=subprocess.PIPE)
          text = out.communicate()[0].decode('utf-8').split("\r\n")
          a = []
          names = []
          images = []
          for each in text:
                a.append(each[3:].split(" "))
                
          for each in a[0]:
                if each!=['']:
                  a[0].remove(each)
                  
          for each in a[0][1::3]:
                names.append(each)      

                
          for each in a[0][0::3]:
                images.append(each.replace("/Users/kausiklakkaraju/Documents/researchPhd/projects/work/MMReasoningDemo-main/demo/demoRecipeQA",""))

          tup = []
          for (name,image) in zip(names,images):
                tup.append((name,image))
          number = len(names)
                
          #return render(request, 'home.html',{'mainData':a[0],'recipes':tup}) 
          return render(request,'home.html',{'UploadedImage':templateurl,'mainData':a,'recipes': tup, 'length':number})

       
      else:
          out= subprocess.Popen([sys.executable,'/Users/kausiklakkaraju/Documents/researchPhd/projects/work/MMReasoningDemo-main/demo/demoRecipeQA/querysys.py',inp],stdout=subprocess.PIPE)
          text = out.communicate()[0].decode('utf-8').split("\r\n")
          a = []
          names = []
          images = []
          for each in text:
                a.append(each[3:].split(" "))
                
          for each in a[0]:
                if each == '':
                      a[0].remove(each)
      
          for each in a[0][1::3]:
                images.append(each.replace("/Users/kausiklakkaraju/Documents/researchPhd/projects/work/MMReasoningDemo-main/demo/demoRecipeQA",""))
                
          for each in a[0][0::3]:
                names.append(each)      
                
          tup = []
          number = len(names)
          for (name,image) in zip(names,images):
                tup.append((name,image))
  
          return render(request, 'home.html',{'text':inp,'mainData':a[0],'recipes':tup, 'length':number})
