# %%
#CV and ML Libraries
import cv2
from PIL import Image
from logo_blend import logo_create
import numpy as np
import matplotlib.pyplot as plt

import os
import csv


inputICs = './component-annotations'
inputLogos = './ocr-annotations'

'''
def logo_create(icVertices,logoVertices,fileName):
    print("Inside logo_create")
    print("Full File Name:"+"./pcb_images/"+fileName+".png")
    print("IC List: "+str(icVertices))
    print("Logo List: "+str(logoVertices))
'''

def extractVertices(vertices):
    vertices = vertices.replace('[','')
    vertices = vertices.replace(']','')

    v = []
    #(0,8) because some components are annotated with a Polygon rather than a Rectanlgle. We dont want those polygons for now.
    for k in range(0,8):
        v.append(vertices.split(',')[k])
    return v

def isLogoInIC(vertIC,vertLogo):
    vertICX = [int(vertIC[0].strip()),int(vertIC[2].strip()),int(vertIC[4].strip()),int(vertIC[6].strip())]
    vertICY = [int(vertIC[1].strip()),int(vertIC[3].strip()),int(vertIC[5].strip()),int(vertIC[7].strip())]

    vertICX.sort()
    vertICY.sort()
    vertLogoX = [int(vertLogo[0].strip()),int(vertLogo[2].strip()),int(vertLogo[4].strip()),int(vertLogo[6].strip())]
    vertLogoY = [int(vertLogo[1].strip()),int(vertLogo[3].strip()),int(vertLogo[5].strip()),int(vertLogo[7].strip())]
    vertLogoX.sort()
    vertLogoY.sort()

    return vertICX[0] <= vertLogoX[0] and vertICY[2] >= vertLogoY[2] and vertICX[2] >= vertLogoX[2] and vertICY[0] <= vertLogoY[0]



def identifyOverlaps():

    #Initiate an Audit File
    auditFile = open("audit.txt", "w")
    auditFile.close()

    for icCSV in os.listdir(inputICs):
        #This Block executes once per Component File (Component File is the File with IC Annotations).
        #icfile = open(inputICs+"\\"+icCSV,mode='r')
        with open(inputICs+"/"+icCSV,mode='r') as icfile:
            #1
            icReader = csv.DictReader(icfile, delimiter=',')

            icList = []
            logoList = []

            fileName = icCSV.split("_")[0]
            print("**********************************************************")
            print("Filename: "+str(fileName))

            auditFile = open("audit.txt", "a")
            auditFile.writelines("**********************************************************\n")
            auditFile.writelines("File Name: "+str(fileName)+"\n")
            auditFile.writelines("\n")
            auditFile.close()

            for logoCSV in os.listdir(inputLogos):
                #This Block executes once per OCR File.
                if logoCSV.split("_")[0] == fileName:
                    with open(inputLogos+"/"+logoCSV,mode='r') as logofile:
                        #logofile = open(inputLogos+"\\"+logoCSV,mode='r')
                        #3
                        #This Block executes once per Component File.
                        logoReader = csv.DictReader(logofile, delimiter=',')


                        for row in icReader:
                            if row["Class"]=="IC":
                                #print("Row: "+str(row))
                                entry = {"Instance ID: ":row["Instance ID"],"Source":row["Source Image Filename"],"Vertices":row["Vertices"]}
                                icList.append(entry)

                        for row in logoReader:
                            if not row["Logo"] == "":
                                entry =  {"Instance ID: ":row["Instance ID"],"Source":row["Source Image Filename"],"Vertices":row["Vertices"]}
                                logoList.append(entry)
                        #4
            #2


        ics = []
        logos = []

        for ic in icList:

            for logo in logoList:
                #print("IC vert: "+ic.get('Vertices'))
                #print("Logo vert: "+logo.get('Vertices'))

                vertIC =  extractVertices(ic.get('Vertices'))
                vertLogo =  extractVertices(logo.get('Vertices'))

                #isIt = isLogoInIC(vertIC,vertLogo)
                #print("Is Logo in IC: "+str(isIt))

                if isLogoInIC(vertIC,vertLogo):
                    #print("Vert IC: "+str(ic))
                    #print("Vert Logo: "+str(logo))
                    ics.append(vertIC)
                    logos.append(vertLogo)

        if len(ics) == 0 and len(logos):
            print("No ICs and Logos Overlap: "+ic.get("Source"))
        else:
            auditFile = open("audit.txt", "a")
            auditFile.writelines("ICS:")
            auditFile.writelines("\n")
            line = ics
            auditFile.writelines(str(line))
            auditFile.writelines("\n")
            auditFile.writelines("Logos")
            auditFile.writelines("\n")
            line = logos
            auditFile.writelines(str(line))
            auditFile.writelines("\n")
            auditFile.close()


            logo_create(ics,logos,fileName)
            return # Debug

if __name__ == '__main__':
    identifyOverlaps()

# %%
