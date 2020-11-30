#CV and ML Libraries
import cv2
from PIL import Image
import numpy as np

import os
import csv

'''
Requirements
1) It should go through all available PCB Annotated Images (49-63 or more if you have it ready)
2) For each PCB identify if Logo annotation overlaps with IC annotation
3) For all overlapping Logo and IC annotations, it should call a function called 'logo_create' with 2 lists as arguments- First list will have vertices of IC, second list will have vertices of Logo.
side note: you don't have to write the function, i have both the algorithms ready, just create a function with a placeholder print function that will say something
'''

inputICs = "./component-annotations"
inputLogos = "./ocr-annotations"

def logo_create(icVertices,logoVertices):
    print("Inside logo_create")
    print("IC List: "+str(icVertices))
    print("Logo List: "+str(logoVertices))

def extractVertices(vertices):
    vertices = vertices.replace('[','')
    vertices = vertices.replace(']','')

    v = []
    for k in range(0,8):
        v.append(vertices.split(',')[k])
    return v

def isLogoInIC(vertIC,vertLogo):
    '''
    x1, y1: 0,1
    x2, y2: 4,5
    '''
    return vertIC[0] <= vertLogo[0] and vertIC[1] <= vertLogo[1] and vertIC[4] >= vertLogo[4] and vertIC[5] >= vertLogo[5]


def identifyOverlaps():

    for icCSV in os.listdir(inputICs):
        #This Block executes once per Component File.

        with open(inputICs+"/"+icCSV,mode='r') as icfile:
            icReader = csv.DictReader(icfile, delimiter=',')


            icList = []
            logoList = []

            fileName = icCSV.split("_")[0]
            print("**********************************************************")
            print("Filename: "+str(fileName))

            for logoCSV in os.listdir(inputLogos):
                #This Block executes once per OCR File.
                if logoCSV.split("_")[0] == fileName:
                    with open(inputLogos+"/"+logoCSV,mode='r') as logofile:
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



            #print("IC List: "+str(icList))
            #print("Logo List: "+str(logoList))

            #This Block executes once per Component File.


            for ic in icList:
                ics = []
                logos = []
                for logo in logoList:
                    #print("IC vert: "+ic.get('Vertices'))
                    #print("Logo vert: "+logo.get('Vertices'))

                    vertIC =  extractVertices(ic.get('Vertices'))
                    vertLogo =  extractVertices(logo.get('Vertices'))

                    if isLogoInIC(vertIC,vertLogo):
                        #print("Vert IC: "+str(ic))
                        #print("Vert Logo: "+str(logo))
                        ics.append(vertIC)
                        logos.append(vertLogo)

            if len(ics) == 0 and len(logos):
                print("No ICs and Logos Overlap: "+ic.get("Source"))
            else:
                logo_create(ics,logos)

if __name__ == '__main__':
    identifyOverlaps()
