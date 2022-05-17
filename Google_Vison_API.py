from glob import glob
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pickle
import openpyxl
import numpy as np
 
# input data
# # 왼쪽: input data에 들어갈 오브젝트/라벨 이름, 오른쪽: input data에 들어갈 순서(1부터 시작)
input_data_object = {
  'Person': 1,
  'Animal': 2
}
 
input_data_label = {
  #1. 자연
  'Plant': 3,
  'Flower': 4,
  'Insect': 5,
  'Arthropod': 6,
  'Pollinator': 7,
  'Nature': 8,
  'Tree': 9,
  'Natural': 10,
  'Landscape': 11,
  'Water': 12,
  'Mountain': 13,
  'Wildlife': 14,
  'Twig': 15,  # (나무-작은가지)
  'Organism': 16,  # (유기체)
  'Terrestrial plant': 17,  # (육상 식물)
  'Natural landscape': 18,
  'Grass': 19,
  'forest': 20,
  'jungle': 21,
  # 2. 사람 - 오브젝트로만 판별
  # 3. 음식
  'Food': 22,
  'Recipe': 23,
  'Ingredient': 24,
  'Cuisine': 25,
  'Tableware': 26,
  # 4. 건물
  'Building': 27,
  'Skyscraper': 28,
  'Tower block': 29,
  'Tower': 30,
  'Urban design': 31,
  'Dusk': 32,
  'Afterglow': 33,
  # 5. 동물 - animals 딕셔너리에서 매칭되는 동물에 대한 score를 아래 'Animal' 키에 저장
  'Animal': 34,
  'Felidae': 35,
  'Carnivore': 36,
  'Terrestrial animal': 37,    # (육상동물)
  'Beak': 38, # 부리
  'Feather': 39
}
 
animals = {
  'Cheetah': 26,
  'Cat': 27,
  'Dog': 28,
  'Fish': 29,
  'Tiger': 30,
  'Lion': 31,
  'Rabbit': 32,
  'Monkey': 33,
  'Bear': 34,
  'Cow': 35,
  'Horse': 36,
  'Giraffe': 37,
  'Snake': 38,
  'Animal': 39,
  'Deer': 40,
  'Zebra': 41,
  'Elephant': 42,
  'Sheep': 43,
  'Mouse': 44,
  'Penguin': 45,
  'Fox': 46,
  'Pig': 47,
  'Koala': 48,
  'Panda': 49,
  'Camel': 50,
  'Wolf': 51,
  'Gorilla': 52,
  'Racoon': 53,
  'Goose': 54,
  'Squirrel': 55,
  'Alligator': 56,
  'Crocodile': 57,
  'Seal': 58,
  'Duck': 59,
  'Bird': 60
}
 
label_weight = [0 for i in range(len(input_data_object) + len(input_data_label))]


def image_info(path):
    #label_weight = [0 for i in range(len(input_data_object) + len(input_data_label))]

    return detect_labels(path)

    # 1차원 배열을 2차원 배열로 확장
    # input_data = np.array(label_weight)
    # input_data = input_data[np.newaxis, :]
    # print(f"업로드한 이미지의 input_data: {input_data}")

    # return input_data


def detect_labels(path):
  label_weight = [0 for i in range(len(input_data_object) + len(input_data_label))]
  """Detects labels in the file."""
  from google.cloud import vision
  import io
  import os
 
  # Set environment variable
  os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "사용자.json"
  client = vision.ImageAnnotatorClient()
 
  with io.open(path, 'rb') as image_file:
      content = image_file.read()
 
  image = vision.Image(content=content)
 
  # step 1. 오브젝트 추출 및 매칭
  # 중요) 오브젝트에 'Person' or 'Animal'이 매칭되면 step 2 건너뜀
  objects = client.object_localization(image=image).localized_object_annotations
 
  top_object_num = 0
  skip_label = False
 
  for object_ in objects:
      print(f"name; {object_.name}")
      if top_object_num == 2:
        break
 
      if object_.name == 'Person' or object_.name == 'Animal':
          skip_label = True
          label_weight[input_data_object[object_.name] - 1] = object_.score
          break
 
      top_object_num += 1
 
  # step 2. 라벨 추출 및 매칭
  if not skip_label:  # step 1에서 오브젝트 매칭 안됐을 경우에 라벨 추출 진행
      response = client.label_detection(image=image)
      labels = response.label_annotations
 
      for label in labels:
            #print(f"name; {object_.name}")
        print(label.description)

        if label.description in input_data_label:
            label_weight[input_data_label[label.description] - 1] = label.score #@
 
      # label과 동물 매칭
      animal_score = score_of_animal(labels)
      if animal_score != 0:
          label_weight[input_data_label['Animal'] - 1] = animal_score
 
      if response.error.message:
          raise Exception(
              '{}\nFor more info on error messages, check: '
              'https://cloud.google.com/apis/design/errors'.format(
                  response.error.message))

  # 1차원 배열을 2차원 배열로 확장
  input_data = np.array(label_weight)
  input_data = input_data[np.newaxis, :]
  # print(f"업로드한 이미지의 input_data: {input_data}")

  return input_data


# 매칭되는 동물 label이 여러개면 그들의 score 중 max값을 return
def score_of_animal(labels):
  animal_score = []
 
  for label in labels:
    if label.description in animals:
      animal_score.append(label.score)
 
  if len(animal_score) > 0:
    return max(animal_score)
 
  return 0


data = detect_labels('C:/Users/PC/Desktop/visionapi/ImageClassificationML/static/image/귀염강아지.jpg')
print(data)

