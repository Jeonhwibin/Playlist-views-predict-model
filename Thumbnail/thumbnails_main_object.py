import os
import pandas as pd
from PIL import Image
import openai
import base64
import io
from tqdm import tqdm

openai.api_key = "api-key" 

image_folder = './images'
csv_folder = './images_csv'
os.makedirs(image_folder, exist_ok=True)
os.makedirs(csv_folder, exist_ok=True)

# 이미지 파일 리스트
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# 1. part CSV에서 처리된 파일명 집합 만들기
part_csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder)
                  if f.startswith('images_val_part_') and f.endswith('.csv')]
processed_files = set()
for csv_file in part_csv_files:
    try:
        df = pd.read_csv(csv_file)
        processed_files.update(df['filename'].astype(str).tolist())
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

# 2. 처리 안 된 파일만 추리기 (filename[:11] 기준)
unprocessed_files = [f for f in image_files if f[:11] not in processed_files]

print(f"전체 이미지 수: {len(image_files)}")
print(f"이미 처리된 이미지 수: {len(processed_files)}")
print(f"처리되지 않은 이미지 수: {len(unprocessed_files)}")

def resize_image_for_api(image_path, max_size=(640, 640), quality=10):
    img = Image.open(image_path)
    img.thumbnail(max_size)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    return buffer.read()

prompt = (
    "아래 기준에 따라 이미지를 스타일별로 분류해줘."
    "1. 사람: 이미지에 사람이 시각적으로 보이면 [1]"
    "2. 동물: 동물의 머리, 몸, 다리 등 신체가 탐지되면 [2]"
    "3. 애니메이션: 만화, 캐릭터, 일러스트, 애니메이션 스타일이면 [3]"
    "4. 풍경: 나무, 건물, 인테리어, 산, 호수, 강, 바다 등 배경 위주면 [4]"
    "- 한 이미지에 여러 클래스를 동시에 포함할 수 있음. 예: 사람이 산에 있으면 [1, 4]"
    "- 애매하거나 분류가 어려우면 [0]으로 반환 "
    "**반드시 파이썬 리스트 형태로 숫자만 반환해줘.**"
    "예시: [1], [2,4], [0]"
    "다른 설명이나 단어는 절대 포함하지 마."
)

def call_openapi(image_path, prompt):
    image_bytes = resize_image_for_api(image_path)
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        max_tokens=10
    )
    result = response.choices[0].message.content.strip()
    try:
        import ast
        cls_list = ast.literal_eval(result)
        if not isinstance(cls_list, list):
            cls_list = [0]
    except:
        cls_list = [0]
    return {"class_list": cls_list}

rows = []
part_num = len(part_csv_files) + 1  # 이어서 part 넘버링

for idx, filename in enumerate(tqdm(unprocessed_files, desc="이미지 분류 진행", unit="장")):
    img_path = os.path.join(image_folder, filename)
    response = call_openapi(img_path, prompt)
    cls_list = response["class_list"]

    row = {
        'video_id': filename[:11],
        'person': int(1 in cls_list),
        'animal': int(2 in cls_list),
        'anime': int(3 in cls_list),
        'landscape': int(4 in cls_list),
        'nonetype': int(cls_list == [0])
    }
    rows.append(row)

    if len(rows) == 100:
        part_csv_path = os.path.join(csv_folder, f'images_val_part_{part_num}.csv')
        pd.DataFrame(rows).to_csv(part_csv_path, index=False)
        print(f'중간 저장: {part_csv_path}')
        rows = []
        part_num += 1

if rows:
    part_csv_path = os.path.join(csv_folder, f'images_val_part_{part_num}.csv')
    pd.DataFrame(rows).to_csv(part_csv_path, index=False)
    print(f'남은 데이터 저장: {part_csv_path}')

print('이미지별 분류 및 분할 저장 완료')

# (선택) 모든 part CSV 병합
csv_files = [os.path.join(csv_folder, f) for f in os.listdir(csv_folder)
             if f.startswith('images_val_part_') and f.endswith('.csv')]
if csv_files:
    dfs = [pd.read_csv(f) for f in csv_files]
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.drop_duplicates(subset='video_id', keep='first')
    
    final_csv_path = os.path.join(csv_folder, 'thumbnails_objects.csv')
    final_df.to_csv(final_csv_path, index=False)
    print(f'최종 병합 CSV 저장: {final_csv_path}')
else:
    print('병합할 part CSV가 없습니다.')

print('전체 코드 실행 완료')
