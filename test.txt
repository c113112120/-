import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 1️⃣ 影像載入與前處理
# ============================================================

image_path = "coin_img.jpg"  # ← 改成你的圖片名稱
image = cv2.imread(image_path)

if image is None:
    print(f"❌ 錯誤：無法讀取圖片 {image_path}")
    exit()

output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 2)

# ============================================================
# 2️⃣ 使用霍夫圓變換偵測硬幣
# ============================================================

circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=60,
    param1=100,
    param2=30,
    minRadius=25,
    maxRadius=90
)

# 初始化統計資料
coin_counts = {'1元': 0, '5元': 0, '10元': 0, '50元': 0}
total_value = 0

# ============================================================
# 3️⃣ 依直徑分類幣別 + 顏色區分
# ============================================================

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    all_coin_data = []

    print("--- 偵測到的硬幣原始數據 ---")

    for (x, y, r) in circles:
        diameter = r * 2
        all_coin_data.append({'x': x, 'y': y, 'r': r, 'd': diameter, 'denomination': '未知'})

    # 排序方便觀察
    all_coin_data.sort(key=lambda c: c['d'])
    diameters = [coin['d'] for coin in all_coin_data]
    print(f"所有直徑: {diameters}")

    # 🎨 每種硬幣使用不同顏色
    color_map = {
        '1元': (255, 0, 0),    # 藍色
        '5元': (0, 255, 255),  # 黃色
        '10元': (255, 0, 255), # 紫色
        '50元': (0, 255, 0)    # 綠色
    }

    # 🎯 根據直徑分類幣別
    for coin in all_coin_data:
        d = coin['d']

        if d < 130:
            coin['denomination'] = '1元'
            coin_counts['1元'] += 1
            total_value += 1
        elif d < 140:
            coin['denomination'] = '5元'
            coin_counts['5元'] += 1
            total_value += 5
        elif d < 165:
            coin['denomination'] = '10元'
            coin_counts['10元'] += 1
            total_value += 10
        else:
            coin['denomination'] = '50元'
            coin_counts['50元'] += 1
            total_value += 50

    print("--- 分類結束 ---")

    # ============================================================
    # 4️⃣ 畫出不同顏色的圓框 + 中文文字
    # ============================================================

    font_path = "C:/Windows/Fonts/msjh.ttc"  # 微軟正黑體
    try:
        font_large = ImageFont.truetype(font_path, 40)
        font_medium = ImageFont.truetype(font_path, 35)
        font_small = ImageFont.truetype(font_path, 30)
    except IOError:
        print(f"⚠️ 找不到字型檔案 {font_path}，改用預設字型。")
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # 先用 OpenCV 畫彩色圓框
    for coin in all_coin_data:
        x, y, r = coin['x'], coin['y'], coin['r']
        denom = coin['denomination']
        color = color_map[denom]
        cv2.circle(output, (x, y), r, color, 4)  # 不同幣別不同顏色

    # 再用 PIL 標文字
    output_pil = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(output_pil)

    for coin in all_coin_data:
        x, y, r = coin['x'], coin['y'], coin['r']
        denom = coin['denomination']
        text_w = font_small.getlength(denom)
        draw.text((x - text_w / 2, y - r - 25),
                  denom, fill=(255, 0, 0), font=font_small)

    # 顯示總數與金額
    draw.text((20, 20), f"總硬幣數: {len(all_coin_data)} 枚",
              fill=(0, 0, 255), font=font_large)
    draw.text((20, 70), f"總金額: ${total_value} 元",
              fill=(0, 0, 255), font=font_medium)

    # 顯示每種幣別數量
    y_offset = 120
    for denom, count in coin_counts.items():
        draw.text((20, y_offset), f"{denom}: {count} 枚",
                  fill=(0, 0, 255), font=font_small)
        y_offset += 35

    output = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)

else:
    print("❌ 沒有偵測到任何硬幣！")

# ============================================================
# 5️⃣ 顯示與儲存結果
# ============================================================

output_filename = "detected_coins_colored.jpg"
cv2.imwrite(output_filename, output)
print(f"✅ 辨識完成，結果已儲存為 '{output_filename}'")

cv2.imshow("Detected Coins - Colored", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
