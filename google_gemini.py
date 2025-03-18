import google.generativeai as genai
import pandas as pd

# 讀取電影數據
movie_data_path = "dataMovie.csv"  # 主要電影資料
movie_code_path = "movie_code.csv"  # 類型分類檔案

df = pd.read_csv(movie_data_path)
movie_code_df = pd.read_csv(movie_code_path)

# **移除前後空格，確保匹配正確**
df["movie_title"] = df["movie_title"].str.strip()
movie_code_df["movie_title"] = movie_code_df["movie_title"].str.strip()

# 設定 Gemini API Key
API_KEY = "AIzaSyA62VmqNk_CMhpdEdQAcvifoqfvqZcUAA0"
genai.configure(api_key=API_KEY)

def chat_with_gemini():
    """ 啟動與 Gemini 的持續對話模式 """
    print("🔹 歡迎使用電影推薦對話系統，輸入 'exit' 可離開 🔹")
    while True:
        movie_name = input("請輸入電影名稱：")
        if movie_name.lower() == "exit":
            print("👋 感謝使用，再見！")
            break
        print(recommend_movies_with_exact_genre(movie_name))

def recommend_movies_with_exact_genre(movie_name, top_n=5):
    """ 先透過類型篩選電影，再用 Gemini 進行文本分析推薦相似的電影 """

    # **步驟 1：找出用戶輸入的電影類型（精確匹配）**
    movie_row = movie_code_df[movie_code_df["movie_title"] == movie_name]
    if movie_row.empty:
        return f"❌ 抱歉，找不到電影「{movie_name}」的類型資訊。"

    # 取得該電影的類型（值為 1 的欄位）
    selected_genres = movie_row.iloc[:, 1:].iloc[0]  # 取得該電影的類型標籤
    matching_genres = selected_genres[selected_genres == 1].index.tolist()  # 找出類型名稱
    
    if not matching_genres:
        return f"❌ 電影「{movie_name}」沒有分類類型，無法進行推薦。"

    # **步驟 2：篩選出完全相同類型的電影**
    genre_filter = (movie_code_df[matching_genres] == 1).all(axis=1) & (movie_code_df.iloc[:, 1:].sum(axis=1) == len(matching_genres))
    filtered_movies = movie_code_df[genre_filter]

    if filtered_movies.empty:
        return f"❌ 沒有找到與「{movie_name}」類型完全相同的電影。"

    # **步驟 3：取得這些電影的劇情描述**
    filtered_movie_titles = filtered_movies["movie_title"].tolist()
    filtered_movie_data = df[df["movie_title"].isin(filtered_movie_titles)][["movie_title", "movie_genre", "movie_description"]]

    if filtered_movie_data.empty:
        return f"❌ 找到了與「{movie_name}」類型完全相同的電影，但無法獲取描述。"

    # **步驟 4：讓 Gemini 進行文本分析**
    movie_info = df[df["movie_title"] == movie_name].iloc[0]  # 確保完全匹配

    context = f"""用戶輸入的電影：{movie_info['movie_title']}
類型：{', '.join(matching_genres)}
描述：{movie_info['movie_description']}

以下是與此電影**完全相同分類**的電影數據，請**只從這些電影**中進行推薦：
"""
    for _, row in filtered_movie_data.iterrows():
        context += f"電影名稱：{row['movie_title']}，類型：{row['movie_genre']}，描述：{row['movie_description']}\n"
    
    context += f"\n請分析電影「{movie_name}」的內容描述，並**只推薦來自上述列表的 {top_n} 部電影**，同時解釋推薦原因。\n"
    
    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(context, stream=True)
    
    for chunk in response:
        if chunk and getattr(chunk, "text", None):  # 確保 chunk 存在且 chunk.text 存在
            text = chunk.text.strip()
            if text:
                print(text, end="", flush=True)
    
    print("\n")  # 確保最後換行





# **啟動對話模式**
chat_with_gemini()


