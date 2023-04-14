# K-means-
主要幫你快速數據處裡，並產生聚類的小工具

`py``
    import matplotlib.pyplot as plt
    import seaborn as sns; sns.set()
    import numpy as np
    import pandas as pd

    df = pd.read_csv("英文魔改版.csv")  ##讀出來有四個特徵值
    df = df.iloc[1:38,4:-1]  ##iloc [前面:]取出想要的行  [:後面]取出想要的列 


    #-----數據預處理-------
    # 找出Age和Cabin的空值
    null_grade1_grade2 = df[['grade1','grade2']].isnull()

    # 刪除含有空值的資料
    #df = df.dropna()

    # 將Age的空值填補成平均值
    df["grade1"] = df["grade1"].fillna(df['grade1'].mean())

    # df["Pclass"] = df["Pclass"].fillna("0")
    # df["Survived"] = df["Survived"].fillna("0")

    # 將若是有類別變項的空值填補成"Null"
    # df["grade2"] = df["grade2"].fillna("Null")

    print(df["grade1"])
    # 將預處理後的資料存入CSV檔案中
    df.to_csv('processed_data.csv', index=False)
    print(df[["grade1","grade2"]].isnull().sum()) #印出還有幾個是空值

    #-----開始聚類-----
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3)  #要分成三組資料
    km.fit(df)  ##()放入訓練資料

    km.labels_  ##上面的訓練會傳回labels_

    # 加入分群結果

    df["class"] = km.labels_  ##分類完之後 可以把分群結果當作目標值 才能做機器學習
    df = df.iloc[:, 3:19]


    ##------視覺化-------


    plt.figure(figsize=(2,2))   # 顯示圖框架大小

    plt.style.use("ggplot")     # 使用ggplot主題樣式
    plt.xlabel("Student's group", fontweight = "bold")                  #設定x座標標題及粗體
    plt.ylabel("Student's score", fontweight = "bold")   #設定y座標標題及粗體
    plt.title("K means scatter plot",
              fontsize = 15, fontweight = "bold")        #設定標題、字大小及粗體

    plt.scatter(df["class"],                    # x軸資料
                df["Grades"],     # y軸資料
                c = "g",                                  # 點顏色
                s = 25,                                   # 點大小
                alpha = 1,                               # 透明度
                marker = "D")                             # 點樣式

    plt.savefig("Scatter of Number of cars and Number of passengers(million).jpg")   #儲存圖檔
    #plt.close()      # 關閉圖表
```    
