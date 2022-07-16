import pandas as pd
import numpy as np
import os
from pprint import pprint


def load_data(data_path, cache_dir):
    """
    加载保存缓存数据集（减少内存开销）
    :param data_path: 数据的路径
    :param cache_dir: 缓存文件的路径
    :return: 用户-评分矩阵
    """
    cache_path = os.path.join(cache_dir, "ratings_matrix_cache")

    print("开始加载数据集：")
    if os.path.exists(cache_path):
        print("加载缓存中")
        ratings_matrix = pd.read_pickle(cache_path)
        print("缓存数据加载完成")
    else:
        print("加载新数据中")
        # 设置要加载的数据的类型
        dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
        # 加载数据，由于数据集分为四列，最后一列是时间相关的数据，所以这里只要前三列
        ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))
        # 透视表格，将movieId转换为列，行为userId
        ratings_matrix = ratings.pivot_table(index=["userId"], columns=["movieId"], values="rating")
        # 存入缓存数据中
        ratings_matrix.to_pickle(cache_path)
        print("数据集加载完成")
    return ratings_matrix


def compute_pearson_similarity(ratings_matrix, based="user"):
    """
    计算皮尔逊相关系数
    :param ratings_matrix:用户-物品评分矩阵
    :param based: "user" or "item"
    :return: 相似度矩阵
    """
    user_similarity_cache_path = os.path.join(".", "user_similarity_cache")
    item_similarity_cache_path = os.path.join(".", "item_similarity_cache")
    # 基于皮尔逊相关系数计算相似度
    # 用户相似度
    if based == "user":
        if os.path.exists(user_similarity_cache_path):
            print("加载缓存的用户相似度矩阵")
            similarity = pd.read_pickle(user_similarity_cache_path)
        else:
            print("开始计算用户相似度矩阵")
            # 相似度方法调用是以列为计算对象
            similarity = ratings_matrix.T.corr()
            similarity.to_pickle(user_similarity_cache_path)
    elif based == "item":
        if os.path.exists(item_similarity_cache_path):
            print("加载缓存的物品相似度矩阵")
            similarity = pd.read_pickle(item_similarity_cache_path)
        else:
            print("开始计算物品相似度矩阵")
            # 相似度方法调用是以列为计算对象
            similarity = ratings_matrix.corr()
            similarity.to_pickle(item_similarity_cache_path)
    else:
        raise Exception(f"Unhandled {based} Value")
    print(f"成功得到{based}相似度矩阵")
    return similarity


def predict(uid, iid, ratings_matrix, user_similarity):
    """
    预测指定用户对指定的物品的评分
    :param uid: 指定用户的id
    :param iid: 指定物品的id
    :param ratings_matrix: 评分矩阵
    :param user_similarity: 用户相似度矩阵
    :return: 预测的评分值
    """
    print(f"开始预测用户{int(uid)}对电影{int(iid)}的评分......")
    # 首先找出用户的相似用户(去掉自身和nan)
    similar_users = user_similarity[uid].drop([uid]).dropna()
    # 相似用户的筛选：只留下相似度大于0（正相关）的用户
    similar_users = similar_users.where(similar_users > 0).dropna()
    if similar_users.empty is True:
        raise Exception(f"用户{int(uid)}没有相似的用户")

    # 从相似用户中找出具有对iid进行评分记录的用户
    ids = set(ratings_matrix[iid].dropna().index) & set(similar_users.index)
    finally_similar_user = similar_users.loc[np.array(list(ids), dtype=np.int64).tolist()]

    # 结合uid用户和其相似用户预测对iid的评分
    # 评分计算的分子和分母
    sum_up = 0
    sum_down = 0
    for sim_uid, similarity in finally_similar_user.items():
        # 相似用户的评分数据
        sim_user_rated_movies = ratings_matrix.loc[sim_uid].dropna()
        # 相似用户对iid的评分
        sum_user_rating_for_item = sim_user_rated_movies[iid]
        # 计算公式中分子的值
        sum_up += similarity * sum_user_rating_for_item
        # 计算公式中分母的值
        sum_down += similarity

    # 计算预测评分并返回
    predict_rating = sum_up / sum_down
    print(f"预计用户{int(uid)}对电影{int(iid)}的评分为{predict_rating:.3f}")
    return predict_rating


def _predict_all(uid, item_ids, ratings_matrix, user_similarity):
    """
    预测单个用户全部的评分
    :param uid:
    :param ratings_matrix:
    :param user_similar:
    :return:
    """
    # 一个一个预测
    for iid in item_ids:
        try:
            rating = predict(uid, iid, ratings_matrix, user_similarity)
        # 报出异常，可以捕获除与程序退出sys.exit()相关之外的所有异常
        except Exception as e:
            print(e)
        else:
            yield uid, iid, rating


def predict_all(uid, ratings_matrix, user_similarity, filter_rule=None):
    """
    预测单个用户对所有电影的评分，同时基于某种规则进行过滤
    :param uid:
    :param ratings_matrix:
    :param user_similarity:
    :param filter_rule: 过滤规则
    :return:
    """
    if filter_rule is None:
        item_ids = ratings_matrix.columns
    elif isinstance(filter_rule, str) and filter_rule == "rated":
        # 过滤已经评分的电影
        # 获取用户对所有电影的评分记录
        user_ratings = ratings_matrix.loc[uid]
        # 由于该数据集的评分范围是1-5，所以不在这个范围的都是没有评分的
        idx = user_ratings < 6
        item_ids = idx.where(idx == False).dropna().index
    elif isinstance(filter_rule, str) and filter_rule == "unpopular":
        # 过滤掉冷门电影
        count = ratings_matrix.count()
        item_ids = count.where(count > 10).dropna().index
    elif isinstance(filter_rule, list) and set(filter_rule) == {"unpopular", "rated"}:
        # 过滤冷门和已经评分的电影
        count = ratings_matrix.count()
        ids1 = count.where(count > 10).dropna().index
        user_ratings = ratings_matrix.loc[uid]
        idx = user_ratings < 6
        ids2 = idx.where(idx == False).dropna().index
        # 取两者的交集
        item_ids = set(ids1) & set(ids2)
    else:
        raise Exception("无效的过滤方式")

    # yield from打开双向通道，把最外层的调用方与最内层的子生成器连接起来，这样二者可以直接发送和产出值，还可以直接传入异常，
    # 而不用在位于中间的协程中添加大量处理异常的样板代码
    yield from _predict_all(uid, item_ids, ratings_matrix, user_similarity)


def top_k_rs_result(k, uid, ratings_matrix, user_similarity):
    results = predict_all(uid, ratings_matrix, user_similarity, filter_rule=["rated", "unpopular"])
    # results是一个三元的迭代器，第一列是userId，第二列是itemId，第三列是评分
    return sorted(results, key=lambda x: x[2], reverse=True)[:k]


if __name__ == "__main__":
    ratings_matrix = load_data("../datasets/ml-latest-small/ratings.csv", ".")
    user_similarity = compute_pearson_similarity(ratings_matrix)
    item_similarity = compute_pearson_similarity(ratings_matrix, "item")
    # predict(1, 3, ratings_matrix, user_similarity)

    # for score in predict_all(1, ratings_matrix, user_similarity, filter_rule=["rated", "unpopular"]):
    #     print(score)

    scores = top_k_rs_result(10, 3, ratings_matrix, user_similarity)
    # pprint是一个美化打印的方法
    pprint(scores)
