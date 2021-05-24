# class宣言
import numpy as np
from random import sample
import open3d as o3d
import copy


# debug-check done
class RegistrationResult:
    def __init__(self, transformation=np.identity(4)):
        self.inlier_rmse = 0.0
        self.fitness = 0.0
        self.transformation = transformation
        self.correspondence_set = []

    def IsBetterRANSACThan(self, other):
        return (self.fitness > other.fitness) or (self.fitness == other.fitness and self.inlier_rmse < other.inlier_rmse)


# def宣言
# debug-check done
def EvaluateRANSACBasedOnCorrespondence(source, target, corres, max_correspondence_distance, transformation):
    result = RegistrationResult(transformation)
    error2 = 0.0
    good = 0
    max_dis2 = max_correspondence_distance * max_correspondence_distance

    for c in corres:
        dis2 = np.linalg.norm(source.points[c[0]] - target.points[c[1]])**2
        if dis2 < max_dis2:
            good = good + 1
            error2 = error2 + dis2
            result.correspondence_set.append(c)

    if good == 0:
        result.fitness = 0.0
        result.inlier_rmse = 0.0
    else:
        result.fitness = good / len(corres)
        result.inlier_rmse = np.sqrt(error2 / good)

    return result


# class 宣言
class RANSACConvergenceCriteria:
    def __init__(self, max_iteration=100000, confidence=0.999):
        self.max_iteration = max_iteration
        self.confidence = confidence


def RegistrationRANSACBasedOnCorrespondence(
        source_p,
        target_p,
        corres,
        max_correspondence_distance,
        ransac_n,
        criteria):

    if ransac_n < 3 or len(corres) < ransac_n or max_correspondence_distance <= 0.0:
        raise Exception(
            "RegistrationRANSACBasedOnCorrespondence's args were not entered correctly.")

    best_result = RegistrationResult()
    exit_itr = -1

    # ransac_corres = CorrespondenceSet() # ここは元のcppのコードを模倣しない
    best_result_local = RegistrationResult()
    exit_itr_local = criteria.max_iteration

    for itr in range(criteria.max_iteration):
        if itr < exit_itr_local:

            ransac_corres = corres[sample(range(len(corres)), k=ransac_n)]
            ransac_corres = o3d.utility.Vector2iVector(ransac_corres)

            transformation = o3d.pipelines.registration.TransformationEstimationPointToPoint().compute_transformation(
                source_p,
                target_p,
                ransac_corres)

            check = True
            check = o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9).Check(
                source_p,
                target_p,
                ransac_corres,
                transformation
            )

            check = o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal(np.pi / 6).Check(
                source_p,
                target_p,
                ransac_corres,
                transformation
            )

            # 仮に上のcheckersの判定がだめだったら, 次の処理をスキップ
            if not check:
                continue

            pcd = copy.deepcopy(source_p)
            pcd.transform(transformation)
            # print(transformation)

            result = EvaluateRANSACBasedOnCorrespondence(
                pcd, target_p, corres, max_correspondence_distance, transformation)

            if result.IsBetterRANSACThan(best_result_local):
                best_result_local = result

                # update exit condition if necessary
                exit_itr_d = np.log(1.0 - criteria.confidence) / \
                    np.log(1.0 - result.fitness ** ransac_n)
                exit_itr_local = int(
                    np.ceil(exit_itr_d)) if exit_itr_d < criteria.max_iteration else exit_itr_local

            if best_result_local.IsBetterRANSACThan(best_result):
                best_result = best_result_local

            if exit_itr_local > exit_itr:
                exit_itr = exit_itr_local

    # print(exit_itr, best_result_local.fitness, best_result_local.inlier_rmse) # <- debug
    return best_result
