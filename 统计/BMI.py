def calculate_bmi(weight, height):
    # weight in kilograms
    # height in meters
    bmi = weight / (height ** 2)
    return bmi

def print_bmi_result(bmi):
    if bmi < 18.5:
        print("您的BMI为：{:.2f}，属于偏瘦范围".format(bmi))
    elif 18.5 <= bmi < 24:
        print("您的BMI为：{:.2f}，属于正常范围".format(bmi))
    elif 24 <= bmi < 28:
        print("您的BMI为：{:.2f}，属于过重范围".format(bmi))
    elif 28 <= bmi < 30:
        print("您的BMI为：{:.2f}，属于轻度肥胖范围".format(bmi))
    else:
        print("您的BMI为：{:.2f}，属于中度及以上肥胖范围".format(bmi))

# 输入体重（千克）和身高（米）
weight = float(input("请输入您的体重（千克）："))
height = float(input("请输入您的身高（米）："))

bmi_value = calculate_bmi(weight, height)
print_bmi_result(bmi_value)