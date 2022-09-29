from data_loader.data_loader import CsvDataLoader


def test_profile_target():

    data_loader = CsvDataLoader(data_path='/Users/pwang/BenWork/Dataset/hospital/aggregate_data.csv')
    target_profile = data_loader.get_columns_class_profile('SEPSIS')
    data_loader.get_class_weight('SEPSIS')
    pos_weight = data_loader.get_pos_weight("SEPSIS", "True", "False")
    print(pos_weight)
    # print(target_profile)

    assert True
