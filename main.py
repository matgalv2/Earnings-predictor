import datetime

from pandas import *
from typing import *

filename = "employments.csv"
dataset = read_csv(f"./resources/{filename}", sep=';')

dataset = dataset.drop(columns='data_source')
dataset: DataFrame = dataset.drop(columns='education')


def getDomainPerColumn(dataFrame: DataFrame):
    return {column_name: set(dataFrame[column_name]) for column_name in domains}


all_columns = ["date_of_employment", "age", 'sex', 'country', "experience_years_it",
               'languages', 'speciality', 'core_programming_language', 'academic_title',
               'education', 'education_towards_it', "rate_per_hour", "salary_monthly",
               'company_country', 'company_type', 'work_form', "team_size", 'team_type',
               'form_of_employment', 'full_time', 'paid_days_off', 'insurance', 'training_sessions',
               'data_source']

domains = [
    'sex', 'country', 'experience_years_it',
    'languages', 'speciality', 'core_programming_language', 'academic_title',
    'education_towards_it', 'company_country', 'company_type', 'work_form', "team_size", 'team_type',
    'form_of_employment', 'full_time', 'paid_days_off', 'insurance', 'training_sessions'
]

enum_columns = ['sex', 'country', 'languages', 'speciality', 'core_programming_language', 'academic_title',
                'company_country', 'company_type', 'work_form', 'team_type', 'form_of_employment']

boolean_columns = ['education_towards_it', 'full_time','paid_days_off', 'insurance', 'training_sessions']

correct_columns = [
    'date_of_employment', 'age', 'experience_years_it', 'rate_per_hour', 'salary_monthly', 'team_size',
    'is_sex_F', 'is_sex_M', 'is_country_Spain', 'is_country_France', 'is_country_Poland',
    'is_country_Greece', 'is_country_Sweden', 'is_country_Germany', 'is_country_United Kingdom',
    'is_country_Russia', 'is_country_Italy', 'is_languages_Polish,English', 'is_languages_Italian',
    'is_languages_Spanish,English', 'is_languages_Swedish', 'is_languages_Russian',
    'is_languages_Spanish', 'is_languages_German,English', 'is_languages_English',
    'is_languages_Greek', 'is_languages_Russian,English', 'is_languages_English,English',
    'is_languages_Italian,English', 'is_languages_Swedish,English', 'is_languages_French',
    'is_languages_French,English', 'is_languages_German', 'is_languages_Greek,English',
    'is_languages_Polish', 'is_speciality_Tech lead', 'is_speciality_DB Administrator',
    'is_speciality_Data quality manager', 'is_speciality_IT Security specialist',
    'is_speciality_Cloud system engineer', 'is_speciality_Data scientist', 'is_speciality_Backend',
    'is_speciality_Computer scientist', 'is_speciality_Applications engineer',
    'is_speciality_Frontend', 'is_speciality_Software Engineer', 'is_speciality_Other',
    'is_speciality_Systems analyst', 'is_speciality_Web administrator', 'is_core_programming_language_Kotlin',
    'is_core_programming_language_Swift', 'is_core_programming_language_Cobol', 'is_core_programming_language_Go',
    'is_core_programming_language_Objective-C', 'is_core_programming_language_JavaScript',
    'is_core_programming_language_R', 'is_core_programming_language_Ruby', 'is_core_programming_language_PHP',
    'is_core_programming_language_Python', 'is_core_programming_language_Java', 'is_core_programming_language_Other',
    'is_academic_title_Master', 'is_academic_title_Bachelor', 'is_academic_title_Licence',
    'is_academic_title_Doctorate', 'is_academic_title_No degree', 'education_towards_it', 'is_company_country_Spain',
    'is_company_country_France', 'is_company_country_Poland', 'is_company_country_Greece', 'is_company_country_Sweden',
    'is_company_country_Germany', 'is_company_country_United Kingdom', 'is_company_country_Russia',
    'is_company_country_Italy', 'is_company_type_Software house', 'is_company_type_Big tech',
    'is_company_type_Startup', 'is_company_type_Company', 'is_company_type_Public institution',
    'is_company_type_Other', 'is_company_type_Corporation', 'is_work_form_hybrid', 'is_work_form_stationary',
    'is_work_form_remote', 'is_team_type_local', 'is_team_type_international', 'is_form_of_employment_contractor',
    'is_form_of_employment_employee', 'full_time', 'paid_days_off','insurance', 'training_sessions'
]
domainsPerColumn = getDomainPerColumn(dataset)

enumsDomains = {enum: domainsPerColumn[enum] for enum in enum_columns}


def mapEnumsInRowForModel(row: Series, domainPerEnumAttribute: Dict = enumsDomains) -> Dict:
    attributes = domainPerEnumAttribute.keys()
    row = row.to_dict()
    for attribute in attributes:
        domain = domainPerEnumAttribute[attribute]
        row_value = row[attribute]
        del row[attribute]
        row.update({f"is_{attribute}_{value}": 1 if row_value == value else 0 for value in domain})
    return row

def changeStructure(row: Series):
    row = row.to_dict()
    print(list(row.keys()))
    attributes = ['full_time','paid_days_off','insurance','training_sessions']
    for attribute in attributes:
        value = row[f"is_{attribute}_True"]
        del row[f"is_{attribute}_True"]
        del row[f"is_{attribute}_False"]
        row[attribute] = value
    print(list(row.keys()))
    return row

def mapNumericValuesInRowForModel(row) -> Dict:
    attribute_date_of_employment = "date_of_employment"
    attribute_age = "age"
    attribute_experienceYearsIt = "experience_years_it"
    attribute_teamSize = "team_size"
    # row = row.to_dict()
    # date_of_employment
    row[attribute_date_of_employment] = datetime.datetime.strptime(row[attribute_date_of_employment], "%Y-%m-%d").date().toordinal()/ datetime.date.max.toordinal()
    # age
    row[attribute_age] = row[attribute_age] / 100
    # experience_years _it
    row[attribute_experienceYearsIt] = row[attribute_experienceYearsIt] / 100
    # team_size
    row[attribute_teamSize] = row[attribute_teamSize] / 100
    return row

def mapBooleanValuesInRowForModel(row) -> Dict:
    for column in boolean_columns:
        row[column] = int(row[column])
    return row



def prepareDataForML(data: DataFrame):
    dataset_ML = DataFrame(columns=correct_columns)
    for i in range(len(data.values)):
        print(f"{i}/{len(data.values)}")
        processed_row = mapBooleanValuesInRowForModel(mapNumericValuesInRowForModel(mapEnumsInRowForModel(data.loc[i])))
        dataset_ML.loc[i] = [processed_row[key] for key in processed_row.keys()]
    dataset_ML.to_csv(path_or_buf="./resources/employments_ML.csv", sep="\t", index=False)


# prepareDataForML(dataset)

# print(domainsPerColumn.keys())
#
# for key in domainsPerColumn.keys():
#     print(key, domainsPerColumn[key])

# processed = read_csv("resources/employments_ML.csv", sep='\t')

# not_full_rows = 0
# data_len = len(processed.values)
# for i in range(data_len):
#     print(f"{i}/{data_len}")
#     sum = 0
#     for column in processed.columns:
#         if processed.loc[i][column] == "":
#             sum += 1
#     if sum != 0:
#         not_full_rows += 1
#
# print(not_full_rows, "<- that many rows")


# with open("resources/results/attributes.txt", mode='a') as file:
#     rate_per_hour_min = min(dataset["rate_per_hour"].to_list())
#     rate_per_hour_max = max(dataset["rate_per_hour"].to_list())
#     print(rate_per_hour_min, rate_per_hour_max)
#     file.write(f"rate_per_hour\t[{rate_per_hour_min}-{rate_per_hour_max}]\tdecimal\n")
#
#     salary_monthly_min = min(dataset["salary_monthly"].to_list())
#     salary_monthly_max = max(dataset["salary_monthly"].to_list())
#     print(salary_monthly_min, salary_monthly_max)
#
#     file.write(f"salary_monthly\t[{salary_monthly_min}-{salary_monthly_max}]\tdecimal(null)\n")
#
#     file.write("attribute\tdomain\ttype\n")
#     domainPerAttribute = getDomainPerColumn(read_csv("resources/employments.csv", sep=';'))
#     for key in domainPerAttribute.keys():
#         file.write(f"{key}\t{[*domainPerAttribute[key]]}\t\n")


