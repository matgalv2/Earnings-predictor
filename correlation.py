import numpy as np
from pandas import read_csv


def correlation(p, q):
    return (np.mean(np.array(p) * np.array(q)) - (np.mean(p) * np.mean(q)))/(np.std(p) * np.std(q))



attributes = [
    'date_of_employment', 'age', 'experience_years_it',
    # 'rate_per_hour', 'salary_monthly',
    'team_size',
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

data = read_csv("./resources/employments_ML2.csv", delimiter='\t').drop(columns="salary_monthly")


with open("./resources/results/correlation.txt", 'a') as file:
    file.write("dependent_variable\tindependent_variable\tvalue\n")
    dependent_variable = "rate_per_hour"
    correlation_values = {}
    i = 0
    dependent_variable_column = [*map(lambda x:float(x), data[dependent_variable].to_list())]
    for attribute in attributes:
        print(f"{i}/{len(attributes)}")
        i += 1
        R = np.corrcoef(dependent_variable_column, [*map(lambda x:float(x), data[attribute].to_list())])
        # R = correlation(dependent_variable_column, [*map(lambda x:float(x), data[attribute].to_list())])
        correlation_values[attribute] = R
        file.write(f"{dependent_variable}\t{attribute}\t{round(R, 5)}\n")



