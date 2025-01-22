import pandas as pd
import numpy as np
import random

num_rows = 100  # Number of test data rows to generate (now 100)
num_cols = 31  # Number of columns, adjust as needed

data = {}

# Add student_id
data['student_id'] = range(1, num_rows + 1)

# Add student_name (generating more diverse random names)
first_names = ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack', 'Kate', 'Liam', 'Mia', 'Noah', 'Olivia', 'Peter', 'Quinn', 'Ryan', 'Sophia', 'Thomas']
last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Wilson', 'Martinez', 'Anderson', 'Taylor', 'Thomas', 'Hernandez', 'Moore', 'Martin', 'Jackson', 'Thompson', 'White']

data['student_name'] = [random.choice(first_names) + " " + random.choice(last_names) for _ in range(num_rows)]

for i in range(num_cols):
    col_name = f"col_{i+1}"  # Generic names
    if i in [0, 5]:  # inv_sum_amount and inv_sum_discount can be higher
        data[col_name] = np.random.randint(500, 3000, size=num_rows)
    elif i in [14, 15]:  # as_min_result and as_max_result
        data[col_name] = np.random.randint(0, 20, size=num_rows)
    elif i in [12, 13]:  # as_min_finish_start and as_max_finish_start
        data[col_name] = np.random.randint(0, 30, size=num_rows)
    elif i in range(23, 31):  # su features
        data[col_name] = np.random.randint(1, 6, size=num_rows)
    else:
        data[col_name] = np.random.randint(0, 5, size=num_rows)

df = pd.DataFrame(data)

# Reorder columns to put student_id and student_name first
cols = ['student_id', 'student_name'] + [col for col in df.columns if col not in ['student_id', 'student_name']]
df = df[cols]

df.columns = ['student_id','student_name','inv_sum_amount','inv_count_payed','inv_count_start','inv_count_return','inv_max_full_paid','inv_sum_discount','inv_count_distinct_partner_campaign_id','inv_count_distinct_group_id','inv_max_is_b2b','inv_max_loyalty_applied','inv_count_distinct_promocode_id','as_count_distinct_course_id','as_min_finish_start','as_max_finish_start','as_min_result','as_max_result','as_count_partner_campaign_id','OL_count_distinct_dod_id','OL_count_distinct_source','OL_count_distinct_content_type_id','OL_count_distinct_object_id','OL_confirmed','vi_avg_M_1','su_min_speech','su_min_structure','su_max_structure','su_min_interaction','su_max_interaction','su_min_quality','su_max_quality','su_comments']

df.to_csv("random_test_data_100.csv", index=False)
print("random_test_data_100.csv created")