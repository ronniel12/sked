
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from tabulate import tabulate

# =============================================
# Penalty and Reward Weights
# =============================================
PENALTY_WEIGHTS = {
    'supervisor': 1000,
    'consecutive': 1000,
    'preference': 800,
    'hours': 200,
    'fairness': 300,
    'two_consecutive_off': 900,
    'one_shift': 900
}

CONSECUTIVE_OFF_REWARD = 300
CONSECUTIVE_OFF_PENALTY = PENALTY_WEIGHTS['two_consecutive_off']

# =============================================
# Data Configuration
# =============================================
employees = [
    {'name': 'Pete', 'position': 'supervisor', 'skills': ['window'], 'min': 112, 'max': 120},
    {'name': 'Massa', 'position': 'supervisor', 'skills': ['window', 'hot', 'cold'], 'min': 112, 'max': 120},
    {'name': 'Mikko', 'position': 'supervisor', 'skills': ['window', 'hot'], 'min': 112, 'max': 120},
    {'name': 'Jimi', 'position': 'cook', 'skills': ['window', 'hot', 'cold'], 'min': 90, 'max': 112},
    {'name': 'Niel', 'position': 'cook', 'skills': ['window', 'hot', 'cold'], 'min': 90, 'max': 112},
    {'name': 'Jed', 'position': 'cook', 'skills': ['window', 'hot'], 'min': 112, 'max': 120},
    {'name': 'Juhi', 'position': 'cook', 'skills': ['cold'], 'min': 90, 'max': 112},
]

day_off_requests = {
    'Mikko': ['2025-03-17']
}

shift_preferences = {
    'Mikko': [('2025-03-27', 'Evening')]
}

closed_days = []

Vacation = {
    'Juhi': ['2025-03-17','2025-03-18','2025-03-19','2025-03-20','2025-03-21']
}

# =============================================
# Schedule Configuration
# =============================================
start_date = datetime(2025, 3, 17)
schedule_duration = 21
all_dates = [start_date + timedelta(days=i) for i in range(schedule_duration)]
active_dates = [d for d in all_dates if d.strftime('%Y-%m-%d') not in closed_days]
num_full_weeks = len(all_dates) // 7

# =============================================
# Model Configuration
# =============================================
model = cp_model.CpModel()
assignments = {}
penalties = []
employee_double_shifts = {e['name']: {} for e in employees}  # Track double shifts

# Create Assignment Variables
for e in employees:
    emp_name = e['name']
    for date in active_dates:
        date_str = date.strftime('%Y-%m-%d')
        day_type = 'short' if date.strftime('%A') in ['Monday', 'Tuesday', 'Sunday'] else 'long'
        for shift in ['Morning', 'Evening']:
            duration = 6 if shift == 'Morning' else (5 if day_type == 'short' else 6)
            for task in ['cold', 'hot', 'window']:
                if task in e['skills']:
                    var = model.NewBoolVar(f"{emp_name}_{date_str}_{shift}_{task}")
                    assignments[(emp_name, date_str, shift, task)] = (var, duration)

# Hard Constraints
# 1. Task Coverage
for date in active_dates:
    date_str = date.strftime('%Y-%m-%d')
    for shift in ['Morning', 'Evening']:
        for task in ['cold', 'hot', 'window']:
            qualified = [var for (emp, d, s, t), (var, _) in assignments.items()
                        if d == date_str and s == shift and t == task]
            if qualified:
                model.AddExactlyOne(qualified)

# 2. No Double Assignments
for e in employees:
    emp_name = e['name']
    for date in active_dates:
        date_str = date.strftime('%Y-%m-%d')
        for shift in ['Morning', 'Evening']:
            tasks = [var for (emp, d, s, t), (var, _) in assignments.items()
                    if emp == emp_name and d == date_str and s == shift]
            if tasks:
                model.AddAtMostOne(tasks)

# 3. Closed Days, Day-Off Requests, and Vacations
for (emp, date_str, shift, task), (var, _) in assignments.items():
    if (date_str in closed_days
        or date_str in day_off_requests.get(emp, [])
        or date_str in Vacation.get(emp, [])):
        model.Add(var == 0)

# 4. Working-Day Indicators
employee_working_vars = {}
for e in employees:
    emp_name = e['name']
    employee_working_vars[emp_name] = {}
    for date in active_dates:
        date_str = date.strftime('%Y-%m-%d')
        working = model.NewBoolVar(f'work_{emp_name}_{date_str}')
        shifts = [var for (emp, d, s, t), (var, _) in assignments.items()
                 if emp == emp_name and d == date_str]
        model.Add(sum(shifts) >= 1).OnlyEnforceIf(working)
        model.Add(sum(shifts) == 0).OnlyEnforceIf(working.Not())
        employee_working_vars[emp_name][date] = working

# 5. Minimum 2 Off Days per Week
for e in employees:
    emp_name = e['name']
    for week in range(num_full_weeks):
        week_dates = all_dates[week*7 : (week+1)*7]
        off_terms = []
        for date in week_dates:
            date_str = date.strftime('%Y-%m-%d')
            if (date_str in closed_days
                or date_str in day_off_requests.get(emp_name, [])
                or date_str in Vacation.get(emp_name, [])):
                off_terms.append(1)
            elif date in active_dates:
                off_terms.append(1 - employee_working_vars[emp_name][date])
        model.Add(sum(off_terms) >= 2)

# New Hard Constraint: Max 2 Consecutive Double Shifts
# --------------------------------------------------
# Track double shifts during variable creation
employee_shift_work_vars = {}
for e in employees:
    emp_name = e['name']
    for date in active_dates:
        date_str = date.strftime('%Y-%m-%d')
        for shift in ['Morning', 'Evening']:
            tasks = [var for (emp, d, s, t), (var, _) in assignments.items()
                    if emp == emp_name and d == date_str and s == shift]
            if tasks:
                work_shift = model.NewBoolVar(f'work_{emp_name}_{date_str}_{shift}')
                model.Add(sum(tasks) >= 1).OnlyEnforceIf(work_shift)
                model.Add(sum(tasks) == 0).OnlyEnforceIf(work_shift.Not())
                employee_shift_work_vars[(emp_name, date_str, shift)] = work_shift

 # Create double shift tracking
morning_var = employee_shift_work_vars.get((emp_name, date_str, 'Morning'))
evening_var = employee_shift_work_vars.get((emp_name, date_str, 'Evening'))
if morning_var is not None and evening_var is not None:
    double_shift = model.NewBoolVar(f'double_{emp_name}_{date_str}')
    # Corrected constraint implementation
    model.Add(morning_var + evening_var == 2).OnlyEnforceIf(double_shift)
    model.Add(double_shift == 0).OnlyEnforceIf(morning_var.Not())
    model.Add(double_shift == 0).OnlyEnforceIf(evening_var.Not())
    employee_double_shifts[emp_name][date_str] = double_shift

# Add consecutive double shift constraint
for emp_name in employee_double_shifts:
    dates = sorted(
        [datetime.strptime(d, '%Y-%m-%d') for d in employee_double_shifts[emp_name].keys()],
        key=lambda x: x
    )

    # Check consecutive triplets
    for i in range(len(dates) - 2):
        d1, d2, d3 = dates[i], dates[i+1], dates[i+2]
        if (d2 - d1).days == 1 and (d3 - d2).days == 1:
            ds1 = employee_double_shifts[emp_name][d1.strftime('%Y-%m-%d')]
            ds2 = employee_double_shifts[emp_name][d2.strftime('%Y-%m-%d')]
            ds3 = employee_double_shifts[emp_name][d3.strftime('%Y-%m-%d')]
            model.AddBoolOr([ds1.Not(), ds2.Not(), ds3.Not()])


# ---------------------------------------------
# Soft Constraints
# ---------------------------------------------

# 1. One Shift Per Day Penalty - FIXED VERSION
employee_shift_work_vars = {}
for e in employees:
    emp_name = e['name']
    for date in active_dates:
        date_str = date.strftime('%Y-%m-%d')
        for shift in ['Morning', 'Evening']:
            tasks = [
                var for (emp, d, s, t), (var, _) in assignments.items()
                if emp == emp_name and d == date_str and s == shift
            ]
            if tasks:  # Only create shift variables if tasks exist
                work_shift = model.NewBoolVar(f'work_{emp_name}_{date_str}_{shift}')
                model.Add(sum(tasks) >= 1).OnlyEnforceIf(work_shift)
                model.Add(sum(tasks) == 0).OnlyEnforceIf(work_shift.Not())
                employee_shift_work_vars[(emp_name, date_str, shift)] = work_shift

for e in employees:
    emp_name = e['name']
    for date in active_dates:
        date_str = date.strftime('%Y-%m-%d')
        # Check for existence in dictionary before using
        morning_var = employee_shift_work_vars.get((emp_name, date_str, 'Morning'))
        evening_var = employee_shift_work_vars.get((emp_name, date_str, 'Evening'))

        # FIX: Check if variables exist rather than evaluating them
        if morning_var is not None and evening_var is not None:
            shift_sum = model.NewIntVar(0, 2, f'shift_sum_{emp_name}_{date_str}')
            model.Add(shift_sum == morning_var + evening_var)
            double_shift = model.NewBoolVar(f'double_shift_{emp_name}_{date_str}')
            model.Add(shift_sum == 2).OnlyEnforceIf(double_shift)
            model.Add(shift_sum != 2).OnlyEnforceIf(double_shift.Not())
            penalties.append(double_shift * PENALTY_WEIGHTS['one_shift'])

# 2. Supervisor Preference
supervisor_penalties = []
for date in active_dates:
    date_str = date.strftime('%Y-%m-%d')
    for shift in ['Morning', 'Evening']:
        supervisor_vars = [
            var for (emp, d, s, t), (var, _) in assignments.items()
            if d == date_str and s == shift and
            any(e['position'] == 'supervisor' and e['name'] == emp for e in employees)
        ]
        if supervisor_vars:
            has_supervisor = model.NewBoolVar(f'super_{date_str}_{shift}')
            model.AddBoolOr(supervisor_vars).OnlyEnforceIf(has_supervisor)
            model.AddBoolAnd([var.Not() for var in supervisor_vars]).OnlyEnforceIf(has_supervisor.Not())
            supervisor_penalties.append(has_supervisor.Not())
penalties.extend([p * PENALTY_WEIGHTS['supervisor'] for p in supervisor_penalties])

# 3. Shift Preferences
for emp, prefs in shift_preferences.items():
    for date_str, shift in prefs:
        pref_vars = [
            var for (e, d, s, t), (var, _) in assignments.items()
            if e == emp and d == date_str and s == shift
        ]
        if pref_vars:
            satisfied = model.NewBoolVar(f'pref_{emp}_{date_str}_{shift}')
            model.Add(sum(pref_vars) >= 1).OnlyEnforceIf(satisfied)
            model.Add(sum(pref_vars) == 0).OnlyEnforceIf(satisfied.Not())
            penalties.append(satisfied.Not() * PENALTY_WEIGHTS['preference'])

# 4. Hour Constraints with Vacation Adjustment
for e in employees:
    emp_name = e['name']
    vacation_days = len(Vacation.get(emp_name, []))
    vacation_hours = 8 * vacation_days
    adjusted_min = max(0, e['min'] - vacation_hours)

    total_expr = sum(
        var * dur for (emp, d, s, t), (var, dur) in assignments.items()
        if emp == emp_name
    )
    under = model.NewIntVar(0, 1000, f'under_{emp_name}')
    over = model.NewIntVar(0, 1000, f'over_{emp_name}')
    model.Add(under >= adjusted_min - total_expr)
    model.Add(over >= total_expr - e['max'])
    penalties.append(under * PENALTY_WEIGHTS['hours'])
    penalties.append(over * PENALTY_WEIGHTS['hours'])

# 5. Fairness Penalty
avg_hours = sum((e['min'] + e['max']) // 2 for e in employees) // len(employees)
abs_diffs = []
for e in employees:
    total_expr = sum(
        var * dur for (emp, d, s, t), (var, dur) in assignments.items()
        if emp == e['name']
    )
    diff = model.NewIntVar(-1000, 1000, f'diff_{e["name"]}')
    abs_diff = model.NewIntVar(0, 1000, f'abs_diff_{e["name"]}')
    model.Add(diff == total_expr - avg_hours)
    model.AddAbsEquality(abs_diff, diff)
    abs_diffs.append(abs_diff)
fairness_pen = model.NewIntVar(0, 1000, 'fairness')
model.Add(fairness_pen == sum(abs_diffs))
penalties.append(fairness_pen * PENALTY_WEIGHTS['fairness'])

# 6. Consecutive Off Days
def to_intvar(x, name):
    if isinstance(x, int):
        return model.NewConstant(x)
    else:
        return x

for e in employees:
    emp_name = e['name']
    for week in range(num_full_weeks):
        week_dates = all_dates[week*7 : (week+1)*7]
        off_indicators = []
        for date in week_dates:
            date_str = date.strftime('%Y-%m-%d')
            if (date_str in closed_days
                or date_str in day_off_requests.get(emp_name, [])
                or date_str in Vacation.get(emp_name, [])):
                off_indicators.append(1)
            else:
                if date in active_dates:
                    off_var = model.NewIntVar(0, 1, f'off_{emp_name}_{date_str}')
                    model.Add(off_var == 1 - employee_working_vars[emp_name][date])
                    off_indicators.append(off_var)
                else:
                    off_indicators.append(1)
        pair_vars = []
        for i in range(len(off_indicators) - 1):
            pair_off = model.NewBoolVar(f'pair_off_{emp_name}_week{week}_{i}')
            off_i = to_intvar(off_indicators[i], f'const_off_{emp_name}_week{week}_{i}')
            off_j = to_intvar(off_indicators[i+1], f'const_off_{emp_name}_week{week}_{i+1}')
            model.Add(pair_off <= off_i)
            model.Add(pair_off <= off_j)
            model.Add(pair_off >= off_i + off_j - 1)
            pair_vars.append(pair_off)
        has_pair = model.NewBoolVar(f'has_pair_{emp_name}_week{week}')
        if pair_vars:
            model.AddMaxEquality(has_pair, pair_vars)
        else:
            model.Add(has_pair == 0)
        reward_penalty = model.NewIntVar(-CONSECUTIVE_OFF_REWARD, CONSECUTIVE_OFF_PENALTY,
                                       f'reward_consec_off_{emp_name}_week{week}')
        model.Add(reward_penalty == -CONSECUTIVE_OFF_REWARD).OnlyEnforceIf(has_pair)
        model.Add(reward_penalty == CONSECUTIVE_OFF_PENALTY).OnlyEnforceIf(has_pair.Not())
        penalties.append(reward_penalty)

# 7. Consecutive Working Days
for e in employees:
    emp_name = e['name']
    working_days = [employee_working_vars[emp_name][d] for d in sorted(employee_working_vars[emp_name].keys())]
    for i in range(len(working_days) - 5):
        window_sum = sum(working_days[i:i+6])
        penalty_var = model.NewIntVar(0, 6, f'consec_penalty_{emp_name}_{i}')
        model.Add(penalty_var >= window_sum - 5)
        model.Add(penalty_var >= 0)
        penalties.append(penalty_var * PENALTY_WEIGHTS['consecutive'])






# Solve and Output
# =============================================
model.Minimize(sum(penalties))
solver = cp_model.CpSolver()
solver.parameters.max_time_in_seconds = 60
solver.parameters.num_search_workers = 8
status = solver.Solve(model)

if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
    schedule_data = {e['name']: {} for e in employees}
    total_hours = {e['name']: 0 for e in employees}

    for (emp, date_str, shift, task), (var, dur) in assignments.items():
        if solver.Value(var):
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            day_of_week = date_obj.strftime('%A')
            time_range = '10:00-16:00' if shift == 'Morning' else \
                        '16:00-21:00' if day_of_week in ['Sunday', 'Monday', 'Tuesday'] else '16:00-22:00'
            entry = f"{time_range} {task}"
            schedule_data[emp].setdefault(date_str, []).append(entry)
            total_hours[emp] += dur

    headers = ['Employee', 'Total Hours'] + [d.strftime('%Y-%m-%d') for d in all_dates]
    table = []
    for e in employees:
        row = [e['name'], total_hours[e['name']]]
        for date in all_dates:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in closed_days:
                row.append('CLOSED')
            elif date_str in day_off_requests.get(e['name'], []):
                row.append('OFF (Requested)')
            elif date_str in Vacation.get(e['name'], []):
                row.append('VACATION')
            else:
                daily_assignments = schedule_data[e['name']].get(date_str, ['OFF'])
                row.append('\n'.join(daily_assignments))
        table.append(row)

    manhours = [
        sum(
            solver.Value(var) * dur
            for (emp, d_str, s, t), (var, dur) in assignments.items()
            if d_str == date.strftime('%Y-%m-%d') and date.strftime('%Y-%m-%d') not in closed_days
        )
        for date in all_dates
    ]
    table.append(['TOTAL HOURS', sum(manhours)] + manhours)
    print(tabulate(table, headers=headers, tablefmt='grid'))
else:
    print("No feasible solution found. Status:", solver.StatusName(status))