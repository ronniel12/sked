from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from ortools.sat.python import cp_model
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import math

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    index_path = Path(__file__).parent / "static" / "index.html"
    with open(index_path) as f:
        return HTMLResponse(content=f.read(), status_code=200)
    

class SolverRequest(BaseModel):
    employees: List[Dict[str, Any]]
    shifts: List[Dict[str, Any]]
    closed_days: List[str]
    start_date: str
    schedule_days: int

class ImprovedScheduleGenerator:
    def __init__(self, request: SolverRequest):
        self.request = request
        self.model = cp_model.CpModel()
        self.all_employees = []
        self.employee_details = {}
        self.shift_structure = {}
        self.task_durations = {}
        self.assignments = {}
        self.penalties = []
        self.total_hours_vars = {}
        self.employee_working_vars = {}
        self.employee_shift_work_vars = {}
        self.employee_double_shifts = {}

        # Constants for penalties (taken from colab.py)
        self.PENALTY_WEIGHTS = {
            'supervisor': 1000,
            'consecutive': 1000,
            'preference': 800,
            'hours': 200,
            'fairness': 300,
            'two_consecutive_off': 900,
            'one_shift': 900,
            'consecutive_double': 1000,  # Strong penalty for consecutive double shifts
        }
        self.CONSECUTIVE_OFF_REWARD = 300
        self.CONSECUTIVE_OFF_PENALTY = self.PENALTY_WEIGHTS['two_consecutive_off']

        try:
            self.start_date = datetime.strptime(request.start_date, "%Y-%m-%d").date()
            self.days = [self.start_date + timedelta(days=i) for i in range(request.schedule_days)]
            self.active_days = [day for day in self.days if day.isoformat() not in request.closed_days]
            self.num_full_weeks = len(self.days) // 7
            self._prepare_data_structures()
        except Exception as e:
            raise ValueError(f"Data validation failed: {str(e)}")

    def _prepare_data_structures(self):
        if not self.request.employees:
            raise ValueError("At least one employee required")
        if not self.request.shifts:
            raise ValueError("At least one shift required")

        # Process employees
        for emp in self.request.employees:
            self.all_employees.append(emp['name'])
            
            skills = emp.get('skills', [])
            if isinstance(skills, str):
                skills = [skills]  # Convert single skill string to list if needed
                
            available_days = [
                day.isoformat() for day in self.days 
                if day.isoformat() not in self.request.closed_days 
                and day.isoformat() not in emp.get('days_off', [])
            ]
            total_days = len(available_days)
            
            # Adjust contract hours based on available days
            min_hours = emp.get('contract', {}).get('minHours', 0)
            max_hours = emp.get('contract', {}).get('maxHours', 168)
            adjusted_min = max(0, math.floor(min_hours * total_days / self.request.schedule_days))
            adjusted_max = math.ceil(max_hours * total_days / self.request.schedule_days)

            # Add preferred shifts handling here
            preferred_shifts = emp.get('preferred_shifts_by_date', {})

            self.employee_details[emp['name']] = {
                'skills': set(skills),
                'position': emp.get('position', 'Employee').lower(),  # Normalize positions
                'contract': emp.get('contract', {}),
                'days_off': set(emp.get('days_off', [])),
                'adjusted_min': adjusted_min,
                'adjusted_max': adjusted_max,
                'available_days': available_days,
                'preferred_shifts_by_date': preferred_shifts  # Added line
            }

        # Process shifts and tasks
        for shift in self.request.shifts:
            shift_name = shift['name']
            tasks_info = []
            for task in shift.get('tasks', []):
                task_schedule = {}
                for day_sched in task.get('schedule', []):
                    try:
                        start = datetime.strptime(day_sched.get('startTime', '00:00'), "%H:%M").time()
                        end = datetime.strptime(day_sched.get('endTime', '00:00'), "%H:%M").time()
                        duration = (datetime.combine(datetime.min, end) - 
                                  datetime.combine(datetime.min, start)).seconds // 3600
                        task_schedule[day_sched['day']] = duration
                    except ValueError:
                        raise ValueError(f"Invalid time format in {shift_name} - {task.get('name', 'unnamed-task')}")

                skill_requirement = task.get('skillRequirement') or task.get('skill_requirement')
                if not skill_requirement:
                    raise ValueError(
                        f"Missing skill requirement in shift '{shift_name}' "
                        f"for task '{task.get('name', 'unnamed-task')}'"
                    )

                tasks_info.append({
                    'name': task['name'],
                    'skill_requirement': skill_requirement,
                    'schedule': task_schedule
                })
                
                # Map task durations
                for day in self.days:
                    day_str = day.isoformat()
                    weekday = day.strftime('%A')
                    if weekday in task_schedule:
                        key = (shift_name, task['name'], day_str)
                        self.task_durations[key] = task_schedule[weekday]

            self.shift_structure[shift_name] = {
                'tasks': tasks_info,
                'required_skills': {task['name']: task['skill_requirement'] for task in tasks_info}
            }

    def _create_assignment_variables(self):
        self.assignments.clear()
        
        # First create all possible assignments
        for emp in self.all_employees:
            emp_info = self.employee_details[emp]
            for day in self.days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in emp_info['days_off']:
                    continue
                    
                for shift_name, shift_data in self.shift_structure.items():
                    for task in shift_data['tasks']:
                        if task['skill_requirement'] in emp_info['skills']:
                            key = (emp, day_str, shift_name, task['name'])
                            var = self.model.NewBoolVar(f'assign_{emp}_{day_str}_{shift_name}_{task["name"]}')
                            duration = self.task_durations.get((shift_name, task['name'], day_str), 0)
                            self.assignments[key] = (var, duration)

        # Add strict preference constraints
        for emp in self.all_employees:
            emp_info = self.employee_details[emp]
            preferred_shifts = emp_info.get('preferred_shifts_by_date', {})
            
            for date_str, preferred_shift in preferred_shifts.items():
                if date_str not in emp_info['available_days']:
                    continue  # Skip preferences on unavailable days
                    
                # Get all assignment variables for this day
                all_day_assignments = [
                    var for (e, d, s, t), (var, _) in self.assignments.items()
                    if e == emp and d == date_str
                ]
                
                # Get preferred shift variables
                pref_shift_vars = [
                    var for (e, d, s, t), (var, _) in self.assignments.items()
                    if e == emp and d == date_str and s == preferred_shift
                ]
                
                # Get other shifts variables
                other_shift_vars = [
                    var for (e, d, s, t), (var, _) in self.assignments.items()
                    if e == emp and d == date_str and s != preferred_shift
                ]
                
                if pref_shift_vars:
                    # If preferred shift is possible, enforce strict constraints:
                    # 1. Must work at least one task in preferred shift
                    # 2. Cannot work any other shifts
                    self.model.Add(sum(pref_shift_vars) >= 1)
                    self.model.Add(sum(other_shift_vars) == 0)
                else:
                    # If preferred shift isn't possible (missing skills), force day off
                    self.model.Add(sum(all_day_assignments) == 0)
                    
    def _force_day_off(self, emp, date_str):
        """Force employee to have day off by setting all assignments to 0"""
        for key in list(self.assignments.keys()):
            if key[0] == emp and key[1] == date_str:
                var, _ = self.assignments[key]
                self.model.Add(var == 0)


    def _create_working_day_indicators(self):
        # Create working day indicators for each employee (similar to colab.py)
        for emp in self.all_employees:
            self.employee_working_vars[emp] = {}
            for day in self.days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                    continue
                    
                working = self.model.NewBoolVar(f'work_{emp}_{day_str}')
                shifts = [var for (e, d, s, t), (var, _) in self.assignments.items()
                         if e == emp and d == day_str]
                
                if shifts:  # Only add constraints if there are shifts available
                    self.model.Add(sum(shifts) >= 1).OnlyEnforceIf(working)
                    self.model.Add(sum(shifts) == 0).OnlyEnforceIf(working.Not())
                    self.employee_working_vars[emp][day_str] = working

    def _create_shift_indicators(self):
        # Create shift indicators for each employee and shift
        for emp in self.all_employees:
            for day in self.active_days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                    continue
                    
                for shift_name in self.shift_structure:
                    tasks = [var for (e, d, s, t), (var, _) in self.assignments.items()
                           if e == emp and d == day_str and s == shift_name]
                    if tasks:
                        work_shift = self.model.NewBoolVar(f'work_{emp}_{day_str}_{shift_name}')
                        self.model.Add(sum(tasks) >= 1).OnlyEnforceIf(work_shift)
                        self.model.Add(sum(tasks) == 0).OnlyEnforceIf(work_shift.Not())
                        self.employee_shift_work_vars[(emp, day_str, shift_name)] = work_shift

    def _create_double_shift_indicators(self):
    # Dynamically get the first two shifts as potential double shifts
        shift_names = list(self.shift_structure.keys())
        if len(shift_names) < 2:
            return  # Not enough shifts for double shifts

        first_shift, second_shift = shift_names[:2]

        # Track double shifts
        for emp in self.all_employees:
            self.employee_double_shifts[emp] = {}
            for day in self.active_days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                    continue
                
                first_var = self.employee_shift_work_vars.get((emp, day_str, first_shift))
                second_var = self.employee_shift_work_vars.get((emp, day_str, second_shift))
                
                if first_var is not None and second_var is not None:
                    double_shift = self.model.NewBoolVar(f'double_{emp}_{day_str}')
                    self.model.AddBoolAnd([first_var, second_var]).OnlyEnforceIf(double_shift)
                    self.model.AddBoolOr([first_var.Not(), second_var.Not()]).OnlyEnforceIf(double_shift.Not())
                    self.employee_double_shifts[emp][day_str] = double_shift

    def _add_task_coverage_constraint(self):
        # Ensure each task is assigned exactly one employee
        for day in self.days:
            day_str = day.isoformat()
            if day_str in self.request.closed_days:
                continue
                
            for shift_name, shift_data in self.shift_structure.items():
                for task in shift_data['tasks']:
                    task_name = task['name']
                    
                    qualified_vars = [
                        var for (emp, d, s, t), (var, _) in self.assignments.items()
                        if d == day_str and s == shift_name and t == task_name
                    ]
                    
                    if qualified_vars:
                        self.model.AddExactlyOne(qualified_vars)

    def _add_one_task_per_employee_constraint(self):
        # An employee can only be assigned one task per shift
        for emp in self.all_employees:
            for day in self.days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                    continue
                    
                for shift_name in self.shift_structure:
                    task_vars = [
                        var for (e, d, s, t), (var, _) in self.assignments.items()
                        if e == emp and d == day_str and s == shift_name
                    ]
                    if task_vars:
                        self.model.AddAtMostOne(task_vars)

    def _add_supervisor_constraint(self):
        # Ensure at least one supervisor per shift
        for day in self.active_days:
            day_str = day.isoformat()
            if day_str in self.request.closed_days:
                continue
                
            for shift_name in self.shift_structure:
                supervisor_vars = []
                for emp in self.all_employees:
                    # Check if employee is a supervisor
                    if self.employee_details[emp]['position'] == 'supervisor':
                        for task_name in self.shift_structure[shift_name]['required_skills']:
                            required_skill = self.shift_structure[shift_name]['required_skills'][task_name]
                            if required_skill in self.employee_details[emp]['skills']:
                                key = (emp, day_str, shift_name, task_name)
                                if key in self.assignments:
                                    supervisor_vars.append(self.assignments[key][0])
                
                if supervisor_vars:
                    # At least one supervisor must be assigned to this shift
                    self.model.AddBoolOr(supervisor_vars)
                else:
                    # If no supervisors are available, we shouldn't create an infeasible problem
                    # Instead, we'll make this a soft constraint with a high penalty
                    has_supervisor = self.model.NewBoolVar(f'has_supervisor_{day_str}_{shift_name}')
                    self.model.Add(has_supervisor == 0)  # Always false since no supervisors
                    self.penalties.append(has_supervisor.Not() * self.PENALTY_WEIGHTS['supervisor'])

    def _add_max_consecutive_days_constraint(self):
        # Maximum 5 consecutive working days
        for emp in self.all_employees:
            working_days = []
            for day in self.days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                    # For closed days or days off, add a constant 0 (not working)
                    working_days.append(0)
                else:
                    working_var = self.employee_working_vars.get(emp, {}).get(day_str)
                    if working_var is not None:
                        working_days.append(working_var)
                    else:
                        # If no working variable exists, add a constant 0
                        working_days.append(0)
            
            # Check consecutive working days
            for i in range(len(working_days) - 5):
                # Get the window of 6 consecutive days
                window = working_days[i:i+6]
                # Filter out constant values
                var_window = [var for var in window if not isinstance(var, int)]
                
                if len(var_window) > 5:  # Only add constraint if we have more than 5 variables
                    self.model.Add(sum(var_window) <= 5)
                    
    def _add_two_days_off_per_week_constraint(self):
        # At least 2 days off per week
        for emp in self.all_employees:
            for week in range(self.num_full_weeks):
                week_days = self.days[week*7:(week+1)*7]
                off_terms = []
                
                for day in week_days:
                    day_str = day.isoformat()
                    if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                        # Definitely off
                        off_terms.append(1)
                    else:
                        working_var = self.employee_working_vars.get(emp, {}).get(day_str)
                        if working_var is not None:
                            # 1 - working_var = off indicator
                            off_var = self.model.NewIntVar(0, 1, f'off_{emp}_{day_str}')
                            self.model.Add(off_var == 1 - working_var)
                            off_terms.append(off_var)
                        else:
                            # If no working variable exists, consider as day off
                            off_terms.append(1)
                
                # Add constraint only if meaningful
                if off_terms:
                    self.model.Add(sum(off_terms) >= 2)

    def _add_contract_hours_constraint(self):
        # Add contract hours constraint as soft constraint with penalties
        for emp in self.all_employees:
            emp_info = self.employee_details[emp]
            emp_vars = []
            emp_coeffs = []
            
            for (e, d, s, t), (var, dur) in self.assignments.items():
                if e == emp:
                    emp_vars.append(var)
                    emp_coeffs.append(dur)
            
            if emp_vars:
                total_hours = self.model.NewIntVar(0, 168*2, f'total_hours_{emp}')
                self.model.Add(total_hours == sum(var * coeff for var, coeff in zip(emp_vars, emp_coeffs)))
                self.total_hours_vars[emp] = total_hours
                
                # Under-hours penalty
                under_diff = self.model.NewIntVar(-168, 168, f'under_diff_{emp}')
                self.model.Add(under_diff == emp_info['adjusted_min'] - total_hours)
                under_hours = self.model.NewIntVar(0, 168, f'under_hours_{emp}')
                self.model.AddMaxEquality(under_hours, [under_diff, 0])
                
                # Over-hours penalty
                over_diff = self.model.NewIntVar(-168, 168, f'over_diff_{emp}')
                self.model.Add(over_diff == total_hours - emp_info['adjusted_max'])
                over_hours = self.model.NewIntVar(0, 168, f'over_hours_{emp}')
                self.model.AddMaxEquality(over_hours, [over_diff, 0])
                
                self.penalties.append(under_hours * self.PENALTY_WEIGHTS['hours'])
                self.penalties.append(over_hours * self.PENALTY_WEIGHTS['hours'])

    def _add_fairness_constraint(self):
        # Add fairness constraint to balance hours between employees
        if len(self.total_hours_vars) >= 2:  # Only add if we have at least 2 employees
            total_hours_list = list(self.total_hours_vars.values())
            max_h = self.model.NewIntVar(0, 168*2, 'max_hours')
            min_h = self.model.NewIntVar(0, 168*2, 'min_hours')
            self.model.AddMaxEquality(max_h, total_hours_list)
            self.model.AddMinEquality(min_h, total_hours_list)
            diff = self.model.NewIntVar(0, 168*2, 'hours_diff')
            self.model.Add(diff == max_h - min_h)
            self.penalties.append(diff * self.PENALTY_WEIGHTS['fairness'])

    def _add_consecutive_off_days_reward(self):
        # Reward consecutive days off
        for emp in self.all_employees:
            for week in range(self.num_full_weeks):
                week_days = self.days[week*7:(week+1)*7]
                off_indicators = []
                
                for day in week_days:
                    day_str = day.isoformat()
                    if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                        off_indicators.append(1)  # Definitely off
                    else:
                        working_var = self.employee_working_vars.get(emp, {}).get(day_str)
                        if working_var is not None:
                            off_var = self.model.NewIntVar(0, 1, f'off_{emp}_{day_str}')
                            self.model.Add(off_var == 1 - working_var)
                            off_indicators.append(off_var)
                        else:
                            off_indicators.append(1)  # No working variable means off
                
                # Check for consecutive days off
                if len(off_indicators) >= 2:
                    pair_vars = []
                    for i in range(len(off_indicators) - 1):
                        pair_off = self.model.NewBoolVar(f'pair_off_{emp}_week{week}_{i}')
                        
                        # Handle both constant and variable cases
                        if isinstance(off_indicators[i], int) and isinstance(off_indicators[i+1], int):
                            if off_indicators[i] == 1 and off_indicators[i+1] == 1:
                                self.model.Add(pair_off == 1)
                            else:
                                self.model.Add(pair_off == 0)
                        elif isinstance(off_indicators[i], int):
                            if off_indicators[i] == 1:
                                self.model.Add(pair_off == off_indicators[i+1])
                            else:
                                self.model.Add(pair_off == 0)
                        elif isinstance(off_indicators[i+1], int):
                            if off_indicators[i+1] == 1:
                                self.model.Add(pair_off == off_indicators[i])
                            else:
                                self.model.Add(pair_off == 0)
                        else:
                            # Both are variables
                            self.model.Add(pair_off <= off_indicators[i])
                            self.model.Add(pair_off <= off_indicators[i+1])
                            self.model.Add(pair_off >= off_indicators[i] + off_indicators[i+1] - 1)
                        
                        pair_vars.append(pair_off)
                    
                    if pair_vars:
                        has_pair = self.model.NewBoolVar(f'has_pair_{emp}_week{week}')
                        self.model.AddMaxEquality(has_pair, pair_vars)
                        reward_penalty = self.model.NewIntVar(
                            -self.CONSECUTIVE_OFF_REWARD, 
                            self.CONSECUTIVE_OFF_PENALTY,
                            f'reward_consec_off_{emp}_week{week}'
                        )
                        self.model.Add(reward_penalty == -self.CONSECUTIVE_OFF_REWARD).OnlyEnforceIf(has_pair)
                        self.model.Add(reward_penalty == self.CONSECUTIVE_OFF_PENALTY).OnlyEnforceIf(has_pair.Not())
                        self.penalties.append(reward_penalty)

    def _add_one_shift_per_day_penalty(self):
        # Penalize working multiple shifts in one day
        for emp in self.all_employees:
            for day in self.active_days:
                day_str = day.isoformat()
                if day_str in self.request.closed_days or day_str in self.employee_details[emp]['days_off']:
                    continue
                
                # Check for multiple shifts per day
                double_shift_var = self.employee_double_shifts.get(emp, {}).get(day_str)
                if double_shift_var is not None:
                    self.penalties.append(double_shift_var * self.PENALTY_WEIGHTS['one_shift'])

    def _add_isolated_workday_penalty(self):
        # Penalize isolated work days
        for emp in self.all_employees:
            for i in range(1, len(self.days) - 1):
                prev_day = self.days[i-1].isoformat()
                curr_day = self.days[i].isoformat()
                next_day = self.days[i+1].isoformat()
                
                if (curr_day in self.request.closed_days or 
                    curr_day in self.employee_details[emp]['days_off'] or
                    prev_day in self.request.closed_days or
                    next_day in self.request.closed_days):
                    continue
                
                prev_off = prev_day in self.employee_details[emp]['days_off']
                next_off = next_day in self.employee_details[emp]['days_off']
                
                # Get working variables if they exist
                prev_work_var = self.employee_working_vars.get(emp, {}).get(prev_day)
                curr_work_var = self.employee_working_vars.get(emp, {}).get(curr_day)
                next_work_var = self.employee_working_vars.get(emp, {}).get(next_day)
                
                if prev_work_var is not None and curr_work_var is not None and next_work_var is not None:
                    # Create off indicators (1 = off, 0 = working)
                    prev_off_var = self.model.NewBoolVar(f'prev_off_{emp}_{curr_day}')
                    next_off_var = self.model.NewBoolVar(f'next_off_{emp}_{curr_day}')
                    
                    self.model.Add(prev_off_var == 1 - prev_work_var)
                    self.model.Add(next_off_var == 1 - next_work_var)
                    
                    # Create isolated work day indicator
                    isolated = self.model.NewBoolVar(f'isolated_{emp}_{curr_day}')
                    self.model.AddBoolAnd([prev_off_var, curr_work_var, next_off_var]).OnlyEnforceIf(isolated)
                    self.model.AddBoolOr([
                        prev_off_var.Not(), curr_work_var.Not(), next_off_var.Not()
                    ]).OnlyEnforceIf(isolated.Not())
                    
                    self.penalties.append(isolated * 300)  # Penalty for isolated work day


    def _add_consecutive_double_shift_constraint(self):
        # Penalize consecutive days with double shifts
        for emp in self.all_employees:
        # Iterate through all consecutive day pairs
            for i in range(len(self.days) - 1):
                day1 = self.days[i]
                day2 = self.days[i + 1]
                day1_str = day1.isoformat()
                day2_str = day2.isoformat()

                # Check if both days are active for the employee
                if (day1_str in self.employee_double_shifts.get(emp, {}) and 
                    day2_str in self.employee_double_shifts.get(emp, {})):
                    
                    # Get double shift variables
                    double1 = self.employee_double_shifts[emp][day1_str]
                    double2 = self.employee_double_shifts[emp][day2_str]

                    # Create penalty variable
                    penalty_var = self.model.NewBoolVar(f'penalty_{emp}_{day1_str}_{day2_str}')
                    self.model.AddBoolAnd([double1, double2]).OnlyEnforceIf(penalty_var)
                    self.model.AddBoolOr([double1.Not(), double2.Not()]).OnlyEnforceIf(penalty_var.Not())
                    
                    # Add weighted penalty
                    self.penalties.append(penalty_var * self.PENALTY_WEIGHTS['consecutive_double'])
                    

    def solve(self):
        # Create all variables
        self._create_assignment_variables()
        self._create_working_day_indicators()
        self._create_shift_indicators()
        self._create_double_shift_indicators()
        
        # Add all constraints
        self._add_task_coverage_constraint()
        self._add_one_task_per_employee_constraint()
        self._add_supervisor_constraint()
        self._add_max_consecutive_days_constraint()
        self._add_two_days_off_per_week_constraint()
        self._add_consecutive_double_shift_constraint()
        
        # Add soft constraints with penalties
        self._add_contract_hours_constraint()
        self._add_fairness_constraint()
        self._add_consecutive_off_days_reward()
        self._add_one_shift_per_day_penalty()
        self._add_isolated_workday_penalty()
        
        # Set objective: minimize total penalties
        if self.penalties:
            self.model.Minimize(sum(self.penalties))
        
        # Solve with parameters from colab.py
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 60  # 1 minute timeout
        solver.parameters.num_search_workers = 8   # Use 8 workers for parallel search
        
        status = solver.Solve(self.model)
        
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            return self._extract_solution(solver)
        else:
            raise ValueError(f"No solution found. Status: {solver.StatusName(status)}")

    def _extract_solution(self, solver: cp_model.CpSolver) -> dict:
        schedule = {}
        total_hours = {}
        daily_hours = {day.isoformat(): 0 for day in self.days}
        violations = []

        for emp in self.all_employees:
            schedule[emp] = {day.isoformat(): [] for day in self.days}
            total_hours[emp] = 0

        # Build task time mapping from shift definitions
        task_times = {}
        for shift in self.request.shifts:
            shift_name = shift['name']
            for task in shift.get('tasks', []):
                task_name = task['name']
                for day_sched in task.get('schedule', []):
                    day_of_week = day_sched['day']
                    start = day_sched.get('startTime', '00:00')
                    end = day_sched.get('endTime', '00:00')
                    # Ensure start and end times are strings
                    if not isinstance(start, str):
                        start = str(start)
                    if not isinstance(end, str):
                        end = str(end)
                    task_times[(shift_name, task_name, day_of_week)] = {
                        'start': start,
                        'end': end
                    }

        for (emp, day_str, shift_name, task_name), (var, duration) in self.assignments.items():
            if solver.Value(var):
                try:
                    # Get day of week (e.g., Monday)
                    date_obj = datetime.strptime(day_str, "%Y-%m-%d")
                    day_of_week = date_obj.strftime('%A')

                    # Get time info from mapping
                    time_info = task_times.get(
                        (shift_name, task_name, day_of_week),
                        {'start': '00:00', 'end': '00:00'}
                    )

                    # Create structured task entry
                    task_entry = {
                        'task': task_name,
                        'shift': shift_name,
                        'start': time_info['start'],
                        'end': time_info['end'],
                        'duration_hours': duration
                    }

                    schedule[emp][day_str].append(task_entry)
                    total_hours[emp] += duration
                    daily_hours[day_str] += duration

                except Exception as e:
                    # Handle any exceptions during processing
                    violations.append(f"Error processing assignment for {emp} on {day_str}: {str(e)}")

        return {
            "schedule": schedule,
            "summary": {
                "total_man_hours": sum(daily_hours.values()),
                "daily_man_hours": daily_hours,
                "total_hours_per_employee": total_hours
            },
            "violations": violations
        }
    


@app.post("/api/generate-schedule")
async def generate_schedule(request: SolverRequest):
    try:
        for shift_idx, shift in enumerate(request.shifts):
            for task_idx, task in enumerate(shift.get('tasks', [])):
                if not task.get('skillRequirement') and not task.get('skill_requirement'):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Missing skill requirement in shift {shift_idx+1} task {task_idx+1}"
                    )
        
        generator = ImprovedScheduleGenerator(request)
        return generator.solve()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)