#!/usr/bin/env python3
import subprocess
import time

# Set variables (corresponding to lines 25-49 in run_exp)
delhs = [60,120,180]
gamma = 0.7
# phi0s = [0,0.5,1,2,3,4,5,6,7,8,9,10,12,14,16,18,25,30,35,40]
phi0s = [60,80]
# phi0s = [12]
# phi0s = [10,12] # To be resubmitted
radius = 1 # default is 1
omegas = [0.25,0.5,1,2] # default is 1
drag = 5e-6 # default 5e-5
res = "T42" # default is T42  
axisymm = True # default is False
Tsfcavg = 310 # default is 310
kadays = 50 # default is 50
ksdays = 50  # Default 7
sigmab = 0.85 # default is 0.85
sigmalat = 20 # default is 20
freediff = False # default is False
diff_coef = 5.0
tauc = 86400 # default is 14400, multiply by 6 to get 1 day
days = 100 # default is 100
runs_per_script = 5 # default is 10
start_analysis = 3 # default is 5

previous_job_id = None

# for phi0 in phi0s:
for i, phi0 in enumerate(phi0s):
    for omega in omegas:
        for delh in delhs:
            # Construct the runname from variables (matching shell script format)
            runname = f"delh{delh}_gamma{gamma}_phi0{phi0}_radius{radius}_omega{omega}_drag{drag}_res{res}_axisymm{axisymm}_Tsfcavg{Tsfcavg}_kadays{kadays}_ksdays{ksdays}_sigmab{sigmab}_sigmalat{sigmalat}_tauc{tauc}_freediff{freediff}_diffcoef{diff_coef}"

            # Read the template script
            with open("run_exp_axisymm", "r") as f:
                template = f.read()

            # Fill in the template with variables
            filled_script = template.format(
                delh=delh,
                gamma=gamma,
                phi0=phi0,
                radius=radius,
                omega=omega,
                drag=drag,
                res=res,
                axisymm=axisymm,
                Tsfcavg=Tsfcavg,
                kadays=kadays,
                ksdays=ksdays,
                sigmab=sigmab,
                sigmalat=sigmalat,
                freediff=freediff,
                diff_coef=diff_coef,
                tauc=tauc,
                days=days,
                runs_per_script=runs_per_script,
                start_analysis=start_analysis,
                runname=runname
            )

            # Write the filled script to a new file
            filled_script_path = f"run_exp_axisymm_{runname}.csh"
            with open(filled_script_path, "w") as f:
                f.write(filled_script)

            # Prepare the sbatch command with dependency if there's a previous job
            cmd = ["sbatch"]
            # if previous_job_id:
                # Add dependency to wait for previous job to complete
                # cmd.extend(["--dependency=afterany:" + previous_job_id])
                # Add a begin time 10 minutes in the future
                # cmd.extend([f"--begin=now+{i*10}minutes"])
            cmd.append(filled_script_path)
            
            print("Running command:", " ".join(cmd))
            # Run sbatch and capture the job ID from its output
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                previous_job_id = result.stdout.strip().split()[-1]
                print(f"Submitted job {previous_job_id}")
            except subprocess.CalledProcessError as e:
                print(f"Error: sbatch failed with exit code {e.returncode}")
                print(f"stderr: {e.stderr}")
                print(f"stdout: {e.stdout}")
                raise