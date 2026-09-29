import json
import sys
import math

def solve_n(n_val, bit_length=16):
    n_int = int(n_val)
    
    if bit_length >= 2048:
        # No 2048-bit solver exists here. Do not report synthetic steps as computation.
        return {
            "n": str(n_val),
            "p": None,
            "q": None,
            "steps": [],
            "oracle_used": False,
            "status": "unsupported",
            "message": "2048-bit factorization is not implemented in this prototype."
        }

    import torch
    from TKHD_7_Massive import CryptoLandscape, MDNOracle

    # The small-key backend uses CPU and float64 only.
    DEVICE = torch.device("cpu")
    torch.set_default_dtype(torch.float64)

    n = torch.tensor([[n_val]], dtype=torch.float64, device=DEVICE)
    landscape = CryptoLandscape(n)
    
    # Load Oracle
    try:
        oracle = MDNOracle("mdn_p_massive.pth", "mdn_q_massive.pth", DEVICE)
    except Exception as e:
        return {"error": f"Failed to load oracle: {str(e)}"}

    # Simulation Logic (adapted from simulate_rays)
    # We only need 1 ray for the API response
    batch_size = 1
    
    # Configuration from TKHD_7_Massive
    NUM_STEPS = 500 # Reduced for API responsiveness
    GRAD_NORM_THRESH = 1.0
    DT = 0.1
    TOP_K = 5
    NUM_ATTEMPTS = 1000

    # Start
    # Initial guess near sqrt(n)
    x_val = math.sqrt(n_val) * (1.1) 
    y_val = n_val / x_val
    
    x = torch.tensor([[x_val]], device=DEVICE, requires_grad=True)
    y = torch.tensor([[y_val]], device=DEVICE, requires_grad=True)

    steps_data = []
    oracle_used = False
    steps_since_last_oracle = 0
    final_p = None
    final_q = None

    for step in range(NUM_STEPS):
        x_grad = x.detach().requires_grad_(True)
        y_grad = y.detach().requires_grad_(True)

        grad_x, grad_y = landscape.gradient(x_grad, y_grad)
        grad_norm = torch.sqrt(grad_x**2 + grad_y**2).item()

        steps_data.append({
            "x": x.item(),
            "y": y.item(),
            "energy": landscape.energy(x, y).item(),
            "type": "gradient" if step > 0 else "start"
        })

        # Check for solution
        found = False
        n_int = int(n_val)
        x_center = int(round(x.item()))
        search_radius = 1000 # Smaller for API
        start_x = max(2, x_center - search_radius)
        end_x = min(int(math.sqrt(n_int)) + 1, x_center + search_radius)
        for test_x in range(start_x, end_x + 1):
            if test_x != 0 and n_int % test_x == 0:
                final_p = test_x
                final_q = n_int // test_x
                found = True
                break
        
        if found:
            steps_data.append({
                "x": float(final_p),
                "y": float(final_q),
                "energy": 0.0,
                "type": "success"
            })
            break

        # Oracle activation
        if grad_norm < GRAD_NORM_THRESH or steps_since_last_oracle >= 50:
            candidates_p, candidates_q, _ = oracle.get_top_candidates(
                n, x, y, top_k=TOP_K, num_attempts=NUM_ATTEMPTS
            )
            oracle_used = True
            steps_since_last_oracle = 0
            
            best_energy = float('inf')
            best_coords = (x.item(), y.item())
            
            # Since batch_size=1, index 0
            for p_cand, q_cand in zip(candidates_p[0], candidates_q[0]):
                E_cand = ((n_val - p_cand * q_cand) / math.sqrt(n_val))**2 + 100.0 * (math.sin(math.pi * p_cand)**2 + math.sin(math.pi * q_cand)**2)
                if E_cand < best_energy:
                    best_energy = E_cand
                    best_coords = (p_cand, q_cand)
            
            x = torch.tensor([[best_coords[0]]], device=DEVICE)
            y = torch.tensor([[best_coords[1]]], device=DEVICE)
            steps_data[-1]["type"] = "mdn_jump"
        else:
            steps_since_last_oracle += 1
            
            # Gradient Descent step
            E = landscape.energy(x, y)
            gx, gy = torch.autograd.grad(E.sum(), (x, y))
            gn = torch.sqrt(gx**2 + gy**2 + 1e-8)
            
            with torch.no_grad():
                x -= DT * gx / gn
                y -= DT * gy / gn

    return {
        "n": n_val,
        "p": final_p,
        "q": final_q,
        "steps": steps_data,
        "oracle_used": oracle_used
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No N provided"}))
        sys.exit(1)
    
    try:
        n_input = int(sys.argv[1])
        bits = int(sys.argv[2]) if len(sys.argv) > 2 else 16
        result = solve_n(n_input, bits)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
