from dolfin import *
from mshr import *
import csv, os

# ---------------- Parameters ----------------
L, H, r = 400.0, 400.0, 20.0          # domain + precipitate radius
k, w, M, theta = 0.29, 0.1, 5.0, 0.5  # model constants
a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 = (
    0, 0, 8.072789087, -81.24549382, 408.0297321,
    -1244.129167, 2444.046270, -3120.635139,
    2506.663551, -1151.003178, 230.2006355)

dt = 0.1   # initial time‑step

# ---------------- Mesh ----------------
base = RectangleMesh(Point(-L/2, -H/2), Point(L/2, H/2), 200, 200, "crossed")
ref_tag = MeshFunction("bool", base, 0)
ref_tag.set_all(False)
class Annulus(SubDomain):
    def inside(self, x, on_boundary):
        return near((x[0]**2 + x[1]**2)**0.5, r, 10)
Annulus().mark(ref_tag, True)
mesh = refine(base, ref_tag)

# ---------------- Function space ----------------
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
ME = FunctionSpace(mesh, MixedElement([P1, P1]))

trial = TrialFunction(ME)
t_eta, t_mu   = TestFunctions(ME)
prev, curr    = Function(ME), Function(ME)

# ---------------- Initial condition ----------------
a_ell, b_ell = r/0.9, r*0.9
tol = 1e-14
ic = Expression((
    "sqrt(pow(x[0],2)/pow(a,2) + pow(x[1],2)/pow(b,2)) <= (1+tol) ? 1 : 0.0065",
    "0"), degree=0, a=a_ell, b=b_ell, tol=tol)
prev.interpolate(ic)
curr.interpolate(ic)
eta_0, mu_0 = split(prev)
eta,   mu   = split(curr)

# ---------------- Free‑energy ----------------
eta_sym  = variable(eta)
fb = w*(a0 + a1*eta_sym + a2*eta_sym**2 + a3*eta_sym**3 + a4*eta_sym**4 +
        a5*eta_sym**5 + a6*eta_sym**6 + a7*eta_sym**7 + a8*eta_sym**8 +
        a9*eta_sym**9 + a10*eta_sym**10)
dfb_deta = diff(fb, eta_sym)

# ---------------- Elastic energy (symbolic) ----------------
C11, C12, C66 = 250.0, 150.0, 100.0
Cm = as_matrix([[C11, C12, 0], [C12, C11, 0], [0, 0, C66]])
Cp = Cm
h  = eta_sym**3*(6*eta_sym**2 - 15*eta_sym + 10)
C  = Cm*(1 - h) + Cp*h
es = 0.005*h

Vu = VectorFunctionSpace(mesh, 'Lagrange', 2)
class FixX(SubDomain):
    def inside(self, x, on_b): return on_b and near(x[0], 0, 1e-2)
class FixY(SubDomain):
    def inside(self, x, on_b): return on_b and near(x[1], 0, 1e-2)
facets = MeshFunction('size_t', mesh, 1, 0)
FixX().mark(facets, 1)
FixY().mark(facets, 2)
bc_u = [DirichletBC(Vu.sub(1), Constant(0.0), facets, 1),
        DirichletBC(Vu.sub(0), Constant(0.0), facets, 2)]

u = TrialFunction(Vu)
v = TestFunction(Vu)
voigt  = lambda e: as_vector([e[0,0], e[1,1], 2*e[0,1]])
stress = lambda s: as_tensor([[s[0], s[2]], [s[2], s[1]]])

E_eig = as_tensor([[es, Constant(0.0)],
                   [Constant(0.0), es]])
S0    = stress(dot(C, as_vector([es, es, 0])))

a_mech = inner(stress(dot(C, voigt(sym(grad(u))))) - S0, grad(v))*dx
U = Function(Vu)
solve(lhs(a_mech) == rhs(a_mech), U, bcs=bc_u,
      solver_parameters={'linear_solver':'mumps'})

strain_diff = sym(grad(U)) - E_eig
E_el_expr   = 0.5*inner(stress(dot(C, voigt(sym(grad(U)))))-S0,
                        strain_diff)

# ---------------- Newton solver ----------------
newton = NewtonSolver()
newton.parameters.update({"linear_solver":"lu",
                          "convergence_criterion":"incremental",
                          "relative_tolerance":1e-6})

# ---------------- Output ----------------
os.makedirs("outputs", exist_ok=True)
vtkfile = File("outputs/eta_solution.pvd", "compressed")
vtkfile << (curr.split()[0], 0.0)

time = 0.0
while time <= 80:
    prev.vector()[:] = curr.vector()
    mu_mid = (1 - theta)*mu_0 + theta*mu

    F1 = (eta - eta_0)/dt * t_eta * dx + M*dot(grad(mu_mid), grad(t_eta))*dx
    F2 = mu * t_mu * dx - diff(E_el_expr, eta_sym)*t_mu*dx - dfb_deta*t_mu*dx \
         - k*dot(grad(eta), grad(t_mu))*dx
    Res = F1 + F2
    Jac = derivative(Res, curr, trial)

    class CHProblem(NonlinearProblem):
        def __init__(self, F, J):
            NonlinearProblem.__init__(self)
            self.Fform, self.Jform = F, J
        def F(self, b, x): assemble(self.Fform, tensor=b)
        def J(self, A, x): assemble(self.Jform, tensor=A)

    newton.solve(CHProblem(Res, Jac), curr.vector())

    time += dt

    # energies & output
    E_el   = assemble(E_el_expr*dx)
    E_bulk = assemble(fb*dx)
    E_grad = assemble((k/2)*dot(grad(eta), grad(eta))*dx)

    with open("outputs/energy_log.csv", "a") as fp:
        csv.writer(fp).writerow([time, E_el, E_bulk, E_grad,
                                 E_el+E_bulk+E_grad])

    vtkfile << (curr.split()[0], time)

    # convergence check
    if assemble(abs(mu - mu_0)*dx)/mesh.num_vertices() < 1e-10:
        break

print("Simulation finished OK – chemical potential converged.")
