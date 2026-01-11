import matplotlib.pyplot as plt

from jaxhmc.eval import run
from jaxhmc.potentials import BananaPotential

potential = BananaPotential()
run(potential)

plt.savefig("examples/banana_samples.png")

plt.show()
