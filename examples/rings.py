import matplotlib.pyplot as plt

from jaxhmc.eval import run
from jaxhmc.potentials import RingsPotential

potential = RingsPotential(sigma=0.1)
run(potential)

plt.savefig("examples/rings_samples.png")

plt.show()
