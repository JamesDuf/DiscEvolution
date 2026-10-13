"""Generate the paired 240-disc grid; uses only the Python standard library."""
from copy import deepcopy
from itertools import product
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MASSES = [0.05, 0.075, 0.1, 0.125, 0.15]
ACCRETION_RATES = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6]
RADII = [50, 100, 150, 200]
# Integer construction avoids accumulated floating-point rounding errors.
PLANET_RADII = [i / 10 for i in range(2, 11)] + [1.5, 2.0] + list(range(3, 11)) + list(range(15, 51, 5))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4, allow_nan=False) + "\n")


def main():
    base = json.loads((HERE / "config.json").read_text())
    manifest = []
    for label, eos_type in [("DZ", "DeadZoneEOS"), ("Irradiated", "IrradiatedEOS")]:
        template = deepcopy(base)
        template["eos"].update(type=eos_type, alpha_SS_dead=1e-5,
                               alpha_SS_active=1e-3 if label == "DZ" else 1e-5)
        template["planets"].update(Rp=PLANET_RADII, Mp=[0.01] * len(PLANET_RADII),
                                   implant_time=[0.0] * len(PLANET_RADII))
        if label == "Irradiated":
            write_json(HERE / "Irradiated_config.json", template)
        for mass, mdot, radius in product(MASSES, ACCRETION_RATES, RADII):
            config = deepcopy(template)
            tag = f"{label}_M{mass:g}_Mdot{mdot:.0e}_Rc{radius}"
            config["disc"].update(M=mass, Mdot=mdot, Rd=radius)
            config["simulation"].update(run_name=tag, output_dir=f"output/{tag}")
            relative = Path("configs") / label / f"{tag}.json"
            write_json(HERE / relative, config)
            manifest.append(relative.as_posix())
    (HERE / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"Generated {len(manifest)} configs, {len(PLANET_RADII)} embryos each.")


if __name__ == "__main__":
    main()
