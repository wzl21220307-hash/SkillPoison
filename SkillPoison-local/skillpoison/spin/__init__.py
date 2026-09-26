"""skillpoison.spin — SPIN attack implementation, organised by paper architecture.

§3.1 Hierarchical Procedure Selection   -> spin.hps
§3.2 Evidence-Grounded Success Attribution -> spin.egsa
§3.4 Cross-Experience Inductive Reinforcement -> spin.ceir
S_f serialization operator              -> spin.serialize
E_f native extractor invocation         -> spin.extract
evaluation protocols                    -> spin.evaluate
verified provenance / conditions        -> spin.registry

See spin/README.md for the layer contract and boundary principles.
"""

__all__ = ["hps", "egsa", "ceir", "serialize", "extract", "evaluate", "registry"]
