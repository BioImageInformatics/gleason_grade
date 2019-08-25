Storage for various model weights

weights to be accessed by name with syntax

[model]-[magnification]-[input size]-[type]-[subtype1]-...-[version]/

ex:

densenet-10x-256-seg-0.1/
  densenet-10x-256-seg-0.1/densenet.ckpt-60450.data-00000-of-00001
  densenet-10x-256-seg-0.1/densenet.ckpt-60450.index
  densenet-10x-256-seg-0.1/densenet.ckpt-60450.meta

we're assuming names duplicating all fields except [version]
are based upon each other


-- > use file names to build meta network < --
