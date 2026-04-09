``nanodisort``
==============

.. autoclass:: nanodisort.DisortState
    :exclude-members: __init__, print_state

    .. automethod:: __init__
    .. automethod:: allocate
    .. automethod:: solve
    .. automethod:: print_state

    .. rubric:: Boolean flags
        :heading-level: 3

    .. autoattribute:: usrtau
    .. autoattribute:: usrang
    .. autoattribute:: lamber
    .. autoattribute:: planck
    .. autoattribute:: spher
    .. autoattribute:: onlyfl
    .. autoattribute:: quiet
    .. autoattribute:: intensity_correction
    .. autoattribute:: old_intensity_correction

    .. rubric:: Dimensions
        :heading-level: 3

    .. autoattribute:: nstr
    .. autoattribute:: nlyr
    .. autoattribute:: nmom
    .. autoattribute:: ntau
    .. autoattribute:: numu
    .. autoattribute:: nphi

    .. rubric:: Boundary conditions
        :heading-level: 3

    .. autoattribute:: fbeam
    .. autoattribute:: umu0
    .. autoattribute:: phi0
    .. autoattribute:: fisot
    .. autoattribute:: fluor
    .. autoattribute:: albedo
    .. autoattribute:: btemp
    .. autoattribute:: ttemp
    .. autoattribute:: temis
    .. autoattribute:: brdf_type

    .. rubric:: Other scalar parameters
        :heading-level: 3

    .. autoattribute:: accur
    .. autoattribute:: wvnmlo
    .. autoattribute:: wvnmhi

    .. rubric:: Optical property arrays
        :heading-level: 3

    .. autoattribute:: dtauc
    .. autoattribute:: ssalb
    .. autoattribute:: pmom

    .. rubric:: Other input arrays
        :heading-level: 3

    .. autoattribute:: umu
    .. autoattribute:: phi
    .. autoattribute:: utau
    .. autoattribute:: temper

    .. rubric:: Output arrays (read-only, always present)
        :heading-level: 3

    .. autoattribute:: rfldir
    .. autoattribute:: rfldn
    .. autoattribute:: flup
    .. autoattribute:: dfdt
    .. autoattribute:: uavg
    .. autoattribute:: uavgdn
    .. autoattribute:: uavgup
    .. autoattribute:: uavgso

    .. rubric:: Output arrays (read-only, intensity)
        :heading-level: 3

    .. autoattribute:: uu
    .. autoattribute:: u0u
    .. autoattribute:: uum

    .. rubric:: Output arrays (read-only, special boundary condition)
        :heading-level: 3

    .. autoattribute:: albmed
    .. autoattribute:: trnmed

.. autoclass:: nanodisort.BRDFType
