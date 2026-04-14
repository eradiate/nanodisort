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
    .. autoattribute:: nphase

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
    .. autoattribute:: phase
    .. autoattribute:: mu_phase

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

.. autoclass:: nanodisort.BatchSolver
    :exclude-members: __init__

    .. automethod:: __init__
    .. automethod:: allocate
    .. automethod:: solve

    .. rubric:: Status
        :heading-level: 3

    .. autoattribute:: nbatch
    .. autoattribute:: nthreads
    .. autoattribute:: allocated
    .. autoattribute:: solved

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

    .. rubric:: Shared boundary conditions
        :heading-level: 3

    .. autoattribute:: umu0
    .. autoattribute:: phi0
    .. autoattribute:: fisot
    .. autoattribute:: fluor
    .. autoattribute:: btemp
    .. autoattribute:: ttemp
    .. autoattribute:: temis

    .. rubric:: Other shared scalar parameters
        :heading-level: 3

    .. autoattribute:: accur
    .. autoattribute:: wvnmlo
    .. autoattribute:: wvnmhi

    .. rubric:: Shared input arrays
        :heading-level: 3

    .. automethod:: set_umu
    .. automethod:: set_phi
    .. automethod:: set_utau
    .. automethod:: set_temper

    .. rubric:: Batched input setters
        :heading-level: 3

    .. automethod:: set_dtauc
    .. automethod:: set_ssalb
    .. automethod:: set_pmom
    .. automethod:: set_fbeam
    .. automethod:: set_albedo
    .. automethod:: set_utau_batched

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

.. autoclass:: nanodisort.BRDFType
