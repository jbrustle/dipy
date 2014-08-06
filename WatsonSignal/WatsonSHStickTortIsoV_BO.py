#!//anaconda/bin/python

from __future__ import division
from WatsonSingleRadTortIsoV_GDP import MeasWatsonSHCylSingleRadTortIsoV_GPD
import numpy as np


def WatsonSHStickTortIsoV_BO(x, grad_dirs, G, delta, smalldel, fibredir, roots):
    """
    Substrate: Impermeable sticks (cylinders with zero radius) in a homogeneous
    background.
    Orientation distribution: Watson's distribution with SH approximation
    Signal approximation: Not applicable
    Notes: This version estimates the hindered diffusivity from the free diffusivity
    and packing density using Szafer et al's tortuosity model for randomly
    packed cylinders.
    This version includes an isotropic diffusion compartment with its own
    diffusivity.
    Includes a free parameter for the measurement at b=0.

    [E,J]=SynthMeasWatsonSHStickTortIsoV_B0(x, protocol, fibredir)
    returns the measurements E according to the model and the Jacobian J of the
    measurements with respect to the parameters.  The Jacobian does not
    include derivates with respect to the fibre direction.

    x is the list of model parameters in SI units:
    x(1) is the volume fraction of the intracellular space.
    x(2) is the free diffusivity of the material inside and outside the cylinders.
    x(3) is the concentration parameter of the Watson's distribution.
    x(4) is the volume fraction of the isotropic compartment.
    x(5) is the diffusivity of the isotropic compartment.
    x(6) is the measurement at b=0.

    protocol is the object containing the acquisition protocol.

    fibredir is a unit vector along the symmetry axis of the Watson's
    distribution.  It must be in Cartesian coordinates [x y z]' with size [3 1].

    """
    xcyl=[x[0], x[1], 0, x[2], x[3], x[4]]

    Enorm = MeasWatsonSHCylSingleRadTortIsoV_GPD(xcyl, grad_dirs, G, delta, smalldel, fibredir, roots)
    S0 = x[5]
    E = Enorm * S0


    return E

def test_WatsonSHStickTortIsoV_BO():


    grad_dirs=[   1.000000000000000,                   0,            0,
    0.132723066185094,  -0.739879368956107,   0.659517328881918,
    -0.918278273681402,   0.379929113233140,  -0.111440033213314,
    -0.965425725572949,  -0.153302956422874,  -0.210834940069123,
    0.608607111564084,   0.784469143801444,  -0.119187021848235,
    0.639150072697892,  -0.437609049774313,  -0.632444071935142,
    0.789541795034582,  -0.209333945656810,   0.576890850238866,
    0.135007939063640,   0.362539836366232,  -0.922137583789607,
    0.344275919703160,  -0.093473978198693,  -0.934203782111941,
    1.000000000000000,                   0,                   0,
    0.682348858017888,  -0.706469852998828,  -0.187883960905391,
    0.141670962386719,  -0.550472853850853,  -0.822744781563346,
    0.699576781174866,   0.167741947530914,  -0.694589782734781,
    -0.591356313217572,   0.690452365704752,  -0.416621220667446,
    0.935168183247626,  -0.136207026689974,  -0.326968064069889,
    0.644819075445448,   0.248547029080625,   0.722795084568837,
    -0.727041998322714,  -0.600674998614243,  -0.332564999232773,
    -0.156172056459332,  -0.940576340037220,   0.301540109012800,
    1.000000000000000,                   0,                   0,
    -0.196180109901600,  -0.149388083688348,  -0.969121542909309,
    0.293367098209200,  -0.952754318949329,   0.078708026348736,
    0.467415038442551,   0.628758051712208,  -0.621439051110257,
    0.242626961848005,  -0.862958864303612,  -0.443208930307396,
    -0.321942099026024,  -0.668161205519403,  -0.670756206317598,
    0.229727986296611,   0.943725943706277,   0.237920985807895,
    -0.391998910872982,   0.321716926852678,  -0.861878804038517,
    0.887685964926641,   0.387798984677675,  -0.248244990191592,
    1.000000000000000,                   0,                   0,
    -0.900301917924429,  -0.079144992784787,  -0.428009960980688,
    0.726183061729557,  -0.538020045734665,  -0.428010036383209,
    0.913388838912018,  -0.376476933603404,  -0.154873972685964,
    0.188307992911240,  -0.565147978725287,  -0.803210969763525,
    -0.103414023139201,  -0.354087079228055,  -0.929477207973337,
    0.333389002305218,   0.942753006518666,  -0.008279000057245,
    0.692523205585932,  -0.712545211529766,  -0.112654033443045,
    -0.684915843591266,  -0.058847986561358,  -0.726241834153981,
    1.000000000000000,                   0,                   0,
    0.611951143007185,  -0.029447006881487,  -0.790347184696650,
    -0.372141070256147,  -0.046290008739045,  -0.927021175011417,
    0.286024139760582,  -0.840328410611452,   0.460476225003474,
    -0.545125332585973,   0.295394180222703,  -0.784589478685248,
    0.974370088752940,   0.203717018556075,   0.095406008690295,
    0.681627050203879,   0.566802041746672,  -0.462731034081530,
    0.833466850506899,  -0.189788965958885,  -0.518953906918879,
    -0.170331024653885,  -0.668514096761406,  -0.723931104782520,
    1.000000000000000,                   0,                   0,
    -0.251529966951857,  -0.886633883506512,  -0.388088949009578,
    0.096343990649190,  -0.851840917323306,  -0.514863950029109,
    -0.269230887958378,   0.605851747872121,  -0.748637688451122,
    -0.464602055814507,  -0.382792045986342,  -0.798508095927977,
    0.972997229516938,  -0.174193041089792,   0.151437035721957,
    0.975728516771059,  -0.001726999144705,  -0.218976891551831,
    0.433122103473316,  -0.708525169267392,  -0.557133133099679,
    0.290163077436701,  -0.216666057822329,  -0.932127248759628,
    1.000000000000000,                   0,                   0,
    -0.287807097298818,   0.952924322154704,  -0.095406032253875,
    0.671101614321369,   0.730094580418416,  -0.128780925990267,
    0.374313914386744,   0.629540856011117,  -0.680857844273871,
    0.756249873276354,   0.398573933211569,   0.518868913053922,
    -0.832060148901753,   0.513260091850725,  -0.210333037640257,
    -0.580720935404672,  -0.782527912957077,  -0.224528975024970,
    0.394573913706497,  -0.885894806254384,  -0.243929946652405,
    0.574959656801509,   0.334516800323971,  -0.746672554304565,
    1.000000000000000,                   0,                   0,
    0.052556987348116,   0.793070809086466,  -0.606857853912947,
    0.048811006319805,   0.963948124807192,  -0.261575033867430,
    0.834780478455198,   0.191763880191910,  -0.516108677551398,
    0.552354026610761,  -0.392401018904704,  -0.735477035433078,
    0.393069051661462,   0.841244110565562,  -0.371221048789957,
    0.302647082000119,   0.187492050799666,  -0.934479253190647,
    -0.580867884748272,   0.782418844757945,  -0.224528955450541,
    0.831553300991582,   0.525224190111758,   0.180717065412903,
    1.000000000000000,                   0,                   0,
    -0.812671010474517,   0.275965003556913,  -0.513234006615073,
    0.098653954863409,   0.487437776985326,  -0.867566603067115,
    -0.596347844875335,   0.588588846893640,  -0.545831858015779,
    0.015014999390016,  -0.988623959837152,  -0.149656993920185,
    -0.506652863195132,  -0.657955822340767,  -0.557132849564680,
    0.891415547131600,   0.406150793662046,  -0.201046897861567,
    -0.200646040920059,  0.295024060167655, -0.934185190519148,
    0,                 0,  -1.000000000000000]

    
    grad_dirs = np.array(grad_dirs)
    grad_dirs=grad_dirs.reshape(81,3) 
   

    G=[     0,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0.023664319132398,
    0,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400,
    0.0400]
    G=np.array(G)

    smalldel=[ 0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192,
    0.029694619030192]
    smalldel = np.array(smalldel)
    delta = smalldel
    roots = 0
    fibredir = [[-0.214994778410205], [0.765937075390002], [-0.605902336849230]]
    x = [0.008986370184599, 0.000000000017000, 0.008508710965541, 0.000004877258795, 0.000000000030000, 8.193333333333335]
    #x = [0.006733692951986,   0.000000000017000,   0.066446333410051,   0.000001258482142,   0.000000000030000,   8.193333333333335]
    #x = [0.292772616103,   0.000000001700,   0.556821335991,   0.176215439118,   0.000000003000,   2652.666666666667]
    #x = [0.9999595735701,   0.0000000017000,   1.0000000000000,   0.6294397716181,   0.0000000030000,   454.8888888888889]
    sample = WatsonSHStickTortIsoV_BO(x, grad_dirs, G, delta, smalldel, fibredir, roots)

    error = abs(8.097560855563191-abs(sample[1])) + abs(8.097562192342671-abs(sample[2])) + abs(8.097562706146647-abs(sample[3]))
    error = error + abs(8.097562221948307-abs(sample[4])) + abs(8.097562783693162-abs(sample[5])) + abs(8.097561892217968-abs(sample[6])) 


    test = True
    if (error > 1.E-10):
        test = False

    return test


