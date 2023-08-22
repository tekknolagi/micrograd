// Value(data=-0.8197584768678694, grad=0)
#include <algorithm>
#include <cassert>
#include <cstring>
#include <initializer_list>

template <typename T = double, int dim = 1>
class Vector {
 public:
  Vector<T, dim>() { std::memset(arr, 0, dim * sizeof(T)); }
  Vector<T, dim>(T other[dim]) {
    for (int i = 0; i < dim; i++) {
      arr[i] = other[i];
    }
  }
  Vector<T, dim>(std::initializer_list<T> other) {
    assert(other.size() == dim && "oh no");
    for (int i = 0; i < dim; i++) {
      arr[i] = other.begin()[i];
    }
  }
  Vector<T, dim> dot(Vector<T, dim> other) {
    T result[dim];
    for (int i = 0; i < dim; i++) {
      arr[i] = other.arr[i];
    }
    return result;
  }
  T sum() {
    T result = 0;
    for (int i = 0; i < dim; i++) {
      result += arr[i];
    }
    return result;
  }
  T& at(int idx) { return arr[idx]; }

 private:
  T arr[dim];
};

double Neuron_0(Vector<double, 2> input) {
  Vector<double, 2> weights = {0.23550571390294128, 0.06653114721000164};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_1(Vector<double, 2> input) {
  Vector<double, 2> weights = {-0.26830328150124894, 0.1715747078045431};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_2(Vector<double, 2> input) {
  Vector<double, 2> weights = {-0.6686254326224383, 0.6487474938152629};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_3(Vector<double, 2> input) {
  Vector<double, 2> weights = {-0.23259038277158273, 0.5792256498313748};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_4(Vector<double, 2> input) {
  Vector<double, 2> weights = {0.8434530197925192, -0.3847332240409951};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_5(Vector<double, 2> input) {
  Vector<double, 2> weights = {0.9844941451716409, -0.5901079958448365};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_6(Vector<double, 2> input) {
  Vector<double, 2> weights = {0.31255526637777775, 0.8246106857787521};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_7(Vector<double, 2> input) {
  Vector<double, 2> weights = {-0.7814232047574572, 0.6408752595662697};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_8(Vector<double, 2> input) {
  Vector<double, 2> weights = {-0.20252189189007108, -0.8693137391598071};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_9(Vector<double, 2> input) {
  Vector<double, 2> weights = {0.39841666323128555, -0.3037961142013801};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_10(Vector<double, 2> input) {
  Vector<double, 2> weights = {-0.19282493884310759, 0.6032250931493106};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_11(Vector<double, 2> input) {
  Vector<double, 2> weights = {0.6001302646227185, 0.32749776568749045};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_12(Vector<double, 2> input) {
  Vector<double, 2> weights = {0.6650130652363544, 0.1889136153241595};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_13(Vector<double, 2> input) {
  Vector<double, 2> weights = {-0.07813264062433589, 0.9151267732861252};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_14(Vector<double, 2> input) {
  Vector<double, 2> weights = {0.5914405264235476, -0.3725442040076463};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_15(Vector<double, 2> input) {
  Vector<double, 2> weights = {0.3810827422406471, 0.8301999957053683};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
Vector<double, 16> Layer_16(Vector<double, 2> input) {
  Vector<double, 16> result;
  result.at(0) = Neuron_0(input);
  result.at(1) = Neuron_1(input);
  result.at(2) = Neuron_2(input);
  result.at(3) = Neuron_3(input);
  result.at(4) = Neuron_4(input);
  result.at(5) = Neuron_5(input);
  result.at(6) = Neuron_6(input);
  result.at(7) = Neuron_7(input);
  result.at(8) = Neuron_8(input);
  result.at(9) = Neuron_9(input);
  result.at(10) = Neuron_10(input);
  result.at(11) = Neuron_11(input);
  result.at(12) = Neuron_12(input);
  result.at(13) = Neuron_13(input);
  result.at(14) = Neuron_14(input);
  result.at(15) = Neuron_15(input);
  return result;
}
double Neuron_17(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      -0.08568482691922008, -0.4702876239420326, -0.598037011209763,
      -0.8653994554527067,  0.05088685407468296, 0.23734644010332318,
      0.15459549089529045,  -0.9122391928398941, -0.18505999501786086,
      0.30584552737905213,  0.23949109098065002, 0.35119774963171047,
      0.26999576683073867,  -0.6059558972032326, -0.4301483303818887,
      -0.09534359352124744};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_18(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      0.833061635489087,   0.5964776511293395,  -0.37143418174288434,
      0.5908148577342738,  0.22158648570764017, -0.1356625769718156,
      0.5808552090645627,  0.09921848842252134, 0.5519936528601597,
      0.11082037875863104, 0.2915133730664663,  0.6968596263439943,
      -0.572699001261544,  0.94892201097003,    0.05815161059370322,
      0.05689619757216291};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_19(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      0.5506426045691593,  -0.8991315551643992,   -0.01068087363780501,
      0.47299771880745967, -0.08962899486130538,  0.797578856715021,
      0.6099780726775426,  -0.024825326467644793, 0.5043619819611675,
      0.45774158735550596, -0.29478212096243595,  0.11675968465796172,
      0.1379511601427985,  -0.48542469349832285,  -0.8664235814101062,
      -0.7390189923668276};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_20(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      -0.8822004511411428, -0.6597694707506181, 0.6399602752689382,
      -0.6162690156778836, 0.9053331545059524,  0.667051974729419,
      0.7551658608563221,  0.10907067581950436, -0.14588865117400673,
      0.2127856122995495,  0.7622713432716846,  0.8620382404752289,
      -0.1401108907535058, 0.48216393547230973, -0.6888711593157701,
      0.2678404966193193};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_21(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      -0.3053994271093132, -0.8631814836201597,  -0.29515687142070823,
      0.35372633701181444, 0.20192101990676137,  0.43475517949093345,
      -0.6169565150718037, -0.03186709594911474, 0.22634427889578657,
      0.10564268012149869, -0.6805473384045992,  0.422794461121468,
      0.6853554447554182,  -0.21409905516555439, -0.6109356015626146,
      0.5254595422399804};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_22(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      0.24979744746643195,  -0.16494497754636983, -0.6818144661499881,
      -0.06157981422579417, 0.3953098897513252,   -0.3566554480884392,
      -0.9395269671087605,  0.19989246416270823,  0.28261231537882425,
      -0.3861199056619302,  0.8859519356381196,   0.09224139623540206,
      0.5616028368830122,   0.7479929232402773,   -0.5498104256800536,
      -0.38944426340050686};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_23(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      -0.11986910432370723, -0.2418861692296186,  0.27309902578900536,
      -0.7118613477995166,  0.640699986750376,    0.5251887402876205,
      -0.5265767665889542,  0.6262237833195563,   -0.8283374538902439,
      0.38807184998509303,  -0.315003423604574,   0.6825221766793921,
      -0.44958052796535997, 0.054321569495217936, 0.18838831645682874,
      -0.22248475258825984};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_24(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      -0.7209187740512764,  0.7176790825016579,  0.008555182533857453,
      -0.24243019229834561, 0.27898488110769337, -0.7726321568042522,
      -0.5139485701725583,  -0.8954946921521039, 0.581615741803986,
      -0.5750613904646755,  0.06993657839881884, 0.8578625660652908,
      0.15993906511777078,  -0.7940725880755064, 0.7128617267763828,
      0.9005136363586974};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_25(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      0.01164370432983386, 0.18191594886177542,   0.28846645419632666,
      0.28220903218440285, -0.007467712517625236, -0.9683122463720533,
      -0.703733854503761,  0.9555727255393986,    0.8304099868721302,
      0.29860725600901694, 0.23053684069095115,   0.8609716364376814,
      0.470379750754194,   -0.958287981521013,    0.5814512996793573,
      -0.6753502452813329};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_26(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      0.5036443505111738,   0.2955698675260916,   0.4217229281756927,
      0.5984472102024547,   -0.07808249126985456, 0.6103717000192679,
      0.34645800749824374,  0.504683663142057,    -0.9498847321986592,
      -0.14743838678191312, 0.5844330583547752,   -0.7950857611747761,
      -0.6601994025531952,  0.43550433241342956,  0.8151878759155218,
      0.2604257711713296};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_27(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      0.7177690445444254,  -0.686154027517816,  -0.6063064618924185,
      -0.843421963461304,  0.10008562568600432, 0.8240605653030353,
      0.15495085113716178, -0.5089384775906294, 0.8286765053073863,
      -0.8822610314096722, -0.5451509553109077, 0.5761953058198175,
      -0.3434024177268147, 0.10319781991345178, -0.05383238577004734,
      -0.6116507489401757};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_28(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      -0.8524536182338882, 0.6964865423661555,   0.32268922233815234,
      -0.2781907279339124, 0.16059637633929102,  -0.9446155428863412,
      0.8742237211553019,  0.3582002209547388,   0.9042806512794279,
      -0.8816062876600146, 0.10041983326299175,  -0.7698683314750423,
      0.30407601555374275, -0.20349872174164796, -0.4433144849231998,
      0.12433118993925452};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_29(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      -0.4258729196203048, 0.3790360826044181,   -0.9859455101873194,
      0.7028423162448694,  -0.40122067203805645, -0.25683960260938843,
      0.5346953520807405,  -0.35517369191511716, 0.5121727526610462,
      -0.8868545578539118, 0.518934991832354,    -0.8928025540682154,
      0.5236713643981046,  0.6018056040412896,   0.24634309741440386,
      -0.20561868737419142};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_30(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      -0.652542799532154,  -0.0065261577446391605, 0.3493423738090866,
      -0.6324860574376863, -0.8530409123740017,    -0.6218486564139833,
      0.9327230982583281,  0.2793857831208002,     0.5689184786100774,
      -0.6840675708965678, -0.5558656769249497,    0.20592862129017364,
      -0.8391389406223104, -0.5529892816922855,    -0.6278982991453468,
      -0.9592572536299122};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_31(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      0.9196221821038293,  0.045865737597233114, 0.5127293960073278,
      -0.7914104355296121, 0.848793948186239,    -0.3571964013350297,
      -0.8965914398912116, 0.4191281777225171,   -0.01884218503850832,
      0.6545963733751365,  -0.3484979864252389,  0.554377859246435,
      0.1689761071111946,  -0.3388547761206535,  0.397274795414263,
      -0.7930174038445066};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
double Neuron_32(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      0.077052593637436,   0.3936052761946094,   -0.8761639684113867,
      0.37877479983298445, -0.20339223773668702, -0.9231467276681604,
      -0.2020186036807059, 0.9605940825345125,   -0.9182348746309268,
      0.22922046437756505, -0.13194342373337498, -0.08703882869490953,
      0.984078978320559,   0.19426273589837106,  0.2294204003823488,
      0.12301724420660465};
  double result = weights.dot(input).sum() + 0;
  result = std::max(result, double{0});
  return result;
}
Vector<double, 16> Layer_33(Vector<double, 16> input) {
  Vector<double, 16> result;
  result.at(0) = Neuron_17(input);
  result.at(1) = Neuron_18(input);
  result.at(2) = Neuron_19(input);
  result.at(3) = Neuron_20(input);
  result.at(4) = Neuron_21(input);
  result.at(5) = Neuron_22(input);
  result.at(6) = Neuron_23(input);
  result.at(7) = Neuron_24(input);
  result.at(8) = Neuron_25(input);
  result.at(9) = Neuron_26(input);
  result.at(10) = Neuron_27(input);
  result.at(11) = Neuron_28(input);
  result.at(12) = Neuron_29(input);
  result.at(13) = Neuron_30(input);
  result.at(14) = Neuron_31(input);
  result.at(15) = Neuron_32(input);
  return result;
}
double Neuron_34(Vector<double, 16> input) {
  Vector<double, 16> weights = {
      0.9128783824023976,  -0.820982404658368,  0.9648285595338895,
      0.3470665940198512,  0.5436156893249604,  0.49097996014038525,
      -0.9353940167321961, -0.707696853463387,  -0.543868634071563,
      0.24162175370353833, -0.6723901907230767, -0.5973053326809556,
      0.6457663814022516,  -0.2271549182489696, -0.3223491002609964,
      -0.2532524374373504};
  double result = weights.dot(input).sum() + 0;
  return result;
}
double Layer_35(Vector<double, 16> input) { return Neuron_34(input); }
double MLP_36(Vector<double, 2> input) {
  Vector<double, 16> result0 = Layer_16(input);
  Vector<double, 16> result1 = Layer_33(result0);
  double result2 = Layer_35(result1);
  return result2;
}

#include <Python.h>

extern "C" {
PyObject* nn_wrapper(PyObject* module, PyObject* obj) {
  if (!PyList_CheckExact(obj)) {
    PyErr_Format(PyExc_TypeError, "expected list");
    return nullptr;
  }
  if (PyList_Size(obj) != 2) {
    PyErr_Format(PyExc_TypeError, "expected list of size 2");
    return nullptr;
  }
  Vector<double, 2> input;
  for (int i = 0; i < 2; i++) {
    PyObject* item_obj = PyList_GetItem(obj, i);
    double item_double = PyFloat_AsDouble(item_obj);
    if (item_double < 0 && PyErr_Occurred()) {
      return nullptr;
    }
    input.at(i) = item_double;
  }
  // TODO(max): Make this able to return multiple outputs?
  double result = MLP_36(input);
  return PyFloat_FromDouble(result);
}

static PyMethodDef nn_methods[] = {
    {"nn", nn_wrapper, METH_O, "doc"},
    {nullptr, nullptr},
};

// clang-format off
static struct PyModuleDef nnmodule = {
    PyModuleDef_HEAD_INIT,
    "nn",
    "doc",
    -1,
    nn_methods,
    NULL,
    NULL,
    NULL,
    NULL
};
// clang-format on

PyObject* PyInit_nn() {
  PyObject* m = PyState_FindModule(&nnmodule);
  if (m != NULL) {
    return m;
  }
  return PyModule_Create(&nnmodule);
}
}
