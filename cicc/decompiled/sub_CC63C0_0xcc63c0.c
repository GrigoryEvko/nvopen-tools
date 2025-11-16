// Function: sub_CC63C0
// Address: 0xcc63c0
//
char *__fastcall sub_CC63C0(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "unknown";
      break;
    case 1:
      result = "gnu";
      break;
    case 2:
      result = "gnut64";
      break;
    case 3:
      result = "gnuabin32";
      break;
    case 4:
      result = "gnuabi64";
      break;
    case 5:
      result = "gnueabi";
      break;
    case 6:
      result = "gnueabit64";
      break;
    case 7:
      result = "gnueabihf";
      break;
    case 8:
      result = "gnueabihft64";
      break;
    case 9:
      result = "gnuf32";
      break;
    case 10:
      result = "gnuf64";
      break;
    case 11:
      result = "gnusf";
      break;
    case 12:
      result = "gnux32";
      break;
    case 13:
      result = "gnu_ilp32";
      break;
    case 14:
      result = "code16";
      break;
    case 15:
      result = "eabi";
      break;
    case 16:
      result = "eabihf";
      break;
    case 17:
      result = "android";
      break;
    case 18:
      result = "musl";
      break;
    case 19:
      result = "muslabin32";
      break;
    case 20:
      result = "muslabi64";
      break;
    case 21:
      result = "musleabi";
      break;
    case 22:
      result = "musleabihf";
      break;
    case 23:
      result = "muslf32";
      break;
    case 24:
      result = "muslsf";
      break;
    case 25:
      result = "muslx32";
      break;
    case 26:
      result = "llvm";
      break;
    case 27:
      result = "msvc";
      break;
    case 28:
      result = "itanium";
      break;
    case 29:
      result = "cygnus";
      break;
    case 30:
      result = "coreclr";
      break;
    case 31:
      result = "simulator";
      break;
    case 32:
      result = "macabi";
      break;
    case 33:
      result = "pixel";
      break;
    case 34:
      result = "vertex";
      break;
    case 35:
      result = "geometry";
      break;
    case 36:
      result = "hull";
      break;
    case 37:
      result = "domain";
      break;
    case 38:
      result = "compute";
      break;
    case 39:
      result = "library";
      break;
    case 40:
      result = "raygeneration";
      break;
    case 41:
      result = "intersection";
      break;
    case 42:
      result = "anyhit";
      break;
    case 43:
      result = "closesthit";
      break;
    case 44:
      result = "miss";
      break;
    case 45:
      result = "callable";
      break;
    case 46:
      result = "mesh";
      break;
    case 47:
      result = "amplification";
      break;
    case 48:
      result = "opencl";
      break;
    case 49:
      result = "ohos";
      break;
    case 50:
      result = "pauthtest";
      break;
    default:
      BUG();
  }
  return result;
}
