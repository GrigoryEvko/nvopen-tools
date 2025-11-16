// Function: sub_16E0EC0
// Address: 0x16e0ec0
//
char *__fastcall sub_16E0EC0(int a1)
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
      result = "gnuabin32";
      break;
    case 3:
      result = "gnuabi64";
      break;
    case 4:
      result = "gnueabi";
      break;
    case 5:
      result = "gnueabihf";
      break;
    case 6:
      result = "gnux32";
      break;
    case 7:
      result = "code16";
      break;
    case 8:
      result = "eabi";
      break;
    case 9:
      result = "eabihf";
      break;
    case 10:
      result = "android";
      break;
    case 11:
      result = "musl";
      break;
    case 12:
      result = "musleabi";
      break;
    case 13:
      result = "musleabihf";
      break;
    case 14:
      result = "msvc";
      break;
    case 15:
      result = "itanium";
      break;
    case 16:
      result = "cygnus";
      break;
    case 17:
      result = "coreclr";
      break;
    case 18:
      result = "simulator";
      break;
  }
  return result;
}
