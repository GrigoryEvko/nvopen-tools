// Function: sub_CC6100
// Address: 0xcc6100
//
char *__fastcall sub_CC6100(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "unknown";
      break;
    case 1:
      result = "darwin";
      break;
    case 2:
      result = "dragonfly";
      break;
    case 3:
      result = "freebsd";
      break;
    case 4:
      result = "fuchsia";
      break;
    case 5:
      result = "ios";
      break;
    case 6:
      result = "kfreebsd";
      break;
    case 7:
      result = "linux";
      break;
    case 8:
      result = "lv2";
      break;
    case 9:
      result = "macosx";
      break;
    case 10:
      result = "netbsd";
      break;
    case 11:
      result = "openbsd";
      break;
    case 12:
      result = "solaris";
      break;
    case 13:
      result = "uefi";
      break;
    case 14:
      result = "windows";
      break;
    case 15:
      result = "zos";
      break;
    case 16:
      result = "haiku";
      break;
    case 17:
      result = "rtems";
      break;
    case 18:
      result = "nacl";
      break;
    case 19:
      result = "aix";
      break;
    case 20:
      result = "cuda";
      break;
    case 21:
      result = "nvcl";
      break;
    case 22:
      result = "directx";
      break;
    case 23:
      result = "amdhsa";
      break;
    case 24:
      result = "ps4";
      break;
    case 25:
      result = "ps5";
      break;
    case 26:
      result = "elfiamcu";
      break;
    case 27:
      result = "tvos";
      break;
    case 28:
      result = "watchos";
      break;
    case 29:
      result = "bridgeos";
      break;
    case 30:
      result = "driverkit";
      break;
    case 31:
      result = "xros";
      break;
    case 32:
      result = "mesa3d";
      break;
    case 33:
      result = "amdpal";
      break;
    case 34:
      result = "hermit";
      break;
    case 35:
      result = "hurd";
      break;
    case 36:
      result = "wasi";
      break;
    case 37:
      result = "emscripten";
      break;
    case 38:
      result = "shadermodel";
      break;
    case 39:
      result = "liteos";
      break;
    case 40:
      result = "serenity";
      break;
    case 41:
      result = "vulkan";
      break;
    default:
      BUG();
  }
  return result;
}
