// Function: sub_CC5ED0
// Address: 0xcc5ed0
//
char *__fastcall sub_CC5ED0(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 1:
    case 2:
    case 36:
    case 37:
      result = (char *)&unk_3F8856D;
      break;
    case 3:
    case 4:
    case 5:
      result = "aarch64";
      break;
    case 6:
      result = "arc";
      break;
    case 7:
      result = "avr";
      break;
    case 8:
    case 9:
      result = "bpf";
      break;
    case 10:
      result = "csky";
      break;
    case 11:
      result = "dx";
      break;
    case 12:
      result = "hexagon";
      break;
    case 13:
    case 14:
      result = "loongarch";
      break;
    case 15:
      result = "m68k";
      break;
    case 16:
    case 17:
    case 18:
    case 19:
      result = "mips";
      break;
    case 21:
      result = "nvgpu";
      break;
    case 22:
    case 23:
    case 24:
    case 25:
      result = "ppc";
      break;
    case 26:
      result = "r600";
      break;
    case 27:
      result = "amdgcn";
      break;
    case 28:
    case 29:
      result = "riscv";
      break;
    case 30:
    case 31:
    case 32:
      result = "sparc";
      break;
    case 33:
      result = "s390";
      break;
    case 38:
    case 39:
      result = (char *)&unk_4458FF5;
      break;
    case 40:
      result = "xcore";
      break;
    case 41:
      result = "xtensa";
      break;
    case 42:
    case 43:
      result = "nvvm";
      break;
    case 44:
    case 45:
      result = "amdil";
      break;
    case 46:
    case 47:
      result = "hsail";
      break;
    case 48:
    case 49:
      result = "spir";
      break;
    case 50:
    case 51:
    case 52:
      result = "spv";
      break;
    case 53:
      result = "kalimba";
      break;
    case 54:
      result = "shave";
      break;
    case 55:
      result = "lanai";
      break;
    case 56:
    case 57:
      result = "wasm";
      break;
    case 58:
      result = "nvsass";
      break;
    case 61:
      result = "ve";
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
