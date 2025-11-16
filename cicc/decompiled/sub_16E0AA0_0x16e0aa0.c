// Function: sub_16E0AA0
// Address: 0x16e0aa0
//
char *__fastcall sub_16E0AA0(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 1:
    case 2:
    case 29:
    case 30:
      result = (char *)&unk_3F8856D;
      break;
    case 3:
    case 4:
      result = "aarch64";
      break;
    case 5:
      result = "arc";
      break;
    case 6:
      result = "avr";
      break;
    case 7:
    case 8:
      result = "bpf";
      break;
    case 9:
      result = "hexagon";
      break;
    case 10:
    case 11:
    case 12:
    case 13:
      result = "mips";
      break;
    case 15:
      result = "nios2";
      break;
    case 16:
    case 17:
    case 18:
      result = "ppc";
      break;
    case 19:
      result = "r600";
      break;
    case 20:
      result = "amdgcn";
      break;
    case 21:
    case 22:
      result = "riscv";
      break;
    case 23:
    case 24:
    case 25:
      result = "sparc";
      break;
    case 26:
      result = "s390";
      break;
    case 31:
    case 32:
      result = (char *)&unk_4458FF5;
      break;
    case 33:
      result = "xcore";
      break;
    case 34:
    case 35:
      result = "nvvm";
      break;
    case 36:
      result = "le32";
      break;
    case 37:
      result = "le64";
      break;
    case 38:
    case 39:
      result = "amdil";
      break;
    case 40:
    case 41:
      result = "hsail";
      break;
    case 42:
    case 43:
      result = "spir";
      break;
    case 44:
      result = "kalimba";
      break;
    case 45:
      result = "shave";
      break;
    case 46:
      result = "lanai";
      break;
    case 47:
    case 48:
      result = "wasm";
      break;
    case 49:
      result = "nvsass";
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
