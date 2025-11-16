// Function: sub_16E0740
// Address: 0x16e0740
//
char *__fastcall sub_16E0740(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "unknown";
      break;
    case 1:
      result = (char *)&unk_3F8856D;
      break;
    case 2:
      result = "armeb";
      break;
    case 3:
      result = "aarch64";
      break;
    case 4:
      result = "aarch64_be";
      break;
    case 5:
      result = "arc";
      break;
    case 6:
      result = "avr";
      break;
    case 7:
      result = "bpfel";
      break;
    case 8:
      result = "bpfeb";
      break;
    case 9:
      result = "hexagon";
      break;
    case 10:
      result = "mips";
      break;
    case 11:
      result = "mipsel";
      break;
    case 12:
      result = "mips64";
      break;
    case 13:
      result = "mips64el";
      break;
    case 14:
      result = "msp430";
      break;
    case 15:
      result = "nios2";
      break;
    case 16:
      result = "powerpc";
      break;
    case 17:
      result = "powerpc64";
      break;
    case 18:
      result = "powerpc64le";
      break;
    case 19:
      result = "r600";
      break;
    case 20:
      result = "amdgcn";
      break;
    case 21:
      result = "riscv32";
      break;
    case 22:
      result = "riscv64";
      break;
    case 23:
      result = "sparc";
      break;
    case 24:
      result = "sparcv9";
      break;
    case 25:
      result = "sparcel";
      break;
    case 26:
      result = "s390x";
      break;
    case 27:
      result = "tce";
      break;
    case 28:
      result = "tcele";
      break;
    case 29:
      result = "thumb";
      break;
    case 30:
      result = "thumbeb";
      break;
    case 31:
      result = "i386";
      break;
    case 32:
      result = "x86_64";
      break;
    case 33:
      result = "xcore";
      break;
    case 34:
      result = "nvptx";
      break;
    case 35:
      result = "nvptx64";
      break;
    case 36:
      result = "le32";
      break;
    case 37:
      result = "le64";
      break;
    case 38:
      result = "amdil";
      break;
    case 39:
      result = "amdil64";
      break;
    case 40:
      result = "hsail";
      break;
    case 41:
      result = "hsail64";
      break;
    case 42:
      result = "spir";
      break;
    case 43:
      result = "spir64";
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
      result = "wasm32";
      break;
    case 48:
      result = "wasm64";
      break;
    case 49:
      result = "nvsass";
      break;
    case 50:
      result = "renderscript32";
      break;
    case 51:
      result = "renderscript64";
      break;
  }
  return result;
}
