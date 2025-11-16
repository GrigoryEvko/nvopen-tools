// Function: sub_CC5B60
// Address: 0xcc5b60
//
char *__fastcall sub_CC5B60(int a1, int a2)
{
  char *result; // rax

  switch ( a1 )
  {
    case 3:
      if ( a2 == 36 )
      {
        result = "arm64ec";
      }
      else
      {
        if ( a2 != 35 )
          goto LABEL_34;
        result = "arm64e";
      }
      break;
    case 11:
      switch ( a2 )
      {
        case 0:
        case 49:
          result = "dxilv1.0";
          break;
        case 50:
          result = "dxilv1.1";
          break;
        case 51:
          result = "dxilv1.2";
          break;
        case 52:
          result = "dxilv1.3";
          break;
        case 53:
          result = "dxilv1.4";
          break;
        case 54:
          result = "dxilv1.5";
          break;
        case 55:
          result = "dxilv1.6";
          break;
        case 56:
          result = "dxilv1.7";
          break;
        case 57:
          result = "dxilv1.8";
          break;
        default:
          goto LABEL_34;
      }
      break;
    case 16:
      if ( a2 != 40 )
        goto LABEL_34;
      result = "mipsisa32r6";
      break;
    case 17:
      if ( a2 != 40 )
        goto LABEL_34;
      result = "mipsisa32r6el";
      break;
    case 18:
      if ( a2 != 40 )
        goto LABEL_34;
      result = "mipsisa64r6";
      break;
    case 19:
      if ( a2 != 40 )
        goto LABEL_34;
      result = "mipsisa64r6el";
      break;
    case 50:
      switch ( a2 )
      {
        case '*':
          result = "spirv1.0";
          break;
        case '+':
          result = "spirv1.1";
          break;
        case ',':
          result = "spirv1.2";
          break;
        case '-':
          result = "spirv1.3";
          break;
        case '.':
          result = "spirv1.4";
          break;
        case '/':
          result = "spirv1.5";
          break;
        case '0':
          result = "spirv1.6";
          break;
        default:
          goto LABEL_34;
      }
      break;
    default:
LABEL_34:
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
          result = "aarch64_32";
          break;
        case 6:
          result = "arc";
          break;
        case 7:
          result = "avr";
          break;
        case 8:
          result = "bpfel";
          break;
        case 9:
          result = "bpfeb";
          break;
        case 10:
          result = "csky";
          break;
        case 11:
          result = "dxil";
          break;
        case 12:
          result = "hexagon";
          break;
        case 13:
          result = "loongarch32";
          break;
        case 14:
          result = "loongarch64";
          break;
        case 15:
          result = "m68k";
          break;
        case 16:
          result = "mips";
          break;
        case 17:
          result = "mipsel";
          break;
        case 18:
          result = "mips64";
          break;
        case 19:
          result = "mips64el";
          break;
        case 20:
          result = "msp430";
          break;
        case 21:
          result = "nvgpu";
          break;
        case 22:
          result = "powerpc";
          break;
        case 23:
          result = "powerpcle";
          break;
        case 24:
          result = "powerpc64";
          break;
        case 25:
          result = "powerpc64le";
          break;
        case 26:
          result = "r600";
          break;
        case 27:
          result = "amdgcn";
          break;
        case 28:
          result = "riscv32";
          break;
        case 29:
          result = "riscv64";
          break;
        case 30:
          result = "sparc";
          break;
        case 31:
          result = "sparcv9";
          break;
        case 32:
          result = "sparcel";
          break;
        case 33:
          result = "s390x";
          break;
        case 34:
          result = "tce";
          break;
        case 35:
          result = "tcele";
          break;
        case 36:
          result = "thumb";
          break;
        case 37:
          result = "thumbeb";
          break;
        case 38:
          result = "i386";
          break;
        case 39:
          result = "x86_64";
          break;
        case 40:
          result = "xcore";
          break;
        case 41:
          result = "xtensa";
          break;
        case 42:
          result = "nvptx";
          break;
        case 43:
          result = "nvptx64";
          break;
        case 44:
          result = "amdil";
          break;
        case 45:
          result = "amdil64";
          break;
        case 46:
          result = "hsail";
          break;
        case 47:
          result = "hsail64";
          break;
        case 48:
          result = "spir";
          break;
        case 49:
          result = "spir64";
          break;
        case 50:
          result = "spirv";
          break;
        case 51:
          result = "spirv32";
          break;
        case 52:
          result = "spirv64";
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
          result = "wasm32";
          break;
        case 57:
          result = "wasm64";
          break;
        case 58:
          result = "nvsass";
          break;
        case 59:
          result = "renderscript32";
          break;
        case 60:
          result = "renderscript64";
          break;
        case 61:
          result = "ve";
          break;
        default:
          BUG();
      }
      return result;
  }
  return result;
}
