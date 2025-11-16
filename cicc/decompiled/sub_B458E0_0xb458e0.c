// Function: sub_B458E0
// Address: 0xb458e0
//
char *__fastcall sub_B458E0(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 1:
      result = "ret";
      break;
    case 2:
      result = "br";
      break;
    case 3:
      result = "switch";
      break;
    case 4:
      result = "indirectbr";
      break;
    case 5:
      result = "invoke";
      break;
    case 6:
      result = "resume";
      break;
    case 7:
      result = "unreachable";
      break;
    case 8:
      result = "cleanupret";
      break;
    case 9:
      result = "catchret";
      break;
    case 10:
      result = "catchswitch";
      break;
    case 11:
      result = "callbr";
      break;
    case 12:
      result = "fneg";
      break;
    case 13:
      result = "add";
      break;
    case 14:
      result = "fadd";
      break;
    case 15:
      result = "sub";
      break;
    case 16:
      result = "fsub";
      break;
    case 17:
      result = "mul";
      break;
    case 18:
      result = "fmul";
      break;
    case 19:
      result = "udiv";
      break;
    case 20:
      result = "sdiv";
      break;
    case 21:
      result = "fdiv";
      break;
    case 22:
      result = "urem";
      break;
    case 23:
      result = "srem";
      break;
    case 24:
      result = "frem";
      break;
    case 25:
      result = "shl";
      break;
    case 26:
      result = "lshr";
      break;
    case 27:
      result = "ashr";
      break;
    case 28:
      result = "and";
      break;
    case 29:
      result = "or";
      break;
    case 30:
      result = "xor";
      break;
    case 31:
      result = "alloca";
      break;
    case 32:
      result = "load";
      break;
    case 33:
      result = "store";
      break;
    case 34:
      result = "getelementptr";
      break;
    case 35:
      result = "fence";
      break;
    case 36:
      result = "cmpxchg";
      break;
    case 37:
      result = "atomicrmw";
      break;
    case 38:
      result = "trunc";
      break;
    case 39:
      result = "zext";
      break;
    case 40:
      result = "sext";
      break;
    case 41:
      result = "fptoui";
      break;
    case 42:
      result = "fptosi";
      break;
    case 43:
      result = "uitofp";
      break;
    case 44:
      result = "sitofp";
      break;
    case 45:
      result = "fptrunc";
      break;
    case 46:
      result = "fpext";
      break;
    case 47:
      result = "ptrtoint";
      break;
    case 48:
      result = "inttoptr";
      break;
    case 49:
      result = "bitcast";
      break;
    case 50:
      result = "addrspacecast";
      break;
    case 51:
      result = "cleanuppad";
      break;
    case 52:
      result = "catchpad";
      break;
    case 53:
      result = "icmp";
      break;
    case 54:
      result = "fcmp";
      break;
    case 55:
      result = "phi";
      break;
    case 56:
      result = "call";
      break;
    case 57:
      result = "select";
      break;
    case 60:
      result = "va_arg";
      break;
    case 61:
      result = "extractelement";
      break;
    case 62:
      result = "insertelement";
      break;
    case 63:
      result = "shufflevector";
      break;
    case 64:
      result = "extractvalue";
      break;
    case 65:
      result = "insertvalue";
      break;
    case 66:
      result = "landingpad";
      break;
    case 67:
      result = "freeze";
      break;
    default:
      result = "<Invalid operator> ";
      break;
  }
  return result;
}
