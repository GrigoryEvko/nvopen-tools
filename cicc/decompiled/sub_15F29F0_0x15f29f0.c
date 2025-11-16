// Function: sub_15F29F0
// Address: 0x15f29f0
//
char *__fastcall sub_15F29F0(int a1)
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
      result = "add";
      break;
    case 12:
      result = "fadd";
      break;
    case 13:
      result = "sub";
      break;
    case 14:
      result = "fsub";
      break;
    case 15:
      result = "mul";
      break;
    case 16:
      result = "fmul";
      break;
    case 17:
      result = "udiv";
      break;
    case 18:
      result = "sdiv";
      break;
    case 19:
      result = "fdiv";
      break;
    case 20:
      result = "urem";
      break;
    case 21:
      result = "srem";
      break;
    case 22:
      result = "frem";
      break;
    case 23:
      result = "shl";
      break;
    case 24:
      result = "lshr";
      break;
    case 25:
      result = "ashr";
      break;
    case 26:
      result = "and";
      break;
    case 27:
      result = "or";
      break;
    case 28:
      result = "xor";
      break;
    case 29:
      result = "alloca";
      break;
    case 30:
      result = "load";
      break;
    case 31:
      result = "store";
      break;
    case 32:
      result = "getelementptr";
      break;
    case 33:
      result = "fence";
      break;
    case 34:
      result = "cmpxchg";
      break;
    case 35:
      result = "atomicrmw";
      break;
    case 36:
      result = "trunc";
      break;
    case 37:
      result = "zext";
      break;
    case 38:
      result = "sext";
      break;
    case 39:
      result = "fptoui";
      break;
    case 40:
      result = "fptosi";
      break;
    case 41:
      result = "uitofp";
      break;
    case 42:
      result = "sitofp";
      break;
    case 43:
      result = "fptrunc";
      break;
    case 44:
      result = "fpext";
      break;
    case 45:
      result = "ptrtoint";
      break;
    case 46:
      result = "inttoptr";
      break;
    case 47:
      result = "bitcast";
      break;
    case 48:
      result = "addrspacecast";
      break;
    case 49:
      result = "cleanuppad";
      break;
    case 50:
      result = "catchpad";
      break;
    case 51:
      result = "icmp";
      break;
    case 52:
      result = "fcmp";
      break;
    case 53:
      result = "phi";
      break;
    case 54:
      result = "call";
      break;
    case 55:
      result = "select";
      break;
    case 58:
      result = "va_arg";
      break;
    case 59:
      result = "extractelement";
      break;
    case 60:
      result = "insertelement";
      break;
    case 61:
      result = "shufflevector";
      break;
    case 62:
      result = "extractvalue";
      break;
    case 63:
      result = "insertvalue";
      break;
    case 64:
      result = "landingpad";
      break;
    default:
      result = "<Invalid operator> ";
      break;
  }
  return result;
}
