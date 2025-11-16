// Function: sub_B4D7D0
// Address: 0xb4d7d0
//
char *__fastcall sub_B4D7D0(int a1)
{
  char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "xchg";
      break;
    case 1:
      result = "add";
      break;
    case 2:
      result = "sub";
      break;
    case 3:
      result = "and";
      break;
    case 4:
      result = "nand";
      break;
    case 5:
      result = "or";
      break;
    case 6:
      result = "xor";
      break;
    case 7:
      result = "max";
      break;
    case 8:
      result = "min";
      break;
    case 9:
      result = "umax";
      break;
    case 10:
      result = "umin";
      break;
    case 11:
      result = "fadd";
      break;
    case 12:
      result = "fsub";
      break;
    case 13:
      result = "fmax";
      break;
    case 14:
      result = "fmin";
      break;
    case 15:
      result = "uinc_wrap";
      break;
    case 16:
      result = "udec_wrap";
      break;
    case 17:
      result = "usub_cond";
      break;
    case 18:
      result = "usub_sat";
      break;
    case 19:
      result = "<invalid operation>";
      break;
    default:
      BUG();
  }
  return result;
}
