// Function: sub_14E6F20
// Address: 0x14e6f20
//
const char *__fastcall sub_14E6F20(int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 1:
      result = "DW_ATE_address";
      break;
    case 2:
      result = "DW_ATE_boolean";
      break;
    case 3:
      result = "DW_ATE_complex_float";
      break;
    case 4:
      result = "DW_ATE_float";
      break;
    case 5:
      result = "DW_ATE_signed";
      break;
    case 6:
      result = "DW_ATE_signed_char";
      break;
    case 7:
      result = "DW_ATE_unsigned";
      break;
    case 8:
      result = "DW_ATE_unsigned_char";
      break;
    case 9:
      result = "DW_ATE_imaginary_float";
      break;
    case 10:
      result = "DW_ATE_packed_decimal";
      break;
    case 11:
      result = "DW_ATE_numeric_string";
      break;
    case 12:
      result = "DW_ATE_edited";
      break;
    case 13:
      result = "DW_ATE_signed_fixed";
      break;
    case 14:
      result = "DW_ATE_unsigned_fixed";
      break;
    case 15:
      result = "DW_ATE_decimal_float";
      break;
    case 16:
      result = "DW_ATE_UTF";
      break;
    case 17:
      result = "DW_ATE_UCS";
      break;
    case 18:
      result = "DW_ATE_ASCII";
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
