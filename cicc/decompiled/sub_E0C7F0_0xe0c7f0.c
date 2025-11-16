// Function: sub_E0C7F0
// Address: 0xe0c7f0
//
const char *__fastcall sub_E0C7F0(int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 1:
      result = "DW_MACRO_GNU_define";
      break;
    case 2:
      result = "DW_MACRO_GNU_undef";
      break;
    case 3:
      result = "DW_MACRO_GNU_start_file";
      break;
    case 4:
      result = "DW_MACRO_GNU_end_file";
      break;
    case 5:
      result = "DW_MACRO_GNU_define_indirect";
      break;
    case 6:
      result = "DW_MACRO_GNU_undef_indirect";
      break;
    case 7:
      result = "DW_MACRO_GNU_transparent_include";
      break;
    case 8:
      result = "DW_MACRO_GNU_define_indirect_alt";
      break;
    case 9:
      result = "DW_MACRO_GNU_undef_indirect_alt";
      break;
    case 10:
      result = "DW_MACRO_GNU_transparent_include_alt";
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
