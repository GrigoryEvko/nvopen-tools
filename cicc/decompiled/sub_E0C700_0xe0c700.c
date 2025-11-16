// Function: sub_E0C700
// Address: 0xe0c700
//
const char *__fastcall sub_E0C700(int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 1:
      result = "DW_MACRO_define";
      break;
    case 2:
      result = "DW_MACRO_undef";
      break;
    case 3:
      result = "DW_MACRO_start_file";
      break;
    case 4:
      result = "DW_MACRO_end_file";
      break;
    case 5:
      result = "DW_MACRO_define_strp";
      break;
    case 6:
      result = "DW_MACRO_undef_strp";
      break;
    case 7:
      result = "DW_MACRO_import";
      break;
    case 8:
      result = "DW_MACRO_define_sup";
      break;
    case 9:
      result = "DW_MACRO_undef_sup";
      break;
    case 10:
      result = "DW_MACRO_import_sup";
      break;
    case 11:
      result = "DW_MACRO_define_strx";
      break;
    case 12:
      result = "DW_MACRO_undef_strx";
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
