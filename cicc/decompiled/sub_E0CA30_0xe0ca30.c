// Function: sub_E0CA30
// Address: 0xe0ca30
//
const char *__fastcall sub_E0CA30(int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 0:
      result = "DW_ATOM_null";
      break;
    case 1:
      result = "DW_ATOM_die_offset";
      break;
    case 2:
      result = "DW_ATOM_cu_offset";
      break;
    case 3:
      result = "DW_ATOM_die_tag";
      break;
    case 4:
    case 5:
      result = "DW_ATOM_type_flags";
      break;
    case 6:
      result = "DW_ATOM_qual_name_hash";
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
