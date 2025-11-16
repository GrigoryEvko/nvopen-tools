// Function: sub_14E9960
// Address: 0x14e9960
//
const char *__fastcall sub_14E9960(int a1)
{
  const char *result; // rax

  switch ( a1 )
  {
    case 1:
      result = "DW_IDX_compile_unit";
      break;
    case 2:
      result = "DW_IDX_type_unit";
      break;
    case 3:
      result = "DW_IDX_die_offset";
      break;
    case 4:
      result = "DW_IDX_parent";
      break;
    case 5:
      result = "DW_IDX_type_hash";
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
