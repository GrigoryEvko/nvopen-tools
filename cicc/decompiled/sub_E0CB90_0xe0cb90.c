// Function: sub_E0CB90
// Address: 0xe0cb90
//
const char *__fastcall sub_E0CB90(unsigned int a1)
{
  const char *result; // rax

  if ( a1 > 5 )
  {
    result = "DW_IDX_GNU_internal";
    if ( a1 == 0x2000 )
      return result;
    if ( a1 == 8193 )
      return "DW_IDX_GNU_external";
    return 0;
  }
  if ( !a1 )
    return 0;
  switch ( a1 )
  {
    case 2u:
      result = "DW_IDX_type_unit";
      break;
    case 3u:
      result = "DW_IDX_die_offset";
      break;
    case 4u:
      result = "DW_IDX_parent";
      break;
    case 5u:
      result = "DW_IDX_type_hash";
      break;
    default:
      result = "DW_IDX_compile_unit";
      break;
  }
  return result;
}
