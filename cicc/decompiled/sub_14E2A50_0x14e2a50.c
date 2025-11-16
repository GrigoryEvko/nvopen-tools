// Function: sub_14E2A50
// Address: 0x14e2a50
//
const char *__fastcall sub_14E2A50(int a1)
{
  const char *result; // rax

  result = "DW_CHILDREN_no";
  if ( a1 )
  {
    result = "DW_CHILDREN_yes";
    if ( a1 != 1 )
      return 0;
  }
  return result;
}
