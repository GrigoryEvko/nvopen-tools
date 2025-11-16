// Function: sub_E0A520
// Address: 0xe0a520
//
const char *__fastcall sub_E0A520(int a1)
{
  const char *result; // rax

  if ( a1 == 2 )
    return "DW_ACCESS_protected";
  result = "DW_ACCESS_private";
  if ( a1 != 3 )
  {
    result = "DW_ACCESS_public";
    if ( a1 != 1 )
      return 0;
  }
  return result;
}
