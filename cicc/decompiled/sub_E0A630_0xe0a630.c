// Function: sub_E0A630
// Address: 0xe0a630
//
const char *__fastcall sub_E0A630(int a1)
{
  const char *result; // rax

  result = "DW_APPLE_ENUM_KIND_Closed";
  if ( a1 )
  {
    result = "DW_APPLE_ENUM_KIND_Open";
    if ( a1 != 1 )
      return 0;
  }
  return result;
}
