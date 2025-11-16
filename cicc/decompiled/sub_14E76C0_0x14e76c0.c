// Function: sub_14E76C0
// Address: 0x14e76c0
//
const char *__fastcall sub_14E76C0(int a1)
{
  const char *result; // rax

  if ( a1 == 1 )
    return "DW_VIRTUALITY_virtual";
  result = "DW_VIRTUALITY_pure_virtual";
  if ( a1 != 2 )
  {
    result = "DW_VIRTUALITY_none";
    if ( a1 )
      return 0;
  }
  return result;
}
