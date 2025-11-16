// Function: sub_2DADC00
// Address: 0x2dadc00
//
_BOOL8 __fastcall sub_2DADC00(_BYTE *a1)
{
  _BOOL8 result; // rax

  result = 0;
  if ( !*a1 )
    return (a1[3] & 0x10) != 0;
  return result;
}
