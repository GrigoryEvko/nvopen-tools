// Function: sub_1643EC0
// Address: 0x1643ec0
//
_BOOL8 __fastcall sub_1643EC0(__int64 a1)
{
  unsigned __int8 v1; // cl
  _BOOL8 result; // rax

  v1 = *(_BYTE *)(a1 + 8);
  result = 1;
  if ( v1 <= 0xCu )
    return ((0x1581uLL >> v1) & 1) == 0;
  return result;
}
