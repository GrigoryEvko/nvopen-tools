// Function: sub_1643F60
// Address: 0x1643f60
//
_BOOL8 __fastcall sub_1643F60(__int64 a1)
{
  unsigned __int8 v1; // cl
  _BOOL8 result; // rax

  v1 = *(_BYTE *)(a1 + 8);
  result = 1;
  if ( v1 <= 0xAu )
    return ((0x581uLL >> v1) & 1) == 0;
  return result;
}
