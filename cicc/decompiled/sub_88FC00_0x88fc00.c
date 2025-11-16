// Function: sub_88FC00
// Address: 0x88fc00
//
__int64 *__fastcall sub_88FC00(__int64 a1, __int64 **a2, int a3)
{
  __int64 *result; // rax

  for ( result = *(__int64 **)(a1 + 240); a2; result = (__int64 *)*result )
  {
    if ( ((_BYTE)a2[3] & 2) != 0 )
      *((_BYTE *)result + 24) |= 2u;
    a2 = (__int64 **)*a2;
  }
  if ( a3 || (result = *(__int64 **)(a1 + 240), (result[3] & 2) != 0) )
    *(_BYTE *)(a1 + 203) |= 0x10u;
  return result;
}
