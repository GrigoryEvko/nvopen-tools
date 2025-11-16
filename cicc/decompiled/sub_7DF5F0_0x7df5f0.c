// Function: sub_7DF5F0
// Address: 0x7df5f0
//
__int64 *__fastcall sub_7DF5F0(__int64 a1, __int64 a2)
{
  __int64 **v2; // rdx
  __int64 *result; // rax
  char v4; // cl

  while ( *(_BYTE *)(a1 + 140) == 12 )
    a1 = *(_QWORD *)(a1 + 160);
  v2 = *(__int64 ***)(a1 + 168);
  result = *v2;
  if ( (*(_BYTE *)(a1 - 8) & 8) != 0 )
  {
    if ( v2[5] )
      result = (__int64 *)*result;
    if ( a2 )
    {
      v4 = *(_BYTE *)(a2 + 174);
      if ( (v4 == 1 || v4 == 2)
        && (((*(_BYTE *)(a2 + 205) & 0x1C) - 8) & 0xF4) == 0
        && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 32LL) + 176LL) & 0x10) != 0 )
      {
        result = (__int64 *)*result;
      }
    }
    if ( ((_BYTE)v2[2] & 0x40) != 0 )
      return (__int64 *)*result;
  }
  return result;
}
