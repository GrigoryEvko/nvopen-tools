// Function: sub_134D0E0
// Address: 0x134d0e0
//
__int64 __fastcall sub_134D0E0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  unsigned int v4; // r8d
  char v6; // bl
  _BYTE v8[80]; // [rsp+0h] [rbp-50h] BYREF

  v4 = 7;
  if ( byte_42880A0[8 * ((*(unsigned __int16 *)(a2 + 18) >> 7) & 7) + 1] )
    return v4;
  if ( *a3 )
  {
    sub_141EDF0(v8, a2, byte_42880A0, a4, 7);
    v6 = sub_134CB50(a1, (__int64)v8, (__int64)a3);
    if ( !v6 || (unsigned __int8)sub_134CBB0(a1, (__int64)a3, 0) )
    {
      return 4;
    }
    else
    {
      if ( v6 != 3 )
        return 6;
      return 2;
    }
  }
  return 6;
}
