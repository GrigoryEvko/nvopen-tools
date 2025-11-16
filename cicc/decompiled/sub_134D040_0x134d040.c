// Function: sub_134D040
// Address: 0x134d040
//
__int64 __fastcall sub_134D040(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  unsigned int v4; // r8d
  char v6; // al
  _BYTE v8[80]; // [rsp+0h] [rbp-50h] BYREF

  v4 = 7;
  if ( byte_42880A0[8 * ((*(unsigned __int16 *)(a2 + 18) >> 7) & 7) + 1] )
    return v4;
  if ( !*a3 )
    return 5;
  sub_141EB40(v8, a2, byte_42880A0, a4, 7);
  v6 = sub_134CB50(a1, (__int64)v8, (__int64)a3);
  v4 = 4;
  if ( !v6 )
    return v4;
  if ( v6 != 3 )
    return 5;
  return 1;
}
