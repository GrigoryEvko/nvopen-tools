// Function: sub_134D1D0
// Address: 0x134d1d0
//
__int64 __fastcall sub_134D1D0(__int64 a1, __int64 a2, _QWORD *a3)
{
  unsigned int v4; // r13d
  _BYTE v6[80]; // [rsp+0h] [rbp-50h] BYREF

  if ( !*a3 )
    return 7;
  sub_141F0A0(v6);
  v4 = sub_134CB50(a1, (__int64)v6, (__int64)a3);
  if ( !(_BYTE)v4 || (unsigned __int8)sub_134CBB0(a1, (__int64)a3, 0) )
    return 4;
  if ( (_BYTE)v4 != 3 )
    return 7;
  return v4;
}
