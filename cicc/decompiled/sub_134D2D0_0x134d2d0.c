// Function: sub_134D2D0
// Address: 0x134d2d0
//
__int64 __fastcall sub_134D2D0(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int64 result; // rax
  _BYTE v5[80]; // [rsp+0h] [rbp-50h] BYREF

  if ( byte_42880A0[8 * ((*(unsigned __int16 *)(a2 + 18) >> 2) & 7) + 2] || !*a3 )
    return 7;
  sub_141F110(v5);
  result = sub_134CB50(a1, (__int64)v5, (__int64)a3);
  if ( !(_BYTE)result )
    return 4;
  if ( (_BYTE)result != 3 )
    return 7;
  return result;
}
