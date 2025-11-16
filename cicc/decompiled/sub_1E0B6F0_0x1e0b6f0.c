// Function: sub_1E0B6F0
// Address: 0x1e0b6f0
//
_QWORD *__fastcall sub_1E0B6F0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r13

  v2 = *(_QWORD **)(a1 + 312);
  if ( v2 )
    *(_QWORD *)(a1 + 312) = *v2;
  else
    v2 = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 120), 200, 8);
  sub_1DD5940((__int64)v2, a1, a2);
  return v2;
}
