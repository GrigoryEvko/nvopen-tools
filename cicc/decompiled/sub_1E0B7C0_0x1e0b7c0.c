// Function: sub_1E0B7C0
// Address: 0x1e0b7c0
//
_QWORD *__fastcall sub_1E0B7C0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // r13

  v2 = *(_QWORD **)(a1 + 224);
  if ( v2 )
    *(_QWORD *)(a1 + 224) = *v2;
  else
    v2 = (_QWORD *)sub_145CBF0((__int64 *)(a1 + 120), 72, 8);
  sub_1E1C580(v2, a1, a2);
  return v2;
}
