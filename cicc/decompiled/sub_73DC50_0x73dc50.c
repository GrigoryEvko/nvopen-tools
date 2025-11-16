// Function: sub_73DC50
// Address: 0x73dc50
//
_BYTE *__fastcall sub_73DC50(__int64 a1, __int64 a2)
{
  _BYTE *v2; // r12

  v2 = sub_73DC30(8u, a2, a1);
  sub_730580(a1, (__int64)v2);
  v2[27] |= 2u;
  *(_QWORD *)(v2 + 28) = *(_QWORD *)(a1 + 28);
  return v2;
}
