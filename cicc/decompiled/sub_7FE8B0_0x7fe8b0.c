// Function: sub_7FE8B0
// Address: 0x7fe8b0
//
_QWORD *__fastcall sub_7FE8B0(__int64 a1, int *a2)
{
  __m128i *v2; // r13
  __int64 v3; // r12
  _BYTE *v5; // rax
  void *v6; // rax
  _BYTE v7[64]; // [rsp+0h] [rbp-40h] BYREF

  v2 = (__m128i *)a2;
  v3 = *(_QWORD *)(a1 + 80);
  qword_4D03F68[6] = *(_QWORD *)(v3 + 80);
  if ( (*(_BYTE *)(a1 + 49) & 4) != 0 )
  {
    v5 = sub_73E830(*(_QWORD *)(v3 + 88));
    v6 = sub_7F0830(v5);
    v2 = (__m128i *)v7;
    sub_7F8BA0((__int64)v6, 0, a2, 0, (__int64)v7, 0);
  }
  return sub_7FE6E0(*(_QWORD *)(a1 + 16), v3 + 8, 1, 0, v2);
}
