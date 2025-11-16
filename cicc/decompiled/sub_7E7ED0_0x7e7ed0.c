// Function: sub_7E7ED0
// Address: 0x7e7ed0
//
__m128i *__fastcall sub_7E7ED0(const __m128i *a1)
{
  __int64 v1; // r14
  __m128i *v2; // r13
  _BYTE *v3; // rbx
  int v4; // eax

  v1 = a1->m128i_i64[0];
  v2 = sub_7E7CA0(a1->m128i_i64[0]);
  if ( (*(_BYTE *)(v1 + 140) & 0xFB) == 8 && (sub_8D4C10(v1, dword_4F077C4 != 2) & 1) != 0 )
    v2[10].m128i_i8[13] |= 8u;
  v3 = sub_731250((__int64)v2);
  *((_QWORD *)v3 + 2) = sub_730FF0(a1);
  sub_7264E0((__int64)a1, 1);
  v4 = sub_8D3410(v1);
  sub_73D8E0((__int64)a1, v4 == 0 ? 73 : 86, v1, 0, (__int64)v3);
  return v2;
}
