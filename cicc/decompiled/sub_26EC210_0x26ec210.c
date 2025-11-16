// Function: sub_26EC210
// Address: 0x26ec210
//
_QWORD *__fastcall sub_26EC210(unsigned __int64 *a1, __m128i *a2)
{
  unsigned __int64 v2; // r12
  unsigned __int64 v3; // r14
  __int64 *v4; // rax
  __int64 v5; // rax
  __int64 v7; // rax
  __m128i v8; // xmm0

  v2 = a2->m128i_i64[1] + 31 * a2->m128i_i64[0];
  v3 = v2 % a1[1];
  v4 = sub_26EC1A0(a1, v3, a2, v2);
  if ( v4 )
  {
    v5 = *v4;
    if ( v5 )
      return (_QWORD *)(v5 + 24);
  }
  v7 = sub_22077B0(0x28u);
  if ( v7 )
    *(_QWORD *)v7 = 0;
  v8 = _mm_loadu_si128(a2);
  *(_DWORD *)(v7 + 24) = 0;
  *(__m128i *)(v7 + 8) = v8;
  return sub_26E92A0(a1, v3, v2, (_QWORD *)v7, 1) + 3;
}
