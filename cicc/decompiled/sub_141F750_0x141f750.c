// Function: sub_141F750
// Address: 0x141f750
//
__m128i *__fastcall sub_141F750(__m128i *a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  int v4; // eax
  __m128i v5; // xmm0
  __m128i v7; // [rsp+0h] [rbp-40h] BYREF
  __int64 v8; // [rsp+10h] [rbp-30h]

  v2 = -1;
  v3 = *(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( *(_BYTE *)(v3 + 16) == 13 )
  {
    v2 = *(_QWORD *)(v3 + 24);
    if ( *(_DWORD *)(v3 + 32) > 0x40u )
      v2 = *(_QWORD *)v2;
  }
  v7 = 0u;
  v8 = 0;
  sub_14A8180(a2, &v7, 0);
  v4 = *(_DWORD *)(a2 + 20);
  v5 = _mm_loadu_si128(&v7);
  a1->m128i_i64[1] = v2;
  a1[1] = v5;
  a1->m128i_i64[0] = *(_QWORD *)(a2 - 24LL * (v4 & 0xFFFFFFF));
  a1[2].m128i_i64[0] = v8;
  return a1;
}
