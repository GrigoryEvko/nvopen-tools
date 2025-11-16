// Function: sub_29B8DF0
// Address: 0x29b8df0
//
unsigned __int64 __fastcall sub_29B8DF0(unsigned __int64 *a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 result; // rax
  unsigned __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r12
  __m128i v8; // xmm0
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // [rsp+0h] [rbp-40h]
  __int64 v11; // [rsp+8h] [rbp-38h]

  if ( a2 > 0x111111111111111LL )
    sub_4262D8((__int64)"vector::reserve");
  v2 = *a1;
  result = 0xEEEEEEEEEEEEEEEFLL * ((__int64)(a1[2] - *a1) >> 3);
  if ( a2 > result )
  {
    v5 = a1[1];
    v11 = 0;
    v10 = v5 - v2;
    if ( a2 )
    {
      v6 = sub_22077B0(120 * a2);
      v5 = a1[1];
      v2 = *a1;
      v11 = v6;
    }
    if ( v2 != v5 )
    {
      v7 = v11;
      do
      {
        if ( v7 )
        {
          *(_QWORD *)v7 = *(_QWORD *)v2;
          *(_QWORD *)(v7 + 8) = *(_QWORD *)(v2 + 8);
          *(_QWORD *)(v7 + 16) = *(_QWORD *)(v2 + 16);
          *(_QWORD *)(v7 + 24) = *(_QWORD *)(v2 + 24);
          *(_QWORD *)(v7 + 32) = *(_QWORD *)(v2 + 32);
          v8 = _mm_loadu_si128((const __m128i *)(v2 + 40));
          *(_QWORD *)(v2 + 32) = 0;
          *(_QWORD *)(v2 + 24) = 0;
          *(_QWORD *)(v2 + 16) = 0;
          *(__m128i *)(v7 + 40) = v8;
          *(_QWORD *)(v7 + 56) = *(_QWORD *)(v2 + 56);
          *(__m128i *)(v7 + 64) = _mm_loadu_si128((const __m128i *)(v2 + 64));
          *(_QWORD *)(v7 + 80) = *(_QWORD *)(v2 + 80);
          *(__m128i *)(v7 + 88) = _mm_loadu_si128((const __m128i *)(v2 + 88));
          *(_QWORD *)(v7 + 104) = *(_QWORD *)(v2 + 104);
          *(_BYTE *)(v7 + 112) = *(_BYTE *)(v2 + 112);
          *(_BYTE *)(v7 + 113) = *(_BYTE *)(v2 + 113);
        }
        v9 = *(_QWORD *)(v2 + 16);
        if ( v9 )
          j_j___libc_free_0(v9);
        v2 += 120LL;
        v7 += 120;
      }
      while ( v2 != v5 );
      v5 = *a1;
    }
    if ( v5 )
      j_j___libc_free_0(v5);
    result = v11 + v10;
    *a1 = v11;
    a1[1] = v11 + v10;
    a1[2] = v11 + 120 * a2;
  }
  return result;
}
