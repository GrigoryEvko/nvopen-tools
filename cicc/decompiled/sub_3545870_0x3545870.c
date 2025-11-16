// Function: sub_3545870
// Address: 0x3545870
//
void __fastcall sub_3545870(_QWORD *a1, __int64 a2)
{
  const __m128i *v2; // r14
  const __m128i *v3; // r13
  __int64 v4; // rax
  __m128i v5; // xmm0
  const __m128i *v6; // r14
  const __m128i *v7; // r13
  __m128i v8; // xmm1
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // [rsp+0h] [rbp-50h] BYREF
  _BYTE v11[20]; // [rsp+8h] [rbp-48h]

  v2 = *(const __m128i **)(a2 + 40);
  v3 = &v2[*(unsigned int *)(a2 + 48)];
  while ( v3 != v2 )
  {
    v10 = a2;
    v4 = v2->m128i_i64[0];
    v5 = _mm_loadu_si128(v2);
    *(_DWORD *)&v11[16] = 0;
    *(__m128i *)v11 = v5;
    if ( ((v4 >> 1) & 3) == 1
      && (*(_WORD *)(*(_QWORD *)(v2->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 68LL) == 68
       || !*(_WORD *)(*(_QWORD *)(v2->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) + 68LL)) )
    {
      v10 = v2->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v11 = a2 & 0xFFFFFFFFFFFFFFF9LL;
      *(_QWORD *)&v11[12] = 0x100000001LL;
    }
    ++v2;
    sub_35456F0(a1, a2, (const __m128i *)&v10);
  }
  v6 = *(const __m128i **)(a2 + 120);
  v7 = &v6[*(unsigned int *)(a2 + 128)];
  while ( v7 != v6 )
  {
    v10 = a2;
    v8 = _mm_loadu_si128(v6);
    *(_DWORD *)&v11[16] = 0;
    *(__m128i *)v11 = v8;
    v9 = v6->m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL;
    v10 = v9;
    *(_QWORD *)v11 = a2 | v8.m128i_i8[0] & 7;
    if ( ((*(__int64 *)v11 >> 1) & 3) == 1
      && (!*(_WORD *)(*(_QWORD *)a2 + 68LL) || *(_WORD *)(*(_QWORD *)a2 + 68LL) == 68) )
    {
      v10 = a2;
      *(_QWORD *)v11 = v9;
      *(_QWORD *)&v11[12] = 0x100000001LL;
    }
    ++v6;
    sub_35456F0(a1, a2, (const __m128i *)&v10);
  }
}
