// Function: sub_2260EF0
// Address: 0x2260ef0
//
__int64 __fastcall sub_2260EF0(unsigned __int64 *a1, unsigned __int64 *a2, int a3)
{
  unsigned __int64 v5; // r12
  unsigned __int64 v6; // rdi
  unsigned __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // r12
  __m128i v10; // xmm1
  __int64 *v11; // rdi
  __int64 v12; // rax
  __m128i v13; // xmm2
  __m128i v14; // xmm3
  __m128i v15; // xmm4
  __m128i v16; // xmm5

  if ( a3 == 1 )
  {
    *a1 = *a2;
    return 0;
  }
  if ( a3 != 2 )
  {
    if ( a3 == 3 )
    {
      v5 = *a1;
      if ( *a1 )
      {
        v6 = *(_QWORD *)(v5 + 96);
        if ( v6 != v5 + 112 )
          j_j___libc_free_0(v6);
        j_j___libc_free_0(v5);
      }
    }
    return 0;
  }
  v7 = *a2;
  v8 = sub_22077B0(0x780u);
  v9 = v8;
  if ( v8 )
  {
    v10 = _mm_loadu_si128((const __m128i *)(v7 + 16));
    v11 = (__int64 *)(v8 + 96);
    v12 = v8 + 112;
    v13 = _mm_loadu_si128((const __m128i *)(v7 + 32));
    v14 = _mm_loadu_si128((const __m128i *)(v7 + 48));
    v15 = _mm_loadu_si128((const __m128i *)(v7 + 64));
    v16 = _mm_loadu_si128((const __m128i *)(v7 + 80));
    *(__m128i *)(v12 - 112) = _mm_loadu_si128((const __m128i *)v7);
    *(__m128i *)(v12 - 96) = v10;
    *(__m128i *)(v12 - 80) = v13;
    *(__m128i *)(v12 - 64) = v14;
    *(__m128i *)(v12 - 48) = v15;
    *(__m128i *)(v12 - 32) = v16;
    *(_QWORD *)(v9 + 96) = v12;
    sub_2260190(v11, *(_BYTE **)(v7 + 96), *(_QWORD *)(v7 + 96) + *(_QWORD *)(v7 + 104));
    qmemcpy((void *)(v9 + 128), (const void *)(v7 + 128), 0x6FCu);
  }
  *a1 = v9;
  return 0;
}
