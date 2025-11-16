// Function: sub_C16ED0
// Address: 0xc16ed0
//
__int64 __fastcall sub_C16ED0(__int64 a1, __int64 a2)
{
  __int64 v3; // rsi
  __int64 v5; // rdx
  __int64 *v6; // rax
  __int64 *v7; // r12
  __m128i v8; // xmm0
  __int64 v9; // rdi
  __int64 v10; // rbx
  __int64 v11; // r15
  _QWORD *v12; // r14
  int v13; // ebx
  __int64 v15; // [rsp+8h] [rbp-58h]
  __int64 *v16; // [rsp+10h] [rbp-50h]
  __int64 *v17; // [rsp+18h] [rbp-48h]
  _QWORD v18[7]; // [rsp+28h] [rbp-38h] BYREF

  v3 = a1 + 16;
  v16 = (__int64 *)(a1 + 16);
  v15 = sub_C8D7D0(a1, a1 + 16, a2, 184, v18);
  v5 = v15;
  v6 = *(__int64 **)a1;
  v7 = (__int64 *)(*(_QWORD *)a1 + 184LL * *(unsigned int *)(a1 + 8));
  if ( *(__int64 **)a1 != v7 )
  {
    do
    {
      if ( v5 )
      {
        *(_QWORD *)v5 = *v6;
        *(_QWORD *)(v5 + 8) = v6[1];
        *(_QWORD *)(v5 + 16) = v6[2];
        v8 = _mm_loadu_si128((const __m128i *)(v6 + 3));
        v6[2] = 0;
        v6[1] = 0;
        *v6 = 0;
        *(__m128i *)(v5 + 24) = v8;
        *(__m128i *)(v5 + 40) = _mm_loadu_si128((const __m128i *)(v6 + 5));
        *(__m128i *)(v5 + 56) = _mm_loadu_si128((const __m128i *)(v6 + 7));
        *(__m128i *)(v5 + 72) = _mm_loadu_si128((const __m128i *)(v6 + 9));
        *(__m128i *)(v5 + 88) = _mm_loadu_si128((const __m128i *)(v6 + 11));
        *(__m128i *)(v5 + 104) = _mm_loadu_si128((const __m128i *)(v6 + 13));
        *(__m128i *)(v5 + 120) = _mm_loadu_si128((const __m128i *)(v6 + 15));
        *(__m128i *)(v5 + 136) = _mm_loadu_si128((const __m128i *)(v6 + 17));
        *(__m128i *)(v5 + 152) = _mm_loadu_si128((const __m128i *)(v6 + 19));
        *(__m128i *)(v5 + 168) = _mm_loadu_si128((const __m128i *)(v6 + 21));
      }
      v6 += 23;
      v5 += 184;
    }
    while ( v7 != v6 );
    v17 = *(__int64 **)a1;
    v7 = (__int64 *)(*(_QWORD *)a1 + 184LL * *(unsigned int *)(a1 + 8));
    if ( *(__int64 **)a1 != v7 )
    {
      do
      {
        v9 = *(v7 - 23);
        v10 = *(v7 - 22);
        v7 -= 23;
        v11 = v9;
        if ( v10 != v9 )
        {
          do
          {
            v12 = *(_QWORD **)(v11 + 8);
            if ( v12 )
            {
              if ( (_QWORD *)*v12 != v12 + 2 )
                j_j___libc_free_0(*v12, v12[2] + 1LL);
              v3 = 32;
              j_j___libc_free_0(v12, 32);
            }
            v11 += 32;
          }
          while ( v10 != v11 );
          v9 = *v7;
        }
        if ( v9 )
        {
          v3 = v7[2] - v9;
          j_j___libc_free_0(v9, v3);
        }
      }
      while ( v17 != v7 );
      v7 = *(__int64 **)a1;
    }
  }
  v13 = v18[0];
  if ( v16 != v7 )
    _libc_free(v7, v3);
  *(_DWORD *)(a1 + 12) = v13;
  *(_QWORD *)a1 = v15;
  return v15;
}
