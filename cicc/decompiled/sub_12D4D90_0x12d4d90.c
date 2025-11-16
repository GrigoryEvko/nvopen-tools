// Function: sub_12D4D90
// Address: 0x12d4d90
//
__int64 __fastcall sub_12D4D90(__int64 *a1, __int64 *a2, int a3)
{
  _QWORD *v5; // r12
  __int64 v6; // rdi
  _QWORD *v7; // rdi
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r12
  __m128i v11; // xmm2
  __int64 *v12; // rdi
  __int64 v13; // rax
  __m128i v14; // xmm3
  __m128i v15; // xmm4
  __m128i v16; // xmm5
  __int64 v17; // rdx
  __int64 v18; // rax
  unsigned __int64 v19; // r14
  __m128i *v20; // rax
  const __m128i *v21; // rcx
  const __m128i *v22; // rdx
  __m128i *v23; // rcx

  if ( a3 == 1 )
  {
    *a1 = *a2;
    return 0;
  }
  if ( a3 != 2 )
  {
    if ( a3 == 3 )
    {
      v5 = (_QWORD *)*a1;
      if ( *a1 )
      {
        v6 = v5[575];
        if ( v6 )
          j_j___libc_free_0(v6, v5[577] - v6);
        v7 = (_QWORD *)v5[10];
        if ( v7 != v5 + 12 )
          j_j___libc_free_0(v7, v5[12] + 1LL);
        j_j___libc_free_0(v5, 4632);
      }
    }
    return 0;
  }
  v8 = *a2;
  v9 = sub_22077B0(4632);
  v10 = v9;
  if ( v9 )
  {
    v11 = _mm_loadu_si128((const __m128i *)(v8 + 16));
    v12 = (__int64 *)(v9 + 80);
    v13 = v9 + 96;
    v14 = _mm_loadu_si128((const __m128i *)(v8 + 32));
    v15 = _mm_loadu_si128((const __m128i *)(v8 + 48));
    v16 = _mm_loadu_si128((const __m128i *)(v8 + 64));
    *(__m128i *)(v13 - 96) = _mm_loadu_si128((const __m128i *)v8);
    *(__m128i *)(v13 - 80) = v11;
    *(__m128i *)(v13 - 64) = v14;
    *(__m128i *)(v13 - 48) = v15;
    *(__m128i *)(v13 - 32) = v16;
    *(_QWORD *)(v10 + 80) = v13;
    sub_12D3F10(v12, *(_BYTE **)(v8 + 80), *(_QWORD *)(v8 + 80) + *(_QWORD *)(v8 + 88));
    v18 = *(_QWORD *)(v8 + 4592);
    qmemcpy((void *)(v10 + 112), (const void *)(v8 + 112), 0x1180u);
    *(_QWORD *)(v10 + 4592) = v18;
    v19 = *(_QWORD *)(v8 + 4608) - *(_QWORD *)(v8 + 4600);
    *(_QWORD *)(v10 + 4600) = 0;
    *(_QWORD *)(v10 + 4608) = 0;
    *(_QWORD *)(v10 + 4616) = 0;
    if ( v19 )
    {
      if ( v19 > 0x7FFFFFFFFFFFFFF0LL )
        sub_4261EA(v10 + 4592, v8 + 4592, v17);
      v20 = (__m128i *)sub_22077B0(v19);
    }
    else
    {
      v19 = 0;
      v20 = 0;
    }
    *(_QWORD *)(v10 + 4600) = v20;
    *(_QWORD *)(v10 + 4608) = v20;
    *(_QWORD *)(v10 + 4616) = (char *)v20 + v19;
    v21 = *(const __m128i **)(v8 + 4608);
    v22 = *(const __m128i **)(v8 + 4600);
    if ( v21 == v22 )
    {
      v23 = v20;
    }
    else
    {
      v23 = (__m128i *)((char *)v20 + (char *)v21 - (char *)v22);
      do
      {
        if ( v20 )
          *v20 = _mm_loadu_si128(v22);
        ++v20;
        ++v22;
      }
      while ( v23 != v20 );
    }
    *(_QWORD *)(v10 + 4608) = v23;
    *(_DWORD *)(v10 + 4624) = *(_DWORD *)(v8 + 4624);
  }
  *a1 = v10;
  return 0;
}
