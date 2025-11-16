// Function: sub_101CAB0
// Address: 0x101cab0
//
unsigned __int8 *__fastcall sub_101CAB0(int a1, __int64 *a2, __int64 *a3, const __m128i *a4, int a5)
{
  __int64 v5; // rdx
  __int64 *v8; // r13
  __int64 v9; // rax
  __int64 **v10; // rbx
  unsigned int v11; // r8d
  unsigned __int8 *v12; // r15
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  unsigned __int64 v15; // xmm2_8
  __m128i v16; // xmm3
  unsigned __int8 *v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  int v20; // edx
  unsigned __int64 v21; // rax
  __m128i v22; // xmm4
  __m128i v23; // xmm5
  unsigned __int64 v24; // xmm6_8
  __m128i v25; // xmm7
  __int64 **v29; // [rsp+10h] [rbp-90h]
  unsigned int v30; // [rsp+1Ch] [rbp-84h]
  unsigned int v31; // [rsp+1Ch] [rbp-84h]
  __m128i v32; // [rsp+20h] [rbp-80h] BYREF
  __m128i v33; // [rsp+30h] [rbp-70h]
  unsigned __int64 v34; // [rsp+40h] [rbp-60h]
  unsigned __int64 v35; // [rsp+48h] [rbp-58h]
  __m128i v36; // [rsp+50h] [rbp-50h]
  __int64 v37; // [rsp+60h] [rbp-40h]

  if ( !a5 )
    return 0;
  v5 = a4[1].m128i_i64[1];
  if ( *(_BYTE *)a2 != 84 )
  {
    v8 = a3;
    if ( sub_FFE760((__int64)a2, (__int64)a3, v5) )
      goto LABEL_4;
    return 0;
  }
  v8 = a2;
  if ( !sub_FFE760((__int64)a3, (__int64)a2, v5) )
    return 0;
LABEL_4:
  v9 = 32LL * (*((_DWORD *)v8 + 1) & 0x7FFFFFF);
  if ( (*((_BYTE *)v8 + 7) & 0x40) != 0 )
  {
    v10 = (__int64 **)*(v8 - 1);
    v29 = &v10[(unsigned __int64)v9 / 8];
  }
  else
  {
    v29 = (__int64 **)v8;
    v10 = (__int64 **)&v8[v9 / 0xFFFFFFFFFFFFFFF8LL];
  }
  if ( v10 == v29 )
    return 0;
  v11 = a5 - 1;
  v12 = 0;
  do
  {
    if ( v8 != *v10 )
    {
      v18 = *(_QWORD *)(*(v8 - 1)
                      + 32LL * *((unsigned int *)v8 + 18)
                      + 8LL * (unsigned int)(((__int64)v10 - *(v8 - 1)) >> 5));
      v19 = *(_QWORD *)(v18 + 48) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v19 == v18 + 48 )
      {
        v21 = 0;
      }
      else
      {
        if ( !v19 )
          BUG();
        v20 = *(unsigned __int8 *)(v19 - 24);
        v21 = v19 - 24;
        if ( (unsigned int)(v20 - 30) >= 0xB )
          v21 = 0;
      }
      if ( v8 == a2 )
      {
        v22 = _mm_loadu_si128(a4);
        v31 = v11;
        v23 = _mm_loadu_si128(a4 + 1);
        v24 = _mm_loadu_si128(a4 + 2).m128i_u64[0];
        v25 = _mm_loadu_si128(a4 + 3);
        v37 = a4[4].m128i_i64[0];
        v34 = v24;
        v35 = v21;
        v32 = v22;
        v33 = v23;
        v36 = v25;
        v17 = sub_101AFF0(a1, *v10, a3, &v32, v11);
        v11 = v31;
        if ( !v17 )
          return 0;
      }
      else
      {
        v13 = _mm_loadu_si128(a4);
        v30 = v11;
        v14 = _mm_loadu_si128(a4 + 1);
        v15 = _mm_loadu_si128(a4 + 2).m128i_u64[0];
        v16 = _mm_loadu_si128(a4 + 3);
        v37 = a4[4].m128i_i64[0];
        v34 = v15;
        v35 = v21;
        v32 = v13;
        v33 = v14;
        v36 = v16;
        v17 = sub_101AFF0(a1, a2, *v10, &v32, v11);
        v11 = v30;
        if ( !v17 )
          return 0;
      }
      if ( v12 && v17 != v12 )
        return 0;
      v12 = v17;
    }
    v10 += 4;
  }
  while ( v29 != v10 );
  return v12;
}
