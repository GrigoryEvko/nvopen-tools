// Function: sub_1ACFB90
// Address: 0x1acfb90
//
const void *__fastcall sub_1ACFB90(__int64 a1, __int64 a2, __int64 a3, const void *a4, size_t a5)
{
  size_t v5; // r15
  __int64 v6; // r13
  __int64 i; // r15
  int v9; // eax
  size_t v10; // r10
  __int64 v11; // r9
  size_t v12; // r11
  __m128i *v13; // rbx
  __int64 v14; // r13
  const void *v15; // rdi
  __int64 v16; // r12
  const void *v17; // rsi
  __int64 v18; // r12
  size_t v19; // rax
  __int64 v20; // r15
  size_t v21; // r14
  int v22; // eax
  size_t v23; // r9
  const void *v24; // rdi
  __int64 v28; // [rsp+18h] [rbp-68h]
  size_t v29; // [rsp+20h] [rbp-60h]
  __int64 v30; // [rsp+20h] [rbp-60h]
  size_t v31; // [rsp+30h] [rbp-50h]
  size_t v32; // [rsp+30h] [rbp-50h]
  __int64 v34; // [rsp+40h] [rbp-40h]
  size_t v35; // [rsp+40h] [rbp-40h]
  __int64 v36; // [rsp+40h] [rbp-40h]

  v5 = a5;
  v6 = a1;
  v28 = a3 & 1;
  v34 = (a3 - 1) / 2;
  if ( a2 < v34 )
  {
    for ( i = a2; ; i = v14 )
    {
      v14 = 2 * (i + 1);
      v11 = v14 - 1;
      v13 = (__m128i *)(a1 + 32 * (i + 1));
      v10 = v13->m128i_u64[1];
      v15 = (const void *)v13->m128i_i64[0];
      v16 = a1 + 16 * (v14 - 1);
      v12 = *(_QWORD *)(v16 + 8);
      v17 = *(const void **)v16;
      if ( v10 <= v12 )
      {
        if ( v10 )
        {
          v29 = *(_QWORD *)(v16 + 8);
          v31 = v13->m128i_u64[1];
          v9 = memcmp(v15, v17, v31);
          v10 = v31;
          v11 = v14 - 1;
          v12 = v29;
          if ( v9 )
            goto LABEL_13;
        }
        if ( v10 == v12 )
          goto LABEL_8;
      }
      else
      {
        if ( !v12 )
          goto LABEL_8;
        v30 = v13->m128i_i64[1];
        v32 = *(_QWORD *)(v16 + 8);
        v9 = memcmp(v15, v17, v32);
        v12 = v32;
        v11 = v14 - 1;
        v10 = v30;
        if ( v9 )
        {
LABEL_13:
          if ( v9 < 0 )
          {
            v13 = (__m128i *)(a1 + 16 * (v14 - 1));
            v14 = v11;
          }
          goto LABEL_8;
        }
      }
      if ( v10 < v12 )
      {
        v13 = (__m128i *)(a1 + 16 * (v14 - 1));
        v14 = v11;
      }
LABEL_8:
      *(__m128i *)(a1 + 16 * i) = _mm_loadu_si128(v13);
      if ( v14 >= v34 )
      {
        v18 = v14;
        v5 = a5;
        v6 = a1;
        if ( v28 )
          goto LABEL_17;
        goto LABEL_33;
      }
    }
  }
  v13 = (__m128i *)(a1 + 16 * a2);
  if ( (a3 & 1) == 0 )
  {
    v18 = a2;
LABEL_33:
    if ( (a3 - 2) / 2 == v18 )
    {
      v18 = 2 * v18 + 1;
      *v13 = _mm_loadu_si128((const __m128i *)(v6 + 16 * v18));
      v13 = (__m128i *)(v6 + 16 * v18);
    }
LABEL_17:
    if ( v18 > a2 )
    {
      v19 = v5;
      v20 = (v18 - 1) / 2;
      v21 = v19;
      while ( 1 )
      {
        v13 = (__m128i *)(v6 + 16 * v20);
        v23 = v13->m128i_u64[1];
        v24 = (const void *)v13->m128i_i64[0];
        if ( v23 <= v21 )
        {
          if ( !v23 || (v35 = v13->m128i_u64[1], v22 = memcmp(v24, a4, v35), v23 = v35, !v22) )
          {
            if ( v23 == v21 )
              goto LABEL_29;
LABEL_22:
            if ( v23 >= v21 )
              goto LABEL_29;
            goto LABEL_23;
          }
        }
        else
        {
          if ( !v21 )
            goto LABEL_29;
          v36 = v13->m128i_i64[1];
          v22 = memcmp(v24, a4, v21);
          v23 = v36;
          if ( !v22 )
            goto LABEL_22;
        }
        if ( v22 >= 0 )
        {
LABEL_29:
          v5 = v21;
          v13 = (__m128i *)(v6 + 16 * v18);
          break;
        }
LABEL_23:
        *(__m128i *)(v6 + 16 * v18) = _mm_loadu_si128(v13);
        v18 = v20;
        if ( a2 >= v20 )
        {
          v5 = v21;
          break;
        }
        v20 = (v20 - 1) / 2;
      }
    }
  }
  v13->m128i_i64[1] = v5;
  v13->m128i_i64[0] = (__int64)a4;
  return a4;
}
