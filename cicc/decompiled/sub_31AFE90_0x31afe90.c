// Function: sub_31AFE90
// Address: 0x31afe90
//
__m128i *__fastcall sub_31AFE90(__m128i *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 *v4; // r14
  __int64 v5; // r13
  __int64 *v7; // rbx
  __int64 v8; // rax
  __int64 *v9; // r13
  __int64 v10; // r15
  __int64 v11; // r15
  __int64 v12; // r15
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 v15; // r15
  __int64 i; // rbx
  __int64 v17; // rax
  __m128i v18; // xmm1
  __m128i v19; // xmm3
  __int64 v21; // rbx
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 j; // rbx
  __int64 v25; // rax
  __m128i v26; // xmm5
  __m128i v27; // xmm7
  __int64 v28; // r13
  __int64 v29; // r13
  __int64 v30; // r13
  __m128i v31; // [rsp+10h] [rbp-90h] BYREF
  __m128i v32; // [rsp+20h] [rbp-80h] BYREF
  __m128i v33; // [rsp+30h] [rbp-70h] BYREF
  __m128i v34; // [rsp+40h] [rbp-60h] BYREF
  __m128i v35; // [rsp+50h] [rbp-50h] BYREF
  __m128i v36[4]; // [rsp+60h] [rbp-40h] BYREF

  v4 = &a2[a3];
  v5 = (8 * a3) >> 5;
  v7 = a2;
  v8 = (8 * a3) >> 3;
  if ( v5 <= 0 )
  {
LABEL_32:
    if ( v8 != 2 )
    {
      if ( v8 != 3 )
      {
        if ( v8 != 1 )
          goto LABEL_35;
        goto LABEL_47;
      }
      v28 = *v7;
      if ( sub_318B630(*v7) && a4 == sub_318B4F0(v28) )
        goto LABEL_9;
      ++v7;
    }
    v29 = *v7;
    if ( sub_318B630(*v7) && a4 == sub_318B4F0(v29) )
      goto LABEL_9;
    ++v7;
LABEL_47:
    v30 = *v7;
    if ( !sub_318B630(*v7) || a4 != sub_318B4F0(v30) )
      goto LABEL_35;
    goto LABEL_9;
  }
  v9 = &a2[4 * v5];
  while ( 1 )
  {
    v13 = *v7;
    if ( sub_318B630(*v7) && a4 == sub_318B4F0(v13) )
      break;
    v10 = v7[1];
    if ( sub_318B630(v10) && a4 == sub_318B4F0(v10) )
    {
      ++v7;
      break;
    }
    v11 = v7[2];
    if ( sub_318B630(v11) && a4 == sub_318B4F0(v11) )
    {
      v7 += 2;
      break;
    }
    v12 = v7[3];
    if ( sub_318B630(v12) && a4 == sub_318B4F0(v12) )
    {
      v7 += 3;
      break;
    }
    v7 += 4;
    if ( v7 == v9 )
    {
      v8 = v4 - v7;
      goto LABEL_32;
    }
  }
LABEL_9:
  if ( v4 != v7 )
  {
    v14 = *v7;
    while ( v4 != ++v7 )
    {
      while ( 1 )
      {
        v15 = *v7;
        if ( !sub_318B630(*v7) || !v15 || a4 != sub_318B4F0(v15) )
          break;
        if ( sub_B445A0(*(_QWORD *)(v14 + 16), *(_QWORD *)(v15 + 16)) )
          v14 = v15;
        if ( v4 == ++v7 )
          goto LABEL_18;
      }
    }
LABEL_18:
    if ( v14 )
    {
      for ( i = v14; sub_318B700(i); i = v17 )
      {
        v14 = i;
        v17 = sub_318B4B0(i);
        if ( !v17 )
          break;
      }
      sub_318B480((__int64)&v33, v14);
      v18 = _mm_loadu_si128(&v34);
      v35 = _mm_loadu_si128(&v33);
      v36[0] = v18;
      sub_371B2F0(&v35);
      v19 = _mm_loadu_si128(v36);
      *a1 = _mm_loadu_si128(&v35);
      a1[1] = v19;
      return a1;
    }
  }
LABEL_35:
  v21 = *(_QWORD *)(a4 + 16) + 48LL;
  sub_371B570(&v35, a4);
  if ( v21 == v35.m128i_i64[1] )
  {
    sub_371B570(a1, a4);
  }
  else
  {
    sub_371B570(&v33, a4);
    v22 = sub_371B3B0(&v33, v33.m128i_i64[1], v34.m128i_i64[0]);
    v23 = v22;
    if ( v22 )
    {
      for ( j = v22; sub_318B700(j); j = v25 )
      {
        v23 = j;
        v25 = sub_318B4B0(j);
        if ( !v25 )
          break;
      }
    }
    sub_318B480((__int64)&v31, v23);
    v26 = _mm_loadu_si128(&v32);
    v35 = _mm_loadu_si128(&v31);
    v36[0] = v26;
    sub_371B2F0(&v35);
    v27 = _mm_loadu_si128(v36);
    *a1 = _mm_loadu_si128(&v35);
    a1[1] = v27;
  }
  return a1;
}
