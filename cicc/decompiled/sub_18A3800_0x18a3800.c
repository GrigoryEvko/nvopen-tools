// Function: sub_18A3800
// Address: 0x18a3800
//
signed __int64 __fastcall sub_18A3800(__m128i *a1, __int64 *a2, __int64 a3)
{
  signed __int64 result; // rax
  __m128i *v4; // r8
  __int64 v5; // r15
  __m128i *v6; // r12
  unsigned __int64 v8; // rdi
  __m128i *v9; // r9
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rax
  bool v12; // dl
  __int64 v13; // rsi
  __int64 v14; // rcx
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  __m128i *v18; // r13
  __m128i *v19; // rax
  __int64 *v20; // r14
  bool v21; // cl
  __int64 v22; // rdx
  bool v23; // dl
  __int64 v24; // rcx
  __m128i v25; // xmm3
  bool v26; // dl
  bool v27; // dl
  __int64 v28; // r12
  __int64 v29; // rsi
  __int64 v30; // rcx
  __int64 v31; // r8
  __m128i v32; // xmm5

  result = (char *)a2 - (char *)a1;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  v4 = (__m128i *)a2;
  v5 = a3;
  v6 = a1 + 1;
  if ( !a3 )
  {
    v20 = a2;
    goto LABEL_36;
  }
  while ( 2 )
  {
    v8 = a1[1].m128i_u64[1];
    --v5;
    v9 = &a1[result >> 5];
    v10 = v9->m128i_u64[1];
    v11 = v4[-1].m128i_u64[1];
    if ( *(_OWORD *)&a1[1] <= *(_OWORD *)v9 )
    {
      v23 = v8 > v11;
      if ( v8 == v11 )
        v23 = a1[1].m128i_i64[0] > (unsigned __int64)v4[-1].m128i_i64[0];
      v24 = a1->m128i_i64[0];
      v15 = a1->m128i_u64[1];
      if ( v23 )
      {
        v25 = _mm_loadu_si128(a1 + 1);
        a1[1].m128i_i64[0] = v24;
        a1[1].m128i_i64[1] = v15;
        *a1 = v25;
        v16 = v4[-1].m128i_u64[1];
      }
      else
      {
        v27 = v10 > v11;
        if ( v10 == v11 )
          v27 = v9->m128i_i64[0] > (unsigned __int64)v4[-1].m128i_i64[0];
        if ( v27 )
        {
          v16 = a1->m128i_u64[1];
          *a1 = _mm_loadu_si128(v4 - 1);
          v4[-1].m128i_i64[0] = v24;
          v4[-1].m128i_i64[1] = v15;
          v15 = a1[1].m128i_u64[1];
        }
        else
        {
          *a1 = _mm_loadu_si128(v9);
          v9->m128i_i64[0] = v24;
          v9->m128i_i64[1] = v15;
          v15 = a1[1].m128i_u64[1];
          v16 = v4[-1].m128i_u64[1];
        }
      }
    }
    else
    {
      v12 = v10 > v11;
      if ( v10 == v11 )
        v12 = v9->m128i_i64[0] > (unsigned __int64)v4[-1].m128i_i64[0];
      v13 = a1->m128i_i64[0];
      v14 = a1->m128i_i64[1];
      if ( v12 )
      {
        *a1 = _mm_loadu_si128(v9);
        v9->m128i_i64[0] = v13;
        v9->m128i_i64[1] = v14;
        v15 = a1[1].m128i_u64[1];
        v16 = v4[-1].m128i_u64[1];
      }
      else
      {
        v26 = v8 > v11;
        if ( v8 == v11 )
          v26 = a1[1].m128i_i64[0] > (unsigned __int64)v4[-1].m128i_i64[0];
        if ( v26 )
        {
          v16 = a1->m128i_u64[1];
          *a1 = _mm_loadu_si128(v4 - 1);
          v4[-1].m128i_i64[0] = v13;
          v4[-1].m128i_i64[1] = v14;
          v15 = a1[1].m128i_u64[1];
        }
        else
        {
          v32 = _mm_loadu_si128(a1 + 1);
          v15 = a1->m128i_u64[1];
          a1[1].m128i_i64[0] = v13;
          a1[1].m128i_i64[1] = v14;
          *a1 = v32;
          v16 = v4[-1].m128i_u64[1];
        }
      }
    }
    v17 = a1->m128i_u64[1];
    v18 = v6;
    v19 = v4;
    while ( 1 )
    {
      v20 = (__int64 *)v18;
      if ( v17 == v15 )
      {
        if ( v18->m128i_i64[0] > (unsigned __int64)a1->m128i_i64[0] )
          goto LABEL_10;
      }
      else if ( v17 < v15 )
      {
        goto LABEL_10;
      }
      for ( --v19; ; --v19 )
      {
        v21 = v17 > v16;
        if ( v17 == v16 )
          v21 = a1->m128i_i64[0] > (unsigned __int64)v19->m128i_i64[0];
        if ( !v21 )
          break;
        v16 = v19[-1].m128i_u64[1];
      }
      if ( v18 >= v19 )
        break;
      v22 = v18->m128i_i64[0];
      *v18 = _mm_loadu_si128(v19);
      v19->m128i_i64[0] = v22;
      v16 = v19[-1].m128i_u64[1];
      v19->m128i_i64[1] = v15;
      v17 = a1->m128i_u64[1];
LABEL_10:
      v15 = v18[1].m128i_u64[1];
      ++v18;
    }
    sub_18A3800(v18, v4, v5);
    result = (char *)v18 - (char *)a1;
    if ( (char *)v18 - (char *)a1 > 256 )
    {
      if ( v5 )
      {
        v4 = v18;
        continue;
      }
LABEL_36:
      v28 = result >> 4;
      v29 = ((result >> 4) - 2) >> 1;
      sub_18A35E0((__int64)a1, v29, result >> 4, a1[v29].m128i_i64[0], a1[v29].m128i_i64[1]);
      do
      {
        --v29;
        sub_18A35E0((__int64)a1, v29, v28, a1[v29].m128i_i64[0], a1[v29].m128i_i64[1]);
      }
      while ( v29 );
      do
      {
        v20 -= 2;
        v30 = *v20;
        v31 = v20[1];
        *(__m128i *)v20 = _mm_loadu_si128(a1);
        result = (signed __int64)sub_18A35E0((__int64)a1, 0, ((char *)v20 - (char *)a1) >> 4, v30, v31);
      }
      while ( (char *)v20 - (char *)a1 > 16 );
    }
    return result;
  }
}
