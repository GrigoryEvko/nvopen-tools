// Function: sub_1AD04B0
// Address: 0x1ad04b0
//
__int64 __fastcall sub_1AD04B0(__m128i *a1, __m128i *a2, __int64 a3)
{
  __int64 result; // rax
  size_t v4; // r12
  void *v5; // r9
  const void *v6; // r8
  size_t v7; // r14
  __m128i *v8; // rbx
  size_t v9; // r13
  const void *v10; // r15
  int v11; // eax
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rax
  void *v15; // rcx
  __int64 v16; // r10
  size_t v17; // r12
  __m128i *v18; // rbx
  __m128i *v19; // r15
  const void *v20; // r13
  size_t v21; // r14
  size_t v22; // r12
  int v23; // eax
  void *v24; // rax
  const void *v25; // rsi
  int v26; // eax
  int v27; // eax
  size_t v28; // rdi
  __int64 v29; // r12
  __int64 i; // rbx
  __m128i *v31; // rbx
  const void *v32; // rcx
  __int64 v33; // r13
  size_t v34; // r8
  int v35; // eax
  int v36; // eax
  __m128i *v37; // [rsp+0h] [rbp-80h]
  __int64 v38; // [rsp+8h] [rbp-78h]
  __m128i *v39; // [rsp+10h] [rbp-70h]
  __m128i *v41; // [rsp+20h] [rbp-60h]
  const void *v42; // [rsp+30h] [rbp-50h]
  void *s2a; // [rsp+38h] [rbp-48h]
  void *s2b; // [rsp+38h] [rbp-48h]
  void *s2; // [rsp+38h] [rbp-48h]
  void *s2c; // [rsp+38h] [rbp-48h]
  void *s2d; // [rsp+38h] [rbp-48h]
  const void *na; // [rsp+40h] [rbp-40h]
  const void *nb; // [rsp+40h] [rbp-40h]
  size_t n; // [rsp+40h] [rbp-40h]
  const void *nc; // [rsp+40h] [rbp-40h]
  const void *nd; // [rsp+40h] [rbp-40h]
  const void *ne; // [rsp+40h] [rbp-40h]
  const void *nf; // [rsp+40h] [rbp-40h]

  result = (char *)a2 - (char *)a1;
  v38 = a3;
  v39 = a2;
  if ( (char *)a2 - (char *)a1 <= 256 )
    return result;
  if ( !a3 )
  {
    v41 = a2;
    goto LABEL_50;
  }
  v37 = a1 + 2;
  while ( 2 )
  {
    --v38;
    v4 = a1[1].m128i_u64[1];
    v5 = (void *)a1[1].m128i_i64[0];
    v6 = (const void *)v39[-1].m128i_i64[0];
    v7 = v39[-1].m128i_u64[1];
    v8 = &a1[(__int64)(((unsigned __int64)((char *)v39 - (char *)a1) >> 63) + v39 - a1) >> 1];
    v9 = v8->m128i_u64[1];
    v10 = (const void *)v8->m128i_i64[0];
    if ( v4 > v9 )
    {
      if ( !v9 )
        goto LABEL_43;
      s2c = (void *)v39[-1].m128i_i64[0];
      nc = (const void *)a1[1].m128i_i64[0];
      v11 = memcmp(nc, (const void *)v8->m128i_i64[0], v8->m128i_u64[1]);
      v5 = (void *)nc;
      v6 = s2c;
      if ( !v11 )
        goto LABEL_8;
    }
    else
    {
      if ( !v4 )
      {
        if ( v9 )
        {
          if ( v7 >= v9 )
            goto LABEL_10;
LABEL_58:
          if ( !v7 )
          {
            if ( v4 )
              goto LABEL_48;
LABEL_64:
            if ( v4 == v7 )
              goto LABEL_48;
            goto LABEL_65;
          }
          s2d = v5;
          ne = v6;
          v12 = memcmp(v10, v6, v7);
          v6 = ne;
          v5 = s2d;
          if ( !v12 )
            goto LABEL_13;
LABEL_60:
          if ( v12 < 0 )
            goto LABEL_14;
LABEL_61:
          if ( v4 > v7 )
          {
            if ( !v7 )
              goto LABEL_48;
            v35 = memcmp(v5, v6, v7);
            if ( !v35 )
            {
LABEL_65:
              if ( v4 < v7 )
                goto LABEL_76;
LABEL_48:
              v15 = (void *)a1->m128i_i64[0];
              v28 = a1->m128i_u64[1];
              s2 = (void *)a1->m128i_i64[0];
              n = v28;
              *a1 = _mm_loadu_si128(a1 + 1);
              a1[1].m128i_i64[0] = (__int64)v15;
              a1[1].m128i_i64[1] = v28;
              v16 = v39[-1].m128i_i64[0];
              v17 = v39[-1].m128i_u64[1];
              goto LABEL_15;
            }
          }
          else
          {
            if ( !v4 )
              goto LABEL_64;
            v35 = memcmp(v5, v6, v4);
            if ( !v35 )
              goto LABEL_64;
          }
          if ( v35 < 0 )
            goto LABEL_76;
          goto LABEL_48;
        }
LABEL_46:
        if ( v4 == v7 )
          goto LABEL_71;
        goto LABEL_47;
      }
      s2a = (void *)v39[-1].m128i_i64[0];
      na = (const void *)a1[1].m128i_i64[0];
      v11 = memcmp(na, (const void *)v8->m128i_i64[0], a1[1].m128i_u64[1]);
      v5 = (void *)na;
      v6 = s2a;
      if ( !v11 )
      {
        if ( v4 == v9 )
        {
          if ( v4 <= v7 )
            goto LABEL_45;
          goto LABEL_68;
        }
LABEL_8:
        if ( v4 < v9 )
          goto LABEL_9;
LABEL_43:
        if ( v4 <= v7 )
        {
          if ( !v4 )
            goto LABEL_46;
LABEL_45:
          nd = v6;
          v27 = memcmp(v5, v6, v4);
          v6 = nd;
          if ( !v27 )
            goto LABEL_46;
LABEL_70:
          if ( v27 < 0 )
            goto LABEL_48;
          goto LABEL_71;
        }
LABEL_68:
        if ( !v7 )
        {
          if ( v9 )
            goto LABEL_14;
LABEL_74:
          if ( v7 == v9 )
            goto LABEL_14;
          goto LABEL_75;
        }
        nf = v6;
        v27 = memcmp(v5, v6, v7);
        v6 = nf;
        if ( !v27 )
        {
LABEL_47:
          if ( v4 < v7 )
            goto LABEL_48;
LABEL_71:
          if ( v7 < v9 )
          {
            if ( !v7 )
              goto LABEL_14;
            v36 = memcmp(v10, v6, v7);
            if ( !v36 )
            {
LABEL_75:
              if ( v7 <= v9 )
                goto LABEL_14;
LABEL_76:
              v16 = a1->m128i_i64[0];
              v17 = a1->m128i_u64[1];
              *a1 = _mm_loadu_si128(v39 - 1);
              v39[-1].m128i_i64[0] = v16;
              v39[-1].m128i_i64[1] = v17;
              v15 = (void *)a1[1].m128i_i64[0];
              s2 = v15;
              n = a1[1].m128i_u64[1];
              goto LABEL_15;
            }
          }
          else
          {
            if ( !v9 )
              goto LABEL_74;
            v36 = memcmp(v10, v6, v9);
            if ( !v36 )
              goto LABEL_74;
          }
          if ( v36 >= 0 )
            goto LABEL_14;
          goto LABEL_76;
        }
        goto LABEL_70;
      }
    }
    if ( v11 >= 0 )
      goto LABEL_43;
LABEL_9:
    if ( v7 < v9 )
      goto LABEL_58;
LABEL_10:
    if ( v9 )
    {
      s2b = v5;
      nb = v6;
      v12 = memcmp(v10, v6, v9);
      v6 = nb;
      v5 = s2b;
      if ( v12 )
        goto LABEL_60;
    }
    if ( v7 == v9 )
      goto LABEL_61;
LABEL_13:
    if ( v7 <= v9 )
      goto LABEL_61;
LABEL_14:
    v13 = a1->m128i_i64[0];
    v14 = a1->m128i_i64[1];
    *a1 = _mm_loadu_si128(v8);
    v8->m128i_i64[1] = v14;
    v8->m128i_i64[0] = v13;
    v15 = (void *)a1[1].m128i_i64[0];
    n = a1[1].m128i_u64[1];
    s2 = v15;
    v16 = v39[-1].m128i_i64[0];
    v17 = v39[-1].m128i_u64[1];
LABEL_15:
    v18 = v37;
    v42 = (const void *)v16;
    v19 = v39;
    v20 = (const void *)a1->m128i_i64[0];
    v21 = v17;
    v22 = a1->m128i_u64[1];
    while ( 1 )
    {
      v41 = v18 - 1;
      if ( v22 < n )
        break;
      if ( n )
      {
        v23 = memcmp(s2, v20, n);
        if ( v23 )
          goto LABEL_24;
      }
      if ( v22 == n )
        goto LABEL_25;
LABEL_19:
      if ( v22 <= n )
        goto LABEL_25;
LABEL_20:
      v24 = (void *)v18->m128i_i64[0];
      ++v18;
      s2 = v24;
      n = v18[-1].m128i_u64[1];
    }
    if ( !v22 )
      goto LABEL_25;
    v23 = memcmp(s2, v20, v22);
    if ( !v23 )
      goto LABEL_19;
LABEL_24:
    if ( v23 < 0 )
      goto LABEL_20;
LABEL_25:
    v25 = v42;
    --v19;
    while ( v21 >= v22 )
    {
      if ( v22 )
      {
        v26 = memcmp(v20, v25, v22);
        if ( v26 )
          goto LABEL_34;
      }
      if ( v21 == v22 )
        goto LABEL_35;
LABEL_29:
      if ( v21 <= v22 )
        goto LABEL_35;
LABEL_30:
      v21 = v19[-1].m128i_u64[1];
      v25 = (const void *)v19[-1].m128i_i64[0];
      --v19;
    }
    if ( !v21 )
      goto LABEL_35;
    v26 = memcmp(v20, v25, v21);
    if ( !v26 )
      goto LABEL_29;
LABEL_34:
    if ( v26 < 0 )
      goto LABEL_30;
LABEL_35:
    if ( v19 > v41 )
    {
      v18[-1] = _mm_loadu_si128(v19);
      v21 = v19[-1].m128i_u64[1];
      v19->m128i_i64[0] = (__int64)s2;
      v19->m128i_i64[1] = n;
      v20 = (const void *)a1->m128i_i64[0];
      v22 = a1->m128i_u64[1];
      v42 = (const void *)v19[-1].m128i_i64[0];
      goto LABEL_20;
    }
    sub_1AD04B0(v41, v39, v38, v15, v6, v5);
    result = (char *)v41 - (char *)a1;
    if ( (char *)v41 - (char *)a1 > 256 )
    {
      if ( v38 )
      {
        v39 = v18 - 1;
        continue;
      }
LABEL_50:
      v29 = result >> 4;
      for ( i = ((result >> 4) - 2) >> 1; ; --i )
      {
        sub_1ACFB90((__int64)a1, i, v29, (const void *)a1[i].m128i_i64[0], a1[i].m128i_u64[1]);
        if ( !i )
          break;
      }
      v31 = v41 - 1;
      do
      {
        v32 = (const void *)v31->m128i_i64[0];
        v33 = (char *)v31 - (char *)a1;
        v34 = v31->m128i_u64[1];
        --v31;
        v31[1] = _mm_loadu_si128(a1);
        result = (__int64)sub_1ACFB90((__int64)a1, 0, v33 >> 4, v32, v34);
      }
      while ( v33 > 16 );
    }
    return result;
  }
}
