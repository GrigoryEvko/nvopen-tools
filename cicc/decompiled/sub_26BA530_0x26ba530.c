// Function: sub_26BA530
// Address: 0x26ba530
//
void __fastcall sub_26BA530(__m128i *a1, unsigned __int64 *a2, __int64 a3)
{
  __int64 v4; // rdi
  __m128i *v5; // r8
  __int64 v6; // r15
  unsigned __int64 v7; // r9
  __m128i *v8; // rdi
  unsigned __int64 v9; // r10
  unsigned __int64 v10; // rax
  bool v11; // cf
  bool v12; // zf
  unsigned __int64 v13; // rcx
  __int64 v14; // rcx
  __int64 v15; // rsi
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  unsigned __int64 v18; // rsi
  __m128i *v19; // r13
  __m128i *v20; // rax
  unsigned __int64 *v21; // r14
  bool v22; // cf
  bool v23; // zf
  __int64 v24; // rdx
  bool v25; // cf
  bool v26; // zf
  unsigned __int64 v27; // rcx
  __int64 v28; // rsi
  __m128i v29; // xmm3
  __int64 v30; // rsi
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // r8
  __m128i v33; // xmm5

  v4 = (char *)a2 - (char *)a1;
  if ( v4 <= 256 )
    return;
  v5 = (__m128i *)a2;
  v6 = a3;
  if ( !a3 )
  {
    v21 = a2;
    goto LABEL_28;
  }
  while ( 2 )
  {
    v7 = a1[1].m128i_u64[1];
    --v6;
    v8 = &a1[v4 >> 5];
    v9 = v8->m128i_u64[1];
    v10 = v5[-1].m128i_u64[1];
    if ( *(_OWORD *)&a1[1] <= *(_OWORD *)v8 )
    {
      v25 = v7 < v10;
      v26 = v7 == v10;
      if ( v7 == v10 )
      {
        v27 = v5[-1].m128i_u64[0];
        v25 = a1[1].m128i_i64[0] < v27;
        v26 = a1[1].m128i_i64[0] == v27;
      }
      v28 = a1->m128i_i64[0];
      v16 = a1->m128i_u64[1];
      if ( !v25 && !v26 )
      {
        v29 = _mm_loadu_si128(a1 + 1);
        a1[1].m128i_i64[0] = v28;
        a1[1].m128i_i64[1] = v16;
        *a1 = v29;
        v17 = v5[-1].m128i_u64[1];
        goto LABEL_8;
      }
      if ( v9 == v10 )
      {
        if ( v8->m128i_i64[0] > (unsigned __int64)v5[-1].m128i_i64[0] )
          goto LABEL_37;
      }
      else if ( v9 > v10 )
      {
LABEL_37:
        v17 = a1->m128i_u64[1];
        *a1 = _mm_loadu_si128(v5 - 1);
        v5[-1].m128i_i64[0] = v28;
        v5[-1].m128i_i64[1] = v16;
        v16 = a1[1].m128i_u64[1];
        goto LABEL_8;
      }
      *a1 = _mm_loadu_si128(v8);
      v8->m128i_i64[0] = v28;
      v8->m128i_i64[1] = v16;
      v16 = a1[1].m128i_u64[1];
      v17 = v5[-1].m128i_u64[1];
      goto LABEL_8;
    }
    v11 = v9 < v10;
    v12 = v9 == v10;
    if ( v9 == v10 )
    {
      v13 = v5[-1].m128i_u64[0];
      v11 = v8->m128i_i64[0] < v13;
      v12 = v8->m128i_i64[0] == v13;
    }
    v14 = a1->m128i_i64[0];
    v15 = a1->m128i_i64[1];
    if ( v11 || v12 )
    {
      if ( v7 == v10 )
      {
        if ( a1[1].m128i_i64[0] > (unsigned __int64)v5[-1].m128i_i64[0] )
          goto LABEL_34;
      }
      else if ( v7 > v10 )
      {
LABEL_34:
        v17 = a1->m128i_u64[1];
        *a1 = _mm_loadu_si128(v5 - 1);
        v5[-1].m128i_i64[0] = v14;
        v5[-1].m128i_i64[1] = v15;
        v16 = a1[1].m128i_u64[1];
        goto LABEL_8;
      }
      v33 = _mm_loadu_si128(a1 + 1);
      a1[1].m128i_i64[0] = v14;
      v16 = v15;
      a1[1].m128i_i64[1] = v15;
      *a1 = v33;
      v17 = v5[-1].m128i_u64[1];
      goto LABEL_8;
    }
    *a1 = _mm_loadu_si128(v8);
    v8->m128i_i64[0] = v14;
    v8->m128i_i64[1] = v15;
    v16 = a1[1].m128i_u64[1];
    v17 = v5[-1].m128i_u64[1];
LABEL_8:
    v18 = a1->m128i_u64[1];
    v19 = a1 + 1;
    v20 = v5;
    while ( 1 )
    {
      v21 = (unsigned __int64 *)v19;
      if ( v18 == v16 )
      {
        if ( v19->m128i_i64[0] > (unsigned __int64)a1->m128i_i64[0] )
          goto LABEL_10;
      }
      else if ( v18 < v16 )
      {
        goto LABEL_10;
      }
      for ( --v20; ; --v20 )
      {
        v22 = v18 < v17;
        v23 = v18 == v17;
        if ( v18 == v17 )
        {
          v22 = a1->m128i_i64[0] < (unsigned __int64)v20->m128i_i64[0];
          v23 = a1->m128i_i64[0] == v20->m128i_i64[0];
        }
        if ( v22 || v23 )
          break;
        v17 = v20[-1].m128i_u64[1];
      }
      if ( v19 >= v20 )
        break;
      v24 = v19->m128i_i64[0];
      *v19 = _mm_loadu_si128(v20);
      v20->m128i_i64[0] = v24;
      v17 = v20[-1].m128i_u64[1];
      v20->m128i_i64[1] = v16;
      v18 = a1->m128i_u64[1];
LABEL_10:
      v16 = v19[1].m128i_u64[1];
      ++v19;
    }
    sub_26BA530(v19, v5, v6);
    v4 = (char *)v19 - (char *)a1;
    if ( (char *)v19 - (char *)a1 > 256 )
    {
      if ( v6 )
      {
        v5 = v19;
        continue;
      }
LABEL_28:
      v30 = ((v4 >> 4) - 2) >> 1;
      sub_26BA020((__int64)a1, v30, v4 >> 4, a1[v30].m128i_u64[0], a1[v30].m128i_u64[1]);
      do
      {
        --v30;
        sub_26BA020((__int64)a1, v30, v4 >> 4, a1[v30].m128i_u64[0], a1[v30].m128i_u64[1]);
      }
      while ( v30 );
      do
      {
        v21 -= 2;
        v31 = *v21;
        v32 = v21[1];
        *(__m128i *)v21 = _mm_loadu_si128(a1);
        sub_26BA020((__int64)a1, 0, ((char *)v21 - (char *)a1) >> 4, v31, v32);
      }
      while ( (char *)v21 - (char *)a1 > 16 );
    }
    break;
  }
}
