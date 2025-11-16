// Function: sub_39908E0
// Address: 0x39908e0
//
void __fastcall sub_39908E0(__m128i *a1, __m128i *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __m128i *v6; // rbx
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rbx
  __m128i v10; // xmm1
  __m128i v11; // xmm0
  __m128i v12; // xmm6
  __m128i v13; // xmm7
  __m128i *v14; // rbx
  unsigned __int64 v15; // r12
  __m128i v16; // xmm1
  __m128i v17; // xmm0
  __m128i v18; // xmm3
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // r12
  __m128i v21; // xmm1
  __m128i v22; // xmm0
  __m128i v23; // xmm1
  __m128i v24; // xmm0
  __m128i v25; // xmm7
  __int64 v26; // r9
  __int128 v27; // rax
  __m128i v28; // xmm5
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int128 v31; // [rsp-10h] [rbp-C0h]
  __int64 v32; // [rsp+10h] [rbp-A0h]
  __m128i *v33; // [rsp+18h] [rbp-98h]
  __m128i *v34; // [rsp+20h] [rbp-90h]
  unsigned __int64 v35; // [rsp+28h] [rbp-88h]
  __m128i *i; // [rsp+30h] [rbp-80h]
  char v37[8]; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int64 v38; // [rsp+48h] [rbp-68h]
  __m128i v39; // [rsp+60h] [rbp-50h] BYREF
  __m128i v40; // [rsp+70h] [rbp-40h]

  v4 = (char *)a2 - (char *)a1;
  v33 = a2;
  v32 = a3;
  if ( (char *)a2 - (char *)a1 <= 512 )
    return;
  if ( !a3 )
  {
    v34 = a2;
    goto LABEL_21;
  }
  while ( 2 )
  {
    --v32;
    v6 = &a1[2 * (v4 >> 6)];
    sub_15B1350(
      (__int64)v37,
      *(unsigned __int64 **)(a1[2].m128i_i64[0] + 24),
      *(unsigned __int64 **)(a1[2].m128i_i64[0] + 32));
    v7 = v38;
    sub_15B1350(
      (__int64)&v39,
      *(unsigned __int64 **)(v6->m128i_i64[0] + 24),
      *(unsigned __int64 **)(v6->m128i_i64[0] + 32));
    if ( v7 >= v39.m128i_i64[1] )
    {
      sub_15B1350(
        (__int64)v37,
        *(unsigned __int64 **)(a1[2].m128i_i64[0] + 24),
        *(unsigned __int64 **)(a1[2].m128i_i64[0] + 32));
      v19 = v38;
      sub_15B1350(
        (__int64)&v39,
        *(unsigned __int64 **)(v33[-2].m128i_i64[0] + 24),
        *(unsigned __int64 **)(v33[-2].m128i_i64[0] + 32));
      if ( v19 < v39.m128i_i64[1] )
        goto LABEL_6;
      sub_15B1350(
        (__int64)v37,
        *(unsigned __int64 **)(v6->m128i_i64[0] + 24),
        *(unsigned __int64 **)(v6->m128i_i64[0] + 32));
      v20 = v38;
      sub_15B1350(
        (__int64)&v39,
        *(unsigned __int64 **)(v33[-2].m128i_i64[0] + 24),
        *(unsigned __int64 **)(v33[-2].m128i_i64[0] + 32));
      if ( v20 < v39.m128i_i64[1] )
      {
LABEL_18:
        v21 = _mm_loadu_si128(a1);
        v22 = _mm_loadu_si128(a1 + 1);
        v39 = v21;
        v40 = v22;
        *a1 = _mm_loadu_si128(v33 - 2);
        a1[1] = _mm_loadu_si128(v33 - 1);
        v33[-2] = v21;
        v33[-1] = v22;
        goto LABEL_7;
      }
LABEL_19:
      v23 = _mm_loadu_si128(a1);
      v24 = _mm_loadu_si128(a1 + 1);
      *a1 = _mm_loadu_si128(v6);
      v25 = _mm_loadu_si128(v6 + 1);
      v39 = v23;
      a1[1] = v25;
      v40 = v24;
      *v6 = v23;
      v6[1] = v24;
      goto LABEL_7;
    }
    sub_15B1350(
      (__int64)v37,
      *(unsigned __int64 **)(v6->m128i_i64[0] + 24),
      *(unsigned __int64 **)(v6->m128i_i64[0] + 32));
    v8 = v38;
    sub_15B1350(
      (__int64)&v39,
      *(unsigned __int64 **)(v33[-2].m128i_i64[0] + 24),
      *(unsigned __int64 **)(v33[-2].m128i_i64[0] + 32));
    if ( v8 < v39.m128i_i64[1] )
      goto LABEL_19;
    sub_15B1350(
      (__int64)v37,
      *(unsigned __int64 **)(a1[2].m128i_i64[0] + 24),
      *(unsigned __int64 **)(a1[2].m128i_i64[0] + 32));
    v9 = v38;
    sub_15B1350(
      (__int64)&v39,
      *(unsigned __int64 **)(v33[-2].m128i_i64[0] + 24),
      *(unsigned __int64 **)(v33[-2].m128i_i64[0] + 32));
    if ( v9 < v39.m128i_i64[1] )
      goto LABEL_18;
LABEL_6:
    v10 = _mm_loadu_si128(a1);
    v11 = _mm_loadu_si128(a1 + 1);
    v12 = _mm_loadu_si128(a1 + 2);
    v13 = _mm_loadu_si128(a1 + 3);
    v39 = v10;
    v40 = v11;
    *a1 = v12;
    a1[1] = v13;
    a1[2] = v10;
    a1[3] = v11;
LABEL_7:
    v14 = v33;
    for ( i = a1 + 2; ; i += 2 )
    {
      v34 = i;
      sub_15B1350(
        (__int64)v37,
        *(unsigned __int64 **)(i->m128i_i64[0] + 24),
        *(unsigned __int64 **)(i->m128i_i64[0] + 32));
      v35 = v38;
      sub_15B1350(
        (__int64)&v39,
        *(unsigned __int64 **)(a1->m128i_i64[0] + 24),
        *(unsigned __int64 **)(a1->m128i_i64[0] + 32));
      if ( v35 < v39.m128i_i64[1] )
        continue;
      do
      {
        v14 -= 2;
        sub_15B1350(
          (__int64)v37,
          *(unsigned __int64 **)(a1->m128i_i64[0] + 24),
          *(unsigned __int64 **)(a1->m128i_i64[0] + 32));
        v15 = v38;
        sub_15B1350(
          (__int64)&v39,
          *(unsigned __int64 **)(v14->m128i_i64[0] + 24),
          *(unsigned __int64 **)(v14->m128i_i64[0] + 32));
      }
      while ( v15 < v39.m128i_i64[1] );
      if ( i >= v14 )
        break;
      v16 = _mm_loadu_si128(i);
      v17 = _mm_loadu_si128(i + 1);
      *i = _mm_loadu_si128(v14);
      v18 = _mm_loadu_si128(v14 + 1);
      v39 = v16;
      i[1] = v18;
      v40 = v17;
      *v14 = v16;
      v14[1] = v17;
    }
    v4 = (char *)i - (char *)a1;
    sub_39908E0(i, v33, v32);
    if ( (char *)i - (char *)a1 > 512 )
    {
      if ( v32 )
      {
        v33 = i;
        continue;
      }
LABEL_21:
      sub_3990780(a1, v34, (unsigned __int64)v34, a4);
      do
      {
        v34 -= 2;
        v27 = (__int128)*v34;
        *v34 = _mm_loadu_si128(a1);
        v28 = _mm_loadu_si128(a1 + 1);
        v29 = v34[1].m128i_i64[0];
        v30 = v34[1].m128i_i64[1];
        v39.m128i_i64[1] = *((_QWORD *)&v27 + 1);
        v34[1] = v28;
        *((_QWORD *)&v31 + 1) = v30;
        *(_QWORD *)&v31 = v29;
        v40.m128i_i64[0] = v29;
        v40.m128i_i64[1] = v30;
        v39.m128i_i64[0] = v27;
        sub_3986080(
          (__int64)a1,
          0,
          ((char *)v34 - (char *)a1) >> 5,
          (char *)v34 - (char *)a1,
          ((char *)v34 - (char *)a1) >> 5,
          v26,
          v27,
          v31);
      }
      while ( (char *)v34 - (char *)a1 > 32 );
    }
    break;
  }
}
