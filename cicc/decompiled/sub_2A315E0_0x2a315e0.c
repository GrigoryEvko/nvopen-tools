// Function: sub_2A315E0
// Address: 0x2a315e0
//
__int64 __fastcall sub_2A315E0(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __m128i *v7; // rbx
  __int64 v8; // r12
  int v9; // eax
  __int64 v10; // r14
  __int64 v11; // r13
  int v12; // eax
  __int64 v13; // rdx
  int v14; // eax
  __int64 *v15; // r13
  __m128i *v16; // rbx
  __int64 v17; // r12
  int v18; // eax
  __m128i *v19; // r14
  __int64 v20; // r12
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  int v24; // eax
  int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rbx
  __int64 i; // r12
  __m128i *v29; // rbx
  __int64 v30; // rax
  __int64 v31; // r13
  __m128i v32; // xmm1
  __int64 v33; // rdx
  __int64 v34; // [rsp+10h] [rbp-80h]
  __m128i *v35; // [rsp+18h] [rbp-78h]
  __int64 v36; // [rsp+20h] [rbp-70h]
  __int64 v37; // [rsp+20h] [rbp-70h]
  __m128i *v38; // [rsp+30h] [rbp-60h]
  __int64 v39; // [rsp+38h] [rbp-58h]
  __m128i v40; // [rsp+40h] [rbp-50h]

  result = (__int64)a2->m128i_i64 - a1;
  v34 = a3;
  v35 = a2;
  if ( (__int64)a2->m128i_i64 - a1 <= 384 )
    return result;
  if ( !a3 )
  {
    v38 = a2;
    goto LABEL_23;
  }
  while ( 2 )
  {
    --v34;
    v7 = (__m128i *)(a1
                   + 8
                   * ((__int64)(0xAAAAAAAAAAAAAAABLL * (((__int64)v35->m128i_i64 - a1) >> 3)) / 2
                    + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v35->m128i_i64 - a1) >> 3)
                      + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v35->m128i_i64 - a1) >> 3)) >> 63))
                     & 0xFFFFFFFFFFFFFFFELL)));
    v8 = *(_QWORD *)(a1 + 24) + 24LL;
    v9 = sub_C4C880(v8, v7->m128i_i64[1] + 24);
    v10 = *(_QWORD *)(a1 + 16);
    v11 = v35[-1].m128i_i64[0] + 24;
    v36 = *(_QWORD *)(a1 + 8);
    v39 = *(_QWORD *)a1;
    if ( v9 < 0 )
    {
      v24 = sub_C4C880(v7->m128i_i64[0] + 24, v11);
      v13 = v36;
      if ( v24 < 0 )
        goto LABEL_6;
      v25 = sub_C4C880(v8, v11);
      v13 = v36;
      if ( v25 < 0 )
      {
LABEL_21:
        *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v35 - 24));
        *(_QWORD *)(a1 + 16) = v35[-1].m128i_i64[1];
        v35[-1].m128i_i64[0] = v13;
        v35[-2].m128i_i64[1] = v39;
        v35[-1].m128i_i64[1] = v10;
        v39 = *(_QWORD *)(a1 + 24);
        goto LABEL_7;
      }
LABEL_20:
      *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 24));
      v26 = *(_QWORD *)(a1 + 40);
      *(_QWORD *)(a1 + 32) = v13;
      *(_QWORD *)(a1 + 40) = v10;
      *(_QWORD *)(a1 + 16) = v26;
      *(_QWORD *)(a1 + 24) = v39;
      v13 = v35[-1].m128i_i64[0];
      goto LABEL_7;
    }
    v12 = sub_C4C880(v8, v11);
    v13 = v36;
    if ( v12 < 0 )
      goto LABEL_20;
    v14 = sub_C4C880(v7->m128i_i64[0] + 24, v11);
    v13 = v36;
    if ( v14 < 0 )
      goto LABEL_21;
LABEL_6:
    *(__m128i *)a1 = _mm_loadu_si128(v7);
    *(_QWORD *)(a1 + 16) = v7[1].m128i_i64[0];
    v7->m128i_i64[1] = v13;
    v7->m128i_i64[0] = v39;
    v7[1].m128i_i64[0] = v10;
    v39 = *(_QWORD *)(a1 + 24);
    v13 = v35[-1].m128i_i64[0];
LABEL_7:
    v15 = (__int64 *)(a1 + 48);
    v16 = v35;
    v17 = *(_QWORD *)(a1 + 8) + 24LL;
    while ( 1 )
    {
      v37 = v13;
      v38 = (__m128i *)(v15 - 3);
      v18 = sub_C4C880(v39 + 24, v17);
      v13 = v37;
      if ( v18 < 0 )
        goto LABEL_14;
      v19 = (__m128i *)((char *)v16 - 24);
      v20 = *(_QWORD *)a1 + 24LL;
      while ( 1 )
      {
        v16 = v19;
        v19 = (__m128i *)((char *)v19 - 24);
        if ( (int)sub_C4C880(v20, v13 + 24) >= 0 )
          break;
        v13 = v19->m128i_i64[1];
      }
      if ( v16 <= v38 )
        break;
      v21 = *(v15 - 2);
      v22 = *(v15 - 1);
      *(__m128i *)(v15 - 3) = _mm_loadu_si128(v16);
      *(v15 - 1) = v16[1].m128i_i64[0];
      v16->m128i_i64[1] = v21;
      v13 = v16[-1].m128i_i64[0];
      v16->m128i_i64[0] = v39;
      v16[1].m128i_i64[0] = v22;
      v17 = *(_QWORD *)(a1 + 8) + 24LL;
LABEL_14:
      v23 = *v15;
      v15 += 3;
      v39 = v23;
    }
    sub_2A315E0(v38, v35, v34);
    result = (__int64)v38->m128i_i64 - a1;
    if ( (__int64)v38->m128i_i64 - a1 > 384 )
    {
      if ( v34 )
      {
        v35 = (__m128i *)(v15 - 3);
        continue;
      }
LABEL_23:
      v27 = 0xAAAAAAAAAAAAAAABLL * (result >> 3);
      for ( i = (v27 - 2) >> 1; ; --i )
      {
        v40 = _mm_loadu_si128((const __m128i *)(a1 + 24 * i));
        sub_2A30CF0(a1, i, v27, a4, a5, a6, v40.m128i_i64[0], v40.m128i_i64[1], *(_QWORD *)(a1 + 24 * i + 16));
        if ( !i )
          break;
      }
      v29 = (__m128i *)((char *)v38 - 24);
      do
      {
        v30 = v29[1].m128i_i64[0];
        v31 = (__int64)v29->m128i_i64 - a1;
        v32 = _mm_loadu_si128(v29);
        *v29 = _mm_loadu_si128((const __m128i *)a1);
        v33 = (__int64)v29->m128i_i64 - a1;
        v29 = (__m128i *)((char *)v29 - 24);
        v29[2].m128i_i64[1] = *(_QWORD *)(a1 + 16);
        result = sub_2A30CF0(
                   a1,
                   0,
                   0xAAAAAAAAAAAAAAABLL * (v33 >> 3),
                   a4,
                   a5,
                   a6,
                   v32.m128i_i64[0],
                   v32.m128i_i64[1],
                   v30);
      }
      while ( v31 > 24 );
    }
    return result;
  }
}
