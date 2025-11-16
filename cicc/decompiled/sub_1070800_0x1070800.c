// Function: sub_1070800
// Address: 0x1070800
//
__int64 __fastcall sub_1070800(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  unsigned __int64 v7; // r14
  __int64 v8; // r12
  __m128i *v9; // rbx
  __int64 v10; // rsi
  __int8 v11; // al
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned __int64 v14; // r12
  __m128i *v15; // rbx
  __m128i *v16; // r15
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int8 v20; // al
  __int64 v21; // rsi
  __int64 v22; // rbx
  __int64 i; // r12
  __m128i *v24; // r14
  __int64 v25; // rax
  __int64 v26; // r12
  __int128 v27; // xmm1
  __int64 v28; // rdx
  char v29; // al
  bool v30; // zf
  char v31; // al
  __int64 v32; // [rsp+18h] [rbp-68h]
  __int64 v33; // [rsp+20h] [rbp-60h]
  __m128i *v34; // [rsp+28h] [rbp-58h]

  result = (__int64)a2->m128i_i64 - a1;
  v33 = a3;
  v34 = a2;
  if ( (__int64)a2->m128i_i64 - a1 <= 384 )
    return result;
  v7 = (unsigned __int64)a2;
  if ( !a3 )
    goto LABEL_21;
  v32 = a1 + 24;
  while ( 2 )
  {
    --v33;
    v8 = (__int64)&v34[-2].m128i_i64[1];
    v9 = (__m128i *)(a1
                   + 8
                   * ((__int64)(0xAAAAAAAAAAAAAAABLL * (((__int64)v34->m128i_i64 - a1) >> 3)) / 2
                    + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v34->m128i_i64 - a1) >> 3)
                      + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v34->m128i_i64 - a1) >> 3)) >> 63))
                     & 0xFFFFFFFFFFFFFFFELL)));
    v10 = (__int64)&v34[-2].m128i_i64[1];
    if ( !sub_10704F0(v32, (__int64)v9) )
    {
      if ( !sub_10704F0(v32, v10) )
      {
        v31 = sub_10704F0((__int64)v9, v8);
        v12 = *(_QWORD *)a1;
        v13 = *(_QWORD *)(a1 + 8);
        v30 = v31 == 0;
        v11 = *(_BYTE *)(a1 + 16);
        if ( v30 )
        {
          *(__m128i *)a1 = _mm_loadu_si128(v9);
          goto LABEL_7;
        }
LABEL_30:
        *(__m128i *)a1 = _mm_loadu_si128((__m128i *)((char *)v34 - 24));
        *(_QWORD *)(a1 + 16) = v34[-1].m128i_i64[1];
        v34[-2].m128i_i64[1] = v12;
        v34[-1].m128i_i64[0] = v13;
        v34[-1].m128i_i8[8] = v11;
        goto LABEL_8;
      }
      v11 = *(_BYTE *)(a1 + 16);
      v12 = *(_QWORD *)a1;
      v13 = *(_QWORD *)(a1 + 8);
      *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 24));
LABEL_20:
      v21 = *(_QWORD *)(a1 + 40);
      *(_QWORD *)(a1 + 24) = v12;
      *(_QWORD *)(a1 + 32) = v13;
      *(_QWORD *)(a1 + 16) = v21;
      *(_BYTE *)(a1 + 40) = v11;
      goto LABEL_8;
    }
    if ( !sub_10704F0((__int64)v9, v10) )
    {
      v29 = sub_10704F0(v32, v8);
      v12 = *(_QWORD *)a1;
      v13 = *(_QWORD *)(a1 + 8);
      v30 = v29 == 0;
      v11 = *(_BYTE *)(a1 + 16);
      if ( !v30 )
        goto LABEL_30;
      *(__m128i *)a1 = _mm_loadu_si128((const __m128i *)(a1 + 24));
      goto LABEL_20;
    }
    v11 = *(_BYTE *)(a1 + 16);
    v12 = *(_QWORD *)a1;
    v13 = *(_QWORD *)(a1 + 8);
    *(__m128i *)a1 = _mm_loadu_si128(v9);
LABEL_7:
    *(_QWORD *)(a1 + 16) = v9[1].m128i_i64[0];
    v9->m128i_i64[0] = v12;
    v9->m128i_i64[1] = v13;
    v9[1].m128i_i8[0] = v11;
LABEL_8:
    v14 = a1 + 24;
    v15 = v34;
    while ( 1 )
    {
      v7 = v14;
      if ( sub_10704F0(v14, a1) )
        goto LABEL_14;
      v16 = (__m128i *)((char *)v15 - 24);
      do
      {
        v17 = (__int64)v16;
        v15 = v16;
        v16 = (__m128i *)((char *)v16 - 24);
      }
      while ( sub_10704F0(a1, v17) );
      if ( v14 >= (unsigned __int64)v15 )
        break;
      v18 = *(_QWORD *)v14;
      v19 = *(_QWORD *)(v14 + 8);
      v20 = *(_BYTE *)(v14 + 16);
      *(__m128i *)v14 = _mm_loadu_si128(v15);
      *(_QWORD *)(v14 + 16) = v15[1].m128i_i64[0];
      v15->m128i_i64[0] = v18;
      v15->m128i_i64[1] = v19;
      v15[1].m128i_i8[0] = v20;
LABEL_14:
      v14 += 24LL;
    }
    sub_1070800(v14, v34, v33);
    result = v14 - a1;
    if ( (__int64)(v14 - a1) > 384 )
    {
      if ( v33 )
      {
        v34 = (__m128i *)v14;
        continue;
      }
LABEL_21:
      v22 = 0xAAAAAAAAAAAAAAABLL * (result >> 3);
      for ( i = (v22 - 2) >> 1; ; --i )
      {
        sub_1070580(
          a1,
          i,
          v22,
          a4,
          a5,
          a6,
          *(_OWORD *)&_mm_loadu_si128((const __m128i *)(a1 + 24 * i)),
          *(_QWORD *)(a1 + 24 * i + 16));
        if ( !i )
          break;
      }
      v24 = (__m128i *)(v7 - 24);
      do
      {
        v25 = v24[1].m128i_i64[0];
        v26 = (__int64)v24->m128i_i64 - a1;
        v27 = (__int128)_mm_loadu_si128(v24);
        *v24 = _mm_loadu_si128((const __m128i *)a1);
        v28 = (__int64)v24->m128i_i64 - a1;
        v24 = (__m128i *)((char *)v24 - 24);
        v24[2].m128i_i64[1] = *(_QWORD *)(a1 + 16);
        result = sub_1070580(a1, 0, 0xAAAAAAAAAAAAAAABLL * (v28 >> 3), a4, a5, a6, v27, v25);
      }
      while ( v26 > 24 );
    }
    return result;
  }
}
