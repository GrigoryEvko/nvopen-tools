// Function: sub_1697010
// Address: 0x1697010
//
__int64 __fastcall sub_1697010(__int64 a1, __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  _QWORD *m128i_i64; // r15
  __int64 v9; // rbx
  __m128i *v10; // r10
  __m128i *v11; // r12
  unsigned __int64 v12; // rsi
  unsigned __int64 v13; // rcx
  unsigned __int64 v14; // r11
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __m128i v17; // xmm5
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // r14
  __m128i *v21; // rdi
  unsigned __int64 *v22; // rcx
  unsigned __int64 v23; // rdx
  unsigned __int64 *v24; // rax
  unsigned __int64 *v25; // rax
  __m128i v26; // xmm0
  __int64 v27; // rdx
  __int64 v28; // rax
  __m128i v29; // xmm4
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rbx
  __int64 i; // r12
  const __m128i *v34; // r15
  __int64 v35; // rax
  __int128 v36; // xmm1
  __int64 v37; // r12
  __m128i v38; // xmm6
  __int64 v39; // rdx
  __int64 v40; // rax
  __m128i v41; // xmm4
  __int64 v42; // rcx
  __int64 v43; // rdx
  __m128i v44; // xmm5
  __int64 v45; // rcx
  __int64 v46; // rdx
  __m128i v47; // xmm7
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // [rsp-60h] [rbp-60h]

  result = (__int64)a2->m128i_i64 - a1;
  if ( (__int64)a2->m128i_i64 - a1 <= 384 )
    return result;
  m128i_i64 = a2->m128i_i64;
  v9 = a3;
  if ( !a3 )
    goto LABEL_24;
  v10 = a2;
  v11 = (__m128i *)(a1 + 48);
  v50 = a1 + 24;
  while ( 2 )
  {
    v12 = *(_QWORD *)(a1 + 24);
    v13 = v10[-2].m128i_u64[1];
    --v9;
    v14 = *(_QWORD *)a1;
    v15 = a1
        + 8
        * ((__int64)(0xAAAAAAAAAAAAAAABLL * (((__int64)v10->m128i_i64 - a1) >> 3)) / 2
         + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v10->m128i_i64 - a1) >> 3)
           + ((0xAAAAAAAAAAAAAAABLL * (((__int64)v10->m128i_i64 - a1) >> 3)) >> 63))
          & 0xFFFFFFFFFFFFFFFELL));
    v16 = *(_QWORD *)v15;
    if ( v12 >= *(_QWORD *)v15 )
    {
      if ( v13 > v12 )
      {
        v47 = _mm_loadu_si128((const __m128i *)(a1 + 32));
        v48 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)a1 = v12;
        v49 = *(_QWORD *)(a1 + 16);
        *(_QWORD *)(a1 + 24) = v14;
        *(_QWORD *)(a1 + 32) = v48;
        *(_QWORD *)(a1 + 40) = v49;
        *(__m128i *)(a1 + 8) = v47;
      }
      else
      {
        if ( v13 <= v16 )
        {
          *(_QWORD *)a1 = v16;
          v44 = _mm_loadu_si128((const __m128i *)(v15 + 8));
          *(_QWORD *)v15 = v14;
          v45 = *(_QWORD *)(a1 + 8);
          v46 = *(_QWORD *)(a1 + 16);
          *(__m128i *)(a1 + 8) = v44;
          *(_QWORD *)(v15 + 8) = v45;
          *(_QWORD *)(v15 + 16) = v46;
        }
        else
        {
          *(_QWORD *)a1 = v13;
          v29 = _mm_loadu_si128(v10 - 1);
          v10[-2].m128i_i64[1] = v14;
          v30 = *(_QWORD *)(a1 + 8);
          v31 = *(_QWORD *)(a1 + 16);
          *(__m128i *)(a1 + 8) = v29;
          v10[-1].m128i_i64[0] = v30;
          v10[-1].m128i_i64[1] = v31;
        }
        v14 = *(_QWORD *)(a1 + 24);
        v12 = *(_QWORD *)a1;
      }
    }
    else if ( v13 > v16 )
    {
      *(_QWORD *)a1 = v16;
      v41 = _mm_loadu_si128((const __m128i *)(v15 + 8));
      *(_QWORD *)v15 = v14;
      v42 = *(_QWORD *)(a1 + 8);
      v43 = *(_QWORD *)(a1 + 16);
      *(__m128i *)(a1 + 8) = v41;
      *(_QWORD *)(v15 + 8) = v42;
      *(_QWORD *)(v15 + 16) = v43;
      v14 = *(_QWORD *)(a1 + 24);
      v12 = *(_QWORD *)a1;
    }
    else if ( v13 <= v12 )
    {
      v38 = _mm_loadu_si128((const __m128i *)(a1 + 32));
      v39 = *(_QWORD *)(a1 + 8);
      *(_QWORD *)a1 = v12;
      v40 = *(_QWORD *)(a1 + 16);
      *(_QWORD *)(a1 + 24) = v14;
      *(_QWORD *)(a1 + 32) = v39;
      *(_QWORD *)(a1 + 40) = v40;
      *(__m128i *)(a1 + 8) = v38;
    }
    else
    {
      *(_QWORD *)a1 = v13;
      v17 = _mm_loadu_si128(v10 - 1);
      v10[-2].m128i_i64[1] = v14;
      v18 = *(_QWORD *)(a1 + 8);
      v19 = *(_QWORD *)(a1 + 16);
      *(__m128i *)(a1 + 8) = v17;
      v10[-1].m128i_i64[0] = v18;
      v10[-1].m128i_i64[1] = v19;
      v14 = *(_QWORD *)(a1 + 24);
      v12 = *(_QWORD *)a1;
    }
    v20 = v50;
    v21 = v11;
    v22 = (unsigned __int64 *)v10;
    while ( 1 )
    {
      m128i_i64 = (_QWORD *)v20;
      if ( v12 > v14 )
        goto LABEL_15;
      v23 = *(v22 - 3);
      v24 = v22 - 3;
      if ( v23 <= v12 )
      {
        v22 -= 3;
        if ( v20 >= (unsigned __int64)v24 )
          break;
        goto LABEL_14;
      }
      v25 = v22 - 6;
      do
      {
        v22 = v25;
        v23 = *v25;
        v25 -= 3;
      }
      while ( v23 > v12 );
      if ( v20 >= (unsigned __int64)v22 )
        break;
LABEL_14:
      v21[-2].m128i_i64[1] = v23;
      v26 = _mm_loadu_si128((const __m128i *)(v22 + 1));
      *v22 = v14;
      v27 = v21[-1].m128i_i64[0];
      v28 = v21[-1].m128i_i64[1];
      v21[-1] = v26;
      v22[1] = v27;
      v22[2] = v28;
      v12 = *(_QWORD *)a1;
LABEL_15:
      v14 = v21->m128i_i64[0];
      v20 += 24LL;
      v21 = (__m128i *)((char *)v21 + 24);
    }
    sub_1697010(v20, v10, v9);
    result = v20 - a1;
    if ( (__int64)(v20 - a1) > 384 )
    {
      if ( v9 )
      {
        v10 = (__m128i *)v20;
        continue;
      }
LABEL_24:
      v32 = 0xAAAAAAAAAAAAAAABLL * (result >> 3);
      for ( i = (v32 - 2) >> 1; ; --i )
      {
        sub_1693910(
          a1,
          i,
          v32,
          a4,
          a5,
          a6,
          *(_OWORD *)&_mm_loadu_si128((const __m128i *)(a1 + 24 * i)),
          *(_QWORD *)(a1 + 24 * i + 16));
        if ( !i )
          break;
      }
      v34 = (const __m128i *)(m128i_i64 - 3);
      do
      {
        v35 = v34[1].m128i_i64[0];
        v36 = (__int128)_mm_loadu_si128(v34);
        v37 = (__int64)v34->m128i_i64 - a1;
        v34 = (const __m128i *)((char *)v34 - 24);
        v34[1].m128i_i64[1] = *(_QWORD *)a1;
        v34[2] = _mm_loadu_si128((const __m128i *)(a1 + 8));
        result = sub_1693910(a1, 0, 0xAAAAAAAAAAAAAAABLL * (v37 >> 3), a4, a5, a6, v36, v35);
      }
      while ( v37 > 24 );
    }
    return result;
  }
}
