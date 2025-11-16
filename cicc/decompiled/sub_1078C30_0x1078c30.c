// Function: sub_1078C30
// Address: 0x1078c30
//
const __m128i *__fastcall sub_1078C30(
        __m128i *a1,
        __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128i *a6,
        const __m128i *a7)
{
  __m128i *v7; // r15
  __m128i *v8; // r13
  __m128i *v9; // r12
  const __m128i *result; // rax
  const __m128i *v11; // rcx
  signed __int64 v12; // rbx
  __m128i *v13; // r8
  __m128i v14; // xmm0
  __m128i v15; // xmm2
  __int64 v16; // rbx
  __m128i *v17; // rbx
  size_t v18; // rdx
  const __m128i *v19; // r14
  __m128i *v20; // rax
  __int64 v21; // r10
  __int64 v22; // r11
  signed __int64 v23; // r8
  unsigned __int64 v24; // rcx
  __m128i *v25; // r10
  size_t v26; // r10
  __m128i *v27; // rsi
  __m128i *v28; // rdi
  __m128i *v29; // rax
  size_t v30; // r9
  __m128i *v31; // rax
  const __m128i *v32; // rax
  __int64 v33; // [rsp+0h] [rbp-70h]
  __int64 v34; // [rsp+0h] [rbp-70h]
  void *dest; // [rsp+8h] [rbp-68h]
  void *desta; // [rsp+8h] [rbp-68h]
  void *destb; // [rsp+8h] [rbp-68h]
  void *destc; // [rsp+8h] [rbp-68h]
  signed __int64 v39; // [rsp+10h] [rbp-60h]
  size_t v40; // [rsp+10h] [rbp-60h]
  signed __int64 v41; // [rsp+10h] [rbp-60h]
  __int64 v42; // [rsp+10h] [rbp-60h]
  signed __int64 v43; // [rsp+10h] [rbp-60h]
  __m128i *v44; // [rsp+10h] [rbp-60h]
  __int64 v45; // [rsp+10h] [rbp-60h]
  __int64 v46; // [rsp+10h] [rbp-60h]
  int v47; // [rsp+18h] [rbp-58h]
  signed __int64 v48; // [rsp+18h] [rbp-58h]
  int v49; // [rsp+18h] [rbp-58h]
  signed __int64 v50; // [rsp+18h] [rbp-58h]
  int v51; // [rsp+18h] [rbp-58h]
  signed __int64 v52; // [rsp+18h] [rbp-58h]
  signed __int64 v53; // [rsp+18h] [rbp-58h]
  signed __int64 v54; // [rsp+18h] [rbp-58h]
  int n; // [rsp+20h] [rbp-50h]
  size_t na; // [rsp+20h] [rbp-50h]
  size_t nb; // [rsp+20h] [rbp-50h]
  int nc; // [rsp+20h] [rbp-50h]
  int nd; // [rsp+20h] [rbp-50h]
  int ne; // [rsp+20h] [rbp-50h]
  __int64 v61; // [rsp+28h] [rbp-48h]
  __int64 v62; // [rsp+28h] [rbp-48h]
  __int64 v63; // [rsp+28h] [rbp-48h]
  __m128i *v64; // [rsp+30h] [rbp-40h]
  void *srca; // [rsp+38h] [rbp-38h]
  __m128i *src; // [rsp+38h] [rbp-38h]

  while ( 1 )
  {
    v7 = (__m128i *)a3;
    v8 = a1;
    v9 = a6;
    result = a7;
    if ( a5 <= (__int64)a7 )
      result = (const __m128i *)a5;
    if ( (__int64)result >= a4 )
    {
      v11 = a2;
      v12 = (char *)a2 - (char *)a1;
      if ( a1 != a2 )
      {
        result = (const __m128i *)memmove(a6, a1, (char *)a2 - (char *)a1);
        v11 = a2;
      }
      v13 = (__m128i *)((char *)v9 + v12);
      if ( v9 == (__m128i *)&v9->m128i_i8[v12] )
        return result;
      while ( v7 != v11 )
      {
        if ( *(_QWORD *)(v11[2].m128i_i64[0] + 160) + v11->m128i_i64[0] < (unsigned __int64)(*(_QWORD *)(v9[2].m128i_i64[0] + 160)
                                                                                           + v9->m128i_i64[0]) )
        {
          v14 = _mm_loadu_si128(v11);
          v11 = (const __m128i *)((char *)v11 + 40);
          *v8 = v14;
          v8[1] = _mm_loadu_si128((const __m128i *)((char *)v11 - 24));
          result = (const __m128i *)v11[-1].m128i_i64[1];
        }
        else
        {
          v15 = _mm_loadu_si128(v9);
          v9 = (__m128i *)((char *)v9 + 40);
          *v8 = v15;
          v8[1] = _mm_loadu_si128((__m128i *)((char *)v9 - 24));
          result = (const __m128i *)v9[-1].m128i_i64[1];
        }
        v8[2].m128i_i64[0] = (__int64)result;
        v8 = (__m128i *)((char *)v8 + 40);
        if ( v13 == v9 )
          return result;
      }
      v27 = v9;
      v28 = v8;
      v18 = (char *)v13 - (char *)v9;
      return (const __m128i *)memmove(v28, v27, v18);
    }
    v16 = a5;
    if ( a5 <= (__int64)a7 )
      break;
    if ( a5 < a4 )
    {
      v63 = a4 / 2;
      src = (__m128i *)((char *)a1 + 40 * (a4 / 2));
      v29 = (__m128i *)sub_1077A50(a2, a3, src);
      v24 = v63;
      v64 = v29;
      v23 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v29 - (char *)a2) >> 3);
    }
    else
    {
      v61 = a5 / 2;
      v64 = (__m128i *)((char *)a2 + 40 * (a5 / 2));
      v20 = (__m128i *)sub_1077AC0(a1, (__int64)a2, v64);
      v23 = v61;
      src = v20;
      v24 = 0xCCCCCCCCCCCCCCCDLL * (((char *)v20 - (char *)a1) >> 3);
    }
    v62 = v21 - v24;
    if ( (__int64)(v21 - v24) > v23 && v22 >= v23 )
    {
      v25 = src;
      if ( !v23 )
        goto LABEL_34;
      v26 = (char *)v64 - (char *)a2;
      if ( v64 == a2 )
      {
        if ( v64 != src )
        {
LABEL_30:
          desta = (void *)v22;
          v40 = v26;
          v48 = v23;
          n = v24;
          memmove((char *)v64 - ((char *)a2 - (char *)src), src, (char *)a2 - (char *)src);
          v22 = (__int64)desta;
          v26 = v40;
          v23 = v48;
          LODWORD(v24) = n;
          goto LABEL_31;
        }
      }
      else
      {
        dest = (void *)v22;
        v39 = v23;
        v47 = v24;
        memmove(v9, a2, (char *)v64 - (char *)a2);
        v26 = (char *)v64 - (char *)a2;
        LODWORD(v24) = v47;
        v23 = v39;
        v22 = (__int64)dest;
        if ( a2 != src )
          goto LABEL_30;
LABEL_31:
        if ( v26 )
        {
          destb = (void *)v22;
          v41 = v23;
          v49 = v24;
          na = v26;
          memmove(src, v9, v26);
          v22 = (__int64)destb;
          v23 = v41;
          LODWORD(v24) = v49;
          v26 = na;
        }
      }
      v25 = (__m128i *)((char *)src + v26);
      goto LABEL_34;
    }
    if ( v22 < v62 )
    {
      v46 = v22;
      v54 = v23;
      ne = v24;
      v32 = sub_1077D10(src, a2, v64);
      v22 = v46;
      v23 = v54;
      LODWORD(v24) = ne;
      v25 = (__m128i *)v32;
      goto LABEL_34;
    }
    v25 = v64;
    if ( !v62 )
      goto LABEL_34;
    v30 = (char *)a2 - (char *)src;
    v25 = (__m128i *)((char *)v64 - ((char *)a2 - (char *)src));
    if ( a2 == src )
    {
      if ( v64 == a2 )
        goto LABEL_34;
    }
    else
    {
      v33 = v22;
      v43 = v23;
      v51 = v24;
      memmove(v9, src, (char *)a2 - (char *)src);
      v30 = (char *)a2 - (char *)src;
      LODWORD(v24) = v51;
      v23 = v43;
      v25 = (__m128i *)((char *)v64 - ((char *)a2 - (char *)src));
      v22 = v33;
      if ( v64 == a2 )
        goto LABEL_46;
    }
    v34 = v22;
    destc = (void *)v30;
    v44 = v25;
    v52 = v23;
    nc = v24;
    memmove(src, a2, (char *)v64 - (char *)a2);
    v22 = v34;
    v30 = (size_t)destc;
    v25 = v44;
    v23 = v52;
    LODWORD(v24) = nc;
LABEL_46:
    if ( v30 )
    {
      v45 = v22;
      v53 = v23;
      nd = v24;
      v31 = (__m128i *)memmove(v25, v9, v30);
      LODWORD(v24) = nd;
      v23 = v53;
      v22 = v45;
      v25 = v31;
    }
LABEL_34:
    v42 = v22;
    v50 = v23;
    nb = (size_t)v25;
    sub_1078C30((_DWORD)a1, (_DWORD)src, (_DWORD)v25, v24, v23, (_DWORD)v9, v22);
    a6 = v9;
    a4 = v62;
    a2 = v64;
    a7 = (const __m128i *)v42;
    a5 = v16 - v50;
    a3 = (__int64)v7;
    a1 = (__m128i *)nb;
  }
  v17 = (__m128i *)a3;
  v18 = a3 - (_QWORD)a2;
  if ( v7 != a2 )
  {
    srca = (void *)v18;
    memmove(a6, a2, v18);
    v18 = (size_t)srca;
  }
  result = (__m128i *)((char *)v9 + v18);
  if ( a1 == a2 )
  {
    if ( v9 == result )
      return result;
    v27 = v9;
    v28 = (__m128i *)((char *)v7 - v18);
    return (const __m128i *)memmove(v28, v27, v18);
  }
  if ( v9 != result )
  {
    v19 = (__m128i *)((char *)a2 - 40);
    while ( 2 )
    {
      result = (const __m128i *)((char *)result - 40);
      while ( 1 )
      {
        v17 = (__m128i *)((char *)v17 - 40);
        if ( *(_QWORD *)(result[2].m128i_i64[0] + 160) + result->m128i_i64[0] >= (unsigned __int64)(*(_QWORD *)(v19[2].m128i_i64[0] + 160)
                                                                                                  + v19->m128i_i64[0]) )
          break;
        *v17 = _mm_loadu_si128(v19);
        v17[1] = _mm_loadu_si128(v19 + 1);
        v17[2].m128i_i64[0] = v19[2].m128i_i64[0];
        if ( a1 == v19 )
        {
          if ( v9 != (__m128i *)&result[2].m128i_u64[1] )
            return (const __m128i *)memmove(
                                      (char *)v17 - ((char *)&result[2].m128i_u64[1] - (char *)v9),
                                      v9,
                                      (char *)&result[2].m128i_u64[1] - (char *)v9);
          return result;
        }
        v19 = (const __m128i *)((char *)v19 - 40);
      }
      *v17 = _mm_loadu_si128(result);
      v17[1] = _mm_loadu_si128(result + 1);
      v17[2].m128i_i64[0] = result[2].m128i_i64[0];
      if ( v9 != result )
        continue;
      break;
    }
  }
  return result;
}
