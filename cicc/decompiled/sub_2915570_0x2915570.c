// Function: sub_2915570
// Address: 0x2915570
//
const __m128i *__fastcall sub_2915570(
        const __m128i *a1,
        const __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128i *a6,
        const __m128i *a7)
{
  const __m128i *result; // rax
  const __m128i *v8; // r14
  const __m128i *v10; // r12
  const __m128i *v11; // rbx
  __int64 v12; // r15
  __int64 *v13; // r9
  __int64 v14; // rbx
  __int64 *i; // r11
  __int64 v16; // r13
  __m128i *v17; // rax
  __int64 v18; // r11
  const __m128i *v19; // r9
  __m128i *v20; // r12
  __int64 v21; // rcx
  __int8 *v22; // r10
  size_t v23; // r10
  __m128i *v24; // rcx
  __int64 v25; // rax
  __m128i v26; // xmm0
  __m128i v27; // xmm1
  const __m128i *v28; // r12
  __m128i *v29; // rbx
  __int64 v30; // rdx
  __m128i *v31; // rsi
  __m128i *v32; // rdi
  size_t v33; // rdx
  __m128i *v34; // rax
  size_t v35; // r8
  __int8 *v36; // rax
  unsigned __int64 v37; // rax
  int v38; // [rsp+8h] [rbp-68h]
  const __m128i *v39; // [rsp+8h] [rbp-68h]
  int v40; // [rsp+10h] [rbp-60h]
  int v41; // [rsp+10h] [rbp-60h]
  int v42; // [rsp+10h] [rbp-60h]
  int v43; // [rsp+10h] [rbp-60h]
  int v44; // [rsp+10h] [rbp-60h]
  size_t v45; // [rsp+18h] [rbp-58h]
  size_t v46; // [rsp+18h] [rbp-58h]
  int v47; // [rsp+18h] [rbp-58h]
  int v48; // [rsp+18h] [rbp-58h]
  size_t v49; // [rsp+18h] [rbp-58h]
  int v50; // [rsp+18h] [rbp-58h]
  int v51; // [rsp+18h] [rbp-58h]
  const __m128i *src; // [rsp+28h] [rbp-48h]
  const __m128i *srcb; // [rsp+28h] [rbp-48h]
  int srcc; // [rsp+28h] [rbp-48h]
  void *srcd; // [rsp+28h] [rbp-48h]
  const __m128i *srca; // [rsp+28h] [rbp-48h]
  void *srce; // [rsp+28h] [rbp-48h]
  int srcf; // [rsp+28h] [rbp-48h]
  int srcg; // [rsp+28h] [rbp-48h]
  int srch; // [rsp+28h] [rbp-48h]
  __m128i *dest; // [rsp+30h] [rbp-40h]
  __m128i *v63; // [rsp+38h] [rbp-38h]

  result = (const __m128i *)a5;
  v8 = a1;
  v10 = a2;
  v11 = (const __m128i *)a3;
  if ( (__int64)a7 <= a5 )
    result = a7;
  if ( (__int64)result >= a4 )
  {
LABEL_22:
    if ( v10 != v8 )
      result = (const __m128i *)memmove(a6, v8, (char *)v10 - (char *)v8);
    v24 = (__m128i *)((char *)a6 + (char *)v10 - (char *)v8);
    if ( v11 == v10 || a6 == v24 )
    {
LABEL_55:
      if ( v24 == a6 )
        return result;
      v31 = a6;
      v32 = (__m128i *)v8;
      v33 = (char *)v24 - (char *)a6;
      return (const __m128i *)memmove(v32, v31, v33);
    }
    while ( v10->m128i_i64[0] >= (unsigned __int64)a6->m128i_i64[0] )
    {
      if ( v10->m128i_i64[0] <= (unsigned __int64)a6->m128i_i64[0] )
      {
        v25 = (v10[1].m128i_i64[0] >> 2) & 1;
        if ( (_BYTE)v25 == ((a6[1].m128i_i64[0] >> 2) & 1) )
        {
          if ( v10->m128i_i64[1] > (unsigned __int64)a6->m128i_i64[1] )
            break;
        }
        else if ( !(_BYTE)v25 )
        {
          break;
        }
      }
      v27 = _mm_loadu_si128(a6);
      a6 = (__m128i *)((char *)a6 + 24);
      v8 = (const __m128i *)((char *)v8 + 24);
      *(__m128i *)((char *)v8 - 24) = v27;
      result = (const __m128i *)a6[-1].m128i_i64[1];
      v8[-1].m128i_i64[1] = (__int64)result;
      if ( v24 == a6 )
        return result;
LABEL_30:
      if ( v11 == v10 )
        goto LABEL_55;
    }
    v26 = _mm_loadu_si128(v10);
    v8 = (const __m128i *)((char *)v8 + 24);
    v10 = (const __m128i *)((char *)v10 + 24);
    *(__m128i *)((char *)v8 - 24) = v26;
    result = (const __m128i *)v10[-1].m128i_i64[1];
    v8[-1].m128i_i64[1] = (__int64)result;
    if ( v24 == a6 )
      return result;
    goto LABEL_30;
  }
  v12 = a5;
  if ( (__int64)a7 >= a5 )
    goto LABEL_39;
  v13 = (__int64 *)a2;
  v14 = a4;
  dest = a6;
  for ( i = (__int64 *)a1; ; i = (__int64 *)srca )
  {
    src = (const __m128i *)v13;
    if ( v14 > v12 )
    {
      v20 = (__m128i *)&i[v14 / 2 + ((v14 + ((unsigned __int64)v14 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
      v34 = (__m128i *)sub_2913650(v13, a3, (unsigned __int64 *)v20);
      v19 = src;
      v21 = v14 / 2;
      v63 = v34;
      v16 = 0xAAAAAAAAAAAAAAABLL * (((char *)v34 - (char *)src) >> 3);
    }
    else
    {
      v16 = v12 / 2;
      v63 = (__m128i *)&v13[v12 / 2 + ((v12 + ((unsigned __int64)v12 >> 63)) & 0xFFFFFFFFFFFFFFFELL)];
      v17 = (__m128i *)sub_29135A0(i, (__int64)v13, (unsigned __int64 *)v63);
      v19 = src;
      v20 = v17;
      v21 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v17->m128i_i64 - v18) >> 3);
    }
    v14 -= v21;
    if ( v14 <= v16 || (__int64)a7 < v16 )
    {
      if ( (__int64)a7 < v14 )
      {
        v51 = v18;
        srch = v21;
        v37 = sub_29133D0(v20, v19, v63);
        LODWORD(v18) = v51;
        LODWORD(v21) = srch;
        v22 = (__int8 *)v37;
      }
      else
      {
        v22 = (__int8 *)v63;
        if ( v14 )
        {
          v35 = (char *)v19 - (char *)v20;
          if ( v19 != v20 )
          {
            v39 = v19;
            v43 = v18;
            v48 = v21;
            srce = (void *)((char *)v19 - (char *)v20);
            memmove(dest, v20, (char *)v19 - (char *)v20);
            v19 = v39;
            LODWORD(v18) = v43;
            LODWORD(v21) = v48;
            v35 = (size_t)srce;
          }
          if ( v19 != v63 )
          {
            v44 = v18;
            v49 = v35;
            srcf = v21;
            memmove(v20, v19, (char *)v63 - (char *)v19);
            LODWORD(v18) = v44;
            v35 = v49;
            LODWORD(v21) = srcf;
          }
          v22 = &v63->m128i_i8[-v35];
          if ( v35 )
          {
            v50 = v18;
            srcg = v21;
            v36 = (__int8 *)memmove((char *)v63 - v35, dest, v35);
            LODWORD(v21) = srcg;
            LODWORD(v18) = v50;
            v22 = v36;
          }
        }
      }
    }
    else
    {
      v22 = (__int8 *)v20;
      if ( v16 )
      {
        v23 = (char *)v63 - (char *)v19;
        if ( v19 != v63 )
        {
          v38 = v18;
          v40 = v21;
          v45 = (char *)v63 - (char *)v19;
          srcb = v19;
          memmove(dest, v19, (char *)v63 - (char *)v19);
          LODWORD(v18) = v38;
          LODWORD(v21) = v40;
          v23 = v45;
          v19 = srcb;
        }
        if ( v19 != v20 )
        {
          v41 = v18;
          v46 = v23;
          srcc = v21;
          memmove((char *)v63 - ((char *)v19 - (char *)v20), v20, (char *)v19 - (char *)v20);
          LODWORD(v18) = v41;
          v23 = v46;
          LODWORD(v21) = srcc;
        }
        if ( v23 )
        {
          v42 = v18;
          v47 = v21;
          srcd = (void *)v23;
          memmove(v20, dest, v23);
          LODWORD(v18) = v42;
          LODWORD(v21) = v47;
          v23 = (size_t)srcd;
        }
        v22 = &v20->m128i_i8[v23];
      }
    }
    v12 -= v16;
    srca = (const __m128i *)v22;
    sub_2915570(v18, (_DWORD)v20, (_DWORD)v22, v21, v16, (_DWORD)dest, (__int64)a7);
    result = (const __m128i *)v12;
    if ( (__int64)a7 <= v12 )
      result = a7;
    if ( (__int64)result >= v14 )
    {
      v11 = (const __m128i *)a3;
      a6 = dest;
      v8 = srca;
      v10 = v63;
      goto LABEL_22;
    }
    if ( (__int64)a7 >= v12 )
      break;
    v13 = (__int64 *)v63;
  }
  v11 = (const __m128i *)a3;
  a6 = dest;
  v8 = srca;
  v10 = v63;
LABEL_39:
  if ( v11 != v10 )
    memmove(a6, v10, (char *)v11 - (char *)v10);
  result = (__m128i *)((char *)a6 + (char *)v11 - (char *)v10);
  if ( v10 == v8 )
  {
    if ( a6 == result )
      return result;
    v33 = (char *)v11 - (char *)v10;
    v32 = (__m128i *)v10;
LABEL_69:
    v31 = a6;
    return (const __m128i *)memmove(v32, v31, v33);
  }
  if ( a6 == result )
    return result;
  v28 = (const __m128i *)((char *)v10 - 24);
  result = (const __m128i *)((char *)result - 24);
  v29 = (__m128i *)&v11[-2].m128i_u64[1];
  while ( 2 )
  {
    if ( result->m128i_i64[0] < (unsigned __int64)v28->m128i_i64[0] )
      goto LABEL_46;
    if ( result->m128i_i64[0] > (unsigned __int64)v28->m128i_i64[0] )
      goto LABEL_51;
    v30 = (result[1].m128i_i64[0] >> 2) & 1;
    if ( (_BYTE)v30 == ((v28[1].m128i_i64[0] >> 2) & 1) )
    {
      if ( result->m128i_i64[1] > (unsigned __int64)v28->m128i_i64[1] )
        goto LABEL_46;
LABEL_51:
      *v29 = _mm_loadu_si128(result);
      v29[1].m128i_i64[0] = result[1].m128i_i64[0];
      if ( a6 == result )
        return result;
      result = (const __m128i *)((char *)result - 24);
LABEL_48:
      v29 = (__m128i *)((char *)v29 - 24);
      continue;
    }
    break;
  }
  if ( (_BYTE)v30 )
    goto LABEL_51;
LABEL_46:
  *v29 = _mm_loadu_si128(v28);
  v29[1].m128i_i64[0] = v28[1].m128i_i64[0];
  if ( v28 != v8 )
  {
    v28 = (const __m128i *)((char *)v28 - 24);
    goto LABEL_48;
  }
  if ( a6 != (__m128i *)&result[1].m128i_u64[1] )
  {
    v33 = (char *)&result[1].m128i_u64[1] - (char *)a6;
    v32 = (__m128i *)((char *)v29 - v33);
    goto LABEL_69;
  }
  return result;
}
