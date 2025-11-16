// Function: sub_349C380
// Address: 0x349c380
//
unsigned __int64 __fastcall sub_349C380(
        __int64 a1,
        const __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        const __m128i *a6,
        __int64 a7)
{
  __int64 v7; // rax
  const __m128i *v9; // r12
  __m128i *v10; // rbx
  __int64 v11; // r13
  unsigned __int64 result; // rax
  __int64 v13; // rdi
  unsigned __int64 v14; // rcx
  const __m128i *v15; // rdx
  const __m128i *v16; // rax
  __m128i v17; // xmm4
  __int64 v18; // rdx
  const __m128i *v19; // rcx
  const __m128i *v20; // rsi
  __int64 v21; // rax
  unsigned int v22; // edx
  __int64 v23; // r14
  __int64 v24; // r10
  __int64 v25; // rax
  const __m128i *v26; // r11
  int v27; // r10d
  unsigned __int64 v28; // r8
  const __m128i *v29; // r15
  __int64 v30; // r12
  unsigned __int64 v31; // rax
  int v32; // r8d
  __int64 v33; // rdx
  unsigned __int64 v34; // rcx
  const __m128i *v35; // rdx
  __m128i v36; // xmm0
  __int64 v37; // rcx
  const __m128i *v38; // rcx
  __m128i v39; // xmm2
  __int64 v40; // rdx
  __int64 v41; // rax
  unsigned int v42; // edx
  __m128i v43; // xmm3
  __int64 v44; // rcx
  __m128i v45; // xmm1
  __int64 v46; // rax
  __int8 *v47; // rcx
  __m128i v48; // xmm7
  unsigned __int64 v49; // rdx
  __m128i v50; // xmm7
  int v52; // [rsp+18h] [rbp-58h]
  unsigned __int64 v53; // [rsp+20h] [rbp-50h]
  __int64 v54; // [rsp+28h] [rbp-48h]
  unsigned __int64 v55; // [rsp+28h] [rbp-48h]
  unsigned __int64 v56; // [rsp+28h] [rbp-48h]
  __m128i *v57; // [rsp+30h] [rbp-40h]
  const __m128i *v58; // [rsp+38h] [rbp-38h]

  v7 = a5;
  v9 = (const __m128i *)a1;
  v10 = (__m128i *)a3;
  if ( a7 <= a5 )
    v7 = a7;
  if ( v7 < a4 )
  {
    v11 = a5;
    if ( a7 >= a5 )
    {
LABEL_5:
      result = 0xAAAAAAAAAAAAAAABLL;
      v13 = (char *)v10 - (char *)a2;
      v14 = 0xAAAAAAAAAAAAAAABLL * (((char *)v10 - (char *)a2) >> 3);
      if ( (char *)v10 - (char *)a2 <= 0 )
        return result;
      v15 = a6;
      v16 = a2;
      do
      {
        v17 = _mm_loadu_si128(v16);
        v15 = (const __m128i *)((char *)v15 + 24);
        v16 = (const __m128i *)((char *)v16 + 24);
        *(__m128i *)((char *)v15 - 24) = v17;
        v15[-1].m128i_i32[2] = v16[-1].m128i_i32[2];
        --v14;
      }
      while ( v14 );
      if ( v13 <= 0 )
        v13 = 24;
      result = (unsigned __int64)a6->m128i_u64 + v13;
      if ( v9 == a2 )
      {
        v49 = 0xAAAAAAAAAAAAAAABLL * (v13 >> 3);
        do
        {
          v50 = _mm_loadu_si128((const __m128i *)(result - 24));
          result -= 24LL;
          v10 = (__m128i *)((char *)v10 - 24);
          *v10 = v50;
          v10[1].m128i_i32[0] = *(_DWORD *)(result + 16);
          --v49;
        }
        while ( v49 );
        return result;
      }
      if ( a6 == (const __m128i *)result )
        return result;
      v18 = *(unsigned int *)(result - 8);
      v19 = (const __m128i *)(result - 24);
      v20 = (const __m128i *)((char *)a2 - 24);
      v21 = a2[-1].m128i_u32[2];
      if ( (unsigned int)v18 <= 6 )
      {
        while ( 1 )
        {
          v22 = dword_44E2140[v18];
          if ( (unsigned int)v21 > 6 )
            goto LABEL_57;
          v10 = (__m128i *)((char *)v10 - 24);
          if ( v22 > dword_44E2140[v21] )
          {
            *v10 = _mm_loadu_si128(v20);
            v10[1].m128i_i32[0] = v20[1].m128i_i32[0];
            if ( v9 == v20 )
            {
              v47 = &v19[1].m128i_i8[8];
              result = 0xAAAAAAAAAAAAAAABLL * ((v47 - (__int8 *)a6) >> 3);
              if ( v47 - (__int8 *)a6 > 0 )
              {
                do
                {
                  v48 = _mm_loadu_si128((const __m128i *)(v47 - 24));
                  v47 -= 24;
                  v10 = (__m128i *)((char *)v10 - 24);
                  *v10 = v48;
                  v10[1].m128i_i32[0] = *((_DWORD *)v47 + 4);
                  --result;
                }
                while ( result );
              }
              return result;
            }
            v20 = (const __m128i *)((char *)v20 - 24);
          }
          else
          {
            *v10 = _mm_loadu_si128(v19);
            result = v19[1].m128i_u32[0];
            v10[1].m128i_i32[0] = result;
            if ( a6 == v19 )
              return result;
            v19 = (const __m128i *)((char *)v19 - 24);
          }
          v18 = v19[1].m128i_u32[0];
          v21 = v20[1].m128i_u32[0];
          if ( (unsigned int)v18 > 6 )
            goto LABEL_57;
        }
      }
      goto LABEL_57;
    }
    v23 = a4;
    v24 = a1;
    v58 = a6;
    while ( 1 )
    {
      v54 = v24;
      if ( v23 > v11 )
      {
        v30 = v23 / 2;
        v29 = (const __m128i *)(v24 + 8 * (v23 / 2 + ((v23 + ((unsigned __int64)v23 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
        v46 = sub_3440630((__int64)a2, a3, (__int64)v29);
        v27 = v54;
        v57 = (__m128i *)v46;
        v28 = 0xAAAAAAAAAAAAAAABLL * ((v46 - (__int64)v26) >> 3);
      }
      else
      {
        v57 = (__m128i *)((char *)a2 + 8 * (v11 / 2)
                                     + 8 * ((v11 + ((unsigned __int64)v11 >> 63)) & 0xFFFFFFFFFFFFFFFELL));
        v25 = sub_34406C0(v24, (__int64)a2, (__int64)v57);
        v27 = v54;
        v28 = v11 / 2;
        v29 = (const __m128i *)v25;
        v30 = 0xAAAAAAAAAAAAAAABLL * ((v25 - v54) >> 3);
      }
      v23 -= v30;
      v52 = v27;
      v55 = v28;
      v31 = sub_349C0F0(v29, v26, v57, v23, v28, v58, a7);
      v32 = v55;
      v53 = v55;
      v56 = v31;
      sub_349C380(v52, (_DWORD)v29, v31, v30, v32, (_DWORD)v58, a7);
      v33 = a7;
      v11 -= v53;
      if ( v11 <= a7 )
        v33 = v11;
      if ( v23 <= v33 )
        break;
      if ( v11 <= a7 )
      {
        v10 = (__m128i *)a3;
        a6 = v58;
        v9 = (const __m128i *)v56;
        a2 = v57;
        goto LABEL_5;
      }
      a2 = v57;
      v24 = v56;
    }
    v10 = (__m128i *)a3;
    a6 = v58;
    v9 = (const __m128i *)v56;
    a2 = v57;
  }
  result = 0xAAAAAAAAAAAAAAABLL;
  v34 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)v9) >> 3);
  if ( (char *)a2 - (char *)v9 <= 0 )
    return result;
  v35 = a6;
  result = (unsigned __int64)v9;
  do
  {
    v36 = _mm_loadu_si128((const __m128i *)result);
    v35 = (const __m128i *)((char *)v35 + 24);
    result += 24LL;
    *(__m128i *)((char *)v35 - 24) = v36;
    v35[-1].m128i_i32[2] = *(_DWORD *)(result - 8);
    --v34;
  }
  while ( v34 );
  v37 = 24;
  if ( (char *)a2 - (char *)v9 > 0 )
    v37 = (char *)a2 - (char *)v9;
  v38 = (const __m128i *)((char *)a6 + v37);
  if ( v10 != a2 )
  {
    if ( a6 == v38 )
      return result;
    while ( 1 )
    {
      v40 = a2[1].m128i_u32[0];
      v41 = a6[1].m128i_u32[0];
      if ( (unsigned int)v40 > 6 )
        break;
      v42 = dword_44E2140[v40];
      if ( (unsigned int)v41 > 6 )
        break;
      if ( v42 > dword_44E2140[v41] )
      {
        v39 = _mm_loadu_si128(a2);
        v9 = (const __m128i *)((char *)v9 + 24);
        a2 = (const __m128i *)((char *)a2 + 24);
        *(__m128i *)((char *)v9 - 24) = v39;
        result = a2[-1].m128i_u32[2];
        v9[-1].m128i_i32[2] = result;
        if ( a6 == v38 )
          return result;
      }
      else
      {
        v43 = _mm_loadu_si128(a6);
        a6 = (const __m128i *)((char *)a6 + 24);
        v9 = (const __m128i *)((char *)v9 + 24);
        *(__m128i *)((char *)v9 - 24) = v43;
        result = a6[-1].m128i_u32[2];
        v9[-1].m128i_i32[2] = result;
        if ( a6 == v38 )
          return result;
      }
      if ( v10 == a2 )
        goto LABEL_43;
    }
LABEL_57:
    BUG();
  }
LABEL_43:
  if ( a6 != v38 )
  {
    v44 = (char *)v38 - (char *)a6;
    result = 0xAAAAAAAAAAAAAAABLL * (v44 >> 3);
    if ( v44 > 0 )
    {
      do
      {
        v45 = _mm_loadu_si128(a6);
        v9 = (const __m128i *)((char *)v9 + 24);
        a6 = (const __m128i *)((char *)a6 + 24);
        *(__m128i *)((char *)v9 - 24) = v45;
        v9[-1].m128i_i32[2] = a6[-1].m128i_i32[2];
        --result;
      }
      while ( result );
    }
  }
  return result;
}
