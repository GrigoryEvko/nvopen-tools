// Function: sub_1B2CC40
// Address: 0x1b2cc40
//
char __fastcall sub_1B2CC40(
        __m128i *a1,
        const __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128i *a6,
        __int64 a7,
        __int64 a8)
{
  __m128i *v9; // r13
  const __m128i *v10; // r12
  __m128i *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // r9
  __int64 v15; // rbx
  __int64 v16; // r15
  __int64 v17; // rax
  const __m128i *v18; // r9
  __m128i *v19; // r12
  __int64 v20; // r11
  __m128i *v21; // r10
  size_t v22; // r10
  __int64 v23; // rax
  char result; // al
  __m128i *v25; // r14
  __m128i v26; // xmm0
  __m128i v27; // xmm3
  size_t v28; // rdx
  __m128i *v29; // r14
  const __m128i *v30; // r12
  __m128i *v31; // rax
  const __m128i *v32; // r13
  const __m128i *v33; // r14
  __int64 v34; // rax
  size_t v35; // r8
  __m128i *v36; // rax
  __m128i *v37; // rax
  int v38; // [rsp+0h] [rbp-80h]
  const __m128i *v39; // [rsp+0h] [rbp-80h]
  size_t v40; // [rsp+8h] [rbp-78h]
  size_t v41; // [rsp+8h] [rbp-78h]
  int v42; // [rsp+8h] [rbp-78h]
  int v43; // [rsp+8h] [rbp-78h]
  size_t v44; // [rsp+8h] [rbp-78h]
  const __m128i *src; // [rsp+18h] [rbp-68h]
  const __m128i *srcb; // [rsp+18h] [rbp-68h]
  int srcc; // [rsp+18h] [rbp-68h]
  void *srcd; // [rsp+18h] [rbp-68h]
  __m128i *srca; // [rsp+18h] [rbp-68h]
  void *srce; // [rsp+18h] [rbp-68h]
  int srcf; // [rsp+18h] [rbp-68h]
  int srcg; // [rsp+18h] [rbp-68h]
  int srch; // [rsp+18h] [rbp-68h]
  __m128i *dest; // [rsp+20h] [rbp-60h]
  const __m128i *v56; // [rsp+28h] [rbp-58h]
  __m128i *v57; // [rsp+30h] [rbp-50h]
  __int64 v58[7]; // [rsp+48h] [rbp-38h] BYREF

  v9 = a1;
  v10 = a2;
  v11 = (__m128i *)a3;
  v12 = a5;
  if ( a7 <= a5 )
    v12 = a7;
  if ( a4 <= v12 )
  {
LABEL_22:
    if ( v10 != v9 )
      memmove(a6, v9, (char *)v10 - (char *)v9);
    result = a8;
    v25 = (__m128i *)((char *)a6 + (char *)v10 - (char *)v9);
    v58[0] = a8;
    if ( a6 != v25 && v11 != v10 )
    {
      do
      {
        result = sub_1B2B020(v58, (__int64)v10, (__int64)a6);
        if ( result )
        {
          v26 = _mm_loadu_si128(v10);
          v9 += 3;
          v10 += 3;
          v9[-3] = v26;
          v9[-2] = _mm_loadu_si128(v10 - 2);
          v9[-1] = _mm_loadu_si128(v10 - 1);
          if ( v25 == a6 )
            return result;
        }
        else
        {
          v27 = _mm_loadu_si128(a6);
          a6 += 3;
          v9 += 3;
          v9[-3] = v27;
          v9[-2] = _mm_loadu_si128(a6 - 2);
          v9[-1] = _mm_loadu_si128(a6 - 1);
          if ( v25 == a6 )
            return result;
        }
      }
      while ( v11 != v10 );
    }
    if ( v25 != a6 )
      return (unsigned __int8)memmove(v9, a6, (char *)v25 - (char *)a6);
  }
  else
  {
    v13 = a5;
    if ( a7 < a5 )
    {
      v57 = a1;
      v14 = (__int64)a2;
      v15 = a4;
      dest = a6;
      while ( 1 )
      {
        src = (const __m128i *)v14;
        if ( v15 > v13 )
        {
          v19 = &v57[((v15 + ((unsigned __int64)v15 >> 63)) & 0xFFFFFFFFFFFFFFFELL) + v15 / 2];
          v34 = sub_1B2C840(v14, a3, (__int64)v19, a8);
          v18 = src;
          v20 = v15 / 2;
          v56 = (const __m128i *)v34;
          v16 = 0xAAAAAAAAAAAAAAABLL * ((v34 - (__int64)src) >> 4);
        }
        else
        {
          v16 = v13 / 2;
          v56 = (const __m128i *)(v14 + 16 * (v13 / 2 + ((v13 + ((unsigned __int64)v13 >> 63)) & 0xFFFFFFFFFFFFFFFELL)));
          v17 = sub_1B2C8E0((__int64)v57, v14, (__int64)v56, a8);
          v18 = src;
          v19 = (__m128i *)v17;
          v20 = 0xAAAAAAAAAAAAAAABLL * ((v17 - (__int64)v57) >> 4);
        }
        v15 -= v20;
        if ( v15 <= v16 || v16 > a7 )
        {
          if ( v15 > a7 )
          {
            srch = v20;
            v37 = sub_1B29FB0(v19, v18, v56);
            LODWORD(v20) = srch;
            v21 = v37;
          }
          else
          {
            v21 = (__m128i *)v56;
            if ( v15 )
            {
              v35 = (char *)v18 - (char *)v19;
              if ( v18 != v19 )
              {
                v39 = v18;
                v43 = v20;
                srce = (void *)((char *)v18 - (char *)v19);
                memmove(dest, v19, (char *)v18 - (char *)v19);
                v18 = v39;
                LODWORD(v20) = v43;
                v35 = (size_t)srce;
              }
              if ( v18 != v56 )
              {
                v44 = v35;
                srcf = v20;
                memmove(v19, v18, (char *)v56 - (char *)v18);
                v35 = v44;
                LODWORD(v20) = srcf;
              }
              v21 = (__m128i *)((char *)v56 - v35);
              if ( v35 )
              {
                srcg = v20;
                v36 = (__m128i *)memmove((char *)v56 - v35, dest, v35);
                LODWORD(v20) = srcg;
                v21 = v36;
              }
            }
          }
        }
        else
        {
          v21 = v19;
          if ( v16 )
          {
            v22 = (char *)v56 - (char *)v18;
            if ( v18 != v56 )
            {
              v38 = v20;
              v40 = (char *)v56 - (char *)v18;
              srcb = v18;
              memmove(dest, v18, (char *)v56 - (char *)v18);
              LODWORD(v20) = v38;
              v22 = v40;
              v18 = srcb;
            }
            if ( v18 != v19 )
            {
              v41 = v22;
              srcc = v20;
              memmove((char *)v56 - ((char *)v18 - (char *)v19), v19, (char *)v18 - (char *)v19);
              v22 = v41;
              LODWORD(v20) = srcc;
            }
            if ( v22 )
            {
              v42 = v20;
              srcd = (void *)v22;
              memmove(v19, dest, v22);
              LODWORD(v20) = v42;
              v22 = (size_t)srcd;
            }
            v21 = (__m128i *)((char *)v19 + v22);
          }
        }
        v13 -= v16;
        srca = v21;
        sub_1B2CC40((_DWORD)v57, (_DWORD)v19, (_DWORD)v21, v20, v16, (_DWORD)dest, a7, a8);
        v23 = a7;
        if ( v13 <= a7 )
          v23 = v13;
        if ( v15 <= v23 )
        {
          v11 = (__m128i *)a3;
          a6 = dest;
          v9 = srca;
          v10 = v56;
          goto LABEL_22;
        }
        if ( v13 <= a7 )
          break;
        v57 = srca;
        v14 = (__int64)v56;
      }
      v11 = (__m128i *)a3;
      a6 = dest;
      v9 = srca;
      v10 = v56;
    }
    v28 = (char *)v11 - (char *)v10;
    if ( v11 != v10 )
    {
      memmove(a6, v10, v28);
      v28 = (char *)v11 - (char *)v10;
    }
    result = a8;
    v29 = (__m128i *)((char *)a6 + v28);
    v58[0] = a8;
    if ( v10 == v9 )
    {
      if ( a6 != v29 )
        return (unsigned __int8)memmove((char *)v11 - v28, a6, v28);
    }
    else if ( a6 != v29 )
    {
      v30 = v10 - 3;
      v31 = v9;
      v32 = v29 - 3;
      v33 = v31;
      while ( 1 )
      {
        while ( 1 )
        {
          v11 -= 3;
          result = sub_1B2B020(v58, (__int64)v32, (__int64)v30);
          if ( result )
            break;
          *v11 = _mm_loadu_si128(v32);
          v11[1] = _mm_loadu_si128(v32 + 1);
          v11[2] = _mm_loadu_si128(v32 + 2);
          if ( a6 == v32 )
            return result;
          v32 -= 3;
        }
        *v11 = _mm_loadu_si128(v30);
        v11[1] = _mm_loadu_si128(v30 + 1);
        v11[2] = _mm_loadu_si128(v30 + 2);
        if ( v30 == v33 )
          break;
        v30 -= 3;
      }
      if ( a6 != &v32[3] )
      {
        v28 = (char *)&v32[3] - (char *)a6;
        return (unsigned __int8)memmove((char *)v11 - v28, a6, v28);
      }
    }
  }
  return result;
}
