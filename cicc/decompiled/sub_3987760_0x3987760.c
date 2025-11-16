// Function: sub_3987760
// Address: 0x3987760
//
void *__fastcall sub_3987760(
        __m128i *a1,
        __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128i *a6,
        __int64 a7,
        void *a8)
{
  const __m128i *v9; // r13
  const __m128i *v10; // r12
  __m128i *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 *v14; // r9
  __int64 v15; // rbx
  __int64 v16; // r15
  __m128i *v17; // rax
  const __m128i *v18; // r9
  __m128i *v19; // r12
  __int64 v20; // r11
  const __m128i *v21; // r10
  size_t v22; // r10
  __int64 v23; // rax
  signed __int64 v24; // r14
  void *result; // rax
  __m128i *v26; // r11
  __m128i v27; // xmm0
  __m128i v28; // xmm1
  __m128i *v29; // r11
  __m128i *v30; // r12
  const __m128i *v31; // r11
  __m128i *v32; // rsi
  __m128i *v33; // rdi
  size_t v34; // rdx
  __int64 *v35; // rax
  size_t v36; // r8
  const __m128i *v37; // rax
  const __m128i *v38; // rax
  int v39; // [rsp+0h] [rbp-80h]
  const __m128i *v40; // [rsp+0h] [rbp-80h]
  size_t v41; // [rsp+8h] [rbp-78h]
  size_t v42; // [rsp+8h] [rbp-78h]
  int v43; // [rsp+8h] [rbp-78h]
  int v44; // [rsp+8h] [rbp-78h]
  size_t v45; // [rsp+8h] [rbp-78h]
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
  __m128i *v57; // [rsp+28h] [rbp-58h]
  char *v58; // [rsp+30h] [rbp-50h]
  _QWORD v59[7]; // [rsp+48h] [rbp-38h] BYREF

  v9 = a1;
  v10 = a2;
  v11 = (__m128i *)a3;
  v12 = a5;
  if ( a7 <= a5 )
    v12 = a7;
  if ( a4 <= v12 )
  {
LABEL_22:
    v24 = (char *)v10 - (char *)v9;
    if ( v10 != v9 )
      memmove(a6, v9, (char *)v10 - (char *)v9);
    result = a8;
    v26 = (__m128i *)((char *)a6 + v24);
    v59[0] = a8;
    if ( a6 != (__m128i *)&a6->m128i_i8[v24] && v11 != v10 )
    {
      do
      {
        result = (void *)sub_3985080((__int64)v59, v10->m128i_i64[0], a6->m128i_i64);
        if ( (_BYTE)result )
        {
          v27 = _mm_loadu_si128(v10);
          ++v9;
          ++v10;
          v9[-1] = v27;
          if ( v26 == a6 )
            return result;
        }
        else
        {
          v28 = _mm_loadu_si128(a6++);
          ++v9;
          v9[-1] = v28;
          if ( v26 == a6 )
            return result;
        }
      }
      while ( v11 != v10 );
    }
    if ( v26 != a6 )
    {
      v32 = a6;
      v33 = (__m128i *)v9;
      v34 = (char *)v26 - (char *)a6;
      return memmove(v33, v32, v34);
    }
  }
  else
  {
    v13 = a5;
    if ( a7 < a5 )
    {
      v58 = (char *)a1;
      v14 = (__int64 *)a2;
      v15 = a4;
      dest = a6;
      while ( 1 )
      {
        src = (const __m128i *)v14;
        if ( v15 > v13 )
        {
          v19 = (__m128i *)&v58[16 * (v15 / 2)];
          v35 = sub_3985650(v14, a3, v19->m128i_i64, (__int64)a8);
          v18 = src;
          v20 = v15 / 2;
          v57 = (__m128i *)v35;
          v16 = ((char *)v35 - (char *)src) >> 4;
        }
        else
        {
          v16 = v13 / 2;
          v57 = (__m128i *)&v14[2 * (v13 / 2)];
          v17 = (__m128i *)sub_39865E0(v58, (__int64)v14, v57->m128i_i64, (__int64)a8);
          v18 = src;
          v19 = v17;
          v20 = ((char *)v17 - v58) >> 4;
        }
        v15 -= v20;
        if ( v15 <= v16 || v16 > a7 )
        {
          if ( v15 > a7 )
          {
            srch = v20;
            v38 = sub_3984680(v19, v18, v57);
            LODWORD(v20) = srch;
            v21 = v38;
          }
          else
          {
            v21 = v57;
            if ( v15 )
            {
              v36 = (char *)v18 - (char *)v19;
              if ( v18 != v19 )
              {
                v40 = v18;
                v44 = v20;
                srce = (void *)((char *)v18 - (char *)v19);
                memmove(dest, v19, (char *)v18 - (char *)v19);
                v18 = v40;
                LODWORD(v20) = v44;
                v36 = (size_t)srce;
              }
              if ( v18 != v57 )
              {
                v45 = v36;
                srcf = v20;
                memmove(v19, v18, (char *)v57 - (char *)v18);
                v36 = v45;
                LODWORD(v20) = srcf;
              }
              v21 = (__m128i *)((char *)v57 - v36);
              if ( v36 )
              {
                srcg = v20;
                v37 = (const __m128i *)memmove((char *)v57 - v36, dest, v36);
                LODWORD(v20) = srcg;
                v21 = v37;
              }
            }
          }
        }
        else
        {
          v21 = v19;
          if ( v16 )
          {
            v22 = (char *)v57 - (char *)v18;
            if ( v18 != v57 )
            {
              v39 = v20;
              v41 = (char *)v57 - (char *)v18;
              srcb = v18;
              memmove(dest, v18, (char *)v57 - (char *)v18);
              LODWORD(v20) = v39;
              v22 = v41;
              v18 = srcb;
            }
            if ( v18 != v19 )
            {
              v42 = v22;
              srcc = v20;
              memmove((char *)v57 - ((char *)v18 - (char *)v19), v19, (char *)v18 - (char *)v19);
              v22 = v42;
              LODWORD(v20) = srcc;
            }
            if ( v22 )
            {
              v43 = v20;
              srcd = (void *)v22;
              memmove(v19, dest, v22);
              LODWORD(v20) = v43;
              v22 = (size_t)srcd;
            }
            v21 = (__m128i *)((char *)v19 + v22);
          }
        }
        v13 -= v16;
        srca = (__m128i *)v21;
        sub_3987760((_DWORD)v58, (_DWORD)v19, (_DWORD)v21, v20, v16, (_DWORD)dest, a7, (__int64)a8);
        v23 = a7;
        if ( v13 <= a7 )
          v23 = v13;
        if ( v15 <= v23 )
        {
          v11 = (__m128i *)a3;
          a6 = dest;
          v9 = srca;
          v10 = v57;
          goto LABEL_22;
        }
        if ( v13 <= a7 )
          break;
        v58 = (char *)srca;
        v14 = (__int64 *)v57;
      }
      v11 = (__m128i *)a3;
      a6 = dest;
      v9 = srca;
      v10 = v57;
    }
    if ( v11 != v10 )
      memmove(a6, v10, (char *)v11 - (char *)v10);
    result = a8;
    v29 = (__m128i *)((char *)a6 + (char *)v11 - (char *)v10);
    v59[0] = a8;
    if ( v10 == v9 )
    {
      if ( a6 != v29 )
      {
        v34 = (char *)v11 - (char *)v10;
        v33 = (__m128i *)v10;
        goto LABEL_58;
      }
    }
    else if ( a6 != v29 )
    {
      v30 = (__m128i *)&v10[-1];
      v31 = v29 - 1;
      while ( 1 )
      {
        while ( 1 )
        {
          --v11;
          result = (void *)sub_3985080((__int64)v59, v31->m128i_i64[0], v30->m128i_i64);
          if ( (_BYTE)result )
            break;
          *v11 = _mm_loadu_si128(v31);
          if ( a6 == v31 )
            return result;
          --v31;
        }
        *v11 = _mm_loadu_si128(v30);
        if ( v30 == v9 )
          break;
        --v30;
      }
      if ( a6 != &v31[1] )
      {
        v34 = (char *)&v31[1] - (char *)a6;
        v33 = (__m128i *)((char *)v11 - v34);
LABEL_58:
        v32 = a6;
        return memmove(v33, v32, v34);
      }
    }
  }
  return result;
}
