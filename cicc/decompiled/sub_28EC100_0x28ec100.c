// Function: sub_28EC100
// Address: 0x28ec100
//
const __m128i *__fastcall sub_28EC100(
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
  __m128i *v11; // rbx
  __int64 v12; // r15
  __m128i *v13; // r9
  __int64 v14; // rbx
  __m128i *v15; // r11
  __int64 v16; // r13
  const __m128i *v17; // r9
  __int64 v18; // r11
  __m128i *v19; // r12
  __int64 v20; // rcx
  const __m128i *v21; // r10
  size_t v22; // r10
  __m128i *v23; // rdx
  __m128i v24; // xmm0
  __m128i v25; // xmm1
  const __m128i *v26; // r12
  size_t v27; // rdx
  __m128i *v28; // rsi
  __m128i *v29; // rdi
  __m128i *v30; // rax
  size_t v31; // r8
  const __m128i *v32; // rax
  const __m128i *v33; // rax
  int v34; // [rsp+8h] [rbp-68h]
  const __m128i *v35; // [rsp+8h] [rbp-68h]
  int v36; // [rsp+10h] [rbp-60h]
  int v37; // [rsp+10h] [rbp-60h]
  int v38; // [rsp+10h] [rbp-60h]
  int v39; // [rsp+10h] [rbp-60h]
  int v40; // [rsp+10h] [rbp-60h]
  size_t v41; // [rsp+18h] [rbp-58h]
  size_t v42; // [rsp+18h] [rbp-58h]
  int v43; // [rsp+18h] [rbp-58h]
  int v44; // [rsp+18h] [rbp-58h]
  size_t v45; // [rsp+18h] [rbp-58h]
  int v46; // [rsp+18h] [rbp-58h]
  int v47; // [rsp+18h] [rbp-58h]
  const __m128i *v49; // [rsp+28h] [rbp-48h]
  int v50; // [rsp+28h] [rbp-48h]
  size_t v51; // [rsp+28h] [rbp-48h]
  const __m128i *v52; // [rsp+28h] [rbp-48h]
  size_t v53; // [rsp+28h] [rbp-48h]
  int v54; // [rsp+28h] [rbp-48h]
  int v55; // [rsp+28h] [rbp-48h]
  int v56; // [rsp+28h] [rbp-48h]
  __m128i *dest; // [rsp+30h] [rbp-40h]
  __m128i *v58; // [rsp+38h] [rbp-38h]

  result = (const __m128i *)a5;
  v8 = a1;
  v10 = a2;
  v11 = (__m128i *)a3;
  if ( (__int64)a7 <= a5 )
    result = a7;
  if ( a4 <= (__int64)result )
  {
LABEL_22:
    if ( v10 != v8 )
      result = (const __m128i *)memmove(a6, v8, (char *)v10 - (char *)v8);
    v23 = (__m128i *)((char *)a6 + (char *)v10 - (char *)v8);
    if ( v11 != v10 && a6 != v23 )
    {
      do
      {
        result = (const __m128i *)a6->m128i_u32[0];
        if ( v10->m128i_i32[0] > (unsigned int)result )
        {
          v24 = _mm_loadu_si128(v10);
          ++v8;
          ++v10;
          v8[-1] = v24;
          if ( v23 == a6 )
            return result;
        }
        else
        {
          v25 = _mm_loadu_si128(a6++);
          ++v8;
          v8[-1] = v25;
          if ( v23 == a6 )
            return result;
        }
      }
      while ( v11 != v10 );
    }
    if ( v23 != a6 )
    {
      v27 = (char *)v23 - (char *)a6;
      v28 = a6;
      v29 = (__m128i *)v8;
      return (const __m128i *)memmove(v29, v28, v27);
    }
  }
  else
  {
    v12 = a5;
    if ( (__int64)a7 < a5 )
    {
      v13 = (__m128i *)a2;
      v14 = a4;
      v15 = (__m128i *)a1;
      dest = a6;
      while ( 1 )
      {
        if ( v12 < v14 )
        {
          v19 = &v15[v14 / 2];
          v30 = (__m128i *)sub_28EA1A0(v13, a3, v19);
          v20 = v14 / 2;
          v58 = v30;
          v16 = v30 - v17;
        }
        else
        {
          v16 = v12 / 2;
          v58 = &v13[v12 / 2];
          v19 = (__m128i *)sub_28EA150(v15, (__int64)v13, v58);
          v20 = ((__int64)v19->m128i_i64 - v18) >> 4;
        }
        v14 -= v20;
        if ( v14 <= v16 || (__int64)a7 < v16 )
        {
          if ( (__int64)a7 < v14 )
          {
            v47 = v18;
            v56 = v20;
            v33 = sub_28E94B0(v19, v17, v58);
            LODWORD(v18) = v47;
            LODWORD(v20) = v56;
            v21 = v33;
          }
          else
          {
            v21 = v58;
            if ( v14 )
            {
              v31 = (char *)v17 - (char *)v19;
              if ( v17 != v19 )
              {
                v35 = v17;
                v39 = v18;
                v44 = v20;
                v53 = (char *)v17 - (char *)v19;
                memmove(dest, v19, (char *)v17 - (char *)v19);
                v17 = v35;
                LODWORD(v18) = v39;
                LODWORD(v20) = v44;
                v31 = v53;
              }
              if ( v17 != v58 )
              {
                v40 = v18;
                v45 = v31;
                v54 = v20;
                memmove(v19, v17, (char *)v58 - (char *)v17);
                LODWORD(v18) = v40;
                v31 = v45;
                LODWORD(v20) = v54;
              }
              v21 = (__m128i *)((char *)v58 - v31);
              if ( v31 )
              {
                v46 = v18;
                v55 = v20;
                v32 = (const __m128i *)memmove((char *)v58 - v31, dest, v31);
                LODWORD(v20) = v55;
                LODWORD(v18) = v46;
                v21 = v32;
              }
            }
          }
        }
        else
        {
          v21 = v19;
          if ( v16 )
          {
            v22 = (char *)v58 - (char *)v17;
            if ( v17 != v58 )
            {
              v34 = v18;
              v36 = v20;
              v41 = (char *)v58 - (char *)v17;
              v49 = v17;
              memmove(dest, v17, (char *)v58 - (char *)v17);
              LODWORD(v18) = v34;
              LODWORD(v20) = v36;
              v22 = v41;
              v17 = v49;
            }
            if ( v17 != v19 )
            {
              v37 = v18;
              v42 = v22;
              v50 = v20;
              memmove((char *)v58 - ((char *)v17 - (char *)v19), v19, (char *)v17 - (char *)v19);
              LODWORD(v18) = v37;
              v22 = v42;
              LODWORD(v20) = v50;
            }
            if ( v22 )
            {
              v38 = v18;
              v43 = v20;
              v51 = v22;
              memmove(v19, dest, v22);
              LODWORD(v18) = v38;
              LODWORD(v20) = v43;
              v22 = v51;
            }
            v21 = (__m128i *)((char *)v19 + v22);
          }
        }
        v12 -= v16;
        v52 = v21;
        sub_28EC100(v18, (_DWORD)v19, (_DWORD)v21, v20, v16, (_DWORD)dest, (__int64)a7);
        result = (const __m128i *)v12;
        if ( (__int64)a7 <= v12 )
          result = a7;
        if ( (__int64)result >= v14 )
        {
          v11 = (__m128i *)a3;
          a6 = dest;
          v8 = v52;
          v10 = v58;
          goto LABEL_22;
        }
        if ( (__int64)a7 >= v12 )
          break;
        v13 = v58;
        v15 = (__m128i *)v52;
      }
      v11 = (__m128i *)a3;
      a6 = dest;
      v8 = v52;
      v10 = v58;
    }
    if ( v11 != v10 )
      memmove(a6, v10, (char *)v11 - (char *)v10);
    result = (__m128i *)((char *)a6 + (char *)v11 - (char *)v10);
    if ( v8 == v10 )
    {
      if ( a6 != result )
      {
        v27 = (char *)v11 - (char *)v10;
        v29 = (__m128i *)v10;
        goto LABEL_58;
      }
    }
    else if ( a6 != result )
    {
      v26 = v10 - 1;
      while ( 1 )
      {
        --result;
        --v11;
        if ( result->m128i_i32[0] > (unsigned __int32)v26->m128i_i32[0] )
          break;
LABEL_42:
        *v11 = _mm_loadu_si128(result);
        if ( a6 == result )
          return result;
      }
      while ( 1 )
      {
        *v11 = _mm_loadu_si128(v26);
        if ( v26 == v8 )
          break;
        --v26;
        --v11;
        if ( result->m128i_i32[0] <= (unsigned __int32)v26->m128i_i32[0] )
          goto LABEL_42;
      }
      if ( a6 != &result[1] )
      {
        v27 = (char *)&result[1] - (char *)a6;
        v29 = (__m128i *)((char *)v11 - v27);
LABEL_58:
        v28 = a6;
        return (const __m128i *)memmove(v29, v28, v27);
      }
    }
  }
  return result;
}
