// Function: sub_28EC640
// Address: 0x28ec640
//
__int64 __fastcall sub_28EC640(
        const __m128i *a1,
        const __m128i *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char *a6,
        __int64 a7)
{
  __int64 result; // rax
  const __m128i *v8; // r14
  const __m128i *v10; // r12
  __m128i *v11; // rbx
  __int64 v12; // r15
  __int64 v13; // r9
  __int64 v14; // rbx
  __int64 v15; // r11
  __int64 v16; // r13
  const __m128i *v17; // r9
  __int64 v18; // r11
  __m128i *v19; // r12
  __int64 v20; // rcx
  const __m128i *v21; // r10
  size_t v22; // r10
  char *v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rax
  const __m128i *v26; // r12
  size_t v27; // rdx
  char *v28; // rsi
  __m128i *v29; // rdi
  __int64 v30; // rax
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
  char *dest; // [rsp+30h] [rbp-40h]
  const __m128i *v58; // [rsp+38h] [rbp-38h]

  result = a5;
  v8 = a1;
  v10 = a2;
  v11 = (__m128i *)a3;
  if ( a7 <= a5 )
    result = a7;
  if ( a4 <= result )
  {
LABEL_22:
    if ( v10 != v8 )
      result = (__int64)memmove(a6, v8, (char *)v10 - (char *)v8);
    v23 = &a6[(char *)v10 - (char *)v8];
    if ( v11 != v10 && a6 != v23 )
    {
      do
      {
        if ( v10->m128i_i32[2] > *((_DWORD *)a6 + 2) )
        {
          v24 = v10->m128i_i64[0];
          ++v8;
          ++v10;
          v8[-1].m128i_i64[0] = v24;
          result = v10[-1].m128i_u32[2];
          v8[-1].m128i_i32[2] = result;
          if ( v23 == a6 )
            return result;
        }
        else
        {
          v25 = *(_QWORD *)a6;
          a6 += 16;
          ++v8;
          v8[-1].m128i_i64[0] = v25;
          result = *((unsigned int *)a6 - 2);
          v8[-1].m128i_i32[2] = result;
          if ( v23 == a6 )
            return result;
        }
      }
      while ( v11 != v10 );
    }
    if ( v23 != a6 )
    {
      v27 = v23 - a6;
      v28 = a6;
      v29 = (__m128i *)v8;
      return (__int64)memmove(v29, v28, v27);
    }
  }
  else
  {
    v12 = a5;
    if ( a7 < a5 )
    {
      v13 = (__int64)a2;
      v14 = a4;
      v15 = (__int64)a1;
      dest = a6;
      while ( 1 )
      {
        if ( v12 < v14 )
        {
          v19 = (__m128i *)(v15 + 16 * (v14 / 2));
          v30 = sub_28EA1F0(v13, a3, (__int64)v19);
          v20 = v14 / 2;
          v58 = (const __m128i *)v30;
          v16 = (v30 - (__int64)v17) >> 4;
        }
        else
        {
          v16 = v12 / 2;
          v58 = (const __m128i *)(v13 + 16 * (v12 / 2));
          v19 = (__m128i *)sub_28EA240(v15, v13, (__int64)v58);
          v20 = ((__int64)v19->m128i_i64 - v18) >> 4;
        }
        v14 -= v20;
        if ( v14 <= v16 || a7 < v16 )
        {
          if ( a7 < v14 )
          {
            v47 = v18;
            v56 = v20;
            v33 = sub_28E9610(v19, v17, v58);
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
              v21 = (const __m128i *)((char *)v58 - v31);
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
        sub_28EC640(v18, (_DWORD)v19, (_DWORD)v21, v20, v16, (_DWORD)dest, a7);
        result = v12;
        if ( a7 <= v12 )
          result = a7;
        if ( result >= v14 )
        {
          v11 = (__m128i *)a3;
          a6 = dest;
          v8 = v52;
          v10 = v58;
          goto LABEL_22;
        }
        if ( a7 >= v12 )
          break;
        v13 = (__int64)v58;
        v15 = (__int64)v52;
      }
      v11 = (__m128i *)a3;
      a6 = dest;
      v8 = v52;
      v10 = v58;
    }
    if ( v11 != v10 )
      memmove(a6, v10, (char *)v11 - (char *)v10);
    result = (__int64)&a6[(char *)v11 - (char *)v10];
    if ( v8 == v10 )
    {
      if ( a6 != (char *)result )
      {
        v27 = (char *)v11 - (char *)v10;
        v29 = (__m128i *)v10;
        goto LABEL_58;
      }
    }
    else if ( a6 != (char *)result )
    {
      v26 = v10 - 1;
      while ( 1 )
      {
        result -= 16;
        --v11;
        if ( *(_DWORD *)(result + 8) > v26->m128i_i32[2] )
          break;
LABEL_42:
        v11->m128i_i64[0] = *(_QWORD *)result;
        v11->m128i_i32[2] = *(_DWORD *)(result + 8);
        if ( a6 == (char *)result )
          return result;
      }
      while ( 1 )
      {
        v11->m128i_i64[0] = v26->m128i_i64[0];
        v11->m128i_i32[2] = v26->m128i_i32[2];
        if ( v26 == v8 )
          break;
        --v26;
        --v11;
        if ( *(_DWORD *)(result + 8) <= v26->m128i_i32[2] )
          goto LABEL_42;
      }
      if ( a6 != (char *)(result + 16) )
      {
        v27 = result + 16 - (_QWORD)a6;
        v29 = (__m128i *)((char *)v11 - v27);
LABEL_58:
        v28 = a6;
        return (__int64)memmove(v29, v28, v27);
      }
    }
  }
  return result;
}
