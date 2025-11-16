// Function: sub_1DB6E70
// Address: 0x1db6e70
//
__int64 __fastcall sub_1DB6E70(__int64 **a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __m128i *v7; // rax
  int v8; // r8d
  __int64 *v9; // r12
  unsigned __int64 v10; // r9
  unsigned __int64 v11; // rcx
  __int64 v12; // rsi
  unsigned __int64 v13; // rdx
  __int64 v14; // rdi
  __m128i *v15; // r10
  __m128i *v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rax
  const __m128i *v19; // rax
  __int64 v20; // rdx
  __m128i *v21; // rax
  __m128i v22; // xmm0
  __int8 *v24; // rbx
  __m128i *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  int v30; // ebx
  __int64 v31; // rax
  __int64 v32; // rax
  int v33; // [rsp+10h] [rbp-60h]
  __m128i v35; // [rsp+20h] [rbp-50h] BYREF
  __int64 v36; // [rsp+30h] [rbp-40h]
  char v37; // [rsp+38h] [rbp-38h] BYREF

  v7 = (__m128i *)sub_1DB3C70(*a1, a2);
  v9 = *a1;
  v10 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v11 = *((unsigned int *)*a1 + 2);
  v12 = **a1;
  v13 = v11;
  v14 = 24 * v11;
  v15 = (__m128i *)(v12 + 24 * v11);
  if ( v7 == v15 )
  {
    if ( !a4 )
    {
      v30 = *((_DWORD *)v9 + 18);
      v31 = sub_145CDC0(0x10u, a3);
      v10 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      a4 = v31;
      if ( v31 )
      {
        *(_DWORD *)v31 = v30;
        *(_QWORD *)(v31 + 8) = a2;
      }
      v32 = *((unsigned int *)v9 + 18);
      if ( (unsigned int)v32 >= *((_DWORD *)v9 + 19) )
      {
        sub_16CD150((__int64)(v9 + 8), v9 + 10, 0, 8, v8, v10);
        v32 = *((unsigned int *)v9 + 18);
        v10 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      }
      *(_QWORD *)(v9[8] + 8 * v32) = a4;
      ++*((_DWORD *)v9 + 18);
      v9 = *a1;
      v13 = *((unsigned int *)*a1 + 2);
    }
    v35.m128i_i64[0] = a2;
    v36 = a4;
    v35.m128i_i64[1] = v10 | 6;
    if ( *((_DWORD *)v9 + 3) <= (unsigned int)v13 )
    {
      sub_16CD150((__int64)v9, v9 + 2, 0, 24, v8, v10);
      v13 = *((unsigned int *)v9 + 2);
    }
    v25 = (__m128i *)(*v9 + 24 * v13);
    v26 = v36;
    *v25 = _mm_loadu_si128(&v35);
    v25[1].m128i_i64[0] = v26;
    ++*((_DWORD *)v9 + 2);
  }
  else
  {
    v16 = v7;
    v17 = v7->m128i_i64[0];
    if ( v10 == (v17 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      a4 = v16[1].m128i_i64[0];
      if ( (*(_DWORD *)(v10 + 24) | (unsigned int)(v17 >> 1) & 3) >= (*(_DWORD *)(v10 + 24) | (unsigned int)(a2 >> 1)
                                                                                            & 3)
        && v17 != a2 )
      {
        *(_QWORD *)(a4 + 8) = a2;
        a4 = v16[1].m128i_i64[0];
        v16->m128i_i64[0] = *(_QWORD *)(a4 + 8);
      }
    }
    else
    {
      if ( a4 )
      {
        v35.m128i_i64[0] = a2;
        v36 = a4;
        v35.m128i_i64[1] = v10 | 6;
        v18 = *((unsigned int *)v9 + 3);
        goto LABEL_5;
      }
      v33 = *((_DWORD *)v9 + 18);
      v27 = sub_145CDC0(0x10u, a3);
      v10 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      a4 = v27;
      if ( v27 )
      {
        *(_QWORD *)(v27 + 8) = a2;
        *(_DWORD *)v27 = v33;
      }
      v28 = *((unsigned int *)v9 + 18);
      if ( (unsigned int)v28 >= *((_DWORD *)v9 + 19) )
      {
        sub_16CD150((__int64)(v9 + 8), v9 + 10, 0, 8, v8, v10);
        v28 = *((unsigned int *)v9 + 18);
        v10 = a2 & 0xFFFFFFFFFFFFFFF8LL;
      }
      *(_QWORD *)(v9[8] + 8 * v28) = a4;
      ++*((_DWORD *)v9 + 18);
      v9 = *a1;
      v11 = *((unsigned int *)*a1 + 2);
      v12 = **a1;
      v35.m128i_i64[0] = a2;
      v35.m128i_i64[1] = v10 | 6;
      v36 = a4;
      LODWORD(v13) = v11;
      v18 = *((unsigned int *)v9 + 3);
      v14 = 24 * v11;
      v15 = (__m128i *)(v12 + 24 * v11);
      if ( v16 != v15 )
      {
LABEL_5:
        if ( v11 >= v18 )
        {
          v24 = &v16->m128i_i8[-v12];
          sub_16CD150((__int64)v9, v9 + 2, 0, 24, v8, v10);
          v12 = *v9;
          LODWORD(v13) = *((_DWORD *)v9 + 2);
          v16 = (__m128i *)&v24[*v9];
          v14 = 24LL * (unsigned int)v13;
          v15 = (__m128i *)(*v9 + v14);
          v19 = (__m128i *)((char *)v15 - 24);
          if ( !v15 )
            goto LABEL_8;
        }
        else
        {
          v19 = (const __m128i *)(v12 + v14 - 24);
          if ( !v15 )
          {
LABEL_8:
            if ( v16 != v19 )
            {
              memmove((void *)(v12 + v14 - ((char *)v19 - (char *)v16)), v16, (char *)v19 - (char *)v16);
              LODWORD(v13) = *((_DWORD *)v9 + 2);
            }
            v20 = (unsigned int)(v13 + 1);
            v21 = &v35;
            *((_DWORD *)v9 + 2) = v20;
            if ( v16 <= &v35 && (unsigned __int64)&v35 < *v9 + 24 * v20 )
              v21 = (__m128i *)&v37;
            v22 = _mm_loadu_si128(v21);
            v16[1].m128i_i64[0] = v21[1].m128i_i64[0];
            *v16 = v22;
            return a4;
          }
        }
        *v15 = _mm_loadu_si128(v19);
        v15[1].m128i_i64[0] = v19[1].m128i_i64[0];
        v12 = *v9;
        LODWORD(v13) = *((_DWORD *)v9 + 2);
        v14 = 24LL * (unsigned int)v13;
        v19 = (const __m128i *)(*v9 + v14 - 24);
        goto LABEL_8;
      }
      if ( (unsigned int)v11 >= (unsigned int)v18 )
      {
        sub_16CD150((__int64)v9, v9 + 2, 0, 24, v8, v10);
        v15 = (__m128i *)(*v9 + 24LL * *((unsigned int *)v9 + 2));
      }
      v29 = v36;
      *v15 = _mm_loadu_si128(&v35);
      v15[1].m128i_i64[0] = v29;
      ++*((_DWORD *)v9 + 2);
    }
  }
  return a4;
}
