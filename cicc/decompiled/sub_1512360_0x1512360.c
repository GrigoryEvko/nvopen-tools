// Function: sub_1512360
// Address: 0x1512360
//
__int64 *__fastcall sub_1512360(__int64 *a1, const __m128i *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  bool v4; // cf
  unsigned __int64 v5; // rax
  __int64 m128i_i64; // r12
  _QWORD *v7; // rbx
  __m128i *v8; // r12
  const __m128i *v9; // r15
  const __m128i *v10; // rdx
  __int64 v11; // rdx
  const __m128i *v12; // rdi
  __int64 v13; // r13
  __int64 v14; // r14
  volatile signed __int32 *v15; // rbx
  signed __int32 v16; // edx
  signed __int32 v17; // edx
  __int8 *v18; // rcx
  __int64 v19; // rbx
  __int64 v20; // r13
  __int64 v21; // rdi
  __int64 v22; // rsi
  const __m128i *v23; // rax
  __m128i *v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rcx
  __int64 v28; // rcx
  const __m128i *v29; // rcx
  __int64 v31; // r12
  __int64 v32; // [rsp+0h] [rbp-60h]
  const __m128i *v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+18h] [rbp-48h]
  const __m128i *v36; // [rsp+20h] [rbp-40h]

  v36 = (const __m128i *)a1[1];
  v34 = (const __m128i *)*a1;
  v2 = 0x2E8BA2E8BA2E8BA3LL * (((__int64)v36->m128i_i64 - *a1) >> 3);
  if ( v2 == 0x1745D1745D1745DLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v3 = 1;
  if ( v2 )
    v3 = 0x2E8BA2E8BA2E8BA3LL * (((__int64)v36->m128i_i64 - *a1) >> 3);
  v4 = __CFADD__(v3, v2);
  v5 = v3 + v2;
  if ( v4 )
  {
    v31 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v5 )
    {
      v32 = 0;
      m128i_i64 = 88;
      v35 = 0;
      goto LABEL_7;
    }
    if ( v5 > 0x1745D1745D1745DLL )
      v5 = 0x1745D1745D1745DLL;
    v31 = 88 * v5;
  }
  v35 = sub_22077B0(v31);
  v32 = v35 + v31;
  m128i_i64 = v35 + 88;
LABEL_7:
  v7 = (_QWORD *)(v35 + (char *)a2 - (char *)v34);
  if ( v7 )
  {
    memset(v7, 0, 0x58u);
    v7[4] = v7 + 6;
  }
  if ( a2 != v34 )
  {
    v8 = (__m128i *)v35;
    v9 = v34 + 3;
    if ( !v35 )
      goto LABEL_32;
LABEL_11:
    v8->m128i_i32[0] = v9[-3].m128i_i32[0];
    v8->m128i_i64[1] = v9[-3].m128i_i64[1];
    v8[1].m128i_i64[0] = v9[-2].m128i_i64[0];
    v8[1].m128i_i64[1] = v9[-2].m128i_i64[1];
    v9[-2].m128i_i64[1] = 0;
    v9[-2].m128i_i64[0] = 0;
    v9[-3].m128i_i64[1] = 0;
    v8[2].m128i_i64[0] = (__int64)v8[3].m128i_i64;
    v10 = (const __m128i *)v9[-1].m128i_i64[0];
    if ( v10 == v9 )
    {
      v8[3] = _mm_loadu_si128(v9);
    }
    else
    {
      v8[2].m128i_i64[0] = (__int64)v10;
      v8[3].m128i_i64[0] = v9->m128i_i64[0];
    }
    v8[2].m128i_i64[1] = v9[-1].m128i_i64[1];
    v11 = v9[1].m128i_i64[0];
    v9[-1].m128i_i64[0] = (__int64)v9;
    v9[-1].m128i_i64[1] = 0;
    v9->m128i_i8[0] = 0;
    v8[4].m128i_i64[0] = v11;
    v8[4].m128i_i64[1] = v9[1].m128i_i64[1];
    v8[5].m128i_i64[0] = v9[2].m128i_i64[0];
    v9[2].m128i_i64[0] = 0;
    v9[1].m128i_i64[1] = 0;
    v9[1].m128i_i64[0] = 0;
    while ( 1 )
    {
      v12 = (const __m128i *)v9[-1].m128i_i64[0];
      if ( v12 != v9 )
        j_j___libc_free_0(v12, v9->m128i_i64[0] + 1);
      v13 = v9[-2].m128i_i64[0];
      v14 = v9[-3].m128i_i64[1];
      if ( v13 != v14 )
      {
        do
        {
          while ( 1 )
          {
            v15 = *(volatile signed __int32 **)(v14 + 8);
            if ( v15 )
            {
              if ( &_pthread_key_create )
              {
                v16 = _InterlockedExchangeAdd(v15 + 2, 0xFFFFFFFF);
              }
              else
              {
                v16 = *((_DWORD *)v15 + 2);
                *((_DWORD *)v15 + 2) = v16 - 1;
              }
              if ( v16 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 16LL))(v15);
                if ( &_pthread_key_create )
                {
                  v17 = _InterlockedExchangeAdd(v15 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v17 = *((_DWORD *)v15 + 3);
                  *((_DWORD *)v15 + 3) = v17 - 1;
                }
                if ( v17 == 1 )
                  break;
              }
            }
            v14 += 16;
            if ( v13 == v14 )
              goto LABEL_27;
          }
          v14 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 24LL))(v15);
        }
        while ( v13 != v14 );
LABEL_27:
        v14 = v9[-3].m128i_i64[1];
      }
      if ( v14 )
        j_j___libc_free_0(v14, v9[-2].m128i_i64[1] - v14);
      v18 = &v8[5].m128i_i8[8];
      if ( a2 == (const __m128i *)&v9[2].m128i_u64[1] )
        break;
      v8 = (__m128i *)((char *)v8 + 88);
      v9 = (const __m128i *)((char *)v9 + 88);
      if ( v18 )
        goto LABEL_11;
LABEL_32:
      v19 = v9[1].m128i_i64[1];
      v20 = v9[1].m128i_i64[0];
      if ( v19 == v20 )
      {
        v22 = v9[2].m128i_i64[0] - v20;
      }
      else
      {
        do
        {
          v21 = *(_QWORD *)(v20 + 8);
          if ( v21 != v20 + 24 )
            j_j___libc_free_0(v21, *(_QWORD *)(v20 + 24) + 1LL);
          v20 += 40;
        }
        while ( v19 != v20 );
        v20 = v9[1].m128i_i64[0];
        v22 = v9[2].m128i_i64[0] - v20;
      }
      if ( v20 )
        j_j___libc_free_0(v20, v22);
    }
    m128i_i64 = (__int64)v8[11].m128i_i64;
  }
  v23 = a2;
  if ( a2 != v36 )
  {
    v24 = (__m128i *)m128i_i64;
    do
    {
      v24->m128i_i32[0] = v23->m128i_i32[0];
      v26 = v23->m128i_i64[1];
      v23->m128i_i64[1] = 0;
      v24->m128i_i64[1] = v26;
      v27 = v23[1].m128i_i64[0];
      v23[1].m128i_i64[0] = 0;
      v24[1].m128i_i64[0] = v27;
      v28 = v23[1].m128i_i64[1];
      v23[1].m128i_i64[1] = 0;
      v24[1].m128i_i64[1] = v28;
      v24[2].m128i_i64[0] = (__int64)v24[3].m128i_i64;
      v29 = (const __m128i *)v23[2].m128i_i64[0];
      if ( v29 == &v23[3] )
      {
        v24[3] = _mm_loadu_si128(v23 + 3);
      }
      else
      {
        v24[2].m128i_i64[0] = (__int64)v29;
        v24[3].m128i_i64[0] = v23[3].m128i_i64[0];
      }
      v25 = v23[2].m128i_i64[1];
      v24 = (__m128i *)((char *)v24 + 88);
      v23 = (const __m128i *)((char *)v23 + 88);
      v24[-3].m128i_i64[0] = v25;
      v24[-2].m128i_i64[1] = v23[-2].m128i_i64[1];
      v24[-1].m128i_i64[0] = v23[-1].m128i_i64[0];
      v24[-1].m128i_i64[1] = v23[-1].m128i_i64[1];
    }
    while ( v23 != v36 );
    m128i_i64 += 88
               * (((0xE8BA2E8BA2E8BA3LL * ((unsigned __int64)((char *)v23 - (char *)a2 - 88) >> 3))
                 & 0x1FFFFFFFFFFFFFFFLL)
                + 1);
  }
  if ( v34 )
    j_j___libc_free_0(v34, a1[2] - (_QWORD)v34);
  *a1 = v35;
  a1[1] = m128i_i64;
  a1[2] = v32;
  return a1;
}
