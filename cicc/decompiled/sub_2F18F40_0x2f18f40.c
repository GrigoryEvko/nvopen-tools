// Function: sub_2F18F40
// Address: 0x2f18f40
//
unsigned __int64 *__fastcall sub_2F18F40(unsigned __int64 *a1, const __m128i *a2, __int64 a3)
{
  unsigned __int64 v3; // r12
  __int64 v4; // rax
  const __m128i *v6; // r14
  __int64 v7; // rdi
  bool v8; // zf
  __int64 v9; // rax
  bool v10; // cf
  unsigned __int64 v11; // rax
  __int8 *v12; // rsi
  __int64 v13; // rbx
  char *v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rdi
  __m128i v17; // xmm4
  int v18; // eax
  unsigned __int64 v19; // r15
  const __m128i *i; // rbx
  __m128i v21; // xmm1
  unsigned __int64 v22; // rdi
  const __m128i *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rcx
  __m128i v26; // xmm0
  const __m128i *v27; // rcx
  unsigned __int64 v29; // rbx
  __int64 v30; // rax
  __int64 v31; // [rsp+0h] [rbp-60h]
  unsigned __int64 v32; // [rsp+10h] [rbp-50h]
  unsigned __int64 v34; // [rsp+20h] [rbp-40h]
  unsigned __int64 v35; // [rsp+28h] [rbp-38h]

  v3 = a1[1];
  v34 = *a1;
  v4 = (__int64)(v3 - *a1) >> 6;
  if ( v4 == 0x1FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = a2;
  v7 = (__int64)(v3 - v34) >> 6;
  v8 = v4 == 0;
  v9 = 1;
  if ( !v8 )
    v9 = (__int64)(v3 - v34) >> 6;
  v10 = __CFADD__(v7, v9);
  v11 = v7 + v9;
  v12 = &a2->m128i_i8[-v34];
  if ( v10 )
  {
    v29 = 0x7FFFFFFFFFFFFFC0LL;
  }
  else
  {
    if ( !v11 )
    {
      v32 = 0;
      v13 = 64;
      v35 = 0;
      goto LABEL_7;
    }
    if ( v11 > 0x1FFFFFFFFFFFFFFLL )
      v11 = 0x1FFFFFFFFFFFFFFLL;
    v29 = v11 << 6;
  }
  v31 = a3;
  v30 = sub_22077B0(v29);
  a3 = v31;
  v35 = v30;
  v32 = v30 + v29;
  v13 = v30 + 64;
LABEL_7:
  v14 = &v12[v35];
  if ( v14 )
  {
    v15 = *(_QWORD *)(a3 + 8);
    *(_QWORD *)v14 = *(_QWORD *)a3;
    *((_QWORD *)v14 + 1) = v14 + 24;
    if ( v15 == a3 + 24 )
    {
      *(__m128i *)(v14 + 24) = _mm_loadu_si128((const __m128i *)(a3 + 24));
    }
    else
    {
      *((_QWORD *)v14 + 1) = v15;
      *((_QWORD *)v14 + 3) = *(_QWORD *)(a3 + 24);
    }
    v16 = *(_QWORD *)(a3 + 16);
    v17 = _mm_loadu_si128((const __m128i *)(a3 + 40));
    *(_QWORD *)(a3 + 8) = a3 + 24;
    v18 = *(_DWORD *)(a3 + 56);
    *(_QWORD *)(a3 + 16) = 0;
    *((_QWORD *)v14 + 2) = v16;
    *(_BYTE *)(a3 + 24) = 0;
    *((_DWORD *)v14 + 14) = v18;
    *(__m128i *)(v14 + 40) = v17;
  }
  if ( a2 != (const __m128i *)v34 )
  {
    v19 = v35;
    for ( i = (const __m128i *)(v34 + 24); ; i += 4 )
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = i[-2].m128i_i64[1];
        *(_QWORD *)(v19 + 8) = v19 + 24;
        v23 = (const __m128i *)i[-1].m128i_i64[0];
        if ( v23 == i )
        {
          *(__m128i *)(v19 + 24) = _mm_loadu_si128(i);
        }
        else
        {
          *(_QWORD *)(v19 + 8) = v23;
          *(_QWORD *)(v19 + 24) = i->m128i_i64[0];
        }
        *(_QWORD *)(v19 + 16) = i[-1].m128i_i64[1];
        v21 = _mm_loadu_si128(i + 1);
        i[-1].m128i_i64[0] = (__int64)i;
        i[-1].m128i_i64[1] = 0;
        i->m128i_i8[0] = 0;
        *(__m128i *)(v19 + 40) = v21;
        *(_DWORD *)(v19 + 56) = i[2].m128i_i32[0];
      }
      v22 = i[-1].m128i_u64[0];
      if ( (const __m128i *)v22 != i )
        j_j___libc_free_0(v22);
      if ( a2 == (const __m128i *)&i[2].m128i_u64[1] )
        break;
      v19 += 64LL;
    }
    v13 = v19 + 128;
  }
  if ( a2 != (const __m128i *)v3 )
  {
    v24 = v13;
    do
    {
      *(_QWORD *)v24 = v6->m128i_i64[0];
      *(_QWORD *)(v24 + 8) = v24 + 24;
      v27 = (const __m128i *)v6->m128i_i64[1];
      if ( v27 == (const __m128i *)&v6[1].m128i_u64[1] )
      {
        *(__m128i *)(v24 + 24) = _mm_loadu_si128((const __m128i *)((char *)v6 + 24));
      }
      else
      {
        *(_QWORD *)(v24 + 8) = v27;
        *(_QWORD *)(v24 + 24) = v6[1].m128i_i64[1];
      }
      v25 = v6[1].m128i_i64[0];
      v26 = _mm_loadu_si128((const __m128i *)((char *)v6 + 40));
      v6 += 4;
      v24 += 64;
      *(_QWORD *)(v24 - 48) = v25;
      LODWORD(v25) = v6[-1].m128i_i32[2];
      *(__m128i *)(v24 - 24) = v26;
      *(_DWORD *)(v24 - 8) = v25;
    }
    while ( v6 != (const __m128i *)v3 );
    v13 += v3 - (_QWORD)a2;
  }
  if ( v34 )
    j_j___libc_free_0(v34);
  *a1 = v35;
  a1[1] = v13;
  a1[2] = v32;
  return a1;
}
