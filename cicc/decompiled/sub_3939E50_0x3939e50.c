// Function: sub_3939E50
// Address: 0x3939e50
//
unsigned __int64 __fastcall sub_3939E50(unsigned __int64 *a1, const __m128i *a2, _DWORD *a3, _QWORD *a4, _QWORD *a5)
{
  const __m128i *v7; // rsi
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdi
  bool v11; // cf
  unsigned __int64 v12; // rax
  __int8 *v13; // r10
  unsigned __int64 v14; // rbx
  __m128i *v15; // r15
  char *v16; // r10
  __int64 v17; // rax
  int v18; // edx
  __m128i *v19; // rdx
  const __m128i *v20; // rax
  unsigned __int64 v22; // rbx
  __int64 v23; // rax
  _QWORD *v24; // [rsp+0h] [rbp-60h]
  _QWORD *v25; // [rsp+8h] [rbp-58h]
  _DWORD *v26; // [rsp+10h] [rbp-50h]
  size_t v27; // [rsp+20h] [rbp-40h]
  unsigned __int64 v28; // [rsp+28h] [rbp-38h]

  v7 = (const __m128i *)a1[1];
  v8 = *a1;
  v9 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v7->m128i_i64 - *a1) >> 3);
  if ( v9 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  if ( v9 )
    v10 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v7->m128i_i64 - v8) >> 3);
  v11 = __CFADD__(v10, v9);
  v12 = v10 - 0x5555555555555555LL * ((__int64)((__int64)v7->m128i_i64 - v8) >> 3);
  v13 = &a2->m128i_i8[-v8];
  if ( v11 )
  {
    v22 = 0x7FFFFFFFFFFFFFF8LL;
  }
  else
  {
    if ( !v12 )
    {
      v28 = 0;
      v14 = 24;
      v15 = 0;
      goto LABEL_7;
    }
    if ( v12 > 0x555555555555555LL )
      v12 = 0x555555555555555LL;
    v22 = 24 * v12;
  }
  v24 = a5;
  v25 = a4;
  v26 = a3;
  v23 = sub_22077B0(v22);
  v13 = &a2->m128i_i8[-v8];
  v15 = (__m128i *)v23;
  a3 = v26;
  a4 = v25;
  v28 = v22 + v23;
  a5 = v24;
  v14 = v23 + 24;
LABEL_7:
  v16 = &v13[(_QWORD)v15];
  if ( v16 )
  {
    v17 = *a5;
    v18 = *a3;
    *((_QWORD *)v16 + 1) = *a4;
    *(_DWORD *)v16 = v18;
    *((_QWORD *)v16 + 2) = v17;
  }
  if ( a2 != (const __m128i *)v8 )
  {
    v19 = v15;
    v20 = (const __m128i *)v8;
    do
    {
      if ( v19 )
      {
        *v19 = _mm_loadu_si128(v20);
        v19[1].m128i_i64[0] = v20[1].m128i_i64[0];
      }
      v20 = (const __m128i *)((char *)v20 + 24);
      v19 = (__m128i *)((char *)v19 + 24);
    }
    while ( v20 != a2 );
    v14 = (unsigned __int64)&v15[3] + 8 * (((unsigned __int64)&a2[-2].m128i_u64[1] - v8) >> 3);
  }
  if ( a2 != v7 )
  {
    v27 = 8 * ((unsigned __int64)((char *)v7 - (char *)a2 - 24) >> 3) + 24;
    memcpy((void *)v14, a2, v27);
    v14 += v27;
  }
  if ( v8 )
    j_j___libc_free_0(v8);
  *a1 = (unsigned __int64)v15;
  a1[1] = v14;
  a1[2] = v28;
  return v28;
}
