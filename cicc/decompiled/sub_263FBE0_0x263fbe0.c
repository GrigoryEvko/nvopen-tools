// Function: sub_263FBE0
// Address: 0x263fbe0
//
void __fastcall sub_263FBE0(unsigned __int64 *a1, const __m128i *a2)
{
  __m128i *v2; // rbx
  unsigned __int64 v3; // r14
  __int64 v4; // rdx
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rcx
  bool v7; // cf
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r8
  __m128i *v11; // r15
  __m128i *v12; // rdx
  __m128i v13; // xmm2
  __m128i *v14; // rdx
  const __m128i *v15; // rax
  unsigned __int64 v16; // r8
  __int64 v17; // rax
  unsigned __int64 v18; // [rsp+0h] [rbp-40h]
  __int8 *v19; // [rsp+0h] [rbp-40h]
  unsigned __int64 v20; // [rsp+8h] [rbp-38h]
  unsigned __int64 v21; // [rsp+8h] [rbp-38h]

  v2 = (__m128i *)a1[1];
  if ( v2 != (__m128i *)a1[2] )
  {
    if ( v2 )
    {
      *v2 = _mm_loadu_si128(a2);
      v2[1].m128i_i64[0] = a2[1].m128i_i64[0];
      v2 = (__m128i *)a1[1];
    }
    a1[1] = (unsigned __int64)&v2[1].m128i_u64[1];
    return;
  }
  v3 = *a1;
  v4 = (__int64)v2->m128i_i64 - *a1;
  v5 = 0xAAAAAAAAAAAAAAABLL * (v4 >> 3);
  if ( v5 == 0x555555555555555LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v6 = 1;
  if ( v5 )
    v6 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v2->m128i_i64 - *a1) >> 3);
  v7 = __CFADD__(v6, v5);
  v8 = v6 - 0x5555555555555555LL * ((__int64)((__int64)v2->m128i_i64 - *a1) >> 3);
  if ( v7 )
  {
    v16 = 0x7FFFFFFFFFFFFFF8LL;
LABEL_25:
    v19 = &v2->m128i_i8[-*a1];
    v21 = v16;
    v17 = sub_22077B0(v16);
    v4 = (__int64)v19;
    v11 = (__m128i *)v17;
    v10 = v17 + v21;
    v9 = v17 + 24;
    goto LABEL_11;
  }
  if ( v8 )
  {
    if ( v8 > 0x555555555555555LL )
      v8 = 0x555555555555555LL;
    v16 = 24 * v8;
    goto LABEL_25;
  }
  v9 = 24;
  v10 = 0;
  v11 = 0;
LABEL_11:
  v12 = (__m128i *)((char *)v11 + v4);
  if ( v12 )
  {
    v13 = _mm_loadu_si128(a2);
    v12[1].m128i_i64[0] = a2[1].m128i_i64[0];
    *v12 = v13;
  }
  if ( v2 != (__m128i *)v3 )
  {
    v14 = v11;
    v15 = (const __m128i *)v3;
    do
    {
      if ( v14 )
      {
        *v14 = _mm_loadu_si128(v15);
        v14[1].m128i_i64[0] = v15[1].m128i_i64[0];
      }
      v15 = (const __m128i *)((char *)v15 + 24);
      v14 = (__m128i *)((char *)v14 + 24);
    }
    while ( v2 != v15 );
    v9 = (unsigned __int64)&v11[3] + 8 * (((unsigned __int64)&v2[-2].m128i_u64[1] - v3) >> 3);
  }
  if ( v3 )
  {
    v18 = v9;
    v20 = v10;
    j_j___libc_free_0(v3);
    v9 = v18;
    v10 = v20;
  }
  *a1 = (unsigned __int64)v11;
  a1[1] = v9;
  a1[2] = v10;
}
