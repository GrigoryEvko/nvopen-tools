// Function: sub_27A5910
// Address: 0x27a5910
//
_QWORD *__fastcall sub_27A5910(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rdx
  const __m128i *v7; // rcx
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  __m128i *v11; // rdi
  __m128i *v12; // rdx
  const __m128i *v13; // rax
  __m128i v15; // [rsp+0h] [rbp-C0h] BYREF
  char v16; // [rsp+10h] [rbp-B0h]
  __int64 v17; // [rsp+20h] [rbp-A0h] BYREF
  __int64 *v18; // [rsp+28h] [rbp-98h]
  __int64 v19; // [rsp+30h] [rbp-90h]
  int v20; // [rsp+38h] [rbp-88h]
  char v21; // [rsp+3Ch] [rbp-84h]
  __int64 v22; // [rsp+40h] [rbp-80h] BYREF
  unsigned __int64 v23; // [rsp+80h] [rbp-40h] BYREF
  const __m128i *v24; // [rsp+88h] [rbp-38h]
  __int64 v25; // [rsp+90h] [rbp-30h]

  v18 = &v22;
  v22 = a2;
  v15.m128i_i64[0] = a2;
  v19 = 0x100000008LL;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v20 = 0;
  v21 = 1;
  v17 = 1;
  v16 = 0;
  sub_27A5770(&v23, &v15);
  sub_C8CD80((__int64)a1, (__int64)(a1 + 4), (__int64)&v17, v3, v4, v5);
  v7 = v24;
  v8 = v23;
  a1[12] = 0;
  a1[13] = 0;
  a1[14] = 0;
  v9 = (unsigned __int64)v7 - v8;
  if ( v7 == (const __m128i *)v8 )
  {
    v11 = 0;
  }
  else
  {
    if ( v9 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, a1 + 4, v6);
    v10 = sub_22077B0((unsigned __int64)v7 - v8);
    v7 = v24;
    v8 = v23;
    v11 = (__m128i *)v10;
  }
  a1[12] = v11;
  a1[13] = v11;
  a1[14] = (char *)v11 + v9;
  if ( v7 != (const __m128i *)v8 )
  {
    v12 = v11;
    v13 = (const __m128i *)v8;
    do
    {
      if ( v12 )
      {
        *v12 = _mm_loadu_si128(v13);
        v12[1].m128i_i64[0] = v13[1].m128i_i64[0];
      }
      v13 = (const __m128i *)((char *)v13 + 24);
      v12 = (__m128i *)((char *)v12 + 24);
    }
    while ( v13 != v7 );
    v11 = (__m128i *)((char *)v11 + 8 * (((unsigned __int64)&v13[-2].m128i_u64[1] - v8) >> 3) + 24);
  }
  a1[13] = v11;
  if ( v8 )
    j_j___libc_free_0(v8);
  if ( !v21 )
    _libc_free((unsigned __int64)v18);
  return a1;
}
