// Function: sub_27A1350
// Address: 0x27a1350
//
_QWORD *__fastcall sub_27A1350(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  unsigned __int64 v8; // rcx
  unsigned __int64 v9; // r8
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  __m128i *v12; // rdi
  __m128i *v13; // rdx
  const __m128i *v14; // rax
  unsigned __int64 v16[18]; // [rsp+0h] [rbp-90h] BYREF

  memset(v16, 0, 0x78u);
  BYTE4(v16[3]) = 1;
  v16[1] = (unsigned __int64)&v16[4];
  LODWORD(v16[2]) = 8;
  sub_C8CD80((__int64)a1, (__int64)(a1 + 4), (__int64)v16, 0, a5, a6);
  v8 = v16[13];
  v9 = v16[12];
  a1[12] = 0;
  a1[13] = 0;
  a1[14] = 0;
  v10 = v8 - v9;
  if ( v8 == v9 )
  {
    v12 = 0;
  }
  else
  {
    if ( v10 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, a1 + 4, v7);
    v11 = sub_22077B0(v8 - v9);
    v8 = v16[13];
    v9 = v16[12];
    v12 = (__m128i *)v11;
  }
  a1[12] = v12;
  a1[13] = v12;
  a1[14] = (char *)v12 + v10;
  if ( v8 != v9 )
  {
    v13 = v12;
    v14 = (const __m128i *)v9;
    do
    {
      if ( v13 )
      {
        *v13 = _mm_loadu_si128(v14);
        v13[1].m128i_i64[0] = v14[1].m128i_i64[0];
      }
      v14 = (const __m128i *)((char *)v14 + 24);
      v13 = (__m128i *)((char *)v13 + 24);
    }
    while ( (const __m128i *)v8 != v14 );
    v12 = (__m128i *)((char *)v12 + 8 * ((v8 - 24 - v9) >> 3) + 24);
  }
  a1[13] = v12;
  if ( v9 )
    j_j___libc_free_0(v9);
  if ( !BYTE4(v16[3]) )
    _libc_free(v16[1]);
  return a1;
}
