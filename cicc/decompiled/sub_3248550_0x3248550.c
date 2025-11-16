// Function: sub_3248550
// Address: 0x3248550
//
__m128i *__fastcall sub_3248550(__m128i *a1, char *a2, unsigned __int8 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rax
  unsigned __int8 *v7; // r13
  __int64 v8; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  _BYTE *v12; // rbx
  _BYTE *v13; // r15
  unsigned __int8 *v14; // r13
  char *v15; // rax
  size_t v16; // rdx
  __int64 v17; // rax
  _BYTE *v18; // [rsp+10h] [rbp-70h] BYREF
  __int64 v19; // [rsp+18h] [rbp-68h]
  _BYTE v20[16]; // [rsp+20h] [rbp-60h] BYREF
  __m128i *v21; // [rsp+30h] [rbp-50h] BYREF
  __int64 v22; // [rsp+38h] [rbp-48h]
  __m128i v23[4]; // [rsp+40h] [rbp-40h] BYREF

  if ( !a3
    || (v6 = *(unsigned __int16 *)(*((_QWORD *)a2 + 10) + 16LL), (unsigned int)v6 > 0x2B)
    || (v7 = a3, v8 = 0xC0206000010LL, !_bittest64(&v8, v6)) )
  {
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    sub_3247510(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
    return a1;
  }
  v22 = 0;
  v21 = v23;
  v18 = v20;
  v19 = 0x100000000LL;
  v10 = 0;
  v23[0].m128i_i8[0] = 0;
  while ( 1 )
  {
    v11 = v10;
    if ( *v7 == 17 )
      break;
    if ( v10 + 1 > (unsigned __int64)HIDWORD(v19) )
    {
      a2 = v20;
      sub_C8D5F0((__int64)&v18, v20, v10 + 1, 8u, a5, a6);
      v10 = (unsigned int)v19;
    }
    *(_QWORD *)&v18[8 * v10] = v7;
    LODWORD(v19) = v19 + 1;
    v7 = (unsigned __int8 *)sub_AF2660(v7);
    if ( !v7 )
    {
      v11 = (unsigned int)v19;
      break;
    }
    v10 = (unsigned int)v19;
  }
  v12 = v18;
  v13 = &v18[8 * v11];
  if ( v18 == v13 )
    goto LABEL_23;
  do
  {
    v14 = (unsigned __int8 *)*((_QWORD *)v13 - 1);
    v15 = (char *)sub_AF5A10(v14, (__int64)a2);
    if ( !v16 )
    {
      if ( *v14 != 21 )
        goto LABEL_18;
      v16 = 21;
      v15 = "(anonymous namespace)";
    }
    if ( v16 > 0x3FFFFFFFFFFFFFFFLL - v22
      || (sub_2241490((unsigned __int64 *)&v21, v15, v16), v22 == 0x3FFFFFFFFFFFFFFFLL || v22 == 4611686018427387902LL) )
    {
      sub_4262D8((__int64)"basic_string::append");
    }
    a2 = "::";
    sub_2241490((unsigned __int64 *)&v21, "::", 2u);
LABEL_18:
    v13 -= 8;
  }
  while ( v12 != v13 );
  v13 = v18;
LABEL_23:
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( v21 == v23 )
  {
    a1[1] = _mm_load_si128(v23);
  }
  else
  {
    a1->m128i_i64[0] = (__int64)v21;
    a1[1].m128i_i64[0] = v23[0].m128i_i64[0];
  }
  v17 = v22;
  v23[0].m128i_i8[0] = 0;
  v22 = 0;
  a1->m128i_i64[1] = v17;
  v21 = v23;
  if ( v13 != v20 )
  {
    _libc_free((unsigned __int64)v13);
    if ( v21 != v23 )
      j_j___libc_free_0((unsigned __int64)v21);
  }
  return a1;
}
