// Function: sub_39A2AF0
// Address: 0x39a2af0
//
__m128i *__fastcall sub_39A2AF0(__m128i *a1, __int64 a2, unsigned __int8 *a3)
{
  unsigned __int8 *v3; // r13
  _BYTE *v4; // rdx
  __int64 v5; // rax
  bool v6; // zf
  int v7; // r8d
  int v8; // r9d
  _BYTE *v10; // r15
  unsigned __int8 *v11; // rbx
  char *v12; // rax
  size_t v13; // rdx
  __m128i *v14; // rax
  _BYTE *v15; // r13
  __int64 v16; // rdx
  _BYTE *v17; // [rsp+10h] [rbp-70h] BYREF
  __int64 v18; // [rsp+18h] [rbp-68h]
  _BYTE v19[16]; // [rsp+20h] [rbp-60h] BYREF
  __m128i *v20; // [rsp+30h] [rbp-50h] BYREF
  __int64 v21; // [rsp+38h] [rbp-48h]
  __m128i v22[4]; // [rsp+40h] [rbp-40h] BYREF

  if ( !a3 || *(_WORD *)(*(_QWORD *)(a2 + 80) + 24LL) != 4 )
  {
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    sub_39A1D60(a1->m128i_i64, byte_3F871B3, (__int64)byte_3F871B3);
    return a1;
  }
  v3 = a3;
  v4 = v19;
  v22[0].m128i_i8[0] = 0;
  v20 = v22;
  v18 = 0x100000000LL;
  v5 = 0;
  v6 = *v3 == 16;
  v21 = 0;
  v17 = v19;
  if ( v6 )
  {
    v15 = v19;
    v16 = 0;
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
LABEL_27:
    a1[1] = _mm_load_si128(v22);
    goto LABEL_24;
  }
  while ( 1 )
  {
    *(_QWORD *)&v4[8 * v5] = v3;
    LODWORD(v18) = v18 + 1;
    if ( !sub_15B0BB0(v3) )
      break;
    v3 = (unsigned __int8 *)sub_15B0BB0(v3);
    if ( *v3 == 16 )
      break;
    v5 = (unsigned int)v18;
    if ( (unsigned int)v18 >= HIDWORD(v18) )
    {
      sub_16CD150((__int64)&v17, v19, 0, 8, v7, v8);
      v5 = (unsigned int)v18;
    }
    v4 = v17;
  }
  v15 = v17;
  v10 = &v17[8 * (unsigned int)v18];
  if ( v10 != v17 )
  {
    while ( 1 )
    {
      v11 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
      v12 = (char *)sub_15B0C30(v11);
      if ( v13 )
        goto LABEL_14;
      if ( *v11 == 20 )
        break;
LABEL_17:
      v10 -= 8;
      if ( v15 == v10 )
      {
        v15 = v17;
        goto LABEL_22;
      }
    }
    v12 = "(anonymous namespace)";
    v13 = 21;
LABEL_14:
    if ( v13 > 0x3FFFFFFFFFFFFFFFLL - v21
      || (sub_2241490((unsigned __int64 *)&v20, v12, v13), v21 == 0x3FFFFFFFFFFFFFFFLL || v21 == 4611686018427387902LL) )
    {
      sub_4262D8((__int64)"basic_string::append");
    }
    sub_2241490((unsigned __int64 *)&v20, "::", 2u);
    goto LABEL_17;
  }
LABEL_22:
  v14 = v20;
  v16 = v21;
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  if ( v14 == v22 )
    goto LABEL_27;
  a1->m128i_i64[0] = (__int64)v14;
  a1[1].m128i_i64[0] = v22[0].m128i_i64[0];
LABEL_24:
  a1->m128i_i64[1] = v16;
  v21 = 0;
  v20 = v22;
  v22[0].m128i_i8[0] = 0;
  if ( v15 != v19 )
  {
    _libc_free((unsigned __int64)v15);
    if ( v20 != v22 )
      j_j___libc_free_0((unsigned __int64)v20);
  }
  return a1;
}
