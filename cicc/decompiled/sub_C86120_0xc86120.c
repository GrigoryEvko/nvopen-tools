// Function: sub_C86120
// Address: 0xc86120
//
__m128i *__fastcall sub_C86120(__m128i *a1, _BYTE *a2, __int64 a3)
{
  const char *v4; // rdi
  char *v5; // rax
  char *v6; // r13
  size_t v7; // rax
  __int64 v8; // rax
  char *name[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v11[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v12[2]; // [rsp+20h] [rbp-50h] BYREF
  __m128i v13[4]; // [rsp+30h] [rbp-40h] BYREF

  if ( a2 )
  {
    name[0] = (char *)v11;
    sub_C85F00((__int64 *)name, a2, (__int64)&a2[a3]);
    v4 = name[0];
  }
  else
  {
    name[1] = 0;
    name[0] = (char *)v11;
    v4 = (const char *)v11;
    LOBYTE(v11[0]) = 0;
  }
  v5 = getenv(v4);
  v6 = v5;
  if ( v5 )
  {
    v12[0] = (__int64)v13;
    v7 = strlen(v5);
    sub_C85F00(v12, v6, (__int64)&v6[v7]);
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( (__m128i *)v12[0] == v13 )
    {
      a1[1] = _mm_load_si128(v13);
    }
    else
    {
      a1->m128i_i64[0] = v12[0];
      a1[1].m128i_i64[0] = v13[0].m128i_i64[0];
    }
    v8 = v12[1];
    a1[2].m128i_i8[0] = 1;
    a1->m128i_i64[1] = v8;
  }
  else
  {
    a1[2].m128i_i8[0] = 0;
  }
  if ( (_QWORD *)name[0] != v11 )
    j_j___libc_free_0(name[0], v11[0] + 1LL);
  return a1;
}
