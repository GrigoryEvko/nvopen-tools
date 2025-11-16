// Function: sub_38E1E90
// Address: 0x38e1e90
//
__m128i *__fastcall sub_38E1E90(__m128i *a1, _DWORD *a2, size_t a3, __int64 a4, __int64 a5)
{
  unsigned __int64 *v6; // rbx
  unsigned __int64 *v7; // r12
  __int64 v8; // rax
  __m128i *result; // rax
  char *v10; // [rsp+0h] [rbp-90h]
  __int64 v11; // [rsp+8h] [rbp-88h]
  const char **v12; // [rsp+10h] [rbp-80h]
  __int64 v13; // [rsp+18h] [rbp-78h]
  unsigned __int64 *v14; // [rsp+20h] [rbp-70h] BYREF
  unsigned __int64 *v15; // [rsp+28h] [rbp-68h]
  __m128i v16; // [rsp+40h] [rbp-50h] BYREF
  __int64 v17; // [rsp+50h] [rbp-40h]

  v10 = (char *)a1[7].m128i_i64[0];
  v11 = a1[7].m128i_i64[1];
  v12 = (const char **)a1[6].m128i_i64[0];
  v13 = a1[6].m128i_i64[1];
  sub_16810B0((__int64 *)&v14, a4, a5);
  sub_1681A40(&v16, (void **)&v14, a2, a3, v10, v11, v12, v13);
  v6 = v15;
  v7 = v14;
  if ( v15 != v14 )
  {
    do
    {
      if ( (unsigned __int64 *)*v7 != v7 + 2 )
        j_j___libc_free_0(*v7);
      v7 += 4;
    }
    while ( v6 != v7 );
    v7 = v14;
  }
  if ( v7 )
    j_j___libc_free_0((unsigned __int64)v7);
  v8 = v17;
  a1[12] = _mm_loadu_si128(&v16);
  a1[13].m128i_i64[0] = v8;
  if ( a3 )
  {
    result = sub_38E1B70((__int64)a1, a2, a3);
    a1[10].m128i_i64[0] = (__int64)result;
  }
  else
  {
    a1[10].m128i_i64[0] = (__int64)xmmword_452E800;
    return xmmword_452E800;
  }
  return result;
}
