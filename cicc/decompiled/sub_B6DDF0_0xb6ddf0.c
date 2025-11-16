// Function: sub_B6DDF0
// Address: 0xb6ddf0
//
__m128i *__fastcall sub_B6DDF0(__m128i *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 *a5, __int64 a6)
{
  unsigned int v6; // eax
  char *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 *v10; // rbx
  __int64 v11; // rcx
  __m128i *v12; // rax
  __int64 *v18; // [rsp+48h] [rbp-A8h]
  char v19; // [rsp+5Fh] [rbp-91h] BYREF
  __m128i *v20; // [rsp+60h] [rbp-90h] BYREF
  __int64 v21; // [rsp+68h] [rbp-88h]
  __m128i v22; // [rsp+70h] [rbp-80h] BYREF
  _QWORD v23[2]; // [rsp+80h] [rbp-70h] BYREF
  __int64 v24; // [rsp+90h] [rbp-60h] BYREF
  _OWORD *v25; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v26; // [rsp+A8h] [rbp-48h]
  _OWORD v27[4]; // [rsp+B0h] [rbp-40h] BYREF

  v25 = v23;
  *(_QWORD *)(__readfsqword(0) - 24) = &v25;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_B5B9E0;
  if ( !&_pthread_key_create )
  {
    v6 = -1;
LABEL_24:
    sub_4264C5(v6);
  }
  v6 = pthread_once(&dword_4F818F8, init_routine);
  if ( v6 )
    goto LABEL_24;
  v19 = 0;
  v7 = sub_B60B70(a2);
  v20 = &v22;
  sub_B5E240((__int64 *)&v20, v7, (__int64)&v7[v8]);
  v18 = (__int64 *)(a3 + 8 * a4);
  if ( (__int64 *)a3 != v18 )
  {
    v10 = (__int64 *)a3;
    do
    {
      sub_B5EBA0((__int64)v23, *v10, &v19, v9);
      v12 = (__m128i *)sub_2241130(v23, 0, 0, ".", 1);
      v25 = v27;
      if ( (__m128i *)v12->m128i_i64[0] == &v12[1] )
      {
        v27[0] = _mm_loadu_si128(v12 + 1);
      }
      else
      {
        v25 = (_OWORD *)v12->m128i_i64[0];
        *(_QWORD *)&v27[0] = v12[1].m128i_i64[0];
      }
      v26 = v12->m128i_i64[1];
      v11 = v26;
      v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
      v12->m128i_i64[1] = 0;
      v12[1].m128i_i8[0] = 0;
      sub_2241490(&v20, v25, v26, v11);
      if ( v25 != v27 )
        j_j___libc_free_0(v25, *(_QWORD *)&v27[0] + 1LL);
      if ( (__int64 *)v23[0] != &v24 )
        j_j___libc_free_0(v23[0], v24 + 1);
      ++v10;
    }
    while ( v18 != v10 );
  }
  if ( v19 )
  {
    if ( !a6 )
      a6 = sub_B6DC00(*a5, a2, a3);
    sub_BAAF70(a1, a5, v20, v21, a2, a6);
    if ( v20 != &v22 )
      j_j___libc_free_0(v20, v22.m128i_i64[0] + 1);
  }
  else
  {
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    if ( v20 == &v22 )
    {
      a1[1] = _mm_load_si128(&v22);
    }
    else
    {
      a1->m128i_i64[0] = (__int64)v20;
      a1[1].m128i_i64[0] = v22.m128i_i64[0];
    }
    a1->m128i_i64[1] = v21;
  }
  return a1;
}
