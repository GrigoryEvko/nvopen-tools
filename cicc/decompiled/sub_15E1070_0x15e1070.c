// Function: sub_15E1070
// Address: 0x15e1070
//
__int64 *__fastcall sub_15E1070(__int64 *a1, int a2, __int64 *a3, __int64 a4)
{
  unsigned int v6; // eax
  __int64 v7; // rdx
  char *v8; // r13
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rcx
  __m128i *v12; // rax
  __int64 *i; // [rsp+8h] [rbp-78h]
  _QWORD v15[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v16; // [rsp+20h] [rbp-60h] BYREF
  _OWORD *v17; // [rsp+30h] [rbp-50h] BYREF
  __int64 v18; // [rsp+38h] [rbp-48h]
  _OWORD v19[4]; // [rsp+40h] [rbp-40h] BYREF

  v17 = v15;
  *(_QWORD *)(__readfsqword(0) - 24) = &v17;
  *(_QWORD *)(__readfsqword(0) - 32) = sub_15DE590;
  if ( !&_pthread_key_create )
  {
    v6 = -1;
LABEL_17:
    sub_4264C5(v6);
  }
  v6 = pthread_once(&dword_4F9E14C, init_routine);
  if ( v6 )
    goto LABEL_17;
  v7 = -1;
  v8 = (&off_4C6F380)[a2];
  *a1 = (__int64)(a1 + 2);
  if ( v8 )
    v7 = (__int64)&v8[strlen(v8)];
  sub_15DE5B0(a1, v8, v7);
  for ( i = &a3[a4]; i != a3; ++a3 )
  {
    sub_15DF750((__int64)v15, *a3, v9, v10);
    v12 = (__m128i *)sub_2241130(v15, 0, 0, ".", 1);
    v17 = v19;
    if ( (__m128i *)v12->m128i_i64[0] == &v12[1] )
    {
      v19[0] = _mm_loadu_si128(v12 + 1);
    }
    else
    {
      v17 = (_OWORD *)v12->m128i_i64[0];
      *(_QWORD *)&v19[0] = v12[1].m128i_i64[0];
    }
    v18 = v12->m128i_i64[1];
    v11 = v18;
    v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
    v12->m128i_i64[1] = 0;
    v12[1].m128i_i8[0] = 0;
    sub_2241490(a1, v17, v18, v11);
    if ( v17 != v19 )
      j_j___libc_free_0(v17, *(_QWORD *)&v19[0] + 1LL);
    if ( (__int64 *)v15[0] != &v16 )
      j_j___libc_free_0(v15[0], v16 + 1);
  }
  return a1;
}
