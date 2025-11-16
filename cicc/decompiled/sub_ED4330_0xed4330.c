// Function: sub_ED4330
// Address: 0xed4330
//
_QWORD *__fastcall sub_ED4330(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, char a5)
{
  __int64 *v5; // r12
  __int64 *v6; // r15
  char v7; // al
  __m128i *v8; // rbx
  __m128i *v9; // r12
  __m128i *v13; // [rsp+20h] [rbp-70h] BYREF
  __m128i *v14; // [rsp+28h] [rbp-68h]
  __int64 v15; // [rsp+30h] [rbp-60h]
  __m128i v16; // [rsp+40h] [rbp-50h] BYREF
  __int64 v17; // [rsp+50h] [rbp-40h] BYREF

  v5 = &a2[a3];
  v13 = 0;
  v14 = 0;
  v15 = 0;
  if ( a2 != v5 )
  {
    v6 = a2;
    do
    {
      sub_ED15E0(v16.m128i_i64, *v6, 0);
      sub_ED42C0(&v13, &v16);
      if ( (__int64 *)v16.m128i_i64[0] != &v17 )
        j_j___libc_free_0(v16.m128i_i64[0], v17 + 1);
      ++v6;
    }
    while ( v5 != v6 );
  }
  v7 = sub_C5E690();
  sub_ED1AF0(a1, (__int64)v13, ((char *)v14 - (char *)v13) >> 5, v7 & a5, a4);
  v8 = v14;
  v9 = v13;
  if ( v14 != v13 )
  {
    do
    {
      if ( (__m128i *)v9->m128i_i64[0] != &v9[1] )
        j_j___libc_free_0(v9->m128i_i64[0], v9[1].m128i_i64[0] + 1);
      v9 += 2;
    }
    while ( v8 != v9 );
    v9 = v13;
  }
  if ( v9 )
    j_j___libc_free_0(v9, v15 - (_QWORD)v9);
  return a1;
}
