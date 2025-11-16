// Function: sub_ED4450
// Address: 0xed4450
//
_QWORD *__fastcall sub_ED4450(_QWORD *a1, __int64 *a2, __int64 a3, __int64 a4, char a5)
{
  __int64 *v5; // r13
  __int64 *v6; // r15
  _BYTE *v7; // rax
  __int64 v8; // rdx
  char v9; // al
  __m128i *v10; // rbx
  __m128i *v11; // r12
  __m128i *v15; // [rsp+20h] [rbp-70h] BYREF
  __m128i *v16; // [rsp+28h] [rbp-68h]
  __int64 v17; // [rsp+30h] [rbp-60h]
  __m128i v18; // [rsp+40h] [rbp-50h] BYREF
  _QWORD v19[8]; // [rsp+50h] [rbp-40h] BYREF

  v5 = &a2[a3];
  v15 = 0;
  v16 = 0;
  v17 = 0;
  if ( a2 != v5 )
  {
    v6 = a2;
    do
    {
      v7 = (_BYTE *)sub_ED1E10(*v6);
      v18.m128i_i64[0] = (__int64)v19;
      sub_ED0450(v18.m128i_i64, v7, (__int64)&v7[v8]);
      sub_ED42C0(&v15, &v18);
      if ( (_QWORD *)v18.m128i_i64[0] != v19 )
        j_j___libc_free_0(v18.m128i_i64[0], v19[0] + 1LL);
      ++v6;
    }
    while ( v5 != v6 );
  }
  v9 = sub_C5E690();
  sub_ED1AF0(a1, (__int64)v15, ((char *)v16 - (char *)v15) >> 5, v9 & a5, a4);
  v10 = v16;
  v11 = v15;
  if ( v16 != v15 )
  {
    do
    {
      if ( (__m128i *)v11->m128i_i64[0] != &v11[1] )
        j_j___libc_free_0(v11->m128i_i64[0], v11[1].m128i_i64[0] + 1);
      v11 += 2;
    }
    while ( v10 != v11 );
    v11 = v15;
  }
  if ( v11 )
    j_j___libc_free_0(v11, v17 - (_QWORD)v11);
  return a1;
}
