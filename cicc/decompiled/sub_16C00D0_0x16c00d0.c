// Function: sub_16C00D0
// Address: 0x16c00d0
//
__m128i *__fastcall sub_16C00D0(__m128i *a1, __m128i *a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx
  __m128i *v6; // rax
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rcx
  _QWORD *v11; // [rsp+0h] [rbp-30h] BYREF
  __int64 v12; // [rsp+8h] [rbp-28h]
  _QWORD v13[4]; // [rsp+10h] [rbp-20h] BYREF

  v3 = sub_22416F0(a2, "-darwin", 0, 7);
  if ( v3 == -1 )
  {
    v8 = sub_22416F0(a2, "-macos", 0, 6);
    if ( v8 != -1 )
    {
      sub_22410F0(a2, v8, 0);
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - a2->m128i_i64[1]) <= 6 )
        sub_4262D8((__int64)"basic_string::append");
      sub_2241490(a2, "-darwin", 7, v9);
      sub_16BFF80((__int64 *)&v11);
      sub_2241490(a2, v11, v12, v10);
      if ( v11 != v13 )
        j_j___libc_free_0(v11, v13[0] + 1LL);
    }
    a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
    v5 = a2->m128i_i64[0];
    v6 = a2 + 1;
    if ( (__m128i *)a2->m128i_i64[0] != &a2[1] )
      goto LABEL_5;
LABEL_9:
    a1[1] = _mm_loadu_si128(a2 + 1);
    goto LABEL_6;
  }
  sub_22410F0(a2, v3 + 7, 0);
  sub_16BFF80((__int64 *)&v11);
  sub_2241490(a2, v11, v12, v4);
  if ( v11 != v13 )
    j_j___libc_free_0(v11, v13[0] + 1LL);
  a1->m128i_i64[0] = (__int64)a1[1].m128i_i64;
  v5 = a2->m128i_i64[0];
  v6 = a2 + 1;
  if ( (__m128i *)a2->m128i_i64[0] == &a2[1] )
    goto LABEL_9;
LABEL_5:
  a1->m128i_i64[0] = v5;
  a1[1].m128i_i64[0] = a2[1].m128i_i64[0];
LABEL_6:
  a1->m128i_i64[1] = a2->m128i_i64[1];
  a2->m128i_i64[0] = (__int64)v6;
  a2->m128i_i64[1] = 0;
  a2[1].m128i_i8[0] = 0;
  return a1;
}
