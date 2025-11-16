// Function: sub_E00FE0
// Address: 0xe00fe0
//
__m128i *__fastcall sub_E00FE0(__m128i *a1, __int64 *a2, unsigned __int64 a3, unsigned int a4)
{
  __int64 v7; // rdi
  __int64 v8; // rdi
  __int64 v9; // rax
  __m128i v11; // [rsp+0h] [rbp-40h] BYREF
  __int64 v12; // [rsp+10h] [rbp-30h]
  __int64 v13; // [rsp+18h] [rbp-28h]

  v7 = *a2;
  if ( *a2 )
    v7 = sub_E00740(v7);
  v11.m128i_i64[0] = v7;
  v8 = a2[1];
  if ( v8 )
    v8 = sub_E00750(v8, a3);
  v9 = a2[2];
  v11.m128i_i64[1] = v8;
  v12 = v9;
  v13 = a2[3];
  sub_E00CC0(a1, &v11, a4);
  return a1;
}
