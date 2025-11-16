// Function: sub_3986F10
// Address: 0x3986f10
//
void __fastcall sub_3986F10(__m128i *src, __m128i *a2, __m128i *a3, __int64 a4)
{
  __int64 *v7; // r14
  __int64 *v8; // rdi
  __int64 v9; // r14
  __int64 v10; // rcx
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12; // [rsp+10h] [rbp-40h]
  const __m128i *v13; // [rsp+18h] [rbp-38h]

  v11 = (char *)a2 - (char *)src;
  v12 = a2 - src;
  v13 = (__m128i *)((char *)a3 + (char *)a2 - (char *)src);
  if ( (char *)a2 - (char *)src <= 96 )
  {
    sub_3986390(src->m128i_i64, a2->m128i_i64, a4);
  }
  else
  {
    v7 = (__int64 *)src;
    do
    {
      v8 = v7;
      v7 += 14;
      sub_3986390(v8, v7, a4);
    }
    while ( (char *)a2 - (char *)v7 > 96 );
    sub_3986390(v7, a2->m128i_i64, a4);
    if ( v11 > 112 )
    {
      v9 = 7;
      do
      {
        sub_3986E50(src, a2, a3, v9, a4);
        v10 = 2 * v9;
        v9 *= 4;
        sub_3986E50(a3, v13, src, v10, a4);
      }
      while ( v12 > v9 );
    }
  }
}
