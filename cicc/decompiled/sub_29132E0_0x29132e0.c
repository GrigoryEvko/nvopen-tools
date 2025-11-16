// Function: sub_29132E0
// Address: 0x29132e0
//
void __fastcall sub_29132E0(__m128i *src, const __m128i *a2, __m128i *a3)
{
  __int64 v4; // rsi
  const __m128i *v7; // r15
  __m128i *v8; // r14
  __m128i *v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rcx

  v4 = (char *)a2 - (char *)src;
  v7 = (__m128i *)((char *)a3 + v4);
  if ( v4 <= 144 )
  {
    sub_2913170(src, a2);
  }
  else
  {
    v8 = src;
    do
    {
      v9 = v8;
      v8 = (__m128i *)((char *)v8 + 168);
      sub_2913170(v9, v8);
    }
    while ( (char *)a2 - (char *)v8 > 144 );
    sub_2913170(v8, a2);
    if ( v4 > 168 )
    {
      v10 = 7;
      do
      {
        sub_29130A0(src, a2, a3->m128i_i8, v10);
        v11 = 2 * v10;
        v10 *= 4;
        sub_29130A0(a3, v7, src->m128i_i8, v11);
      }
      while ( (__int64)(0xAAAAAAAAAAAAAAABLL * (v4 >> 3)) > v10 );
    }
  }
}
