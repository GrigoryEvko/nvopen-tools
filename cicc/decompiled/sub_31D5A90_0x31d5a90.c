// Function: sub_31D5A90
// Address: 0x31d5a90
//
void __fastcall sub_31D5A90(__m128i *src, __m128i *a2, __m128i *a3)
{
  __int64 v4; // rsi
  const __m128i *v7; // r15
  char *v8; // r14
  char *v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rcx

  v4 = (char *)a2 - (char *)src;
  v7 = (__m128i *)((char *)a3 + v4);
  if ( v4 <= 144 )
  {
    sub_31D59B0(src->m128i_i8, a2->m128i_i8);
  }
  else
  {
    v8 = (char *)src;
    do
    {
      v9 = v8;
      v8 += 168;
      sub_31D59B0(v9, v8);
    }
    while ( (char *)a2 - v8 > 144 );
    sub_31D59B0(v8, a2->m128i_i8);
    if ( v4 > 168 )
    {
      v10 = 7;
      do
      {
        sub_31D5160(src, a2, a3->m128i_i8, v10);
        v11 = 2 * v10;
        v10 *= 4;
        sub_31D5160(a3, v7, src->m128i_i8, v11);
      }
      while ( (__int64)(0xAAAAAAAAAAAAAAABLL * (v4 >> 3)) > v10 );
    }
  }
}
