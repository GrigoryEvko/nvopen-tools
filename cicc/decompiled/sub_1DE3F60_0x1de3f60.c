// Function: sub_1DE3F60
// Address: 0x1de3f60
//
void __fastcall sub_1DE3F60(__m128i *src, __m128i *a2, __m128i *a3)
{
  __int64 v4; // rsi
  const __m128i *v7; // r15
  unsigned __int64 *v8; // r14
  unsigned __int64 *v9; // rdi
  __int64 v10; // r14
  __int64 v11; // rcx

  v4 = (char *)a2 - (char *)src;
  v7 = (__m128i *)((char *)a3 + v4);
  if ( v4 <= 144 )
  {
    sub_1DE3E80((unsigned __int64 *)src, (unsigned __int64 *)a2);
  }
  else
  {
    v8 = (unsigned __int64 *)src;
    do
    {
      v9 = v8;
      v8 += 21;
      sub_1DE3E80(v9, v8);
    }
    while ( (char *)a2 - (char *)v8 > 144 );
    sub_1DE3E80(v8, (unsigned __int64 *)a2);
    if ( v4 > 168 )
    {
      v10 = 7;
      do
      {
        sub_1DE38C0(src, a2, a3->m128i_i8, v10);
        v11 = 2 * v10;
        v10 *= 4;
        sub_1DE38C0(a3, v7, src->m128i_i8, v11);
      }
      while ( (__int64)(0xAAAAAAAAAAAAAAABLL * (v4 >> 3)) > v10 );
    }
  }
}
