// Function: sub_3440350
// Address: 0x3440350
//
__int64 __fastcall sub_3440350(const __m128i *a1, const __m128i *a2, __int64 a3, __int64 a4)
{
  const __m128i *v4; // r11
  const __m128i *v6; // r15
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // r12
  const __m128i *v11; // rdi

  v4 = a2;
  v6 = a1;
  v7 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)a1) >> 3);
  v8 = 2 * a4;
  if ( 2 * a4 <= v7 )
  {
    v9 = 48 * a4;
    v10 = 24 * a4;
    do
    {
      v11 = v6;
      v6 = (const __m128i *)((char *)v6 + v9);
      a3 = sub_3440210(v11, (const __m128i *)((char *)v6 + v10 - v9), (const __m128i *)((char *)v6 + v10 - v9), v6, a3);
      v7 = 0xAAAAAAAAAAAAAAABLL * (((char *)v4 - (char *)v6) >> 3);
    }
    while ( v7 >= v8 );
  }
  if ( v7 > a4 )
    v7 = a4;
  return sub_3440210(v6, (const __m128i *)((char *)v6 + 24 * v7), (const __m128i *)((char *)v6 + 24 * v7), v4, a3);
}
