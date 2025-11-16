// Function: sub_1EF8540
// Address: 0x1ef8540
//
__m128i *__fastcall sub_1EF8540(const __m128i *a1, const __m128i *a2, __m128i *a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r13
  const __m128i *v7; // rbx
  __int64 v8; // r15
  __int64 v9; // r14
  const __m128i *v10; // rdi

  v5 = 0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - (char *)a1) >> 3);
  v6 = 2 * a4;
  v7 = a1;
  if ( 2 * a4 <= v5 )
  {
    v8 = 80 * a4;
    v9 = 40 * a4;
    do
    {
      v10 = v7;
      v7 = (const __m128i *)((char *)v7 + v8);
      a3 = sub_1EF82D0(v10, (const __m128i *)((char *)v7 + v9 - v8), (const __m128i *)((char *)v7 + v9 - v8), v7, a3);
      v5 = 0xCCCCCCCCCCCCCCCDLL * (((char *)a2 - (char *)v7) >> 3);
    }
    while ( v6 <= v5 );
  }
  if ( a4 <= v5 )
    v5 = a4;
  return sub_1EF82D0(v7, (const __m128i *)((char *)v7 + 40 * v5), (const __m128i *)((char *)v7 + 40 * v5), a2, a3);
}
