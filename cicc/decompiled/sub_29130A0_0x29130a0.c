// Function: sub_29130A0
// Address: 0x29130a0
//
char *__fastcall sub_29130A0(const __m128i *src, const __m128i *a2, char *a3, __int64 a4)
{
  __int64 v5; // rax
  const __m128i *v6; // r13
  __int64 v7; // rbx
  __int64 v8; // r15
  __int64 v9; // r14
  const __m128i *v10; // rdi

  v5 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)src) >> 3);
  v6 = src;
  v7 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 48 * a4;
    v9 = 24 * a4;
    do
    {
      v10 = v6;
      v6 = (const __m128i *)((char *)v6 + v8);
      a3 = sub_2912FB0(v10, (const __m128i *)((char *)v6 + v9 - v8), (const __m128i *)((char *)v6 + v9 - v8), v6, a3);
      v5 = 0xAAAAAAAAAAAAAAABLL * (((char *)a2 - (char *)v6) >> 3);
    }
    while ( v7 <= v5 );
  }
  if ( a4 <= v5 )
    v5 = a4;
  return sub_2912FB0(v6, (const __m128i *)((char *)v6 + 24 * v5), (const __m128i *)((char *)v6 + 24 * v5), a2, a3);
}
