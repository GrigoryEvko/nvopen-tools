// Function: sub_3986E50
// Address: 0x3986e50
//
char *__fastcall sub_3986E50(const __m128i *src, const __m128i *a2, __m128i *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rax
  const __m128i *v6; // r15
  __int64 v9; // r14
  __int64 v10; // rbx
  const __m128i *v11; // rdi
  __int64 v12; // rsi
  __int64 v15; // [rsp+8h] [rbp-38h]

  v5 = a2 - src;
  v6 = src;
  v15 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v9 = 32 * a4;
    v10 = -1 * a4;
    do
    {
      v11 = v6;
      v6 = (const __m128i *)((char *)v6 + v9);
      a3 = (__m128i *)sub_3986C10(v11, &v6[v10], &v6[v10], v6, a3, a5);
      v5 = a2 - v6;
    }
    while ( v5 >= v15 );
  }
  v12 = a4;
  if ( v5 <= a4 )
    v12 = v5;
  return sub_3986C10(v6, &v6[v12], &v6[v12], a2, a3, a5);
}
