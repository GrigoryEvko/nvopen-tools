// Function: sub_2443870
// Address: 0x2443870
//
char *__fastcall sub_2443870(const __m128i *src, const __m128i *a2, __m128i *a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // r14
  const __m128i *v8; // rbx
  __int64 v9; // r15
  __int64 v10; // r13
  const __m128i *v11; // rdi
  __int64 v12; // rsi

  v5 = a2 - src;
  v6 = 2 * a4;
  v8 = src;
  if ( 2 * a4 <= v5 )
  {
    v9 = 32 * a4;
    v10 = -1 * a4;
    do
    {
      v11 = v8;
      v8 = (const __m128i *)((char *)v8 + v9);
      a3 = (__m128i *)sub_2443570(v11, &v8[v10], &v8[v10], v8, a3);
      v5 = a2 - v8;
    }
    while ( v6 <= v5 );
  }
  v12 = a4;
  if ( a4 > v5 )
    v12 = v5;
  return sub_2443570(v8, &v8[v12], &v8[v12], a2, a3);
}
