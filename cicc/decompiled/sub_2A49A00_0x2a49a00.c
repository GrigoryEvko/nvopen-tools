// Function: sub_2A49A00
// Address: 0x2a49a00
//
char *__fastcall sub_2A49A00(const __m128i *src, const __m128i *a2, __m128i *a3, __int64 a4, __int64 a5)
{
  const __m128i *v5; // r15
  __int64 v6; // rax
  __int64 v8; // rbx
  __int64 v9; // r13
  const __m128i *v10; // rdi
  __int64 v13; // [rsp+8h] [rbp-38h]

  v5 = src;
  v6 = 0xAAAAAAAAAAAAAAABLL * (a2 - src);
  v8 = 2 * a4;
  if ( 2 * a4 <= v6 )
  {
    v13 = 48 * a4;
    v9 = 96 * a4;
    do
    {
      v10 = v5;
      v5 = (const __m128i *)((char *)v5 + v9);
      a3 = (__m128i *)sub_2A49900(
                        v10,
                        (const __m128i *)((char *)v5 + v13 - v9),
                        (const __m128i *)((char *)v5 + v13 - v9),
                        v5,
                        a3,
                        a5);
      v6 = 0xAAAAAAAAAAAAAAABLL * (a2 - v5);
    }
    while ( v6 >= v8 );
  }
  if ( v6 > a4 )
    v6 = a4;
  return sub_2A49900(v5, &v5[3 * v6], &v5[3 * v6], a2, a3, a5);
}
