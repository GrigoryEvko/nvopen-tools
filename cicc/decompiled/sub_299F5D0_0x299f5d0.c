// Function: sub_299F5D0
// Address: 0x299f5d0
//
char *__fastcall sub_299F5D0(
        const __m128i *src,
        const __m128i *a2,
        char *a3,
        __int64 a4,
        unsigned __int8 (__fastcall *a5)(const __m128i *, const __m128i *))
{
  const __m128i *v5; // r15
  __int64 v6; // rax
  __int64 v9; // rcx
  __int64 v10; // r11
  __int64 v11; // rbx
  __int64 v12; // r13
  const __m128i *v13; // rdi
  __int64 v16; // [rsp+8h] [rbp-38h]

  v5 = src;
  v6 = 0x6DB6DB6DB6DB6DB7LL * (((char *)a2 - (char *)src) >> 3);
  v9 = 2 * a4;
  v16 = v9;
  if ( v9 <= v6 )
  {
    v10 = a4;
    v11 = 56 * a4;
    v12 = 8 * (16 * v10 - v9);
    do
    {
      v13 = v5;
      v5 = (const __m128i *)((char *)v5 + v12);
      a3 = sub_299F4D0(
             v13,
             (const __m128i *)((char *)v5 + v11 - v12),
             (const __m128i *)((char *)v5 + v11 - v12),
             v5,
             a3,
             a5);
      v6 = 0x6DB6DB6DB6DB6DB7LL * (((char *)a2 - (char *)v5) >> 3);
    }
    while ( v6 >= v16 );
  }
  if ( v6 > a4 )
    v6 = a4;
  return sub_299F4D0(v5, (const __m128i *)((char *)v5 + 56 * v6), (const __m128i *)((char *)v5 + 56 * v6), a2, a3, a5);
}
