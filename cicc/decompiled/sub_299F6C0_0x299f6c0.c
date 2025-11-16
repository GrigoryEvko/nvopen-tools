// Function: sub_299F6C0
// Address: 0x299f6c0
//
void __fastcall sub_299F6C0(
        __m128i *src,
        __m128i *a2,
        __m128i *a3,
        unsigned __int8 (__fastcall *a4)(__m128i *, __int8 *))
{
  char *v7; // r15
  char *v8; // rdi
  __int64 v9; // r15
  __int64 v10; // rcx
  __int64 v11; // [rsp+8h] [rbp-48h]
  __int64 v12; // [rsp+10h] [rbp-40h]
  const __m128i *v13; // [rsp+18h] [rbp-38h]

  v11 = (char *)a2 - (char *)src;
  v13 = (__m128i *)((char *)a3 + (char *)a2 - (char *)src);
  v12 = 0x6DB6DB6DB6DB6DB7LL * (((char *)a2 - (char *)src) >> 3);
  if ( (char *)a2 - (char *)src <= 336 )
  {
    sub_299EA40(src->m128i_i8, a2->m128i_i8, a4);
  }
  else
  {
    v7 = (char *)src;
    do
    {
      v8 = v7;
      v7 += 392;
      sub_299EA40(v8, v7, a4);
    }
    while ( (char *)a2 - v7 > 336 );
    sub_299EA40(v7, a2->m128i_i8, a4);
    if ( v11 > 392 )
    {
      v9 = 7;
      do
      {
        sub_299F5D0(src, a2, a3->m128i_i8, v9, (unsigned __int8 (__fastcall *)(const __m128i *, const __m128i *))a4);
        v10 = 2 * v9;
        v9 *= 4;
        sub_299F5D0(a3, v13, src->m128i_i8, v10, (unsigned __int8 (__fastcall *)(const __m128i *, const __m128i *))a4);
      }
      while ( v12 > v9 );
    }
  }
}
