// Function: sub_2A49AE0
// Address: 0x2a49ae0
//
void __fastcall sub_2A49AE0(__m128i *src, const __m128i *a2, __m128i *a3, __int64 a4)
{
  __m128i *v7; // r15
  __m128i *v8; // rdi
  __int64 v9; // r15
  __int64 v10; // rcx
  __int64 v11; // [rsp+8h] [rbp-48h]
  signed __int64 v12; // [rsp+10h] [rbp-40h]
  const __m128i *v13; // [rsp+18h] [rbp-38h]

  v11 = (char *)a2 - (char *)src;
  v13 = (__m128i *)((char *)a3 + (char *)a2 - (char *)src);
  v12 = 0xAAAAAAAAAAAAAAABLL * (a2 - src);
  if ( (char *)a2 - (char *)src <= 288 )
  {
    sub_2A45D80(src, a2, a4);
  }
  else
  {
    v7 = src;
    do
    {
      v8 = v7;
      v7 += 21;
      sub_2A45D80(v8, v7, a4);
    }
    while ( (char *)a2 - (char *)v7 > 288 );
    sub_2A45D80(v7, a2, a4);
    if ( v11 > 336 )
    {
      v9 = 7;
      do
      {
        sub_2A49A00(src, a2, a3, v9, a4);
        v10 = 2 * v9;
        v9 *= 4;
        sub_2A49A00(a3, v13, src, v10, a4);
      }
      while ( v12 > v9 );
    }
  }
}
