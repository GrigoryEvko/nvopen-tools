// Function: sub_235CCD0
// Address: 0x235ccd0
//
unsigned __int64 *__fastcall sub_235CCD0(
        unsigned __int64 *a1,
        __int64 a2,
        unsigned __int64 *a3,
        const __m128i *a4,
        __int64 a5)
{
  const __m128i *v5; // r15
  const __m128i *v6; // r14
  unsigned __int64 v8; // rax
  __int64 v10[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a4;
  v6 = (const __m128i *)((char *)a4 + 40 * a5);
  if ( a4 == v6 )
  {
LABEL_6:
    v10[0] = 0;
    *a1 = 1;
    sub_9C66B0(v10);
  }
  else
  {
    while ( 1 )
    {
      sub_235B6A0(v10, a2, a3, v5);
      v8 = v10[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v10[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        break;
      v5 = (const __m128i *)((char *)v5 + 40);
      if ( v6 == v5 )
        goto LABEL_6;
    }
    v10[0] = 0;
    *a1 = v8 | 1;
    sub_9C66B0(v10);
  }
  return a1;
}
