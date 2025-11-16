// Function: sub_2377250
// Address: 0x2377250
//
unsigned __int64 *__fastcall sub_2377250(
        unsigned __int64 *a1,
        __m128i *a2,
        unsigned __int64 *a3,
        __int64 a4,
        __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // r14
  unsigned __int64 v8; // rax
  __int64 v10[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = a4;
  v6 = a4 + 40 * a5;
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
      sub_2368220(v10, a2, a3, v5);
      v8 = v10[0] & 0xFFFFFFFFFFFFFFFELL;
      if ( (v10[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
        break;
      v5 += 40;
      if ( v6 == v5 )
        goto LABEL_6;
    }
    v10[0] = 0;
    *a1 = v8 | 1;
    sub_9C66B0(v10);
  }
  return a1;
}
