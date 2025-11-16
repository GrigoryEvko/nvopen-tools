// Function: sub_16C8EF0
// Address: 0x16c8ef0
//
__int64 __fastcall sub_16C8EF0(
        void *a1,
        size_t a2,
        _QWORD *a3,
        __int64 a4,
        const __m128i *a5,
        unsigned int a6,
        __int64 a7,
        __int64 a8,
        int a9,
        __int64 **a10,
        _BYTE *a11)
{
  __int64 result; // rax
  unsigned int v14; // edx
  __pid_t v17[3]; // [rsp+14h] [rbp-5Ch] BYREF
  __m128i v18; // [rsp+20h] [rbp-50h] BYREF

  sub_16C7610(v17);
  if ( a5[1].m128i_i8[0] )
    v18 = _mm_loadu_si128(a5);
  if ( (unsigned __int8)sub_16C8300(v17, a1, a2, a3, a4, (__int64)&v18, a7, a8, (__m128i **)a10) )
  {
    if ( a11 )
      *a11 = 0;
    sub_16C7BA0(v17, a6, a6 == 0, a10);
    return v14;
  }
  else
  {
    result = 0xFFFFFFFFLL;
    if ( a11 )
      *a11 = 1;
  }
  return result;
}
