// Function: sub_16C8E00
// Address: 0x16c8e00
//
__int64 __fastcall sub_16C8E00(
        void *a1,
        size_t a2,
        _QWORD *a3,
        __int64 a4,
        const __m128i *a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __m128i **a9,
        _BYTE *a10)
{
  char v12; // al
  __int64 v15; // [rsp+24h] [rbp-5Ch] BYREF
  __m128i v16; // [rsp+30h] [rbp-50h] BYREF
  __int8 v17; // [rsp+40h] [rbp-40h]

  sub_16C7610(&v15);
  if ( !a10 )
  {
    v17 = a5[1].m128i_i8[0];
    if ( !v17 )
    {
      sub_16C8300((__pid_t *)&v15, a1, a2, a3, a4, (__int64)&v16, a7, a8, a9);
      return v15;
    }
    goto LABEL_8;
  }
  *a10 = 0;
  v17 = a5[1].m128i_i8[0];
  if ( v17 )
LABEL_8:
    v16 = _mm_loadu_si128(a5);
  v12 = sub_16C8300((__pid_t *)&v15, a1, a2, a3, a4, (__int64)&v16, a7, a8, a9);
  if ( a10 && v12 != 1 )
    *a10 = 1;
  return v15;
}
