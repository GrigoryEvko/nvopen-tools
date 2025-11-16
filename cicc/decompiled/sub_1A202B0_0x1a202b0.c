// Function: sub_1A202B0
// Address: 0x1a202b0
//
unsigned __int64 __fastcall sub_1A202B0(
        __int64 a1,
        __int64 a2,
        __int64 *a3,
        const __m128i *a4,
        unsigned int a5,
        _QWORD *a6,
        unsigned __int64 a7)
{
  unsigned __int64 v7; // rax
  __int64 v8; // r11

  v7 = *(unsigned __int8 *)(a2 + 8);
  if ( (unsigned __int8)v7 <= 0x10u && (v8 = 100990, _bittest64(&v8, v7)) )
    return sub_1A1F8D0(a1, a2, a3, a4, a5, a6);
  else
    return sub_1A1FCD0(a1, a2, a3, (__int64)a4, a5, a6, a7);
}
