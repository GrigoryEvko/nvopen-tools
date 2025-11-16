// Function: sub_33F6C50
// Address: 0x33f6c50
//
__m128i *__fastcall sub_33F6C50(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        int a8)
{
  __int64 v8; // r14

  v8 = *(_QWORD *)(a2 + 40);
  return sub_33F65D0(
           a1,
           *(_QWORD *)v8,
           *(_QWORD *)(v8 + 8),
           a4,
           *(_QWORD *)(v8 + 40),
           *(_QWORD *)(v8 + 48),
           a5,
           a6,
           a7,
           *(_OWORD *)(v8 + 160),
           *(unsigned __int16 *)(a2 + 96),
           *(_QWORD *)(a2 + 104),
           *(const __m128i **)(a2 + 112),
           a8,
           (*(_BYTE *)(a2 + 33) & 4) != 0,
           (*(_BYTE *)(a2 + 33) & 8) != 0);
}
