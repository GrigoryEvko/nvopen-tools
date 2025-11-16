// Function: sub_33E95C0
// Address: 0x33e95c0
//
__m128i *__fastcall sub_33E95C0(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int64 a5,
        unsigned __int64 a6,
        __int128 a7,
        int a8)
{
  __int64 v8; // rax
  unsigned __int16 *v9; // r12

  v8 = *(_QWORD *)(a2 + 40);
  v9 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16LL * a3);
  return sub_33E8F60(
           a1,
           *v9,
           *((_QWORD *)v9 + 1),
           a4,
           *(_QWORD *)v8,
           *(_QWORD *)(v8 + 8),
           a5,
           a6,
           a7,
           *(_OWORD *)(v8 + 120),
           *(_OWORD *)(v8 + 160),
           *(unsigned __int16 *)(a2 + 96),
           *(_QWORD *)(a2 + 104),
           *(const __m128i **)(a2 + 112),
           a8,
           (*(_BYTE *)(a2 + 33) >> 2) & 3,
           (*(_BYTE *)(a2 + 33) & 0x10) != 0);
}
