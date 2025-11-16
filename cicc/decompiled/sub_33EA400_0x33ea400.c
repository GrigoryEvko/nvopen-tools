// Function: sub_33EA400
// Address: 0x33ea400
//
__m128i *__fastcall sub_33EA400(
        __int64 *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        int a8)
{
  __int64 v11; // rbx
  __int64 v12; // rdi
  __int16 v13; // ax
  unsigned __int8 v14; // r11
  __int16 v17; // [rsp+Eh] [rbp-52h]
  _OWORD v18[5]; // [rsp+10h] [rbp-50h] BYREF

  v11 = 16LL * a3;
  v12 = *(_QWORD *)(a2 + 112);
  v13 = *(_WORD *)(v12 + 32) & 0x3CF;
  v18[0] = _mm_loadu_si128((const __m128i *)(v12 + 40));
  v18[1] = _mm_loadu_si128((const __m128i *)(v12 + 56));
  v17 = v13;
  v14 = sub_2EAC4F0(v12);
  return sub_33EA290(
           a1,
           a8,
           (*(_BYTE *)(a2 + 33) >> 2) & 3,
           *(unsigned __int16 *)(*(_QWORD *)(a2 + 48) + v11),
           *(_QWORD *)(*(_QWORD *)(a2 + 48) + v11 + 8),
           a4,
           *(_OWORD *)*(_QWORD *)(a2 + 40),
           a5,
           a6,
           a7,
           *(_OWORD *)*(_QWORD *)(a2 + 112),
           *(_QWORD *)(*(_QWORD *)(a2 + 112) + 16LL),
           *(unsigned __int16 *)(a2 + 96),
           *(_QWORD *)(a2 + 104),
           v14,
           v17,
           (__int64)v18,
           0);
}
