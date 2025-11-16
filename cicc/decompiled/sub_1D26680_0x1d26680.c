// Function: sub_1D26680
// Address: 0x1d26680
//
__int64 __fastcall sub_1D26680(
        _QWORD *a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        unsigned int a8)
{
  __int64 v11; // rbx
  __int64 v12; // rdi
  unsigned __int16 v13; // ax
  int v14; // r11d
  unsigned __int16 v17; // [rsp+Eh] [rbp-52h]
  __m128i v18; // [rsp+10h] [rbp-50h] BYREF
  __int64 v19; // [rsp+20h] [rbp-40h]

  v11 = 16LL * a3;
  v12 = *(_QWORD *)(a2 + 104);
  v13 = *(_WORD *)(v12 + 32) & 0x1CF;
  v18 = _mm_loadu_si128((const __m128i *)(v12 + 40));
  v17 = v13;
  v19 = *(_QWORD *)(v12 + 56);
  v14 = sub_1E34390();
  return sub_1D264C0(
           a1,
           a8,
           (*(_BYTE *)(a2 + 27) >> 2) & 3,
           *(unsigned __int8 *)(*(_QWORD *)(a2 + 40) + v11),
           *(_QWORD *)(*(_QWORD *)(a2 + 40) + v11 + 8),
           a4,
           *(_OWORD *)*(_QWORD *)(a2 + 32),
           a5,
           a6,
           a7,
           *(_OWORD *)*(_QWORD *)(a2 + 104),
           *(_QWORD *)(*(_QWORD *)(a2 + 104) + 16LL),
           *(unsigned __int8 *)(a2 + 88),
           *(_QWORD *)(a2 + 96),
           v14,
           v17,
           (__int64)&v18,
           0);
}
