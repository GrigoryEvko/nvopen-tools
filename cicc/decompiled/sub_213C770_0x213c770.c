// Function: sub_213C770
// Address: 0x213c770
//
__int64 *__fastcall sub_213C770(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v5; // rax
  __m128i v6; // xmm0
  __m128i v7; // xmm1
  __int64 v8; // rax
  unsigned int v9; // ecx
  __int128 v11; // [rsp+0h] [rbp-30h] BYREF
  __int128 v12; // [rsp+10h] [rbp-20h] BYREF

  v5 = *(_QWORD *)(a2 + 32);
  v6 = _mm_loadu_si128((const __m128i *)(v5 + 80));
  v7 = _mm_loadu_si128((const __m128i *)(v5 + 120));
  v8 = *(_QWORD *)(v5 + 40);
  v11 = (__int128)v6;
  v9 = *(_DWORD *)(v8 + 84);
  v12 = (__int128)v7;
  sub_213C0E0(a1, (__int64)&v11, (__int64)&v12, v9, v6, *(double *)v7.m128i_i64, a5);
  return sub_1D2E370(
           *(_QWORD **)(a1 + 8),
           (__int64 *)a2,
           **(_QWORD **)(a2 + 32),
           *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
           *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
           *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
           v11,
           v12,
           *(_OWORD *)(*(_QWORD *)(a2 + 32) + 160LL));
}
