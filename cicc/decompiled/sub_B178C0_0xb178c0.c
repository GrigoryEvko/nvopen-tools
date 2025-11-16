// Function: sub_B178C0
// Address: 0xb178c0
//
void *__fastcall sub_B178C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __m128i v8; // xmm0
  __int64 v9; // rdx
  __int64 v11; // [rsp+8h] [rbp-48h]
  __m128i v12[4]; // [rsp+10h] [rbp-40h] BYREF

  v11 = *(_QWORD *)(a5 + 40);
  sub_B157E0((__int64)v12, (_QWORD *)(a5 + 48));
  v8 = _mm_loadu_si128(v12);
  v9 = *(_QWORD *)(*(_QWORD *)(a5 + 40) + 72LL);
  *(_QWORD *)(a1 + 88) = 0x400000000LL;
  *(_QWORD *)(a1 + 40) = a2;
  *(_QWORD *)(a1 + 16) = v9;
  *(_QWORD *)(a1 + 424) = v11;
  *(_QWORD *)(a1 + 48) = a3;
  *(_QWORD *)(a1 + 56) = a4;
  *(_DWORD *)(a1 + 8) = 15;
  *(_BYTE *)(a1 + 12) = 2;
  *(_BYTE *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = a1 + 96;
  *(_BYTE *)(a1 + 416) = 0;
  *(_DWORD *)(a1 + 420) = -1;
  *(_QWORD *)a1 = &unk_49D9DE8;
  *(__m128i *)(a1 + 24) = v8;
  return &unk_49D9DE8;
}
