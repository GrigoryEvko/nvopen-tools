// Function: sub_E98A20
// Address: 0xe98a20
//
__int64 __fastcall sub_E98A20(__int64 a1, __int64 a2)
{
  __m128i v2; // xmm0
  __m128i v3; // xmm1
  __m128i v5; // [rsp+0h] [rbp-20h] BYREF
  __m128i v6; // [rsp+10h] [rbp-10h] BYREF

  *(_QWORD *)(a1 + 8) = a2;
  v5 = 0u;
  v6 = 0u;
  v2 = _mm_loadu_si128(&v5);
  v3 = _mm_loadu_si128(&v6);
  *(_QWORD *)a1 = &unk_49E3C78;
  *(_QWORD *)(a1 + 48) = a1 + 64;
  *(_QWORD *)(a1 + 56) = 0x100000000LL;
  *(_QWORD *)(a1 + 120) = a1 + 136;
  *(_WORD *)(a1 + 276) = 1;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_DWORD *)(a1 + 272) = 0;
  *(_BYTE *)(a1 + 278) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(__m128i *)(a1 + 136) = v2;
  *(__m128i *)(a1 + 152) = v3;
  *(_QWORD *)(a1 + 128) = 0x400000001LL;
  return 0x400000001LL;
}
