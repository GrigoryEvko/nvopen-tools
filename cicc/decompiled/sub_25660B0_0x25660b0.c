// Function: sub_25660B0
// Address: 0x25660b0
//
__int64 __fastcall sub_25660B0(__m128i *a1, __int64 a2)
{
  __int64 result; // rax
  __m128i v3; // xmm0

  if ( (unsigned __int8)sub_2509800(a1) != 1 )
    BUG();
  result = sub_A777F0(0xC8u, *(__int64 **)(a2 + 128));
  if ( result )
  {
    v3 = _mm_loadu_si128(a1);
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 40) = result + 56;
    *(_QWORD *)(result + 48) = 0x200000000LL;
    *(_WORD *)(result + 96) = 256;
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = 0;
    *(_DWORD *)(result + 32) = 0;
    *(_QWORD *)result = off_4A1E470;
    *(_QWORD *)(result + 88) = &unk_4A1E4F8;
    *(_QWORD *)(result + 104) = 0;
    *(_QWORD *)(result + 112) = result + 136;
    *(_QWORD *)(result + 120) = 8;
    *(_DWORD *)(result + 128) = 0;
    *(_BYTE *)(result + 132) = 1;
    *(__m128i *)(result + 72) = v3;
  }
  return result;
}
