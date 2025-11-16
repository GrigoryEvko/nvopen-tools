// Function: sub_2565F60
// Address: 0x2565f60
//
__int64 __fastcall sub_2565F60(__m128i *a1, __int64 a2)
{
  __int64 result; // rax
  __m128i v3; // xmm0

  if ( (unsigned __int8)sub_2509800(a1) != 5 )
    BUG();
  result = sub_A777F0(0x130u, *(__int64 **)(a2 + 128));
  if ( result )
  {
    v3 = _mm_loadu_si128(a1);
    *(_QWORD *)(result + 40) = result + 56;
    *(_QWORD *)(result + 48) = 0x200000000LL;
    *(_WORD *)(result + 96) = 256;
    *(_QWORD *)(result + 88) = &unk_4A1E5E0;
    *(_QWORD *)(result + 168) = result + 184;
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = 0;
    *(_DWORD *)(result + 32) = 0;
    *(_QWORD *)result = off_4A1E558;
    *(_QWORD *)(result + 104) = 0;
    *(_QWORD *)(result + 112) = 0;
    *(_QWORD *)(result + 120) = 0;
    *(_DWORD *)(result + 128) = 0;
    *(_QWORD *)(result + 136) = 0;
    *(_QWORD *)(result + 144) = 0;
    *(_QWORD *)(result + 152) = 0;
    *(_DWORD *)(result + 160) = 0;
    *(_QWORD *)(result + 176) = 0x400000000LL;
    *(_QWORD *)(result + 216) = 0;
    *(_QWORD *)(result + 224) = 0;
    *(_QWORD *)(result + 232) = 0;
    *(_DWORD *)(result + 240) = 0;
    *(_QWORD *)(result + 248) = result + 264;
    *(_QWORD *)(result + 256) = 0x400000000LL;
    *(_BYTE *)(result + 296) = 1;
    *(__m128i *)(result + 72) = v3;
  }
  return result;
}
