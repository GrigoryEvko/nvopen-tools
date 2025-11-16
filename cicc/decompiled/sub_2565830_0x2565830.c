// Function: sub_2565830
// Address: 0x2565830
//
__int64 __fastcall sub_2565830(__m128i *a1, __int64 a2)
{
  __int64 result; // rax
  __m128i v3; // xmm3
  __m128i v4; // xmm6
  __m128i v5; // xmm2
  __m128i v6; // xmm4
  __m128i v7; // xmm5
  __m128i v8; // xmm0
  __m128i v9; // xmm1

  switch ( (unsigned __int8)sub_2509800(a1) )
  {
    case 0u:
      BUG();
    case 1u:
      result = sub_A777F0(0x148u, *(__int64 **)(a2 + 128));
      if ( result )
      {
        v5 = _mm_loadu_si128(a1);
        *(_QWORD *)(result + 40) = result + 56;
        *(_QWORD *)(result + 48) = 0x200000000LL;
        *(_QWORD *)(result + 136) = result + 152;
        *(_QWORD *)(result + 248) = result + 264;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 24) = 0;
        *(_DWORD *)(result + 32) = 0;
        *(_WORD *)(result + 96) = 256;
        *(_QWORD *)(result + 104) = 0;
        *(_QWORD *)(result + 112) = 0;
        *(_QWORD *)(result + 120) = 0;
        *(_DWORD *)(result + 128) = 0;
        *(_QWORD *)(result + 144) = 0x800000000LL;
        *(_QWORD *)(result + 216) = 0;
        *(_QWORD *)(result + 224) = 0;
        *(_QWORD *)(result + 232) = 0;
        *(_DWORD *)(result + 240) = 0;
        *(_QWORD *)(result + 256) = 0x800000000LL;
        *(_QWORD *)result = off_4A1DE18;
        *(_QWORD *)(result + 88) = &unk_4A1DEA0;
        *(__m128i *)(result + 72) = v5;
      }
      break;
    case 2u:
      result = sub_A777F0(0x148u, *(__int64 **)(a2 + 128));
      if ( result )
      {
        v6 = _mm_loadu_si128(a1);
        *(_QWORD *)(result + 40) = result + 56;
        *(_QWORD *)(result + 48) = 0x200000000LL;
        *(_QWORD *)(result + 136) = result + 152;
        *(_QWORD *)(result + 248) = result + 264;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 24) = 0;
        *(_DWORD *)(result + 32) = 0;
        *(_WORD *)(result + 96) = 256;
        *(_QWORD *)(result + 104) = 0;
        *(_QWORD *)(result + 112) = 0;
        *(_QWORD *)(result + 120) = 0;
        *(_DWORD *)(result + 128) = 0;
        *(_QWORD *)(result + 144) = 0x800000000LL;
        *(_QWORD *)(result + 216) = 0;
        *(_QWORD *)(result + 224) = 0;
        *(_QWORD *)(result + 232) = 0;
        *(_DWORD *)(result + 240) = 0;
        *(_QWORD *)(result + 256) = 0x800000000LL;
        *(_QWORD *)result = off_4A1E1B8;
        *(_QWORD *)(result + 88) = &unk_4A1E240;
        *(__m128i *)(result + 72) = v6;
      }
      break;
    case 3u:
      result = sub_A777F0(0x148u, *(__int64 **)(a2 + 128));
      if ( result )
      {
        v7 = _mm_loadu_si128(a1);
        *(_WORD *)(result + 96) = 256;
        *(_QWORD *)(result + 40) = result + 56;
        *(_QWORD *)(result + 248) = result + 264;
        *(_QWORD *)(result + 48) = 0x200000000LL;
        *(_QWORD *)(result + 136) = result + 152;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 24) = 0;
        *(_DWORD *)(result + 32) = 0;
        *(_QWORD *)(result + 104) = 0;
        *(_QWORD *)(result + 112) = 0;
        *(_QWORD *)(result + 120) = 0;
        *(_DWORD *)(result + 128) = 0;
        *(_QWORD *)(result + 144) = 0x800000000LL;
        *(_QWORD *)(result + 216) = 0;
        *(_QWORD *)(result + 224) = 0;
        *(_QWORD *)(result + 232) = 0;
        *(_DWORD *)(result + 240) = 0;
        *(_QWORD *)(result + 256) = 0x800000000LL;
        *(_QWORD *)result = off_4A1E2A0;
        *(_QWORD *)(result + 88) = &unk_4A1E328;
        *(__m128i *)(result + 72) = v7;
      }
      break;
    case 4u:
      result = sub_A777F0(0x148u, *(__int64 **)(a2 + 128));
      if ( result )
      {
        v8 = _mm_loadu_si128(a1);
        *(_QWORD *)(result + 40) = result + 56;
        *(_QWORD *)(result + 48) = 0x200000000LL;
        *(_QWORD *)(result + 136) = result + 152;
        *(_QWORD *)(result + 248) = result + 264;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 24) = 0;
        *(_DWORD *)(result + 32) = 0;
        *(_WORD *)(result + 96) = 256;
        *(_QWORD *)(result + 104) = 0;
        *(_QWORD *)(result + 112) = 0;
        *(_QWORD *)(result + 120) = 0;
        *(_DWORD *)(result + 128) = 0;
        *(_QWORD *)(result + 144) = 0x800000000LL;
        *(_QWORD *)(result + 216) = 0;
        *(_QWORD *)(result + 224) = 0;
        *(_QWORD *)(result + 232) = 0;
        *(_DWORD *)(result + 240) = 0;
        *(_QWORD *)(result + 256) = 0x800000000LL;
        *(_QWORD *)result = off_4A1E388;
        *(_QWORD *)(result + 88) = &unk_4A1E410;
        *(__m128i *)(result + 72) = v8;
      }
      break;
    case 5u:
      result = sub_A777F0(0x148u, *(__int64 **)(a2 + 128));
      if ( result )
      {
        v9 = _mm_loadu_si128(a1);
        *(_QWORD *)(result + 40) = result + 56;
        *(_QWORD *)(result + 248) = result + 264;
        *(_QWORD *)(result + 48) = 0x200000000LL;
        *(_QWORD *)(result + 136) = result + 152;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 24) = 0;
        *(_DWORD *)(result + 32) = 0;
        *(_WORD *)(result + 96) = 256;
        *(_QWORD *)(result + 104) = 0;
        *(_QWORD *)(result + 112) = 0;
        *(_QWORD *)(result + 120) = 0;
        *(_DWORD *)(result + 128) = 0;
        *(_QWORD *)(result + 144) = 0x800000000LL;
        *(_QWORD *)(result + 216) = 0;
        *(_QWORD *)(result + 224) = 0;
        *(_QWORD *)(result + 232) = 0;
        *(_DWORD *)(result + 240) = 0;
        *(_QWORD *)(result + 256) = 0x800000000LL;
        *(_QWORD *)result = off_4A1DFE8;
        *(_QWORD *)(result + 88) = &unk_4A1E070;
        *(__m128i *)(result + 72) = v9;
      }
      break;
    case 6u:
      result = sub_A777F0(0x148u, *(__int64 **)(a2 + 128));
      if ( result )
      {
        v3 = _mm_loadu_si128(a1);
        *(_QWORD *)(result + 40) = result + 56;
        *(_QWORD *)(result + 248) = result + 264;
        *(_QWORD *)(result + 48) = 0x200000000LL;
        *(_QWORD *)(result + 136) = result + 152;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 24) = 0;
        *(_DWORD *)(result + 32) = 0;
        *(_WORD *)(result + 96) = 256;
        *(_QWORD *)(result + 104) = 0;
        *(_QWORD *)(result + 112) = 0;
        *(_QWORD *)(result + 120) = 0;
        *(_DWORD *)(result + 128) = 0;
        *(_QWORD *)(result + 144) = 0x800000000LL;
        *(_QWORD *)(result + 216) = 0;
        *(_QWORD *)(result + 224) = 0;
        *(_QWORD *)(result + 232) = 0;
        *(_DWORD *)(result + 240) = 0;
        *(_QWORD *)(result + 256) = 0x800000000LL;
        *(_QWORD *)result = off_4A1DF00;
        *(_QWORD *)(result + 88) = &unk_4A1DF88;
        *(__m128i *)(result + 72) = v3;
      }
      break;
    case 7u:
      result = sub_A777F0(0x148u, *(__int64 **)(a2 + 128));
      if ( result )
      {
        v4 = _mm_loadu_si128(a1);
        *(_QWORD *)(result + 40) = result + 56;
        *(_QWORD *)(result + 48) = 0x200000000LL;
        *(_WORD *)(result + 96) = 256;
        *(_QWORD *)(result + 136) = result + 152;
        *(_QWORD *)(result + 248) = result + 264;
        *(_QWORD *)(result + 8) = 0;
        *(_QWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 24) = 0;
        *(_DWORD *)(result + 32) = 0;
        *(_QWORD *)(result + 104) = 0;
        *(_QWORD *)(result + 112) = 0;
        *(_QWORD *)(result + 120) = 0;
        *(_DWORD *)(result + 128) = 0;
        *(_QWORD *)(result + 144) = 0x800000000LL;
        *(_QWORD *)(result + 216) = 0;
        *(_QWORD *)(result + 224) = 0;
        *(_QWORD *)(result + 232) = 0;
        *(_DWORD *)(result + 240) = 0;
        *(_QWORD *)(result + 256) = 0x800000000LL;
        *(_QWORD *)result = off_4A1E0D0;
        *(_QWORD *)(result + 88) = &unk_4A1E158;
        *(__m128i *)(result + 72) = v4;
      }
      break;
    default:
      result = 0;
      break;
  }
  return result;
}
