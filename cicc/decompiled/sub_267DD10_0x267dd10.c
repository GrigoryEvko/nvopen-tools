// Function: sub_267DD10
// Address: 0x267dd10
//
__int64 __fastcall sub_267DD10(const __m128i *a1, __int64 a2)
{
  __int64 v2; // rdx
  _BYTE *v3; // rax
  __int64 result; // rax
  __m128i v5; // xmm0

  v2 = a1->m128i_i64[0] & 3;
  if ( v2 == 2 || v2 == 3 || (v3 = (_BYTE *)(a1->m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL)) == 0 || *v3 || v2 == 1 )
    BUG();
  result = sub_A777F0(0x238u, *(__int64 **)(a2 + 128));
  if ( result )
  {
    v5 = _mm_loadu_si128(a1);
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 40) = result + 56;
    *(_WORD *)(result + 96) = 256;
    *(_QWORD *)(result + 48) = 0x200000000LL;
    *(_QWORD *)(result + 120) = result + 144;
    *(_QWORD *)result = off_4A20118;
    *(_QWORD *)(result + 168) = result + 192;
    *(_QWORD *)(result + 88) = &unk_4A201C8;
    *(_QWORD *)(result + 16) = 0;
    *(_QWORD *)(result + 24) = 0;
    *(_DWORD *)(result + 32) = 0;
    *(_DWORD *)(result + 104) = 65793;
    *(_QWORD *)(result + 112) = 0;
    *(_QWORD *)(result + 128) = 2;
    *(_DWORD *)(result + 136) = 0;
    *(_BYTE *)(result + 140) = 1;
    *(_QWORD *)(result + 160) = 0;
    *(_QWORD *)(result + 176) = 4;
    *(_DWORD *)(result + 184) = 0;
    *(_BYTE *)(result + 188) = 1;
    *(_QWORD *)(result + 224) = 0;
    *(_QWORD *)(result + 232) = 0;
    *(_QWORD *)(result + 240) = 0;
    *(_DWORD *)(result + 248) = 0;
    *(_QWORD *)(result + 256) = 0;
    *(_QWORD *)(result + 264) = 0;
    *(_QWORD *)(result + 272) = 0;
    *(_DWORD *)(result + 280) = 0;
    *(_QWORD *)(result + 288) = 0;
    *(_QWORD *)(result + 296) = 0;
    *(__m128i *)(result + 72) = v5;
    *(_QWORD *)(result + 304) = 0;
    *(_QWORD *)(result + 320) = result + 336;
    *(_DWORD *)(result + 312) = 0;
    *(_QWORD *)(result + 328) = 0x1000000000LL;
    *(_QWORD *)(result + 464) = 0;
    *(_QWORD *)(result + 472) = 0;
    *(_QWORD *)(result + 480) = result + 504;
    *(_QWORD *)(result + 488) = 8;
    *(_DWORD *)(result + 496) = 0;
    *(_BYTE *)(result + 500) = 1;
  }
  return result;
}
