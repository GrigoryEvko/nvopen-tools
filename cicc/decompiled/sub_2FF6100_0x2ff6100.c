// Function: sub_2FF6100
// Address: 0x2ff6100
//
__int64 __fastcall sub_2FF6100(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int128 a8,
        __int64 a9,
        __int64 a10,
        unsigned int a11)
{
  __m128i v11; // xmm0
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 result; // rax

  *(_QWORD *)(a1 + 20) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)a1 = &unk_4A2D968;
  v11 = _mm_loadu_si128((const __m128i *)&a8);
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 272) = a7;
  v12 = a9;
  *(_DWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 312) = v12;
  v13 = a10;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 320) = v13;
  result = a11;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_DWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_QWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = a2;
  *(_QWORD *)(a1 + 256) = a5;
  *(_QWORD *)(a1 + 264) = a6;
  *(_QWORD *)(a1 + 280) = a3;
  *(_QWORD *)(a1 + 288) = a4;
  *(_DWORD *)(a1 + 328) = result;
  *(__m128i *)(a1 + 296) = v11;
  return result;
}
