// Function: sub_CAD7C0
// Address: 0xcad7c0
//
__int64 __fastcall sub_CAD7C0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7)
{
  __m128i v7; // xmm0
  __int64 result; // rax

  *(_QWORD *)(a1 + 8) = a3;
  v7 = _mm_loadu_si128((const __m128i *)&a7);
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)a1 = &unk_49DCC58;
  *(_DWORD *)(a1 + 32) = a2;
  *(_QWORD *)(a1 + 40) = a4;
  *(_QWORD *)(a1 + 48) = a5;
  *(__m128i *)(a1 + 56) = v7;
  result = *(_QWORD *)(sub_CAD7B0(a1, a2, a3, a4, a5) + 8);
  *(_QWORD *)(a1 + 16) = result;
  *(_QWORD *)(a1 + 24) = result;
  return result;
}
