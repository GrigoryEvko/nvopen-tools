// Function: sub_E82D80
// Address: 0xe82d80
//
__int64 __fastcall sub_E82D80(__int64 a1, int a2, int a3, int a4, int a5, __int64 a6, __int128 a7)
{
  __int64 result; // rax
  __m128i v8; // xmm0

  result = *(_QWORD *)(*(_QWORD *)(a1 + 296) + 24LL);
  v8 = _mm_loadu_si128((const __m128i *)&a7);
  *(_BYTE *)(result + 1952) = 1;
  *(_DWORD *)(result + 1956) = a2;
  *(_DWORD *)(result + 1960) = a3;
  *(_DWORD *)(result + 1964) = a4;
  *(_DWORD *)(result + 1968) = a5;
  *(__m128i *)(result + 1972) = v8;
  return result;
}
