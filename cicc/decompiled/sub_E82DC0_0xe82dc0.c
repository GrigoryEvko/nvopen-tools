// Function: sub_E82DC0
// Address: 0xe82dc0
//
__int64 __fastcall sub_E82DC0(__int64 a1, int a2, int a3, int a4, int a5, __int64 a6, __int128 a7)
{
  __int64 result; // rax
  __m128i v8; // xmm0

  result = *(_QWORD *)(*(_QWORD *)(a1 + 296) + 24LL);
  v8 = _mm_loadu_si128((const __m128i *)&a7);
  *(_BYTE *)(result + 1988) = 1;
  *(_DWORD *)(result + 1992) = a2;
  *(_DWORD *)(result + 1996) = a3;
  *(_DWORD *)(result + 2000) = a4;
  *(_DWORD *)(result + 2004) = a5;
  *(__m128i *)(result + 2008) = v8;
  return result;
}
