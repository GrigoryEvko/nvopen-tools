// Function: sub_87DC80
// Address: 0x87dc80
//
__int64 __fastcall sub_87DC80(__int64 a1, int a2, __int64 a3, int a4)
{
  _BYTE *v4; // rdx
  __int64 result; // rax
  __m128i v6; // xmm3
  __int64 v7; // rax

  v4 = *(_BYTE **)(a1 + 24);
  result = 0;
  if ( (v4[82] & 4) != 0 && (!a2 || v4[80] != 16 || (v4[96] & 0x20) == 0) )
  {
    if ( a4 )
      sub_6854C0(0x10Au, (FILE *)(a1 + 8), (__int64)v4);
    *(__m128i *)a1 = _mm_loadu_si128(xmmword_4F06660);
    *(__m128i *)(a1 + 16) = _mm_loadu_si128(&xmmword_4F06660[1]);
    *(__m128i *)(a1 + 32) = _mm_loadu_si128(&xmmword_4F06660[2]);
    v6 = _mm_loadu_si128(&xmmword_4F06660[3]);
    v7 = *(_QWORD *)dword_4F07508;
    *(_BYTE *)(a1 + 17) |= 0x20u;
    *(__m128i *)(a1 + 48) = v6;
    *(_QWORD *)(a1 + 8) = v7;
    return 1;
  }
  return result;
}
