// Function: sub_8842A0
// Address: 0x8842a0
//
__int64 __fastcall sub_8842A0(__int64 a1, __int64 a2, _DWORD *a3)
{
  __m128i v4; // xmm3
  __int64 result; // rax
  __int64 v6; // r14
  __int64 v7; // [rsp-10h] [rbp-30h]

  if ( a3 )
  {
    *a3 = 0;
    if ( (*(_BYTE *)(a2 + 82) & 4) != 0 )
    {
      *a3 = 1;
LABEL_4:
      *(__m128i *)a1 = _mm_loadu_si128(xmmword_4F06660);
      *(__m128i *)(a1 + 16) = _mm_loadu_si128(&xmmword_4F06660[1]);
      *(__m128i *)(a1 + 32) = _mm_loadu_si128(&xmmword_4F06660[2]);
      v4 = _mm_loadu_si128(&xmmword_4F06660[3]);
      result = *(_QWORD *)dword_4F07508;
      *(_BYTE *)(a1 + 17) |= 0x20u;
      *(__m128i *)(a1 + 48) = v4;
      *(_QWORD *)(a1 + 8) = result;
      return result;
    }
  }
  else if ( (*(_BYTE *)(a2 + 82) & 4) != 0 )
  {
    sub_6854C0(0x10Au, (FILE *)(a1 + 8), a2);
    goto LABEL_4;
  }
  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( (*(_BYTE *)(result + 6) & 2) == 0 && (*(_BYTE *)(a2 + 81) & 0x10) != 0 )
  {
    v6 = *(_QWORD *)(a1 + 24);
    result = sub_883A10(v6, a2, 0);
    if ( !(_DWORD)result )
    {
      sub_87D9B0(v6, a2, 0, (FILE *)(a1 + 8), a1, 3, 0, a3);
      return v7;
    }
  }
  return result;
}
