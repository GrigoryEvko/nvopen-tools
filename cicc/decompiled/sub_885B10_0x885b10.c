// Function: sub_885B10
// Address: 0x885b10
//
_QWORD *__fastcall sub_885B10(__int64 a1)
{
  __int64 v1; // rdx
  __m128i v2; // xmm3
  __int64 v3; // rax
  _QWORD *result; // rax

  v1 = *(_QWORD *)a1;
  *(__m128i *)a1 = _mm_loadu_si128(xmmword_4F06660);
  *(__m128i *)(a1 + 16) = _mm_loadu_si128(&xmmword_4F06660[1]);
  *(__m128i *)(a1 + 32) = _mm_loadu_si128(&xmmword_4F06660[2]);
  v2 = _mm_loadu_si128(&xmmword_4F06660[3]);
  *(_BYTE *)(a1 + 17) |= 0x60u;
  v3 = *(_QWORD *)dword_4F07508;
  *(_QWORD *)a1 = v1;
  *(__m128i *)(a1 + 48) = v2;
  *(_QWORD *)(a1 + 8) = v3;
  result = sub_885AD0(0xDu, a1, 0, 1);
  *(_QWORD *)(a1 + 24) = result;
  return result;
}
