// Function: sub_6E3CB0
// Address: 0x6e3cb0
//
__int64 __fastcall sub_6E3CB0(__int64 a1, __int64 *a2, _QWORD *a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // rcx
  const __m128i *v6; // rdx

  result = qword_4D03C50;
  if ( (*(_BYTE *)(qword_4D03C50 + 19LL) & 2) != 0 )
  {
    result = *(unsigned __int8 *)(a1 + 24);
    if ( (_BYTE)result )
    {
      if ( (_BYTE)result == 1 && (*(_BYTE *)(a1 + 59) & 1) != 0 )
      {
        return sub_6E3AC0(a1, a2, 0, a2);
      }
      else
      {
        result = sub_6E3C60(a1, a2, a3, a4);
        if ( *(_BYTE *)(a1 + 24) == 5 )
        {
          v5 = *(_QWORD *)(a1 + 56);
          v6 = *(const __m128i **)(a1 + 80);
          result = *(_QWORD *)(v5 + 120);
          if ( result )
          {
            if ( (const __m128i *)result != v6 )
            {
              *(__m128i *)(result + 352) = _mm_loadu_si128(v6 + 22);
              *(__m128i *)(result + 368) = _mm_loadu_si128(v6 + 23);
            }
          }
          else
          {
            *(_QWORD *)(v5 + 120) = v6;
          }
        }
      }
    }
  }
  return result;
}
