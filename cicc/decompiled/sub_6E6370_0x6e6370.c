// Function: sub_6E6370
// Address: 0x6e6370
//
__int64 __fastcall sub_6E6370(__m128i *a1, __int64 a2)
{
  __int64 v2; // r8
  __int64 v3; // r9
  unsigned int *v4; // rdx
  __int64 result; // rax
  unsigned int v6; // [rsp+Ch] [rbp-14h] BYREF

  if ( (unsigned int)sub_6E6010() )
  {
    v4 = 0;
    v6 = 0;
    if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
      v4 = &v6;
    sub_8842A0(a1, a2, v4);
    result = v6;
    if ( v6 )
      return sub_6E50A0();
  }
  else
  {
    result = sub_87DC80(a1, 0, 0, 0, v2, v3);
    if ( (_DWORD)result )
    {
      *a1 = _mm_loadu_si128(xmmword_4F06660);
      a1[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
      a1[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
      result = *(_QWORD *)dword_4F07508;
      a1[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
      a1[1].m128i_i8[1] |= 0x20u;
      a1->m128i_i64[1] = result;
    }
  }
  return result;
}
