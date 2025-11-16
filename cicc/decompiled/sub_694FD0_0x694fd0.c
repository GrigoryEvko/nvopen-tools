// Function: sub_694FD0
// Address: 0x694fd0
//
unsigned __int8 *__fastcall sub_694FD0(__int64 a1, char *a2, __m128i *a3)
{
  __int64 v4; // rax
  size_t v5; // rax
  unsigned __int8 *result; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rcx

  *a3 = _mm_loadu_si128(xmmword_4F06660);
  a3[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
  a3[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
  v4 = *(_QWORD *)&dword_4F077C8;
  a3[3] = _mm_loadu_si128(&xmmword_4F06660[3]);
  a3->m128i_i64[1] = v4;
  v5 = strlen(a2);
  sub_878540(a2, v5);
  result = (unsigned __int8 *)sub_7D2AC0(a3, a1, 0);
  if ( result )
  {
    if ( (result[81] & 0x10) == 0
      || (v7 = result[80], (unsigned __int8)v7 > 0x14u)
      || (v8 = 1180672, !_bittest64(&v8, v7)) )
    {
      if ( (result[84] & 2) == 0 )
        return 0;
    }
  }
  return result;
}
