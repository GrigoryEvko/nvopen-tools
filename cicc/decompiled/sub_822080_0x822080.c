// Function: sub_822080
// Address: 0x822080
//
__int64 __fastcall sub_822080(__int64 a1, unsigned __int64 a2, _QWORD *a3, __m128i *a4)
{
  __m128i v4; // xmm3
  __int64 result; // rax
  unsigned __int64 v7; // rbx
  int v9[13]; // [rsp+1Ch] [rbp-34h] BYREF

  *a3 = 0;
  *a4 = _mm_loadu_si128(xmmword_4F06660);
  a4[1] = _mm_loadu_si128(&xmmword_4F06660[1]);
  a4[2] = _mm_loadu_si128(&xmmword_4F06660[2]);
  v4 = _mm_loadu_si128(&xmmword_4F06660[3]);
  result = 0;
  a4->m128i_i32[2] = 0;
  a4->m128i_i16[6] = 1;
  a4[3] = v4;
  if ( a2 )
  {
    v7 = 0;
    while ( !dword_4F055C0[*(char *)(a1 + v7) + 128]
         && (unsigned int)sub_7B3CF0((unsigned __int8 *)(a1 + v7), v9, v7 == 0) )
    {
      v7 += v9[0];
      if ( a2 <= v7 )
      {
        *a3 = sub_87A510(a1, a2, a4);
        return 1;
      }
    }
    return 0;
  }
  return result;
}
