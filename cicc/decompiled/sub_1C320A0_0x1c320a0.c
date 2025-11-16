// Function: sub_1C320A0
// Address: 0x1c320a0
//
__int64 __fastcall sub_1C320A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // eax
  __int64 v5; // r13
  __int64 i; // r12
  __int64 result; // rax
  __int64 v8; // rax
  __m128i *v9; // rdx
  __m128i si128; // xmm0

  if ( *(_BYTE *)(a2 + 16) == 5 )
  {
    v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    if ( v4 )
    {
      v5 = v4 - 1;
      for ( i = 0; ; ++i )
      {
        sub_1C320A0(a1, *(_QWORD *)(a2 + 24 * (i - v4)), a3);
        if ( i == v5 )
          break;
        v4 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
      }
    }
  }
  result = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 15 )
  {
    result = *(unsigned int *)(result + 8);
    if ( (unsigned int)result > 0x1FF && (unsigned int)result >> 8 != 4 )
    {
      v8 = sub_1C31E60(a1, a3, 0);
      v9 = *(__m128i **)(v8 + 24);
      if ( *(_QWORD *)(v8 + 16) - (_QWORD)v9 <= 0x35u )
      {
        sub_16E7EE0(v8, "Invalid address space for global constant initializer\n", 0x36u);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_42D0500);
        v9[3].m128i_i32[0] = 1702521196;
        v9[3].m128i_i16[2] = 2674;
        *v9 = si128;
        v9[1] = _mm_load_si128((const __m128i *)&xmmword_42D0510);
        v9[2] = _mm_load_si128((const __m128i *)&xmmword_42D0520);
        *(_QWORD *)(v8 + 24) += 54LL;
      }
      return sub_1C31880(a1);
    }
  }
  return result;
}
