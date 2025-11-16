// Function: sub_E4D960
// Address: 0xe4d960
//
_BYTE *__fastcall sub_E4D960(__int64 a1)
{
  _BYTE *result; // rax
  __int64 v3; // rdi
  __m128i *v4; // rdx
  __m128i si128; // xmm0

  result = *(_BYTE **)(a1 + 312);
  if ( *((_DWORD *)result + 44) == 1 )
  {
    v3 = *(_QWORD *)(a1 + 304);
    v4 = *(__m128i **)(v3 + 32);
    if ( *(_QWORD *)(v3 + 24) - (_QWORD)v4 <= 0x16u )
    {
      sub_CB6200(v3, "\t.intel_syntax noprefix", 0x17u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F7F830);
      v4[1].m128i_i32[0] = 1701998703;
      v4[1].m128i_i16[2] = 26982;
      v4[1].m128i_i8[6] = 120;
      *v4 = si128;
      *(_QWORD *)(v3 + 32) += 23LL;
    }
    return sub_E4D880(a1);
  }
  return result;
}
