// Function: sub_2EAC4C0
// Address: 0x2eac4c0
//
__int64 __fastcall sub_2EAC4C0(__m128i *a1, const __m128i *a2)
{
  __int64 result; // rax

  result = a2[2].m128i_u8[2];
  if ( (unsigned __int8)result >= (unsigned int)a1[2].m128i_i8[2] )
  {
    a1[2].m128i_i8[2] = result;
    *a1 = _mm_loadu_si128(a2);
    a1[1].m128i_i32[0] = a2[1].m128i_i32[0];
    result = a2[1].m128i_u8[4];
    a1[1].m128i_i8[4] = result;
  }
  return result;
}
