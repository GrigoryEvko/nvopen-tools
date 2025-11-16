// Function: sub_1A1AC50
// Address: 0x1a1ac50
//
__int64 __fastcall sub_1A1AC50(unsigned __int64 *a1)
{
  __int64 v1; // r8
  unsigned __int64 v2; // rdx
  unsigned __int64 v3; // rsi
  __int64 v4; // rcx
  __int64 result; // rax
  __m128i v6; // xmm0

  v1 = a1[2];
  v2 = *a1;
  v3 = a1[1];
  v4 = (v1 >> 2) & 1;
  while ( 1 )
  {
    if ( *(a1 - 3) > v2 )
      goto LABEL_4;
    if ( *(a1 - 3) < v2 )
    {
LABEL_7:
      *a1 = v2;
      a1[1] = v3;
      a1[2] = v1;
      return result;
    }
    result = ((__int64)*(a1 - 1) >> 2) & 1;
    if ( (_BYTE)v4 == (_BYTE)result )
      break;
    if ( (_BYTE)v4 )
      goto LABEL_7;
LABEL_4:
    v6 = _mm_loadu_si128((const __m128i *)(a1 - 3));
    result = *(a1 - 1);
    a1 -= 3;
    a1[5] = result;
    *(__m128i *)(a1 + 3) = v6;
  }
  if ( *(a1 - 2) < v3 )
    goto LABEL_4;
  *a1 = v2;
  a1[1] = v3;
  a1[2] = v1;
  return result;
}
