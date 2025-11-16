// Function: sub_371D0A0
// Address: 0x371d0a0
//
__int64 __fastcall sub_371D0A0(__int64 *a1)
{
  __int64 v1; // rsi
  __int64 v2; // rdx
  unsigned int v3; // ecx
  __m128i v4; // xmm0
  __int64 v5; // rax
  __int64 result; // rax

  v1 = *a1;
  v2 = a1[1];
  v3 = *((_DWORD *)a1 + 4);
  while ( 1 )
  {
    result = *(unsigned int *)(*(a1 - 2) + 72);
    if ( *(_DWORD *)(v2 + 72) == (_DWORD)result )
      break;
    if ( *(_DWORD *)(v2 + 72) >= (unsigned int)result )
      goto LABEL_6;
LABEL_3:
    v4 = _mm_loadu_si128((const __m128i *)(a1 - 3));
    v5 = *(a1 - 1);
    a1 -= 3;
    a1[5] = v5;
    *(__m128i *)(a1 + 3) = v4;
  }
  if ( v3 < *((_DWORD *)a1 - 2) )
    goto LABEL_3;
LABEL_6:
  *a1 = v1;
  a1[1] = v2;
  *((_DWORD *)a1 + 4) = v3;
  return result;
}
