// Function: sub_771200
// Address: 0x771200
//
unsigned __int64 __fastcall sub_771200(__int64 a1, int a2, unsigned int a3)
{
  unsigned int v3; // r8d
  unsigned __int64 result; // rax
  unsigned int v5; // eax
  unsigned int v6; // r9d
  unsigned __int64 *v7; // rcx

  v3 = a2 & (a3 + 1);
  result = *(_QWORD *)(a1 + 16LL * v3);
  while ( 1 )
  {
    v5 = a2 & (result >> 3);
    v6 = a2 & (v3 + 1);
    v7 = (unsigned __int64 *)(a1 + 16LL * v6);
    if ( (v5 > a3 || v5 <= v3 && a3 >= v3) && (a3 >= v3 || v5 <= v3) )
      break;
    *(__m128i *)(a1 + 16LL * a3) = _mm_loadu_si128((const __m128i *)(a1 + 16LL * v3));
    *(_QWORD *)(a1 + 16LL * v3) = 0;
    result = *v7;
    if ( !*v7 )
      return result;
LABEL_3:
    a3 = v3;
    v3 = v6;
  }
  result = *v7;
  if ( *v7 )
  {
    v3 = a3;
    goto LABEL_3;
  }
  return result;
}
