// Function: sub_2045380
// Address: 0x2045380
//
bool __fastcall sub_2045380(__int64 *a1)
{
  unsigned __int64 v1; // r8
  __int64 v2; // r9
  unsigned int v3; // esi
  unsigned int v4; // ecx
  bool result; // al
  __m128i v6; // xmm0
  unsigned __int64 *v7; // rdx
  bool v8; // cf

  v1 = *a1;
  v2 = a1[1];
  v3 = *((_DWORD *)a1 + 4);
  v4 = *((_DWORD *)a1 + 5);
  while ( 1 )
  {
    v7 = (unsigned __int64 *)a1;
    v8 = *((_DWORD *)a1 - 1) < v4;
    if ( *((_DWORD *)a1 - 1) == v4 )
    {
      v8 = *((_DWORD *)a1 - 2) < v3;
      if ( *((_DWORD *)a1 - 2) == v3 )
        break;
    }
    result = v8;
    a1 -= 3;
    if ( !v8 )
      goto LABEL_7;
LABEL_3:
    v6 = _mm_loadu_si128((const __m128i *)a1);
    a1[5] = a1[2];
    *(__m128i *)(a1 + 3) = v6;
  }
  result = *(a1 - 3) > v1;
  a1 -= 3;
  if ( result )
    goto LABEL_3;
LABEL_7:
  *v7 = v1;
  v7[1] = v2;
  *((_DWORD *)v7 + 4) = v3;
  *((_DWORD *)v7 + 5) = v4;
  return result;
}
