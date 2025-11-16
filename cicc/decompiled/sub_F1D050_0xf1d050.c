// Function: sub_F1D050
// Address: 0xf1d050
//
__int64 __fastcall sub_F1D050(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  bool v5; // zf
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 i; // rdx
  _QWORD *v9; // rax
  _QWORD *v10; // [rsp+8h] [rbp-28h] BYREF

  v4 = a2;
  v5 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v5 )
  {
    result = *(_QWORD *)(a1 + 16);
    v7 = 56LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = a1 + 16;
    v7 = 224;
  }
  for ( i = result + v7; i != result; result += 56 )
  {
    if ( result )
    {
      *(_QWORD *)result = -4096;
      *(_QWORD *)(result + 8) = 0;
      *(_BYTE *)(result + 32) = 0;
      *(_QWORD *)(result + 40) = 0;
    }
  }
  if ( a2 != a3 )
  {
    do
    {
      result = *(_QWORD *)v4;
      if ( *(_QWORD *)v4 == -4096 )
      {
        if ( !*(_QWORD *)(v4 + 8) && !*(_BYTE *)(v4 + 32) )
          goto LABEL_15;
      }
      else if ( result == -8192
             && !*(_QWORD *)(v4 + 8)
             && *(_BYTE *)(v4 + 32)
             && !*(_QWORD *)(v4 + 16)
             && !*(_QWORD *)(v4 + 24) )
      {
LABEL_15:
        if ( !*(_QWORD *)(v4 + 40) )
          goto LABEL_11;
      }
      sub_F15DB0(a1, (__int64 *)v4, &v10);
      v9 = v10;
      *v10 = *(_QWORD *)v4;
      *(__m128i *)(v9 + 1) = _mm_loadu_si128((const __m128i *)(v4 + 8));
      *(__m128i *)(v9 + 3) = _mm_loadu_si128((const __m128i *)(v4 + 24));
      v9[5] = *(_QWORD *)(v4 + 40);
      v10[6] = *(_QWORD *)(v4 + 48);
      result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
      *(_DWORD *)(a1 + 8) = result;
LABEL_11:
      v4 += 56;
    }
    while ( a3 != v4 );
  }
  return result;
}
