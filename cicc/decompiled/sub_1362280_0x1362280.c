// Function: sub_1362280
// Address: 0x1362280
//
__int64 __fastcall sub_1362280(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  bool v5; // zf
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 i; // rdx
  __m128i *v9; // rax
  __m128i *v10; // [rsp+8h] [rbp-28h] BYREF

  v4 = a2;
  v5 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v5 )
  {
    result = *(_QWORD *)(a1 + 16);
    v7 = 88LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    result = a1 + 16;
    v7 = 704;
  }
  for ( i = result + v7; i != result; result += 88 )
  {
    if ( result )
    {
      *(_QWORD *)result = -8;
      *(_QWORD *)(result + 8) = 0;
      *(_QWORD *)(result + 16) = 0;
      *(_QWORD *)(result + 24) = 0;
      *(_QWORD *)(result + 32) = 0;
      *(_QWORD *)(result + 40) = -8;
      *(_QWORD *)(result + 48) = 0;
      *(_QWORD *)(result + 56) = 0;
      *(_QWORD *)(result + 64) = 0;
      *(_QWORD *)(result + 72) = 0;
    }
  }
  if ( a2 != a3 )
  {
    do
    {
      result = *(_QWORD *)v4;
      if ( *(_QWORD *)v4 == -8 )
      {
        if ( !*(_QWORD *)(v4 + 8)
          && !*(_QWORD *)(v4 + 16)
          && !*(_QWORD *)(v4 + 24)
          && !*(_QWORD *)(v4 + 32)
          && *(_QWORD *)(v4 + 40) == -8 )
        {
          goto LABEL_19;
        }
      }
      else if ( result == -16
             && !*(_QWORD *)(v4 + 8)
             && !*(_QWORD *)(v4 + 16)
             && !*(_QWORD *)(v4 + 24)
             && !*(_QWORD *)(v4 + 32)
             && *(_QWORD *)(v4 + 40) == -16 )
      {
LABEL_19:
        if ( !*(_QWORD *)(v4 + 48) && !*(_QWORD *)(v4 + 56) && !*(_QWORD *)(v4 + 64) && !*(_QWORD *)(v4 + 72) )
          goto LABEL_12;
      }
      sub_1361B70(a1, (__int64 *)v4, (__int64 **)&v10);
      v9 = v10;
      *v10 = _mm_loadu_si128((const __m128i *)v4);
      v9[1] = _mm_loadu_si128((const __m128i *)(v4 + 16));
      v9[2].m128i_i64[0] = *(_QWORD *)(v4 + 32);
      *(__m128i *)((char *)v9 + 40) = _mm_loadu_si128((const __m128i *)(v4 + 40));
      *(__m128i *)((char *)v9 + 56) = _mm_loadu_si128((const __m128i *)(v4 + 56));
      v9[4].m128i_i64[1] = *(_QWORD *)(v4 + 72);
      v9[5].m128i_i8[0] = *(_BYTE *)(v4 + 80);
      result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
      *(_DWORD *)(a1 + 8) = result;
LABEL_12:
      v4 += 88;
    }
    while ( a3 != v4 );
  }
  return result;
}
