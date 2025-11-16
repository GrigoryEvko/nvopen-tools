// Function: sub_7B0D00
// Address: 0x7b0d00
//
__int64 __fastcall sub_7B0D00(int a1, int a2)
{
  __int64 v2; // r8
  const __m128i *v3; // r9
  __m128i v4; // xmm4
  __int64 v5; // rbx
  const __m128i *v7; // r11
  _QWORD *v8; // r10
  const __m128i *v9; // rax
  __int64 v10; // r10

  v2 = sub_727A60();
  v3 = *(const __m128i **)(unk_4F064B0 + 56LL);
  *(__m128i *)v2 = _mm_loadu_si128(v3);
  *(__m128i *)(v2 + 16) = _mm_loadu_si128(v3 + 1);
  *(__m128i *)(v2 + 32) = _mm_loadu_si128(v3 + 2);
  *(__m128i *)(v2 + 48) = _mm_loadu_si128(v3 + 3);
  v4 = _mm_loadu_si128(v3 + 4);
  *(_DWORD *)(v2 + 32) = a2;
  v5 = unk_4F064B0;
  *(_DWORD *)(v2 + 24) = a1;
  *(_DWORD *)(v2 + 28) = -1;
  *(_QWORD *)(v2 + 40) = 0;
  *(_QWORD *)(v2 + 48) = 0;
  *(__m128i *)(v2 + 64) = v4;
  v3[3].m128i_i64[1] = v2;
  if ( *(_QWORD *)(v2 + 56) )
  {
LABEL_2:
    *(_QWORD *)(v5 + 56) = v2;
    return v2;
  }
  if ( dword_4F17FD8 <= 0 )
    v7 = (const __m128i *)qword_4F07280;
  else
    v7 = *(const __m128i **)(v5 - 56);
  if ( (const __m128i *)v7[3].m128i_i64[0] != v3 )
  {
    v7 = (const __m128i *)qword_4F07280;
    if ( qword_4F07280 )
    {
      while ( v3 != v7 )
      {
        if ( v3 == (const __m128i *)v7[3].m128i_i64[0] )
          goto LABEL_16;
        v8 = (_QWORD *)v7[2].m128i_i64[1];
        v7 = (const __m128i *)v7[3].m128i_i64[1];
        if ( v8 )
        {
          while ( 1 )
          {
            v9 = (const __m128i *)sub_7AD800(v8, (__int64)v3);
            v8 = *(_QWORD **)(v10 + 56);
            if ( v9 )
              break;
            if ( !v8 )
              goto LABEL_12;
          }
          v7 = v9;
          goto LABEL_16;
        }
LABEL_12:
        if ( !v7 )
          goto LABEL_2;
      }
    }
    goto LABEL_2;
  }
LABEL_16:
  v7[3].m128i_i64[0] = v2;
  *(_QWORD *)(v5 + 56) = v2;
  return v2;
}
