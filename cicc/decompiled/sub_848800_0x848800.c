// Function: sub_848800
// Address: 0x848800
//
__int64 __fastcall sub_848800(__int64 a1, __int64 a2, _BYTE *a3, unsigned int a4, __m128i *a5)
{
  unsigned int v5; // r9d
  __int64 v7; // rdi
  char v8; // al

  if ( !*(_BYTE *)(a1 + 8) )
  {
    v7 = *(_QWORD *)(a1 + 24);
    *a5 = _mm_loadu_si128((const __m128i *)(v7 + 8));
    a5[1] = _mm_loadu_si128((const __m128i *)(v7 + 24));
    a5[2] = _mm_loadu_si128((const __m128i *)(v7 + 40));
    a5[3] = _mm_loadu_si128((const __m128i *)(v7 + 56));
    a5[4] = _mm_loadu_si128((const __m128i *)(v7 + 72));
    a5[5] = _mm_loadu_si128((const __m128i *)(v7 + 88));
    a5[6] = _mm_loadu_si128((const __m128i *)(v7 + 104));
    a5[7] = _mm_loadu_si128((const __m128i *)(v7 + 120));
    a5[8] = _mm_loadu_si128((const __m128i *)(v7 + 136));
    v8 = *(_BYTE *)(v7 + 24);
    if ( v8 == 2 )
    {
      a5[9] = _mm_loadu_si128((const __m128i *)(v7 + 152));
      a5[10] = _mm_loadu_si128((const __m128i *)(v7 + 168));
      a5[11] = _mm_loadu_si128((const __m128i *)(v7 + 184));
      a5[12] = _mm_loadu_si128((const __m128i *)(v7 + 200));
      a5[13] = _mm_loadu_si128((const __m128i *)(v7 + 216));
      a5[14] = _mm_loadu_si128((const __m128i *)(v7 + 232));
      a5[15] = _mm_loadu_si128((const __m128i *)(v7 + 248));
      a5[16] = _mm_loadu_si128((const __m128i *)(v7 + 264));
      a5[17] = _mm_loadu_si128((const __m128i *)(v7 + 280));
      a5[18] = _mm_loadu_si128((const __m128i *)(v7 + 296));
      a5[19] = _mm_loadu_si128((const __m128i *)(v7 + 312));
      a5[20] = _mm_loadu_si128((const __m128i *)(v7 + 328));
      a5[21] = _mm_loadu_si128((const __m128i *)(v7 + 344));
    }
    else if ( v8 == 5 || v8 == 1 )
    {
      a5[9].m128i_i64[0] = *(_QWORD *)(v7 + 152);
      return sub_843D70(a5, a2, a3, a4);
    }
    return sub_843D70(a5, a2, a3, a4);
  }
  v5 = 0;
  if ( a2 )
    v5 = ((*(_BYTE *)(a2 + 34) & 0x20) != 0) << 6;
  return sub_839D30(
           a1,
           *(__m128i **)(a2 + 8),
           0,
           ((unsigned int)qword_4F077B4 | dword_4F077BC | dword_4D04964) != 0,
           ((unsigned int)qword_4F077B4 | dword_4F077BC | dword_4D04964) == 0,
           v5,
           1,
           0,
           0,
           (__int64)a5,
           0,
           0);
}
