// Function: sub_6E6610
// Address: 0x6e6610
//
void __fastcall sub_6E6610(_QWORD *a1, __m128i *a2, int a3)
{
  __int64 v4; // r13
  char v5; // al
  __int64 v6; // rax

  sub_6E65B0((__int64)a1);
  v4 = a1[3];
  *a2 = _mm_loadu_si128((const __m128i *)(v4 + 8));
  a2[1] = _mm_loadu_si128((const __m128i *)(v4 + 24));
  a2[2] = _mm_loadu_si128((const __m128i *)(v4 + 40));
  a2[3] = _mm_loadu_si128((const __m128i *)(v4 + 56));
  a2[4] = _mm_loadu_si128((const __m128i *)(v4 + 72));
  a2[5] = _mm_loadu_si128((const __m128i *)(v4 + 88));
  a2[6] = _mm_loadu_si128((const __m128i *)(v4 + 104));
  a2[7] = _mm_loadu_si128((const __m128i *)(v4 + 120));
  a2[8] = _mm_loadu_si128((const __m128i *)(v4 + 136));
  v5 = *(_BYTE *)(v4 + 24);
  if ( v5 == 2 )
  {
    a2[9] = _mm_loadu_si128((const __m128i *)(v4 + 152));
    a2[10] = _mm_loadu_si128((const __m128i *)(v4 + 168));
    a2[11] = _mm_loadu_si128((const __m128i *)(v4 + 184));
    a2[12] = _mm_loadu_si128((const __m128i *)(v4 + 200));
    a2[13] = _mm_loadu_si128((const __m128i *)(v4 + 216));
    a2[14] = _mm_loadu_si128((const __m128i *)(v4 + 232));
    a2[15] = _mm_loadu_si128((const __m128i *)(v4 + 248));
    a2[16] = _mm_loadu_si128((const __m128i *)(v4 + 264));
    a2[17] = _mm_loadu_si128((const __m128i *)(v4 + 280));
    a2[18] = _mm_loadu_si128((const __m128i *)(v4 + 296));
    a2[19] = _mm_loadu_si128((const __m128i *)(v4 + 312));
    a2[20] = _mm_loadu_si128((const __m128i *)(v4 + 328));
    a2[21] = _mm_loadu_si128((const __m128i *)(v4 + 344));
    goto LABEL_4;
  }
  if ( v5 != 5 && v5 != 1 )
  {
LABEL_4:
    sub_832D70(a1, a2);
    if ( !a3 )
      return;
LABEL_7:
    *(_QWORD *)(v4 + 96) = 0;
    sub_6E2DD0(v4 + 8, 0);
    v6 = sub_72C930(v4 + 8);
    *(_DWORD *)(v4 + 25) &= 0xFFE9C500;
    *(_QWORD *)(v4 + 8) = v6;
    sub_6E1990(a1);
    return;
  }
  a2[9].m128i_i64[0] = *(_QWORD *)(v4 + 152);
  sub_832D70(a1, a2);
  if ( a3 )
    goto LABEL_7;
}
