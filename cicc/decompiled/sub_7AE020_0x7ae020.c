// Function: sub_7AE020
// Address: 0x7ae020
//
void __fastcall sub_7AE020(__int64 a1, __m128i *a2)
{
  char v2; // al
  __m128i *v3; // rax
  __m128i *v4; // rax
  __m128i *v5; // rax
  __int64 v6; // rax

  *a2 = _mm_loadu_si128((const __m128i *)a1);
  a2[1] = _mm_loadu_si128((const __m128i *)(a1 + 16));
  a2[2] = _mm_loadu_si128((const __m128i *)(a1 + 32));
  a2[3] = _mm_loadu_si128((const __m128i *)(a1 + 48));
  a2[4] = _mm_loadu_si128((const __m128i *)(a1 + 64));
  a2[5] = _mm_loadu_si128((const __m128i *)(a1 + 80));
  a2[6] = _mm_loadu_si128((const __m128i *)(a1 + 96));
  v2 = *(_BYTE *)(a1 + 26);
  switch ( v2 )
  {
    case 2:
      v3 = (__m128i *)sub_7ADFF0();
      a2[3].m128i_i64[0] = (__int64)v3;
      sub_72A510(*(const __m128i **)(a1 + 48), v3);
      a2->m128i_i64[0] = 0;
      break;
    case 8:
      v4 = (__m128i *)sub_7ADFF0();
      a2[3].m128i_i64[0] = (__int64)v4;
      sub_72A510(*(const __m128i **)(a1 + 48), v4);
      v5 = (__m128i *)sub_7ADFF0();
      a2[3].m128i_i64[1] = (__int64)v5;
      sub_72A510(*(const __m128i **)(a1 + 56), v5);
      a2->m128i_i64[0] = 0;
      break;
    case 3:
      v6 = sub_853ED0(*(_QWORD *)(a1 + 48));
      a2->m128i_i64[0] = 0;
      a2[3].m128i_i64[0] = v6;
      break;
    default:
      a2->m128i_i64[0] = 0;
      break;
  }
}
