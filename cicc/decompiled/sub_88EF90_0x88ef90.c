// Function: sub_88EF90
// Address: 0x88ef90
//
const __m128i *__fastcall sub_88EF90(__int64 a1)
{
  __m128i *v2; // rbx
  char v3; // al
  const __m128i *v4; // rax
  const __m128i *result; // rax
  __int64 v6; // rdi

  v2 = *(__m128i **)(a1 + 88);
  v2[7].m128i_i8[9] = ((*(_BYTE *)(a1 + 56) & 0x10) != 0) | v2[7].m128i_i8[9] & 0xFE;
  v3 = *(_BYTE *)(*(_QWORD *)(a1 + 8) + 80LL);
  switch ( v3 )
  {
    case 3:
      v2[7].m128i_i8[8] = 1;
      v2[8].m128i_i64[0] = *(_QWORD *)(a1 + 64);
      v2[8].m128i_i64[1] = *(_QWORD *)(a1 + 80);
      v4 = (const __m128i *)sub_72A270(*(_QWORD *)(a1 + 64), 6);
      *v2 = _mm_loadu_si128(v4);
      v2[1] = _mm_loadu_si128(v4 + 1);
      v2[2] = _mm_loadu_si128(v4 + 2);
      v2[3] = _mm_loadu_si128(v4 + 3);
      v2[4] = _mm_loadu_si128(v4 + 4);
      v2[5] = _mm_loadu_si128(v4 + 5);
      v2[6] = _mm_loadu_si128(v4 + 6);
      result = *(const __m128i **)(a1 + 64);
      if ( (result[10].m128i_i8[1] & 4) != 0 )
        v2[7].m128i_i8[9] |= 2u;
      break;
    case 19:
      v2[7].m128i_i8[8] = 3;
      v6 = *(_QWORD *)(*(_QWORD *)(a1 + 64) + 104LL);
      v2[8].m128i_i64[0] = v6;
      v2[8].m128i_i64[1] = *(_QWORD *)(a1 + 80);
      result = (const __m128i *)sub_72A270(v6, 59);
      *v2 = _mm_loadu_si128(result);
      v2[1] = _mm_loadu_si128(result + 1);
      v2[2] = _mm_loadu_si128(result + 2);
      v2[3] = _mm_loadu_si128(result + 3);
      v2[4] = _mm_loadu_si128(result + 4);
      v2[5] = _mm_loadu_si128(result + 5);
      v2[6] = _mm_loadu_si128(result + 6);
      break;
    case 2:
      v2[7].m128i_i8[8] = 2;
      v2[8].m128i_i64[0] = *(_QWORD *)(a1 + 64);
      v2[8].m128i_i64[1] = *(_QWORD *)(a1 + 80);
      result = (const __m128i *)sub_72A270(*(_QWORD *)(a1 + 64), 2);
      *v2 = _mm_loadu_si128(result);
      v2[1] = _mm_loadu_si128(result + 1);
      v2[2] = _mm_loadu_si128(result + 2);
      v2[3] = _mm_loadu_si128(result + 3);
      v2[4] = _mm_loadu_si128(result + 4);
      v2[5] = _mm_loadu_si128(result + 5);
      v2[6] = _mm_loadu_si128(result + 6);
      break;
    default:
      sub_721090();
  }
  return result;
}
