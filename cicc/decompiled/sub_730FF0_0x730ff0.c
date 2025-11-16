// Function: sub_730FF0
// Address: 0x730ff0
//
void *__fastcall sub_730FF0(const __m128i *a1)
{
  int v1; // r12d
  void *result; // rax
  int v3; // edx
  __m128i v4; // xmm4
  __m128i *v5; // rcx
  int v6; // edx
  __m128i v7; // xmm6
  const __m128i *v8; // rdx
  __m128i *v9; // rdx
  int v10; // ecx
  __m128i v11; // xmm4
  const __m128i *v12; // rcx
  __m128i *v13; // rcx
  int v14; // edx
  __m128i v15; // xmm4
  const __m128i *v16; // rdx

  v1 = a1[1].m128i_u8[8];
  result = sub_726700(v1);
  switch ( v1 )
  {
    case 7:
      v9 = (__m128i *)*((_QWORD *)result + 7);
      *(__m128i *)result = _mm_loadu_si128(a1);
      *((__m128i *)result + 1) = _mm_loadu_si128(a1 + 1);
      v10 = *((_DWORD *)result + 6);
      *((__m128i *)result + 2) = _mm_loadu_si128(a1 + 2);
      *((__m128i *)result + 3) = _mm_loadu_si128(a1 + 3);
      v11 = _mm_loadu_si128(a1 + 4);
      *((_QWORD *)result + 2) = 0;
      *((_DWORD *)result + 6) = v10 & 0xFFF3FBFF | 0x80000;
      *((_QWORD *)result + 10) = 0;
      *((__m128i *)result + 4) = v11;
      v12 = (const __m128i *)a1[3].m128i_i64[1];
      *v9 = _mm_loadu_si128(v12);
      v9[1] = _mm_loadu_si128(v12 + 1);
      v9[2] = _mm_loadu_si128(v12 + 2);
      v9[3].m128i_i64[0] = v12[3].m128i_i64[0];
      *((_QWORD *)result + 7) = v9;
      break;
    case 8:
      v5 = (__m128i *)*((_QWORD *)result + 7);
      *(__m128i *)result = _mm_loadu_si128(a1);
      *((__m128i *)result + 1) = _mm_loadu_si128(a1 + 1);
      v6 = *((_DWORD *)result + 6);
      *((__m128i *)result + 2) = _mm_loadu_si128(a1 + 2);
      *((__m128i *)result + 3) = _mm_loadu_si128(a1 + 3);
      v7 = _mm_loadu_si128(a1 + 4);
      *((_QWORD *)result + 2) = 0;
      *((_DWORD *)result + 6) = v6 & 0xFFF3FBFF | 0x80000;
      *((_QWORD *)result + 10) = 0;
      *((__m128i *)result + 4) = v7;
      v8 = (const __m128i *)a1[3].m128i_i64[1];
      if ( v8 )
      {
        *v5 = _mm_loadu_si128(v8);
        v5[1].m128i_i64[0] = v8[1].m128i_i64[0];
      }
      else
      {
        v5 = 0;
      }
      *((_QWORD *)result + 7) = v5;
      break;
    case 9:
      v13 = (__m128i *)*((_QWORD *)result + 7);
      *(__m128i *)result = _mm_loadu_si128(a1);
      *((__m128i *)result + 1) = _mm_loadu_si128(a1 + 1);
      v14 = *((_DWORD *)result + 6);
      *((__m128i *)result + 2) = _mm_loadu_si128(a1 + 2);
      *((__m128i *)result + 3) = _mm_loadu_si128(a1 + 3);
      v15 = _mm_loadu_si128(a1 + 4);
      *((_QWORD *)result + 2) = 0;
      *((_DWORD *)result + 6) = v14 & 0xFFF3FBFF | 0x80000;
      *((_QWORD *)result + 10) = 0;
      *((__m128i *)result + 4) = v15;
      v16 = (const __m128i *)a1[3].m128i_i64[1];
      *v13 = _mm_loadu_si128(v16);
      v13[1] = _mm_loadu_si128(v16 + 1);
      *((_QWORD *)result + 7) = v13;
      break;
    default:
      *(__m128i *)result = _mm_loadu_si128(a1);
      *((__m128i *)result + 1) = _mm_loadu_si128(a1 + 1);
      v3 = *((_DWORD *)result + 6);
      *((__m128i *)result + 2) = _mm_loadu_si128(a1 + 2);
      *((__m128i *)result + 3) = _mm_loadu_si128(a1 + 3);
      v4 = _mm_loadu_si128(a1 + 4);
      *((_QWORD *)result + 2) = 0;
      *((_DWORD *)result + 6) = v3 & 0xFFF3FBFF | 0x80000;
      *((_QWORD *)result + 10) = 0;
      *((__m128i *)result + 4) = v4;
      break;
  }
  return result;
}
