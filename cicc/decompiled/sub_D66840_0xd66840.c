// Function: sub_D66840
// Address: 0xd66840
//
__m128i *__fastcall sub_D66840(__m128i *a1, _BYTE *a2)
{
  __m128i *result; // rax
  __m128i v3; // xmm0
  __m128i v4; // xmm1
  __m128i v5; // xmm2
  __m128i v6; // xmm3
  __m128i v7; // xmm4
  __m128i v8; // xmm5
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __m128i v11; // xmm3
  __m128i v12; // xmm4
  __m128i v13; // xmm5
  __m128i v14; // xmm6
  __m128i v15; // xmm6
  __m128i v16; // xmm7
  __m128i v17; // xmm0
  __m128i v18; // [rsp+0h] [rbp-40h] BYREF
  __m128i v19; // [rsp+10h] [rbp-30h] BYREF
  __m128i v20; // [rsp+20h] [rbp-20h] BYREF

  switch ( *a2 )
  {
    case '=':
      sub_D665A0(&v18, (__int64)a2);
      v3 = _mm_loadu_si128(&v18);
      v4 = _mm_loadu_si128(&v19);
      a1[3].m128i_i8[0] = 1;
      v5 = _mm_loadu_si128(&v20);
      *a1 = v3;
      a1[1] = v4;
      a1[2] = v5;
      result = a1;
      break;
    case '>':
      sub_D66630(&v18, (__int64)a2);
      v6 = _mm_loadu_si128(&v18);
      v7 = _mm_loadu_si128(&v19);
      a1[3].m128i_i8[0] = 1;
      v8 = _mm_loadu_si128(&v20);
      *a1 = v6;
      a1[1] = v7;
      a1[2] = v8;
      result = a1;
      break;
    case 'A':
      sub_D66720(&v18, (__int64)a2);
      v9 = _mm_loadu_si128(&v18);
      v10 = _mm_loadu_si128(&v19);
      a1[3].m128i_i8[0] = 1;
      v11 = _mm_loadu_si128(&v20);
      *a1 = v9;
      a1[1] = v10;
      a1[2] = v11;
      result = a1;
      break;
    case 'B':
      sub_D667B0(&v18, (__int64)a2);
      v12 = _mm_loadu_si128(&v18);
      v13 = _mm_loadu_si128(&v19);
      a1[3].m128i_i8[0] = 1;
      v14 = _mm_loadu_si128(&v20);
      *a1 = v12;
      a1[1] = v13;
      a1[2] = v14;
      result = a1;
      break;
    case 'Y':
      sub_D666C0(&v18, (__int64)a2);
      v15 = _mm_loadu_si128(&v18);
      v16 = _mm_loadu_si128(&v19);
      a1[3].m128i_i8[0] = 1;
      v17 = _mm_loadu_si128(&v20);
      *a1 = v15;
      a1[1] = v16;
      a1[2] = v17;
      result = a1;
      break;
    default:
      a1[3].m128i_i8[0] = 0;
      result = a1;
      break;
  }
  return result;
}
