// Function: sub_18FCF20
// Address: 0x18fcf20
//
__m128i *__fastcall sub_18FCF20(__m128i *a1, __int64 a2)
{
  __m128i *result; // rax
  __int64 v3; // rax
  __m128i v4; // xmm0
  __m128i v5; // xmm1
  __int64 v6; // rax
  __m128i v7; // xmm2
  __m128i v8; // xmm3
  __int64 v9; // rax
  __m128i v10; // xmm6
  __m128i v11; // xmm7
  __int64 v12; // rax
  __m128i v13; // xmm4
  __m128i v14; // xmm5
  __m128i v15; // [rsp+0h] [rbp-40h] BYREF
  __m128i v16; // [rsp+10h] [rbp-30h] BYREF
  __int64 v17; // [rsp+20h] [rbp-20h]

  switch ( *(_BYTE *)(a2 + 16) )
  {
    case '6':
      sub_141EB40(&v15, (__int64 *)a2);
      goto LABEL_4;
    case '7':
      sub_141EDF0(&v15, a2);
      v6 = v17;
      v7 = _mm_loadu_si128(&v15);
      a1[2].m128i_i8[8] = 1;
      v8 = _mm_loadu_si128(&v16);
      a1[2].m128i_i64[0] = v6;
      *a1 = v7;
      a1[1] = v8;
      result = a1;
      break;
    case ':':
      sub_141F110(&v15, a2);
      v9 = v17;
      v10 = _mm_loadu_si128(&v15);
      a1[2].m128i_i8[8] = 1;
      v11 = _mm_loadu_si128(&v16);
      a1[2].m128i_i64[0] = v9;
      *a1 = v10;
      a1[1] = v11;
      result = a1;
      break;
    case ';':
      sub_141F3C0(&v15, a2);
LABEL_4:
      v3 = v17;
      v4 = _mm_loadu_si128(&v15);
      a1[2].m128i_i8[8] = 1;
      v5 = _mm_loadu_si128(&v16);
      a1[2].m128i_i64[0] = v3;
      *a1 = v4;
      a1[1] = v5;
      result = a1;
      break;
    case 'R':
      sub_141F0A0(&v15, a2);
      v12 = v17;
      v13 = _mm_loadu_si128(&v15);
      a1[2].m128i_i8[8] = 1;
      v14 = _mm_loadu_si128(&v16);
      a1[2].m128i_i64[0] = v12;
      *a1 = v13;
      a1[1] = v14;
      result = a1;
      break;
    default:
      a1[2].m128i_i8[8] = 0;
      result = a1;
      break;
  }
  return result;
}
