// Function: sub_732B40
// Address: 0x732b40
//
unsigned __int64 __fastcall sub_732B40(const __m128i *a1, __m128i *a2)
{
  __int64 v2; // rcx
  __int8 v3; // dl
  int v4; // eax
  unsigned __int64 result; // rax
  __m128i v6; // xmm5
  __int64 v7; // rax
  _QWORD *i; // rax
  __int64 v9; // rdx

  v2 = a2[1].m128i_i64[0];
  v3 = a2[2].m128i_i8[9];
  *a2 = _mm_loadu_si128(a1);
  a2[1] = _mm_loadu_si128(a1 + 1);
  a2[2] = _mm_loadu_si128(a1 + 2);
  v4 = a2[2].m128i_u8[9];
  a2[3] = _mm_loadu_si128(a1 + 3);
  LODWORD(result) = v3 & 1 | v4 & 0xFFFFFFFE;
  a2[4] = _mm_loadu_si128(a1 + 4);
  v6 = _mm_loadu_si128(a1 + 5);
  a2[2].m128i_i8[9] = result;
  a2[1].m128i_i64[0] = v2;
  a2[5] = v6;
  switch ( a2[2].m128i_i8[8] )
  {
    case 1:
    case 3:
    case 4:
      *(_QWORD *)(a2[4].m128i_i64[1] + 24) = a2;
      result = a2[5].m128i_u64[0];
      if ( result )
        goto LABEL_8;
      return result;
    case 2:
      *(_QWORD *)(*(_QWORD *)a2[4].m128i_i64[1] + 24LL) = a2;
      result = *(_QWORD *)(a2[4].m128i_i64[1] + 8);
      if ( result )
        goto LABEL_8;
      return result;
    case 5:
    case 0xC:
    case 0xE:
      goto LABEL_7;
    case 7:
      result = a2[4].m128i_u64[1];
      *(_QWORD *)(result + 128) = a2;
      return result;
    case 0xB:
      v9 = *(_QWORD *)(a2[5].m128i_i64[0] + 8);
      result = a2[4].m128i_u64[1];
      if ( v9 )
        *(_QWORD *)(v9 + 80) = a2;
      while ( result )
      {
        *(_QWORD *)(result + 24) = a2;
        result = *(_QWORD *)(result + 16);
      }
      a1[4].m128i_i64[1] = 0;
      return result;
    case 0xD:
      *(_QWORD *)(a2[4].m128i_i64[1] + 24) = a2;
      result = *(_QWORD *)a2[5].m128i_i64[0];
      if ( !result )
        return result;
      goto LABEL_8;
    case 0xF:
      result = a2[5].m128i_u64[0];
      *(_QWORD *)result = a2;
      return result;
    case 0x10:
      for ( i = *(_QWORD **)a2[5].m128i_i64[0]; i; i = (_QWORD *)i[3] )
        *(_QWORD *)(*i + 72LL) = a2;
LABEL_7:
      result = a2[4].m128i_u64[1];
LABEL_8:
      *(_QWORD *)(result + 24) = a2;
      break;
    case 0x13:
      v7 = a2[4].m128i_i64[1];
      *(_QWORD *)(*(_QWORD *)(v7 + 8) + 24LL) = a2;
      for ( result = *(_QWORD *)(v7 + 16); result; result = *(_QWORD *)result )
        *(_QWORD *)(*(_QWORD *)(result + 24) + 24LL) = a2;
      break;
    default:
      result = (unsigned int)result;
      break;
  }
  return result;
}
