// Function: sub_13575E0
// Address: 0x13575e0
//
__int64 __fastcall sub_13575E0(_QWORD *a1, __int64 a2, const __m128i *a3, __int64 a4)
{
  int v4; // eax
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  __int64 v7; // rdx
  __int64 result; // rax
  __int64 v9; // r8
  __m128i v10; // [rsp+0h] [rbp-30h] BYREF
  __m128i v11; // [rsp+10h] [rbp-20h]
  __int64 v12; // [rsp+20h] [rbp-10h]

  v4 = *(unsigned __int8 *)(a2 + 16);
  if ( a3[2].m128i_i8[8] )
  {
    v5 = _mm_loadu_si128(a3);
    v6 = _mm_loadu_si128(a3 + 1);
    v7 = a3[2].m128i_i64[0];
    v10 = v5;
    v12 = v7;
    v11 = v6;
    goto LABEL_3;
  }
  if ( (unsigned __int8)v4 > 0x17u )
  {
    v9 = a2 | 4;
    if ( (_BYTE)v4 == 78 )
      goto LABEL_9;
    if ( (_BYTE)v4 == 29 )
    {
      v9 = a2 & 0xFFFFFFFFFFFFFFFBLL;
LABEL_9:
      if ( (v9 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        return sub_134CC90((__int64)a1, v9) & 7;
    }
  }
  v10.m128i_i64[0] = 0;
  v10.m128i_i64[1] = -1;
  v11 = 0u;
  v12 = 0;
LABEL_3:
  switch ( v4 )
  {
    case 29:
      result = sub_134F0E0(a1, a2 & 0xFFFFFFFFFFFFFFFBLL, (__int64)&v10);
      break;
    case 33:
      result = sub_134D290((__int64)a1, a2, &v10);
      break;
    case 54:
      result = sub_134D040((__int64)a1, a2, &v10, a4);
      break;
    case 55:
      result = sub_134D0E0((__int64)a1, a2, &v10, a4);
      break;
    case 57:
      result = sub_134D190((__int64)a1, a2, &v10);
      break;
    case 58:
      result = sub_134D2D0((__int64)a1, a2, &v10);
      break;
    case 59:
      result = sub_134D360((__int64)a1, a2, &v10);
      break;
    case 74:
      result = sub_134D250((__int64)a1, a2, &v10);
      break;
    case 78:
      result = sub_134F0E0(a1, a2 | 4, (__int64)&v10);
      break;
    case 82:
      result = sub_134D1D0((__int64)a1, a2, &v10);
      break;
    default:
      result = 4;
      break;
  }
  return result;
}
