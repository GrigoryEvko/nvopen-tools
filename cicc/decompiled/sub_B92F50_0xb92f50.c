// Function: sub_B92F50
// Address: 0xb92f50
//
__int64 __fastcall sub_B92F50(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 i; // rdx
  unsigned int v10; // eax
  unsigned int v11; // ebx
  __int8 v12; // al
  __int64 v13; // rdi
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // rdx
  __int64 j; // rdx

  result = (unsigned __int32)a1[1].m128i_i32[2] >> 1;
  if ( !(_DWORD)result )
    return result;
  if ( (_BYTE)a2 )
    return sub_B92A60(a1, a2, a3, a4, a5);
  ++a1[1].m128i_i64[0];
  if ( (a1[1].m128i_i8[8] & 1) != 0 )
  {
    result = (__int64)a1[2].m128i_i64;
    v8 = 96;
    goto LABEL_9;
  }
  v7 = a1[2].m128i_u32[2];
  if ( 4 * (int)result >= (unsigned int)v7 || (unsigned int)v7 <= 0x40 )
  {
    result = a1[2].m128i_i64[0];
    v8 = 24 * v7;
LABEL_9:
    for ( i = result + v8; i != result; result += 24 )
      *(_QWORD *)result = -4096;
    a1[1].m128i_i64[1] &= 1uLL;
    return result;
  }
  v10 = result - 1;
  if ( v10 )
  {
    _BitScanReverse(&v10, v10);
    v11 = 1 << (33 - (v10 ^ 0x1F));
    if ( v11 - 5 > 0x3A )
    {
      if ( (_DWORD)v7 == v11 )
        return (__int64)sub_B92A00((__int64)a1[1].m128i_i64);
      sub_C7D6A0(a1[2].m128i_i64[0], 24 * v7, 8);
      v12 = a1[1].m128i_i8[8] | 1;
      a1[1].m128i_i8[8] = v12;
      if ( v11 <= 4 )
        goto LABEL_18;
      v13 = 24LL * v11;
    }
    else
    {
      v11 = 64;
      sub_C7D6A0(a1[2].m128i_i64[0], 24 * v7, 8);
      v12 = a1[1].m128i_i8[8];
      v13 = 1536;
    }
    a1[1].m128i_i8[8] = v12 & 0xFE;
    v14 = sub_C7D670(v13, 8);
    a1[2].m128i_i32[2] = v11;
    a1[2].m128i_i64[0] = v14;
  }
  else
  {
    sub_C7D6A0(a1[2].m128i_i64[0], 24 * v7, 8);
    a1[1].m128i_i8[8] |= 1u;
  }
LABEL_18:
  v15 = (a1[1].m128i_i64[1] & 1) == 0;
  a1[1].m128i_i64[1] &= 1uLL;
  if ( v15 )
  {
    result = a1[2].m128i_i64[0];
    v16 = 24LL * a1[2].m128i_u32[2];
  }
  else
  {
    result = (__int64)a1[2].m128i_i64;
    v16 = 96;
  }
  for ( j = result + v16; j != result; result += 24 )
  {
    if ( result )
      *(_QWORD *)result = -4096;
  }
  return result;
}
