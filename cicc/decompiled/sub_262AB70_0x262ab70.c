// Function: sub_262AB70
// Address: 0x262ab70
//
__m128i *__fastcall sub_262AB70(__int64 *a1, __int64 a2)
{
  _BYTE *v4; // rax
  __m128i *v5; // rsi
  _BYTE *v6; // rdi
  __int64 v7; // r14
  __m128i *result; // rax
  __int64 v9; // r12
  __int64 v10; // r14
  __int64 v11; // r12
  unsigned __int64 *v12; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v13; // [rsp+18h] [rbp-48h]
  __m128i v14; // [rsp+20h] [rbp-40h] BYREF

  a1[1] = (__int64)(a1 + 3);
  *a1 = a2;
  a1[2] = 0x400000000LL;
  a1[7] = (__int64)(a1 + 9);
  a1[8] = 0x400000000LL;
  a1[13] = 0;
  a1[14] = 0;
  a1[15] = 0;
  a1[16] = 0;
  a1[17] = 0;
  a1[18] = 0;
  v13 = (unsigned __int64 *)(a1 + 13);
  v12 = (unsigned __int64 *)(a1 + 16);
  v4 = sub_BAA9B0(a2, (__int64)(a1 + 1), 0);
  if ( v4 )
    sub_B30290((__int64)v4);
  v5 = (__m128i *)(a1 + 7);
  v6 = sub_BAA9B0(a2, (__int64)(a1 + 7), 1);
  if ( v6 )
    sub_B30290((__int64)v6);
  v7 = *(_QWORD *)(a2 + 48);
  result = &v14;
  v9 = a2 + 40;
  if ( v7 != a2 + 40 )
  {
    while ( 1 )
    {
      if ( !v7 )
        BUG();
      result = (__m128i *)sub_BD3990(*(unsigned __int8 **)(v7 - 80), (__int64)v5);
      if ( result->m128i_i8[0] )
        goto LABEL_10;
      v14.m128i_i64[0] = v7 - 48;
      v5 = (__m128i *)a1[14];
      v14.m128i_i64[1] = (__int64)result;
      if ( v5 == (__m128i *)a1[15] )
      {
        result = (__m128i *)sub_262A870(v13, v5, &v14);
        v7 = *(_QWORD *)(v7 + 8);
        if ( v9 == v7 )
          break;
      }
      else
      {
        if ( v5 )
        {
          *v5 = _mm_loadu_si128(&v14);
          v5 = (__m128i *)a1[14];
        }
        a1[14] = (__int64)++v5;
LABEL_10:
        v7 = *(_QWORD *)(v7 + 8);
        if ( v9 == v7 )
          break;
      }
    }
  }
  v10 = *(_QWORD *)(a2 + 64);
  v11 = a2 + 56;
  if ( a2 + 56 != v10 )
  {
    while ( 1 )
    {
      if ( !v10 )
        BUG();
      result = (__m128i *)sub_BD3990(*(unsigned __int8 **)(v10 - 88), (__int64)v5);
      if ( result->m128i_i8[0] )
        goto LABEL_20;
      v14.m128i_i64[0] = v10 - 56;
      v5 = (__m128i *)a1[17];
      v14.m128i_i64[1] = (__int64)result;
      if ( v5 == (__m128i *)a1[18] )
      {
        result = (__m128i *)sub_262A9F0(v12, v5, &v14);
        v10 = *(_QWORD *)(v10 + 8);
        if ( v11 == v10 )
          return result;
      }
      else
      {
        if ( v5 )
        {
          *v5 = _mm_loadu_si128(&v14);
          v5 = (__m128i *)a1[17];
        }
        a1[17] = (__int64)++v5;
LABEL_20:
        v10 = *(_QWORD *)(v10 + 8);
        if ( v11 == v10 )
          return result;
      }
    }
  }
  return result;
}
