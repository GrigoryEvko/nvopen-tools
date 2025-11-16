// Function: sub_21E3B00
// Address: 0x21e3b00
//
__m128i *__fastcall sub_21E3B00(__m128i *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, double a6, __m128i a7)
{
  __int64 v7; // r15
  __int64 v10; // rsi
  __int32 v11; // edx
  __m128i v12; // xmm1
  char v14; // al
  __int64 v16; // rsi
  __int64 v17; // [rsp+10h] [rbp-A0h]
  __m128i v18; // [rsp+50h] [rbp-60h] BYREF
  __m128i v19; // [rsp+60h] [rbp-50h] BYREF
  __int64 v20; // [rsp+70h] [rbp-40h] BYREF
  int v21; // [rsp+78h] [rbp-38h]

  v7 = a2 - 448;
  v17 = *(_QWORD *)(a2 - 176);
  v18.m128i_i64[0] = 0;
  v18.m128i_i32[2] = 0;
  v19.m128i_i64[0] = 0;
  v19.m128i_i32[2] = 0;
  if ( sub_21C2A00(a2 - 448, a3, a4, (__int64)&v18) )
  {
    v10 = *(_QWORD *)(a3 + 72);
    v20 = v10;
    if ( v10 )
      sub_1623A60((__int64)&v20, v10, 2);
LABEL_4:
    v21 = *(_DWORD *)(a3 + 64);
    v19.m128i_i64[0] = sub_1D38BB0(v17, 0, (__int64)&v20, 6, 0, 1, a5, a6, a7, 0);
    v19.m128i_i32[2] = v11;
    if ( v20 )
      sub_161E7C0((__int64)&v20, v20);
    goto LABEL_6;
  }
  if ( *(_BYTE *)(*(_QWORD *)(a2 + 16) + 936LL) )
    v14 = sub_21C2BC0(v7, a3, a3, a4, (__int64)&v18, (__int64)&v19, a5, a6, a7);
  else
    v14 = sub_21C2BA0(v7, a3, a3, a4, (__int64)&v18, (__int64)&v19, a5, a6, a7);
  if ( !v14
    && !(*(_BYTE *)(*(_QWORD *)(a2 + 16) + 936LL)
       ? sub_21C2F80(v7, a3, a3, a4, (__int64)&v18, (__int64)&v19, a5, a6, a7)
       : (unsigned __int8)sub_21C2F60(v7, a3, a3, a4, (__int64)&v18, (__int64)&v19, a5, a6, a7)) )
  {
    v16 = *(_QWORD *)(a3 + 72);
    v18.m128i_i64[0] = a3;
    v18.m128i_i32[2] = a4;
    v20 = v16;
    if ( v16 )
      sub_1623A60((__int64)&v20, v16, 2);
    goto LABEL_4;
  }
LABEL_6:
  v12 = _mm_loadu_si128(&v19);
  *a1 = _mm_loadu_si128(&v18);
  a1[1] = v12;
  return a1;
}
