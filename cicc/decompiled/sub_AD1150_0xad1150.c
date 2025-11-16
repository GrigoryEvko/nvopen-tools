// Function: sub_AD1150
// Address: 0xad1150
//
__int64 __fastcall sub_AD1150(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  unsigned int v7; // eax
  __m128i v8; // xmm0
  __int64 v9; // rdx
  __int64 v10; // r9
  int v11; // r8d
  __int64 v12; // rax
  __int64 *v13; // rsi
  __int64 v14; // rax
  int v15; // r11d
  _QWORD *v16; // rcx
  __int64 v17; // r11
  _QWORD *v18; // rax
  int i; // [rsp+20h] [rbp-80h]
  int v21; // [rsp+24h] [rbp-7Ch]
  __m128i v22; // [rsp+30h] [rbp-70h] BYREF
  __int64 v23; // [rsp+40h] [rbp-60h]
  int v24; // [rsp+50h] [rbp-50h] BYREF
  __m128i v25; // [rsp+58h] [rbp-48h]
  __int64 v26; // [rsp+68h] [rbp-38h]

  v22.m128i_i64[0] = a2;
  v22.m128i_i64[1] = (__int64)a3;
  v23 = a4;
  v24 = sub_AC5F60(a3, (__int64)&a3[a4]);
  v7 = sub_AC7240(v22.m128i_i64, &v24);
  v8 = _mm_loadu_si128(&v22);
  v9 = *(unsigned int *)(a1 + 24);
  v10 = *(_QWORD *)(a1 + 8);
  v24 = v7;
  v26 = v23;
  v25 = v8;
  if ( !(_DWORD)v9 )
    return sub_AD0DB0(a1, a2, a3, a4, (__int64)&v24);
  v11 = v9 - 1;
  v12 = ((_DWORD)v9 - 1) & v7;
  v13 = (__int64 *)(v10 + 8 * v12);
  v21 = v12;
  v14 = *v13;
  if ( *v13 == -4096 )
    return sub_AD0DB0(a1, a2, a3, a4, (__int64)&v24);
  for ( i = 1; ; ++i )
  {
    if ( v14 == -8192 )
      goto LABEL_6;
    if ( v25.m128i_i64[0] != *(_QWORD *)(v14 + 8) )
      goto LABEL_6;
    v15 = *(_DWORD *)(v14 + 4) & 0x7FFFFFF;
    if ( v23 != v15 )
      goto LABEL_6;
    if ( !v15 )
      break;
    v16 = (_QWORD *)v25.m128i_i64[1];
    v17 = v25.m128i_i64[1] + 8 + 8LL * (unsigned int)(v15 - 1);
    v18 = (_QWORD *)(-32 * v23 + v14);
    while ( *v16 == *v18 )
    {
      ++v16;
      v18 += 4;
      if ( v16 == (_QWORD *)v17 )
        goto LABEL_15;
    }
LABEL_6:
    v13 = (__int64 *)(v10 + 8LL * (v11 & (unsigned int)(v21 + i)));
    v21 = v11 & (v21 + i);
    v14 = *v13;
    if ( *v13 == -4096 )
      return sub_AD0DB0(a1, a2, a3, a4, (__int64)&v24);
  }
LABEL_15:
  if ( v13 == (__int64 *)(v10 + 8 * v9) )
    return sub_AD0DB0(a1, a2, a3, a4, (__int64)&v24);
  return *v13;
}
