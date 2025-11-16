// Function: sub_2120200
// Address: 0x2120200
//
__int64 __fastcall sub_2120200(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 v6; // rsi
  __int64 v7; // rsi
  unsigned __int8 *v8; // rax
  unsigned int v9; // r13d
  __int64 v10; // rax
  __int64 *v11; // rbx
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r15
  __int64 v14; // r14
  __int128 v15; // rax
  __int128 v16; // rax
  __int64 v17; // r14
  const void **v19; // [rsp+8h] [rbp-78h]
  const void **v20; // [rsp+10h] [rbp-70h]
  unsigned __int8 v21; // [rsp+1Fh] [rbp-61h]
  __int64 v22; // [rsp+20h] [rbp-60h] BYREF
  int v23; // [rsp+28h] [rbp-58h]
  _BYTE v24[8]; // [rsp+30h] [rbp-50h] BYREF
  unsigned __int8 v25; // [rsp+38h] [rbp-48h]
  const void **v26; // [rsp+40h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 72);
  v22 = v6;
  if ( v6 )
    sub_1623A60((__int64)&v22, v6, 2);
  v7 = *(_QWORD *)a1;
  v23 = *(_DWORD *)(a2 + 64);
  v8 = *(unsigned __int8 **)(a2 + 40);
  v21 = *v8;
  v19 = (const void **)*((_QWORD *)v8 + 1);
  sub_1F40D10((__int64)v24, v7, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v8, (__int64)v19);
  v9 = v25;
  v20 = v26;
  v10 = sub_1D309E0(
          *(__int64 **)(a1 + 8),
          *(unsigned __int16 *)(a2 + 24),
          (__int64)&v22,
          v25,
          v26,
          0,
          *(double *)a3.m128i_i64,
          a4,
          *(double *)a5.m128i_i64,
          *(_OWORD *)*(_QWORD *)(a2 + 32));
  v11 = *(__int64 **)(a1 + 8);
  v13 = v12;
  v14 = v10;
  *(_QWORD *)&v15 = sub_1D38E70((__int64)v11, 0, (__int64)&v22, 0, a3, a4, a5);
  *(_QWORD *)&v16 = sub_1D332F0(v11, 154, (__int64)&v22, v21, v19, 0, *(double *)a3.m128i_i64, a4, a5, v14, v13, v15);
  v17 = sub_1D309E0(v11, 157, (__int64)&v22, v9, v20, 0, *(double *)a3.m128i_i64, a4, *(double *)a5.m128i_i64, v16);
  if ( v22 )
    sub_161E7C0((__int64)&v22, v22);
  return v17;
}
