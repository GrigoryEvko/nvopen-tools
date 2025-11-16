// Function: sub_2176390
// Address: 0x2176390
//
__int64 *__fastcall sub_2176390(__int64 a1, __m128i a2, double a3, __m128i a4, __int64 a5, __int64 *a6)
{
  __int64 v7; // rsi
  __int64 v8; // r10
  unsigned __int8 *v9; // rax
  const void **v10; // r15
  unsigned int v11; // r14d
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 *v16; // r14
  __int64 v18; // [rsp+8h] [rbp-48h]
  __int64 v19; // [rsp+10h] [rbp-40h] BYREF
  int v20; // [rsp+18h] [rbp-38h]

  v7 = *(_QWORD *)(a1 + 72);
  v19 = v7;
  if ( v7 )
    sub_1623A60((__int64)&v19, v7, 2);
  v8 = *(_QWORD *)(a1 + 32);
  v20 = *(_DWORD *)(a1 + 64);
  v9 = *(unsigned __int8 **)(a1 + 40);
  v18 = v8;
  v10 = (const void **)*((_QWORD *)v9 + 1);
  v11 = *v9;
  v12 = sub_1D38BB0((__int64)a6, 0, (__int64)&v19, *v9, v10, 0, a2, a3, a4, 0);
  v14 = sub_1D332F0(
          a6,
          53,
          (__int64)&v19,
          v11,
          v10,
          0,
          *(double *)a2.m128i_i64,
          a3,
          a4,
          v12,
          v13,
          *(_OWORD *)(v18 + 80));
  v16 = sub_1D2B8F0(
          a6,
          224,
          (__int64)&v19,
          *(unsigned __int8 *)(a1 + 88),
          *(_QWORD *)(a1 + 96),
          *(_QWORD *)(a1 + 104),
          *(_OWORD *)*(_QWORD *)(a1 + 32),
          *(_OWORD *)(*(_QWORD *)(a1 + 32) + 40LL),
          (__int64)v14,
          v15);
  if ( v19 )
    sub_161E7C0((__int64)&v19, v19);
  return v16;
}
