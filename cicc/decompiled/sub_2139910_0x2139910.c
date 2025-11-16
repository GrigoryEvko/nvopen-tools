// Function: sub_2139910
// Address: 0x2139910
//
__int64 *__fastcall sub_2139910(__int64 a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 *v6; // r12
  unsigned __int64 v7; // rdx
  unsigned __int64 v8; // r13
  __int64 *v9; // rax
  __int64 v10; // rsi
  __int64 *v11; // r14
  __int64 v12; // rdx
  __int64 v13; // r15
  __int64 *v14; // r11
  __int64 v15; // rcx
  const void **v16; // r8
  __int64 v17; // rsi
  __int64 *v18; // r12
  __int128 v20; // [rsp-10h] [rbp-70h]
  __int64 v21; // [rsp+0h] [rbp-60h]
  const void **v22; // [rsp+8h] [rbp-58h]
  __int64 *v23; // [rsp+10h] [rbp-50h]
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF
  int v25; // [rsp+28h] [rbp-38h]

  v6 = sub_2139210(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, a4, a5);
  v8 = v7;
  v9 = sub_2139210(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL), a3, a4, a5);
  v10 = *(_QWORD *)(a2 + 72);
  v11 = v9;
  v13 = v12;
  v14 = *(__int64 **)(a1 + 8);
  v15 = *(unsigned __int8 *)(v6[5] + 16LL * (unsigned int)v8);
  v16 = *(const void ***)(v6[5] + 16LL * (unsigned int)v8 + 8);
  v24 = v10;
  if ( v10 )
  {
    v21 = v15;
    v22 = v16;
    v23 = v14;
    sub_1623A60((__int64)&v24, v10, 2);
    v15 = v21;
    v16 = v22;
    v14 = v23;
  }
  *((_QWORD *)&v20 + 1) = v13;
  *(_QWORD *)&v20 = v11;
  v17 = *(unsigned __int16 *)(a2 + 24);
  v25 = *(_DWORD *)(a2 + 64);
  v18 = sub_1D332F0(v14, v17, (__int64)&v24, v15, v16, 0, *(double *)a3.m128i_i64, a4, a5, (__int64)v6, v8, v20);
  if ( v24 )
    sub_161E7C0((__int64)&v24, v24);
  return v18;
}
