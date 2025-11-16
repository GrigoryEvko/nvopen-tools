// Function: sub_2139A10
// Address: 0x2139a10
//
__int64 *__fastcall sub_2139A10(__int64 *a1, __int64 a2, __m128i a3, double a4, __m128i a5)
{
  __int64 *v6; // r12
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // r14
  __int64 v11; // r15
  __int64 *v12; // r9
  __int64 v13; // rsi
  __int64 *v14; // r10
  const void **v15; // r8
  __int64 v16; // rcx
  __int64 *v17; // r12
  __int64 *v19; // rax
  unsigned int v20; // edx
  __int128 v21; // [rsp-10h] [rbp-90h]
  __int64 v22; // [rsp+0h] [rbp-80h]
  const void **v23; // [rsp+8h] [rbp-78h]
  __int64 *v24; // [rsp+10h] [rbp-70h]
  __int64 v25; // [rsp+20h] [rbp-60h]
  unsigned __int64 v26; // [rsp+28h] [rbp-58h]
  __int64 v27; // [rsp+30h] [rbp-50h] BYREF
  int v28; // [rsp+38h] [rbp-48h]

  v6 = sub_2139210((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a3, a4, a5);
  v7 = *(_QWORD *)(a2 + 32);
  v9 = v8;
  v10 = *(_QWORD *)(v7 + 40);
  v11 = *(_QWORD *)(v7 + 48);
  v26 = v10;
  v25 = *(unsigned int *)(v7 + 48);
  sub_1F40D10(
    (__int64)&v27,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    *(unsigned __int8 *)(*(_QWORD *)(v26 + 40) + 16 * v25),
    *(_QWORD *)(*(_QWORD *)(v26 + 40) + 16 * v25 + 8));
  v12 = a1;
  if ( (_BYTE)v27 == 1 )
  {
    v19 = sub_2139210((__int64)a1, v10, v11, a3, a4, a5);
    v12 = a1;
    v26 = (unsigned __int64)v19;
    v25 = v20;
  }
  v13 = *(_QWORD *)(a2 + 72);
  v14 = (__int64 *)v12[1];
  v15 = *(const void ***)(v6[5] + 16LL * (unsigned int)v9 + 8);
  v16 = *(unsigned __int8 *)(v6[5] + 16LL * (unsigned int)v9);
  v27 = v13;
  if ( v13 )
  {
    v22 = v16;
    v23 = v15;
    v24 = v14;
    sub_1623A60((__int64)&v27, v13, 2);
    v16 = v22;
    v15 = v23;
    v14 = v24;
  }
  v28 = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v21 + 1) = v11 & 0xFFFFFFFF00000000LL | v25;
  *(_QWORD *)&v21 = v26;
  v17 = sub_1D332F0(v14, 124, (__int64)&v27, v16, v15, 0, *(double *)a3.m128i_i64, a4, a5, (__int64)v6, v9, v21);
  if ( v27 )
    sub_161E7C0((__int64)&v27, v27);
  return v17;
}
