// Function: sub_213D000
// Address: 0x213d000
//
__int64 *__fastcall sub_213D000(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // r13
  __int64 *v12; // rdi
  __int64 v13; // rax
  __int64 *v14; // r15
  __int64 v15; // r12
  unsigned int v16; // edx
  unsigned __int64 v17; // r13
  unsigned __int8 *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r9
  __int128 v21; // rax
  __int64 *v22; // r12
  __int128 v24; // [rsp-10h] [rbp-60h]
  __int64 v25; // [rsp+0h] [rbp-50h]
  __int64 v26; // [rsp+8h] [rbp-48h]
  __int64 v27; // [rsp+10h] [rbp-40h] BYREF
  int v28; // [rsp+18h] [rbp-38h]

  v7 = sub_2138AD0(a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = *(_QWORD *)(a2 + 72);
  v9 = v7;
  v11 = v10;
  v27 = v8;
  if ( v8 )
    sub_1623A60((__int64)&v27, v8, 2);
  v12 = *(__int64 **)(a1 + 8);
  v28 = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v24 + 1) = v11;
  *(_QWORD *)&v24 = v9;
  v13 = sub_1D309E0(
          v12,
          144,
          (__int64)&v27,
          **(unsigned __int8 **)(a2 + 40),
          *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL),
          0,
          a3,
          a4,
          *(double *)a5.m128i_i64,
          v24);
  v14 = *(__int64 **)(a1 + 8);
  v15 = v13;
  v25 = v13;
  v26 = v16;
  v17 = v16 | v11 & 0xFFFFFFFF00000000LL;
  v18 = (unsigned __int8 *)(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 32) + 8LL));
  *(_QWORD *)&v21 = sub_1D2EF30(v14, *v18, *((_QWORD *)v18 + 1), v19, v16, v20);
  v22 = sub_1D332F0(
          v14,
          148,
          (__int64)&v27,
          *(unsigned __int8 *)(*(_QWORD *)(v25 + 40) + 16 * v26),
          *(const void ***)(*(_QWORD *)(v25 + 40) + 16 * v26 + 8),
          0,
          a3,
          a4,
          a5,
          v15,
          v17,
          v21);
  if ( v27 )
    sub_161E7C0((__int64)&v27, v27);
  return v22;
}
