// Function: sub_1F87630
// Address: 0x1f87630
//
__int64 *__fastcall sub_1F87630(__int64 **a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v5; // rax
  __int64 *result; // rax
  __int64 v9; // rax
  __int64 v10; // r12
  __int64 v11; // r13
  __int64 v12; // rsi
  __int64 *v13; // rdi
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rdx
  __int64 v18; // r13
  __int128 v19; // [rsp-88h] [rbp-88h]
  __int128 v20; // [rsp-78h] [rbp-78h]
  unsigned int v21; // [rsp-68h] [rbp-68h]
  const void **v22; // [rsp-60h] [rbp-60h]
  __int64 v23; // [rsp-58h] [rbp-58h]
  __int64 *v24; // [rsp-58h] [rbp-58h]
  unsigned __int64 v25; // [rsp-50h] [rbp-50h]
  __int64 v26; // [rsp-48h] [rbp-48h] BYREF
  int v27; // [rsp-40h] [rbp-40h]

  v5 = *(_QWORD *)(a2 + 48);
  if ( !v5 || *(_QWORD *)(v5 + 32) )
    return 0;
  if ( !sub_1D18C00(**(_QWORD **)(a2 + 32), 1, *(_DWORD *)(*(_QWORD *)(a2 + 32) + 8LL)) )
    return 0;
  v9 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 32LL);
  v10 = *(_QWORD *)(v9 + 40);
  v11 = *(_QWORD *)(v9 + 48);
  if ( !(unsigned __int8)sub_1F70310(v10, v11, 1u) )
    return 0;
  v12 = *(_QWORD *)(a2 + 72);
  v26 = v12;
  if ( v12 )
    sub_1623A60((__int64)&v26, v12, 2);
  v13 = *a1;
  v27 = *(_DWORD *)(a2 + 64);
  v21 = **(unsigned __int8 **)(a2 + 40);
  v22 = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
  v14 = sub_1D309E0(
          v13,
          145,
          (__int64)&v26,
          **(unsigned __int8 **)(a2 + 40),
          v22,
          0,
          a3,
          a4,
          *(double *)a5.m128i_i64,
          *(_OWORD *)*(_QWORD *)(**(_QWORD **)(a2 + 32) + 32LL));
  *((_QWORD *)&v19 + 1) = v11;
  *(_QWORD *)&v19 = v10;
  v25 = v15;
  v23 = v14;
  v16 = sub_1D309E0(*a1, 145, (__int64)&v26, v21, v22, 0, a3, a4, *(double *)a5.m128i_i64, v19);
  v18 = v17;
  sub_1F81BC0((__int64)a1, v23);
  sub_1F81BC0((__int64)a1, v16);
  *((_QWORD *)&v20 + 1) = v18;
  *(_QWORD *)&v20 = v16;
  result = sub_1D332F0(*a1, 118, (__int64)&v26, v21, v22, 0, a3, a4, a5, v23, v25, v20);
  if ( v26 )
  {
    v24 = result;
    sub_161E7C0((__int64)&v26, v26);
    return v24;
  }
  return result;
}
