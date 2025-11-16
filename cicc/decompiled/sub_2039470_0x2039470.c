// Function: sub_2039470
// Address: 0x2039470
//
__int64 *__fastcall sub_2039470(__int64 *a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // r13
  __int64 *v13; // r10
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rsi
  __int64 *v17; // r12
  __int128 v19; // [rsp-10h] [rbp-80h]
  __int64 v20; // [rsp+0h] [rbp-70h]
  __int64 v21; // [rsp+8h] [rbp-68h]
  __int64 *v22; // [rsp+10h] [rbp-60h]
  const void **v23; // [rsp+18h] [rbp-58h]
  __int64 v24; // [rsp+20h] [rbp-50h] BYREF
  int v25; // [rsp+28h] [rbp-48h]
  const void **v26; // [rsp+30h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v24,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v6 = (unsigned __int8)v25;
  v23 = v26;
  v7 = sub_20363F0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v8 = *(_QWORD *)(a2 + 72);
  v9 = v7;
  v10 = *(_QWORD *)(a2 + 32);
  v12 = v11;
  v13 = (__int64 *)a1[1];
  v14 = *(_QWORD *)(v10 + 40);
  v15 = *(_QWORD *)(v10 + 48);
  v24 = v8;
  if ( v8 )
  {
    v20 = v14;
    v21 = v15;
    v22 = v13;
    sub_1623A60((__int64)&v24, v8, 2);
    v14 = v20;
    v15 = v21;
    v13 = v22;
  }
  *((_QWORD *)&v19 + 1) = v15;
  *(_QWORD *)&v19 = v14;
  v16 = *(unsigned __int16 *)(a2 + 24);
  v25 = *(_DWORD *)(a2 + 64);
  v17 = sub_1D332F0(v13, v16, (__int64)&v24, v6, v23, 0, a3, a4, a5, v9, v12, v19);
  if ( v24 )
    sub_161E7C0((__int64)&v24, v24);
  return v17;
}
