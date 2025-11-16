// Function: sub_2039930
// Address: 0x2039930
//
__int64 __fastcall sub_2039930(__int64 *a1, __int64 a2, double a3, double a4, double a5)
{
  __int64 v6; // rax
  __int64 *v7; // r12
  __int64 v8; // rcx
  const void **v9; // r8
  __int64 v10; // r14
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rsi
  __int64 v14; // r14
  __int128 v16; // [rsp-10h] [rbp-70h]
  const void **v17; // [rsp+0h] [rbp-60h]
  __int64 v18; // [rsp+0h] [rbp-60h]
  __int64 v19; // [rsp+8h] [rbp-58h]
  const void **v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+10h] [rbp-50h] BYREF
  int v22; // [rsp+18h] [rbp-48h]
  const void **v23; // [rsp+20h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v21,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v17 = v23;
  v19 = (unsigned __int8)v22;
  v6 = sub_20363F0((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v7 = (__int64 *)a1[1];
  v8 = v19;
  v9 = v17;
  v10 = v6;
  v12 = v11;
  v21 = *(_QWORD *)(a2 + 72);
  if ( v21 )
  {
    v18 = v19;
    v20 = v9;
    sub_1623A60((__int64)&v21, v21, 2);
    v8 = v18;
    v9 = v20;
  }
  *((_QWORD *)&v16 + 1) = v12;
  *(_QWORD *)&v16 = v10;
  v13 = *(unsigned __int16 *)(a2 + 24);
  v22 = *(_DWORD *)(a2 + 64);
  v14 = sub_1D309E0(v7, v13, (__int64)&v21, v8, v9, 0, a3, a4, a5, v16);
  if ( v21 )
    sub_161E7C0((__int64)&v21, v21);
  return v14;
}
