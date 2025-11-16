// Function: sub_213BC50
// Address: 0x213bc50
//
__int64 *__fastcall sub_213BC50(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  __int64 *v9; // r9
  __int64 v10; // r10
  __int128 *v11; // r14
  __int64 v12; // r11
  __int64 v13; // rcx
  const void **v14; // r8
  __int64 *v15; // r12
  __int128 v17; // [rsp-30h] [rbp-A0h]
  __int128 v18; // [rsp-20h] [rbp-90h]
  __int64 v19; // [rsp+8h] [rbp-68h]
  __int64 v20; // [rsp+10h] [rbp-60h]
  __int64 v21; // [rsp+18h] [rbp-58h]
  const void **v22; // [rsp+20h] [rbp-50h]
  __int64 *v23; // [rsp+28h] [rbp-48h]
  __int64 v24; // [rsp+30h] [rbp-40h] BYREF
  int v25; // [rsp+38h] [rbp-38h]

  v3 = sub_2138AD0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v5 = v4;
  v6 = sub_2138AD0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 120LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 128LL));
  v8 = *(_QWORD *)(a2 + 72);
  v9 = *(__int64 **)(a1 + 8);
  v10 = v6;
  v11 = *(__int128 **)(a2 + 32);
  v12 = v7;
  v13 = *(unsigned __int8 *)(*(_QWORD *)(v3 + 40) + 16LL * (unsigned int)v5);
  v14 = *(const void ***)(*(_QWORD *)(v3 + 40) + 16LL * (unsigned int)v5 + 8);
  v24 = v8;
  if ( v8 )
  {
    v21 = v7;
    v19 = v13;
    v20 = v6;
    v22 = v14;
    v23 = v9;
    sub_1623A60((__int64)&v24, v8, 2);
    v13 = v19;
    v10 = v20;
    v12 = v21;
    v14 = v22;
    v9 = v23;
  }
  v25 = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v18 + 1) = v12;
  *(_QWORD *)&v18 = v10;
  *((_QWORD *)&v17 + 1) = v5;
  *(_QWORD *)&v17 = v3;
  v15 = sub_1D36A20(
          v9,
          136,
          (__int64)&v24,
          v13,
          v14,
          (__int64)v9,
          *v11,
          *(__int128 *)((char *)v11 + 40),
          v17,
          v18,
          v11[10]);
  if ( v24 )
    sub_161E7C0((__int64)&v24, v24);
  return v15;
}
