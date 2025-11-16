// Function: sub_2034810
// Address: 0x2034810
//
__int64 *__fastcall sub_2034810(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v4; // r13
  __int64 v5; // rax
  __int128 *v6; // r11
  __int64 v7; // r14
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // r15
  __int64 v11; // rcx
  const void **v12; // r8
  __int64 *v13; // r12
  __int128 v15; // [rsp-30h] [rbp-A0h]
  __int128 v16; // [rsp-20h] [rbp-90h]
  __int64 v17; // [rsp+0h] [rbp-70h]
  __int128 *v18; // [rsp+10h] [rbp-60h]
  const void **v19; // [rsp+18h] [rbp-58h]
  __int64 v20; // [rsp+20h] [rbp-50h]
  __int64 *v21; // [rsp+28h] [rbp-48h]
  __int64 v22; // [rsp+30h] [rbp-40h] BYREF
  int v23; // [rsp+38h] [rbp-38h]

  v2 = sub_2032580(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 88LL));
  v4 = v3;
  v20 = *(_QWORD *)(a2 + 32);
  v21 = *(__int64 **)(a1 + 8);
  v5 = sub_2032580(a1, *(_QWORD *)(v20 + 120), *(_QWORD *)(v20 + 128));
  v6 = *(__int128 **)(a2 + 32);
  v7 = v5;
  v8 = v20;
  v10 = v9;
  v11 = *(unsigned __int8 *)(*(_QWORD *)(v2 + 40) + 16LL * (unsigned int)v4);
  v12 = *(const void ***)(*(_QWORD *)(v2 + 40) + 16LL * (unsigned int)v4 + 8);
  v22 = *(_QWORD *)(a2 + 72);
  if ( v22 )
  {
    v17 = v11;
    v18 = v6;
    v19 = v12;
    sub_1623A60((__int64)&v22, v22, 2);
    v11 = v17;
    v8 = v20;
    v6 = v18;
    v12 = v19;
  }
  v23 = *(_DWORD *)(a2 + 64);
  *((_QWORD *)&v16 + 1) = v10;
  *(_QWORD *)&v16 = v7;
  *((_QWORD *)&v15 + 1) = v4;
  *(_QWORD *)&v15 = v2;
  v13 = sub_1D36A20(
          v21,
          136,
          (__int64)&v22,
          v11,
          v12,
          v8,
          *v6,
          *(__int128 *)((char *)v6 + 40),
          v15,
          v16,
          *(_OWORD *)(v8 + 160));
  if ( v22 )
    sub_161E7C0((__int64)&v22, v22);
  return v13;
}
