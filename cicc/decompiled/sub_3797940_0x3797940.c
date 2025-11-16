// Function: sub_3797940
// Address: 0x3797940
//
unsigned __int8 *__fastcall sub_3797940(__int64 a1, __int64 a2)
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
  __int64 v12; // r8
  unsigned __int8 *v13; // r12
  __int128 v15; // [rsp-30h] [rbp-A0h]
  __int128 v16; // [rsp-20h] [rbp-90h]
  __int64 v17; // [rsp+0h] [rbp-70h]
  __int128 *v18; // [rsp+10h] [rbp-60h]
  __int64 v19; // [rsp+18h] [rbp-58h]
  __int64 v20; // [rsp+20h] [rbp-50h]
  _QWORD *v21; // [rsp+28h] [rbp-48h]
  __int64 v22; // [rsp+30h] [rbp-40h] BYREF
  int v23; // [rsp+38h] [rbp-38h]

  v2 = sub_37946F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v4 = v3;
  v20 = *(_QWORD *)(a2 + 40);
  v21 = *(_QWORD **)(a1 + 8);
  v5 = sub_37946F0(a1, *(_QWORD *)(v20 + 120), *(_QWORD *)(v20 + 128));
  v6 = *(__int128 **)(a2 + 40);
  v7 = v5;
  v8 = v20;
  v10 = v9;
  v11 = *(unsigned __int16 *)(*(_QWORD *)(v2 + 48) + 16LL * (unsigned int)v4);
  v12 = *(_QWORD *)(*(_QWORD *)(v2 + 48) + 16LL * (unsigned int)v4 + 8);
  v22 = *(_QWORD *)(a2 + 80);
  if ( v22 )
  {
    v17 = v11;
    v18 = v6;
    v19 = v12;
    sub_B96E90((__int64)&v22, v22, 1);
    v11 = v17;
    v8 = v20;
    v6 = v18;
    v12 = v19;
  }
  v23 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v16 + 1) = v10;
  *(_QWORD *)&v16 = v7;
  *((_QWORD *)&v15 + 1) = v4;
  *(_QWORD *)&v15 = v2;
  v13 = sub_33FC1D0(
          v21,
          207,
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
    sub_B91220((__int64)&v22, v22);
  return v13;
}
