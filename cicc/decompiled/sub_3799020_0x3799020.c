// Function: sub_3799020
// Address: 0x3799020
//
__int64 __fastcall sub_3799020(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int128 *v4; // r12
  __int64 v5; // rsi
  __int64 v6; // r14
  __int64 v7; // rdx
  __int64 v8; // r15
  _QWORD *v9; // r9
  __int64 v10; // r10
  __int64 v11; // r11
  unsigned int v12; // ecx
  __int64 v13; // r8
  __int64 v14; // r14
  __int128 v16; // [rsp-30h] [rbp-A0h]
  __int128 v17; // [rsp-20h] [rbp-90h]
  unsigned int v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+10h] [rbp-60h]
  __int64 v20; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+20h] [rbp-50h]
  _QWORD *v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h] BYREF
  int v24; // [rsp+38h] [rbp-38h]

  v3 = sub_37946F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v4 = *(__int128 **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = v3;
  v8 = v7;
  v9 = *(_QWORD **)(a1 + 8);
  v10 = *(_QWORD *)v4;
  v11 = *((_QWORD *)v4 + 1);
  v12 = *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v4 + 48LL) + 16LL * *((unsigned int *)v4 + 2));
  v13 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)v4 + 48LL) + 16LL * *((unsigned int *)v4 + 2) + 8);
  v23 = v5;
  if ( v5 )
  {
    v18 = v12;
    v19 = v10;
    v20 = v11;
    v21 = v13;
    v22 = v9;
    sub_B96E90((__int64)&v23, v5, 1);
    v12 = v18;
    v10 = v19;
    v11 = v20;
    v13 = v21;
    v9 = v22;
  }
  v24 = *(_DWORD *)(a2 + 72);
  *((_QWORD *)&v17 + 1) = v8;
  *(_QWORD *)&v17 = v6;
  *((_QWORD *)&v16 + 1) = v11;
  *(_QWORD *)&v16 = v10;
  v14 = sub_340F900(v9, 0x9Du, (__int64)&v23, v12, v13, (__int64)v9, v16, v17, v4[5]);
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  return v14;
}
