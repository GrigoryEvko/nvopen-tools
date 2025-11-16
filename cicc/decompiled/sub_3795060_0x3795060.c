// Function: sub_3795060
// Address: 0x3795060
//
__int64 __fastcall sub_3795060(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // r13
  __int128 v6; // rax
  __int64 v7; // rsi
  _QWORD *v8; // r15
  __int64 v9; // r9
  __int64 v10; // r10
  __int64 v11; // r11
  unsigned int v12; // ecx
  __int64 v13; // r8
  unsigned int v14; // esi
  __int64 v15; // r12
  __int128 v17; // [rsp-10h] [rbp-90h]
  unsigned int v18; // [rsp+8h] [rbp-78h]
  __int64 v19; // [rsp+10h] [rbp-70h]
  __int64 v20; // [rsp+18h] [rbp-68h]
  unsigned int v21; // [rsp+24h] [rbp-5Ch]
  __int64 v22; // [rsp+28h] [rbp-58h]
  __int128 v23; // [rsp+30h] [rbp-50h]
  __int64 v24; // [rsp+40h] [rbp-40h] BYREF
  int v25; // [rsp+48h] [rbp-38h]

  v3 = sub_37946F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v5 = v4;
  *(_QWORD *)&v6 = sub_37946F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *(_QWORD **)(a1 + 8);
  v23 = v6;
  *(_QWORD *)&v6 = *(_QWORD *)(a2 + 40);
  v9 = *(unsigned int *)(a2 + 28);
  v10 = *(_QWORD *)(v6 + 80);
  v11 = *(_QWORD *)(v6 + 88);
  v12 = *(unsigned __int16 *)(*(_QWORD *)(v3 + 48) + 16LL * (unsigned int)v5);
  v13 = *(_QWORD *)(*(_QWORD *)(v3 + 48) + 16LL * (unsigned int)v5 + 8);
  v24 = v7;
  if ( v7 )
  {
    v18 = v12;
    v19 = v10;
    v20 = v11;
    v21 = v9;
    v22 = v13;
    sub_B96E90((__int64)&v24, v7, 1);
    v12 = v18;
    v10 = v19;
    v11 = v20;
    v9 = v21;
    v13 = v22;
  }
  *((_QWORD *)&v17 + 1) = v11;
  *(_QWORD *)&v17 = v10;
  v14 = *(_DWORD *)(a2 + 24);
  v25 = *(_DWORD *)(a2 + 72);
  v15 = sub_340EC60(v8, v14, (__int64)&v24, v12, v13, v9, v3, v5, v23, v17);
  if ( v24 )
    sub_B91220((__int64)&v24, v24);
  return v15;
}
