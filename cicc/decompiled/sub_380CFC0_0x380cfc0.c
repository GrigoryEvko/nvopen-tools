// Function: sub_380CFC0
// Address: 0x380cfc0
//
__int64 __fastcall sub_380CFC0(__int64 a1, __int64 a2)
{
  __int16 *v3; // rax
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rsi
  _QWORD *v8; // r9
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // r15
  __int64 v13; // rax
  unsigned int v14; // r8d
  unsigned int v15; // ecx
  __int128 v16; // rax
  __int64 v17; // r12
  __int128 v19; // [rsp-20h] [rbp-90h]
  unsigned int v20; // [rsp+Ch] [rbp-64h]
  __int128 v21; // [rsp+10h] [rbp-60h]
  _QWORD *v22; // [rsp+20h] [rbp-50h]
  unsigned int v23; // [rsp+20h] [rbp-50h]
  unsigned __int16 v24; // [rsp+28h] [rbp-48h]
  _QWORD *v25; // [rsp+28h] [rbp-48h]
  __int64 v26; // [rsp+30h] [rbp-40h] BYREF
  int v27; // [rsp+38h] [rbp-38h]

  v3 = *(__int16 **)(a2 + 48);
  v4 = *((_QWORD *)v3 + 1);
  v24 = *v3;
  *(_QWORD *)&v21 = sub_380AAE0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  *((_QWORD *)&v21 + 1) = v5;
  v6 = sub_380AAE0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *(_QWORD **)(a1 + 8);
  v9 = v6;
  v10 = *(_QWORD *)(a2 + 40);
  v12 = v11;
  v26 = v7;
  v13 = *(_QWORD *)(v10 + 80);
  v14 = *(_DWORD *)(v13 + 96);
  if ( v7 )
  {
    v20 = *(_DWORD *)(v13 + 96);
    v22 = v8;
    sub_B96E90((__int64)&v26, v7, 1);
    v14 = v20;
    v8 = v22;
  }
  v15 = v24;
  v25 = v8;
  v23 = v15;
  v27 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v16 = sub_33ED040(v8, v14);
  *((_QWORD *)&v19 + 1) = v12;
  *(_QWORD *)&v19 = v9;
  v17 = sub_340F900(v25, 0xD0u, (__int64)&v26, v23, v4, (__int64)v25, v21, v19, v16);
  if ( v26 )
    sub_B91220((__int64)&v26, v26);
  return v17;
}
