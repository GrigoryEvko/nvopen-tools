// Function: sub_380E320
// Address: 0x380e320
//
__int64 __fastcall sub_380E320(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rsi
  _QWORD *v9; // r9
  __int64 v10; // r10
  __int64 v11; // rax
  __int128 *v12; // rcx
  __int64 v13; // r11
  unsigned int v14; // r15d
  __int64 v15; // r8
  __int64 v16; // r12
  __int128 v18; // [rsp-20h] [rbp-90h]
  __int128 v19; // [rsp-10h] [rbp-80h]
  __int64 v20; // [rsp+0h] [rbp-70h]
  __int64 v21; // [rsp+8h] [rbp-68h]
  __int64 v22; // [rsp+18h] [rbp-58h]
  __int128 *v23; // [rsp+20h] [rbp-50h]
  _QWORD *v24; // [rsp+28h] [rbp-48h]
  __int64 v25; // [rsp+30h] [rbp-40h] BYREF
  int v26; // [rsp+38h] [rbp-38h]

  HIWORD(v14) = 0;
  v3 = sub_380AAE0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v5 = v4;
  v6 = sub_380AAE0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *(_QWORD **)(a1 + 8);
  v10 = v6;
  v11 = *(_QWORD *)(v3 + 48);
  v12 = *(__int128 **)(a2 + 40);
  v13 = v7;
  LOWORD(v14) = *(_WORD *)v11;
  v15 = *(_QWORD *)(v11 + 8);
  v25 = v8;
  if ( v8 )
  {
    v21 = v7;
    v20 = v10;
    v22 = v15;
    v23 = v12;
    v24 = v9;
    sub_B96E90((__int64)&v25, v8, 1);
    v10 = v20;
    v13 = v21;
    v15 = v22;
    v12 = v23;
    v9 = v24;
  }
  *((_QWORD *)&v19 + 1) = v13;
  *(_QWORD *)&v19 = v10;
  *((_QWORD *)&v18 + 1) = v5;
  v26 = *(_DWORD *)(a2 + 72);
  *(_QWORD *)&v18 = v3;
  v16 = sub_340F900(v9, 0xCDu, (__int64)&v25, v14, v15, (__int64)v9, *v12, v18, v19);
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  return v16;
}
