// Function: sub_3794F40
// Address: 0x3794f40
//
__int64 __fastcall sub_3794F40(__int64 a1, __int64 a2)
{
  __int64 v3; // r12
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // rdx
  __int128 v7; // rax
  __int64 v8; // rsi
  _QWORD *v9; // r11
  __int64 v10; // r9
  unsigned int v11; // ecx
  __int64 v12; // r8
  unsigned int v13; // esi
  __int64 v14; // r12
  unsigned int v16; // [rsp+8h] [rbp-78h]
  unsigned int v17; // [rsp+14h] [rbp-6Ch]
  __int64 v18; // [rsp+18h] [rbp-68h]
  _QWORD *v19; // [rsp+20h] [rbp-60h]
  __int128 v20; // [rsp+30h] [rbp-50h]
  __int128 v21; // [rsp+40h] [rbp-40h]
  __int64 v22; // [rsp+50h] [rbp-30h] BYREF
  int v23; // [rsp+58h] [rbp-28h]

  v3 = sub_37946F0(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v5 = v4;
  *(_QWORD *)&v21 = sub_37946F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  *((_QWORD *)&v21 + 1) = v6;
  *(_QWORD *)&v7 = sub_37946F0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 88LL));
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *(_QWORD **)(a1 + 8);
  v20 = v7;
  v10 = *(unsigned int *)(a2 + 28);
  *(_QWORD *)&v7 = *(_QWORD *)(v3 + 48) + 16LL * (unsigned int)v5;
  v11 = *(unsigned __int16 *)v7;
  v12 = *(_QWORD *)(v7 + 8);
  v22 = v8;
  if ( v8 )
  {
    v16 = v11;
    v17 = v10;
    v18 = v12;
    v19 = v9;
    sub_B96E90((__int64)&v22, v8, 1);
    v11 = v16;
    v10 = v17;
    v12 = v18;
    v9 = v19;
  }
  v13 = *(_DWORD *)(a2 + 24);
  v23 = *(_DWORD *)(a2 + 72);
  v14 = sub_340EC60(v9, v13, (__int64)&v22, v11, v12, v10, v3, v5, v21, v20);
  if ( v22 )
    sub_B91220((__int64)&v22, v22);
  return v14;
}
