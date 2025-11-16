// Function: sub_17DA100
// Address: 0x17da100
//
unsigned __int64 __fastcall sub_17DA100(__int128 a1)
{
  __int64 v1; // r12
  _QWORD *v2; // rax
  __int64 *v3; // r15
  __int64 v4; // rax
  __int64 *v5; // rax
  _QWORD *v6; // r10
  __int64 v7; // r11
  __int64 v8; // r9
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 result; // rax
  __int64 v17; // [rsp+0h] [rbp-E0h]
  __int64 v18; // [rsp+8h] [rbp-D8h]
  __int64 v19; // [rsp+8h] [rbp-D8h]
  _QWORD *v20; // [rsp+10h] [rbp-D0h]
  __int64 v21; // [rsp+10h] [rbp-D0h]
  __int64 ***v22; // [rsp+18h] [rbp-C8h]
  __int64 v23; // [rsp+18h] [rbp-C8h]
  __int64 v24; // [rsp+18h] [rbp-C8h]
  _BYTE v25[16]; // [rsp+20h] [rbp-C0h] BYREF
  __int16 v26; // [rsp+30h] [rbp-B0h]
  __int64 v27[2]; // [rsp+40h] [rbp-A0h] BYREF
  __int16 v28; // [rsp+50h] [rbp-90h]
  __int64 v29[16]; // [rsp+60h] [rbp-80h] BYREF

  v1 = *((_QWORD *)&a1 + 1);
  sub_17CE510((__int64)v29, *((__int64 *)&a1 + 1), 0, 0, 0);
  if ( (*(_BYTE *)(*((_QWORD *)&a1 + 1) + 23LL) & 0x40) != 0 )
    v2 = *(_QWORD **)(*((_QWORD *)&a1 + 1) - 8LL);
  else
    v2 = (_QWORD *)(*((_QWORD *)&a1 + 1) - 24LL * (*(_DWORD *)(*((_QWORD *)&a1 + 1) + 20LL) & 0xFFFFFFF));
  *((_QWORD *)&a1 + 1) = *v2;
  v3 = sub_17D4DA0(a1);
  if ( (*(_BYTE *)(v1 + 23) & 0x40) != 0 )
    v4 = *(_QWORD *)(v1 - 8);
  else
    v4 = v1 - 24LL * (*(_DWORD *)(v1 + 20) & 0xFFFFFFF);
  *((_QWORD *)&a1 + 1) = *(_QWORD *)(v4 + 24);
  v5 = sub_17D4DA0(a1);
  v6 = *(_QWORD **)(v1 - 48);
  v7 = *(_QWORD *)(v1 - 24);
  v8 = (__int64)v5;
  if ( *v6 != *v3 )
  {
    v22 = (__int64 ***)v5;
    v18 = *(_QWORD *)(v1 - 24);
    v28 = 257;
    v9 = sub_17CE200(v29, (__int64)v6, (__int64 **)*v3, 0, v27);
    v28 = 257;
    v20 = (_QWORD *)v9;
    v10 = sub_17CE200(v29, v18, *v22, 0, v27);
    v6 = v20;
    v8 = (__int64)v22;
    v7 = v10;
  }
  v17 = v7;
  v19 = (__int64)v6;
  v23 = v8;
  v28 = 257;
  v11 = sub_1281C00(v29, (__int64)v3, v8, (__int64)v27);
  v28 = 257;
  v21 = v11;
  v12 = sub_1281C00(v29, v19, v23, (__int64)v27);
  v28 = 257;
  v24 = v12;
  v13 = sub_1281C00(v29, (__int64)v3, v17, (__int64)v27);
  v28 = 257;
  v26 = 257;
  v14 = sub_156D390(v29, v24, v13, (__int64)v25);
  v15 = sub_156D390(v29, v21, v14, (__int64)v27);
  sub_17D4920(a1, (__int64 *)v1, v15);
  result = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(result + 156) )
    result = sub_17D9C10((_QWORD *)a1, v1);
  if ( v29[0] )
    return sub_161E7C0((__int64)v29, v29[0]);
  return result;
}
