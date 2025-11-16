// Function: sub_2478D50
// Address: 0x2478d50
//
void __fastcall sub_2478D50(__int64 *a1, __int64 a2)
{
  __int64 *v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 **v7; // r15
  __int64 v8; // rbx
  __int64 v9; // rsi
  _QWORD *v10; // rax
  _BYTE *v11; // rcx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  _BYTE *v17; // rax
  __int64 v18; // rax
  __int64 v19; // [rsp+10h] [rbp-140h]
  __int64 v20; // [rsp+18h] [rbp-138h]
  __int64 v21; // [rsp+20h] [rbp-130h] BYREF
  __int64 v22; // [rsp+28h] [rbp-128h]
  _QWORD v23[4]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v24; // [rsp+50h] [rbp-100h]
  _BYTE v25[32]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v26; // [rsp+80h] [rbp-D0h]
  unsigned int *v27[2]; // [rsp+90h] [rbp-C0h] BYREF
  char v28; // [rsp+A0h] [rbp-B0h] BYREF
  void *v29; // [rsp+110h] [rbp-40h]

  sub_23D0AB0((__int64)v27, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v19 = sub_246F3F0((__int64)a1, *v3);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(_QWORD *)(a2 - 8);
  else
    v4 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v20 = sub_246F3F0((__int64)a1, *(_QWORD *)(v4 + 32));
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(_QWORD *)(a2 - 8);
  else
    v5 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v6 = sub_246F3F0((__int64)a1, *(_QWORD *)(v5 + 64));
  v26 = 257;
  v7 = *(__int64 ***)(v6 + 8);
  v8 = v6;
  v24 = 257;
  v9 = *(_QWORD *)(v6 + 8);
  v10 = sub_2463540(a1, v9);
  v11 = v10;
  if ( v10 )
    v11 = (_BYTE *)sub_AD6530((__int64)v10, v9);
  v12 = sub_92B530(v27, 0x21u, v8, v11, (__int64)v23);
  v13 = sub_24633A0((__int64 *)v27, 0x28u, v12, v7, (__int64)v25, 0, v22, 0);
  v14 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v26 = 257;
  v23[0] = v19;
  HIDWORD(v22) = 0;
  v15 = *(_QWORD *)(a2 + 32 * (2 - v14));
  v23[1] = v20;
  v23[2] = v15;
  v21 = *(_QWORD *)(v13 + 8);
  v16 = *(_QWORD *)(a2 - 32);
  if ( !v16 || *(_BYTE *)v16 || *(_QWORD *)(v16 + 24) != *(_QWORD *)(a2 + 80) )
    BUG();
  v17 = (_BYTE *)sub_B33D10((__int64)v27, *(_DWORD *)(v16 + 36), (__int64)&v21, 1, (int)v23, 3, v22, (__int64)v25);
  v26 = 257;
  v18 = sub_A82480(v27, v17, (_BYTE *)v13, (__int64)v25);
  sub_246EF60((__int64)a1, a2, v18);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, a2);
  nullsub_61();
  v29 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v27[0] != &v28 )
    _libc_free((unsigned __int64)v27[0]);
}
