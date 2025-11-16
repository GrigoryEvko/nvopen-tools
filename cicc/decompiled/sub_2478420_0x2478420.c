// Function: sub_2478420
// Address: 0x2478420
//
void __fastcall sub_2478420(__int64 *a1, __int64 a2)
{
  __int64 **v3; // r15
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // rbx
  _QWORD *v7; // rax
  _BYTE *v8; // rcx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  bool v11; // zf
  _BYTE *v12; // r15
  __int64 *v13; // rdx
  __int64 v14; // rax
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rsi
  _BYTE *v19; // rax
  __int64 v20; // rax
  int v21; // [rsp+18h] [rbp-128h]
  _QWORD v22[4]; // [rsp+20h] [rbp-120h] BYREF
  __int16 v23; // [rsp+40h] [rbp-100h]
  _BYTE v24[32]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v25; // [rsp+70h] [rbp-D0h]
  unsigned int *v26[2]; // [rsp+80h] [rbp-C0h] BYREF
  char v27; // [rsp+90h] [rbp-B0h] BYREF
  void *v28; // [rsp+100h] [rbp-40h]

  sub_23D0AB0((__int64)v26, a2, 0, 0, 0);
  v3 = (__int64 **)sub_2463540(a1, *(_QWORD *)(a2 + 8));
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v4 = *(_QWORD *)(a2 - 8);
  else
    v4 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v5 = sub_246F3F0((__int64)a1, *(_QWORD *)(v4 + 32));
  v25 = 257;
  v6 = v5;
  v23 = 257;
  v7 = sub_2463540(a1, (__int64)v3);
  v8 = v7;
  if ( v7 )
    v8 = (_BYTE *)sub_AD6530((__int64)v7, (__int64)v3);
  v9 = sub_92B530(v26, 0x21u, v6, v8, (__int64)v22);
  v10 = sub_24633A0((__int64 *)v26, 0x28u, v9, v3, (__int64)v24, 0, v21, 0);
  v11 = (*(_BYTE *)(a2 + 7) & 0x40) == 0;
  v25 = 257;
  v12 = (_BYTE *)v10;
  if ( v11 )
    v13 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  else
    v13 = *(__int64 **)(a2 - 8);
  v14 = sub_246F3F0((__int64)a1, *v13);
  v15 = *(_DWORD *)(a2 + 4);
  v22[0] = v14;
  v16 = 1LL - (v15 & 0x7FFFFFF);
  v17 = *(_QWORD *)(a2 - 32);
  v22[1] = *(_QWORD *)(a2 + 32 * v16);
  if ( !v17 )
    goto LABEL_11;
  if ( *(_BYTE *)v17 || (v18 = *(_QWORD *)(a2 + 80), *(_QWORD *)(v17 + 24) != v18) )
  {
    LODWORD(v17) = 0;
LABEL_11:
    v18 = 0;
  }
  v19 = (_BYTE *)sub_921880(v26, v18, v17, (int)v22, 2, (__int64)v24, 0);
  v25 = 257;
  v20 = sub_A82480(v26, v12, v19, (__int64)v24);
  sub_246EF60((__int64)a1, a2, v20);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, a2);
  nullsub_61();
  v28 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v26[0] != &v27 )
    _libc_free((unsigned __int64)v26[0]);
}
