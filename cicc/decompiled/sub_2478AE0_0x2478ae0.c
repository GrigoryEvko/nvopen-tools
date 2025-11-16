// Function: sub_2478AE0
// Address: 0x2478ae0
//
void __fastcall sub_2478AE0(__int64 *a1, __int64 a2, int a3)
{
  __int64 **v4; // r15
  __int64 *v5; // rdx
  __int64 v6; // r8
  __int64 v7; // rdx
  _BYTE *v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  _BYTE *v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rsi
  unsigned __int64 v15; // r15
  __int64 **v16; // rax
  unsigned __int64 v17; // rax
  unsigned int v18; // r14d
  __int64 *v19; // rax
  _BYTE *v20; // [rsp+8h] [rbp-138h]
  __int64 v21; // [rsp+8h] [rbp-138h]
  int v22; // [rsp+18h] [rbp-128h]
  int v23[8]; // [rsp+20h] [rbp-120h] BYREF
  __int16 v24; // [rsp+40h] [rbp-100h]
  _BYTE v25[32]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v26; // [rsp+70h] [rbp-D0h]
  unsigned int *v27[2]; // [rsp+80h] [rbp-C0h] BYREF
  char v28; // [rsp+90h] [rbp-B0h] BYREF
  void *v29; // [rsp+100h] [rbp-40h]

  if ( a3 )
  {
    v18 = 2 * a3;
    v19 = (__int64 *)sub_BCCE00(*(_QWORD **)(a1[1] + 72), 2 * a3);
    v4 = (__int64 **)sub_BCDA70(v19, 0x40 / v18);
  }
  else
  {
    v4 = *(__int64 ***)(a2 + 8);
  }
  sub_23D0AB0((__int64)v27, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(__int64 **)(a2 - 8);
  else
    v5 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v6 = sub_246F3F0((__int64)a1, *v5);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v7 = *(_QWORD *)(a2 - 8);
  else
    v7 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v20 = (_BYTE *)v6;
  v8 = (_BYTE *)sub_246F3F0((__int64)a1, *(_QWORD *)(v7 + 32));
  v26 = 257;
  v9 = sub_A82480(v27, v20, v8, (__int64)v25);
  v26 = 257;
  v10 = sub_24633A0((__int64 *)v27, 0x31u, v9, v4, (__int64)v25, 0, v23[0], 0);
  v24 = 257;
  v26 = 257;
  v21 = v10;
  v11 = (_BYTE *)sub_AD6530((__int64)v4, 257);
  v12 = sub_92B530(v27, 0x21u, v21, v11, (__int64)v23);
  v13 = sub_24633A0((__int64 *)v27, 0x28u, v12, v4, (__int64)v25, 0, v22, 0);
  v14 = *(_QWORD *)(a2 + 8);
  v15 = v13;
  v26 = 257;
  v16 = (__int64 **)sub_2463540(a1, v14);
  v17 = sub_24633A0((__int64 *)v27, 0x31u, v15, v16, (__int64)v25, 0, v23[0], 0);
  sub_246EF60((__int64)a1, a2, v17);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, a2);
  nullsub_61();
  v29 = &unk_49DA100;
  nullsub_63();
  if ( (char *)v27[0] != &v28 )
    _libc_free((unsigned __int64)v27[0]);
}
