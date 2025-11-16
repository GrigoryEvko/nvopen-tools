// Function: sub_24745F0
// Address: 0x24745f0
//
void __fastcall sub_24745F0(__int64 a1, __int64 a2)
{
  __int64 *v3; // rdx
  int v4; // eax
  __int64 v5; // rbx
  unsigned __int8 *v6; // r10
  __int64 (__fastcall *v7)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v8; // rax
  _BYTE *v9; // r15
  __int64 v10; // rax
  _BYTE *v11; // rax
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned int *v16; // rbx
  __int64 v17; // rdx
  unsigned int v18; // esi
  __int64 v19; // rax
  _BYTE *v20; // [rsp+8h] [rbp-138h]
  _BYTE *v21; // [rsp+10h] [rbp-130h]
  unsigned int *v22; // [rsp+10h] [rbp-130h]
  unsigned __int8 *v23; // [rsp+18h] [rbp-128h]
  unsigned __int8 *v24; // [rsp+18h] [rbp-128h]
  char v25[32]; // [rsp+20h] [rbp-120h] BYREF
  __int16 v26; // [rsp+40h] [rbp-100h]
  _BYTE v27[32]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v28; // [rsp+70h] [rbp-D0h]
  unsigned int *v29; // [rsp+80h] [rbp-C0h] BYREF
  unsigned int v30; // [rsp+88h] [rbp-B8h]
  char v31; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v32; // [rsp+B8h] [rbp-88h]
  __int64 v33; // [rsp+C0h] [rbp-80h]
  __int64 v34; // [rsp+D0h] [rbp-70h]
  __int64 v35; // [rsp+D8h] [rbp-68h]
  void *v36; // [rsp+100h] [rbp-40h]

  sub_23D0AB0((__int64)&v29, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v20 = (_BYTE *)sub_246F3F0(a1, *v3);
  v4 = *(_DWORD *)(a2 + 4);
  v26 = 257;
  v5 = *(_QWORD *)(a2 - 32LL * (v4 & 0x7FFFFFF));
  v6 = (unsigned __int8 *)sub_AD62B0(*(_QWORD *)(v5 + 8));
  v7 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v34 + 16LL);
  if ( v7 != sub_9202E0 )
  {
    v24 = v6;
    v19 = v7(v34, 30u, (_BYTE *)v5, v6);
    v6 = v24;
    v9 = (_BYTE *)v19;
    goto LABEL_9;
  }
  if ( *(_BYTE *)v5 <= 0x15u && *v6 <= 0x15u )
  {
    v23 = v6;
    if ( (unsigned __int8)sub_AC47B0(30) )
      v8 = sub_AD5570(30, v5, v23, 0, 0);
    else
      v8 = sub_AABE40(0x1Eu, (unsigned __int8 *)v5, v23);
    v6 = v23;
    v9 = (_BYTE *)v8;
LABEL_9:
    if ( v9 )
      goto LABEL_10;
  }
  v28 = 257;
  v9 = (_BYTE *)sub_B504D0(30, v5, (__int64)v6, (__int64)v27, 0, 0);
  (*(void (__fastcall **)(__int64, _BYTE *, char *, __int64, __int64))(*(_QWORD *)v35 + 16LL))(v35, v9, v25, v32, v33);
  v15 = 4LL * v30;
  v16 = v29;
  v22 = &v29[v15];
  while ( v22 != v16 )
  {
    v17 = *((_QWORD *)v16 + 1);
    v18 = *v16;
    v16 += 4;
    sub_B99FD0((__int64)v9, v18, v17);
  }
LABEL_10:
  v28 = 257;
  v10 = sub_A82480(&v29, v9, v20, (__int64)v27);
  v21 = (_BYTE *)sub_B34860((__int64)&v29, v10);
  v11 = (_BYTE *)sub_B34870((__int64)&v29, (__int64)v20);
  v28 = 257;
  v12 = sub_A82350(&v29, v21, v11, (__int64)v27);
  sub_246EF60(a1, a2, v12);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v13 = *(__int64 **)(a2 - 8);
  else
    v13 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v14 = sub_246EE10(a1, *v13);
  sub_246F1C0(a1, a2, v14);
  nullsub_61();
  v36 = &unk_49DA100;
  nullsub_63();
  if ( v29 != (unsigned int *)&v31 )
    _libc_free((unsigned __int64)v29);
}
