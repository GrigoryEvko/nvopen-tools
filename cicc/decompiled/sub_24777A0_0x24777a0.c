// Function: sub_24777A0
// Address: 0x24777a0
//
void __fastcall sub_24777A0(__int64 *a1, __int64 a2)
{
  __int64 *v3; // rdx
  _BYTE *v4; // r15
  __int64 v5; // rdx
  _BYTE *v6; // rax
  _BYTE *v7; // r15
  _QWORD *v8; // rax
  __int64 v9; // rax
  unsigned __int8 *v10; // r11
  __int64 (__fastcall *v11)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rsi
  _QWORD *v15; // rax
  _BYTE *v16; // rcx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // r15
  unsigned int *v21; // r15
  __int64 v22; // rdx
  unsigned int v23; // esi
  __int64 v24; // rax
  __int64 v25; // [rsp+8h] [rbp-138h]
  unsigned __int8 *v26; // [rsp+18h] [rbp-128h]
  __int64 v27; // [rsp+18h] [rbp-128h]
  unsigned int *v28; // [rsp+18h] [rbp-128h]
  unsigned __int8 *v29; // [rsp+18h] [rbp-128h]
  char v30[32]; // [rsp+20h] [rbp-120h] BYREF
  __int16 v31; // [rsp+40h] [rbp-100h]
  _BYTE v32[32]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v33; // [rsp+70h] [rbp-D0h]
  unsigned int *v34; // [rsp+80h] [rbp-C0h] BYREF
  unsigned int v35; // [rsp+88h] [rbp-B8h]
  char v36; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v37; // [rsp+B8h] [rbp-88h]
  __int64 v38; // [rsp+C0h] [rbp-80h]
  _QWORD *v39; // [rsp+C8h] [rbp-78h]
  __int64 v40; // [rsp+D0h] [rbp-70h]
  __int64 v41; // [rsp+D8h] [rbp-68h]
  void *v42; // [rsp+100h] [rbp-40h]

  sub_23D0AB0((__int64)&v34, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(__int64 **)(a2 - 8);
  else
    v3 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v4 = (_BYTE *)sub_246F3F0((__int64)a1, *v3);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(_QWORD *)(a2 - 8);
  else
    v5 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v6 = (_BYTE *)sub_246F3F0((__int64)a1, *(_QWORD *)(v5 + 32));
  v33 = 257;
  v7 = (_BYTE *)sub_A82480(&v34, v4, v6, (__int64)v32);
  v8 = sub_2463540(a1, *(_QWORD *)(a2 + 8));
  v31 = 257;
  v25 = (__int64)v8;
  v9 = sub_BCB2E0(v39);
  v10 = (unsigned __int8 *)sub_ACD640(v9, 0, 0);
  v11 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int8 *))(*(_QWORD *)v40 + 96LL);
  if ( v11 != sub_948070 )
  {
    v29 = v10;
    v24 = v11(v40, v7, v10);
    v10 = v29;
    v13 = v24;
LABEL_9:
    if ( v13 )
      goto LABEL_10;
    goto LABEL_19;
  }
  if ( *v7 <= 0x15u && *v10 <= 0x15u )
  {
    v26 = v10;
    v12 = sub_AD5840((__int64)v7, v10, 0);
    v10 = v26;
    v13 = v12;
    goto LABEL_9;
  }
LABEL_19:
  v27 = (__int64)v10;
  v33 = 257;
  v19 = sub_BD2C40(72, 2u);
  v13 = (__int64)v19;
  if ( v19 )
    sub_B4DE80((__int64)v19, (__int64)v7, v27, (__int64)v32, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v41 + 16LL))(v41, v13, v30, v37, v38);
  v20 = 4LL * v35;
  v28 = &v34[v20];
  if ( v34 != &v34[v20] )
  {
    v21 = v34;
    do
    {
      v22 = *((_QWORD *)v21 + 1);
      v23 = *v21;
      v21 += 4;
      sub_B99FD0(v13, v23, v22);
    }
    while ( v28 != v21 );
  }
LABEL_10:
  v33 = 257;
  v14 = *(_QWORD *)(v13 + 8);
  v15 = sub_2463540(a1, v14);
  v16 = v15;
  if ( v15 )
    v16 = (_BYTE *)sub_AD6530((__int64)v15, v14);
  v17 = sub_92B530(&v34, 0x21u, v13, v16, (__int64)v32);
  v18 = sub_2464970(a1, &v34, v17, v25, 1);
  sub_246EF60((__int64)a1, a2, v18);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, a2);
  nullsub_61();
  v42 = &unk_49DA100;
  nullsub_63();
  if ( v34 != (unsigned int *)&v36 )
    _libc_free((unsigned __int64)v34);
}
