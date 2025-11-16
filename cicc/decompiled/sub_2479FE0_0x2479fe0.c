// Function: sub_2479FE0
// Address: 0x2479fe0
//
void __fastcall sub_2479FE0(__int64 *a1, __int64 a2, char a3)
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
  __int64 v15; // rax
  __int64 v16; // r10
  _BYTE *v17; // r11
  __int64 (__fastcall *v18)(__int64, unsigned int, unsigned __int8 *, _BYTE *, unsigned __int8); // rax
  char v19; // al
  unsigned __int8 *v20; // r10
  unsigned __int8 *v21; // rdx
  unsigned __int8 *v22; // rsi
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rsi
  __int64 **v26; // rax
  unsigned __int64 v27; // rax
  unsigned int *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  _BYTE *v31; // [rsp+10h] [rbp-140h]
  __int64 v32; // [rsp+10h] [rbp-140h]
  unsigned __int8 *v33; // [rsp+10h] [rbp-140h]
  unsigned __int8 *v34; // [rsp+10h] [rbp-140h]
  unsigned int *v35; // [rsp+10h] [rbp-140h]
  _BYTE *v36; // [rsp+10h] [rbp-140h]
  unsigned int v37; // [rsp+18h] [rbp-138h]
  unsigned __int8 *v38; // [rsp+18h] [rbp-138h]
  unsigned __int8 *v39; // [rsp+18h] [rbp-138h]
  __int64 v40; // [rsp+18h] [rbp-138h]
  unsigned int *v41; // [rsp+18h] [rbp-138h]
  int v42; // [rsp+28h] [rbp-128h]
  int v43[8]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v44; // [rsp+50h] [rbp-100h]
  _BYTE v45[32]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v46; // [rsp+80h] [rbp-D0h]
  unsigned int *v47; // [rsp+90h] [rbp-C0h] BYREF
  unsigned int v48; // [rsp+98h] [rbp-B8h]
  char v49; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v50; // [rsp+C8h] [rbp-88h]
  __int64 v51; // [rsp+D0h] [rbp-80h]
  __int64 v52; // [rsp+E0h] [rbp-70h]
  __int64 v53; // [rsp+E8h] [rbp-68h]
  void *v54; // [rsp+110h] [rbp-40h]

  if ( a3 )
    v4 = (__int64 **)sub_BCCE00(*(_QWORD **)(a1[1] + 72), 0x40u);
  else
    v4 = *(__int64 ***)(a2 + 8);
  v37 = sub_BCB060((__int64)v4) - 16;
  sub_23D0AB0((__int64)&v47, a2, 0, 0, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v5 = *(__int64 **)(a2 - 8);
  else
    v5 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  v6 = sub_246F3F0((__int64)a1, *v5);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v7 = *(_QWORD *)(a2 - 8);
  else
    v7 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v31 = (_BYTE *)v6;
  v8 = (_BYTE *)sub_246F3F0((__int64)a1, *(_QWORD *)(v7 + 32));
  v46 = 257;
  v9 = sub_A82480(&v47, v31, v8, (__int64)v45);
  v46 = 257;
  v10 = sub_24633A0((__int64 *)&v47, 0x31u, v9, v4, (__int64)v45, 0, v43[0], 0);
  v46 = 257;
  v44 = 257;
  v32 = v10;
  v11 = (_BYTE *)sub_AD6530((__int64)v4, 49);
  v12 = sub_92B530(&v47, 0x21u, v32, v11, (__int64)v43);
  v13 = sub_24633A0((__int64 *)&v47, 0x28u, v12, v4, (__int64)v45, 0, v42, 0);
  v44 = 257;
  v14 = v37;
  v38 = (unsigned __int8 *)v13;
  v15 = sub_AD64C0(*(_QWORD *)(v13 + 8), v14, 0);
  v16 = (__int64)v38;
  v17 = (_BYTE *)v15;
  v18 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, _BYTE *, unsigned __int8))(*(_QWORD *)v52 + 24LL);
  if ( v18 != sub_920250 )
  {
    v36 = v17;
    v30 = v18(v52, 26u, v38, v17, 0);
    v17 = v36;
    v16 = (__int64)v38;
    v24 = v30;
    goto LABEL_13;
  }
  if ( *v38 <= 0x15u && *v17 <= 0x15u )
  {
    v33 = v38;
    v39 = v17;
    v19 = sub_AC47B0(26);
    v20 = v33;
    v21 = v39;
    v22 = v33;
    v34 = v39;
    v40 = (__int64)v20;
    if ( v19 )
      v23 = sub_AD5570(26, (__int64)v22, v21, 0, 0);
    else
      v23 = sub_AABE40(0x1Au, v22, v21);
    v16 = v40;
    v17 = v34;
    v24 = v23;
LABEL_13:
    if ( v24 )
      goto LABEL_14;
  }
  v46 = 257;
  v24 = sub_B504D0(26, v16, (__int64)v17, (__int64)v45, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, int *, __int64, __int64))(*(_QWORD *)v53 + 16LL))(v53, v24, v43, v50, v51);
  v28 = v47;
  v29 = 4LL * v48;
  v35 = &v47[v29];
  if ( v47 != &v47[v29] )
  {
    do
    {
      v41 = v28;
      sub_B99FD0(v24, *v28, *((_QWORD *)v28 + 1));
      v28 = v41 + 4;
    }
    while ( v35 != v41 + 4 );
  }
LABEL_14:
  v25 = *(_QWORD *)(a2 + 8);
  v46 = 257;
  v26 = (__int64 **)sub_2463540(a1, v25);
  v27 = sub_24633A0((__int64 *)&v47, 0x31u, v24, v26, (__int64)v45, 0, v43[0], 0);
  sub_246EF60((__int64)a1, a2, v27);
  if ( *(_DWORD *)(a1[1] + 4) )
    sub_2477350((__int64)a1, a2);
  nullsub_61();
  v54 = &unk_49DA100;
  nullsub_63();
  if ( v47 != (unsigned int *)&v49 )
    _libc_free((unsigned __int64)v47);
}
