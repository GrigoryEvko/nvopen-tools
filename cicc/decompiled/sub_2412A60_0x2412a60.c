// Function: sub_2412A60
// Address: 0x2412a60
//
__int64 __fastcall sub_2412A60(__int64 a1, __int64 a2, __int16 a3, unsigned __int8 a4, __int64 *a5)
{
  _BYTE *v6; // rax
  __int64 v7; // r12
  __int64 (__fastcall *v8)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v9; // rax
  _BYTE **v10; // rcx
  __int64 v11; // r14
  int v12; // ecx
  __int64 v13; // rsi
  __int64 v14; // r12
  __int64 v16; // r11
  unsigned int *v17; // r15
  unsigned int *v18; // r12
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int64 v21; // rdx
  int v22; // eax
  char v23; // al
  int v24; // edx
  __int64 v27; // [rsp+18h] [rbp-138h]
  _BYTE *v28; // [rsp+20h] [rbp-130h] BYREF
  __int64 v29; // [rsp+28h] [rbp-128h] BYREF
  char v30[32]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v31; // [rsp+50h] [rbp-100h]
  unsigned __int64 v32; // [rsp+60h] [rbp-F0h] BYREF
  unsigned int v33; // [rsp+68h] [rbp-E8h]
  unsigned __int64 v34; // [rsp+70h] [rbp-E0h]
  unsigned int v35; // [rsp+78h] [rbp-D8h]
  __int16 v36; // [rsp+80h] [rbp-D0h]
  unsigned int *v37; // [rsp+90h] [rbp-C0h] BYREF
  int v38; // [rsp+98h] [rbp-B8h]
  char v39; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v40; // [rsp+C8h] [rbp-88h]
  __int64 v41; // [rsp+D0h] [rbp-80h]
  __int64 v42; // [rsp+E0h] [rbp-70h]
  __int64 v43; // [rsp+E8h] [rbp-68h]
  void *v44; // [rsp+110h] [rbp-40h]

  if ( !a2 )
    BUG();
  sub_2412230((__int64)&v37, *(_QWORD *)(a2 + 16), a2, a3, 0, a2, 0, 0);
  v31 = 257;
  v6 = (_BYTE *)sub_ACD640(*(_QWORD *)(a1 + 64), 1, 0);
  v7 = *(_QWORD *)(a1 + 24);
  v28 = v6;
  v27 = *a5;
  v8 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v42 + 64LL);
  if ( v8 == sub_920540 )
  {
    if ( sub_BCEA30(v7) )
      goto LABEL_11;
    if ( *(_BYTE *)v27 > 0x15u )
      goto LABEL_11;
    v9 = sub_240D5B0(&v28, (__int64)&v29);
    if ( v10 != v9 )
      goto LABEL_11;
    LOBYTE(v36) = 0;
    v11 = sub_AD9FD0(v7, (unsigned __int8 *)v27, (__int64 *)&v28, 1, 0, (__int64)&v32, 0);
    if ( (_BYTE)v36 )
    {
      LOBYTE(v36) = 0;
      if ( v35 > 0x40 && v34 )
        j_j___libc_free_0_0(v34);
      if ( v33 > 0x40 && v32 )
        j_j___libc_free_0_0(v32);
    }
  }
  else
  {
    v11 = v8(v42, v7, (_BYTE *)v27, &v28, 1, 0);
  }
  if ( v11 )
    goto LABEL_8;
LABEL_11:
  v36 = 257;
  v11 = (__int64)sub_BD2C40(88, 2u);
  if ( !v11 )
    goto LABEL_14;
  v16 = *(_QWORD *)(v27 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 > 1 )
  {
    v21 = *((_QWORD *)v28 + 1);
    v22 = *(unsigned __int8 *)(v21 + 8);
    if ( v22 == 17 )
    {
      v23 = 0;
    }
    else
    {
      if ( v22 != 18 )
        goto LABEL_13;
      v23 = 1;
    }
    v24 = *(_DWORD *)(v21 + 32);
    BYTE4(v29) = v23;
    LODWORD(v29) = v24;
    v16 = sub_BCE1B0((__int64 *)v16, v29);
  }
LABEL_13:
  sub_B44260(v11, v16, 34, 2u, 0, 0);
  *(_QWORD *)(v11 + 72) = v7;
  *(_QWORD *)(v11 + 80) = sub_B4DC50(v7, (__int64)&v28, 1);
  sub_B4D9A0(v11, v27, (__int64 *)&v28, 1, (__int64)&v32);
LABEL_14:
  sub_B4DDE0(v11, 0);
  (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v43 + 16LL))(v43, v11, v30, v40, v41);
  v17 = v37;
  v18 = &v37[4 * v38];
  if ( v37 != v18 )
  {
    do
    {
      v19 = *((_QWORD *)v17 + 1);
      v20 = *v17;
      v17 += 4;
      sub_B99FD0(v11, v20, v19);
    }
    while ( v18 != v17 );
  }
LABEL_8:
  v12 = a4;
  *a5 = v11;
  v13 = *(_QWORD *)(a1 + 24);
  BYTE1(v12) = 1;
  v36 = 257;
  v14 = sub_A82CA0(&v37, v13, v11, v12, 0, (__int64)&v32);
  nullsub_61();
  v44 = &unk_49DA100;
  nullsub_63();
  if ( v37 != (unsigned int *)&v39 )
    _libc_free((unsigned __int64)v37);
  return v14;
}
