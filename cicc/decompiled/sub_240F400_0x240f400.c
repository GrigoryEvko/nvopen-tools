// Function: sub_240F400
// Address: 0x240f400
//
__int64 __fastcall sub_240F400(__int64 a1, unsigned int a2, __int64 *a3)
{
  __int64 v4; // r13
  __int64 v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rax
  _BYTE *v8; // rax
  _QWORD *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 (__fastcall *v13)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  __int64 v14; // r14
  __int64 v16; // r11
  __int64 v17; // rbx
  __int64 v18; // r12
  __int64 v19; // rdx
  unsigned int v20; // esi
  __int64 v21; // rax
  int v22; // edx
  int v23; // edx
  char v24; // dl
  __int64 v25; // [rsp+8h] [rbp-B8h]
  __int64 v26; // [rsp+18h] [rbp-A8h]
  _BYTE *v27; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v28; // [rsp+28h] [rbp-98h]
  _QWORD v29[4]; // [rsp+30h] [rbp-90h] BYREF
  char v30; // [rsp+50h] [rbp-70h]
  char v31; // [rsp+51h] [rbp-6Fh]
  unsigned __int64 v32; // [rsp+60h] [rbp-60h] BYREF
  unsigned int v33; // [rsp+68h] [rbp-58h]
  unsigned __int64 v34; // [rsp+70h] [rbp-50h]
  unsigned int v35; // [rsp+78h] [rbp-48h]
  __int16 v36; // [rsp+80h] [rbp-40h]

  v4 = *(_QWORD *)(a1 + 88);
  v31 = 1;
  v29[0] = "_dfsarg_o";
  v5 = *(_QWORD *)(a1 + 96);
  v6 = (_QWORD *)a3[9];
  v30 = 3;
  v25 = v5;
  v7 = sub_BCB2E0(v6);
  v8 = (_BYTE *)sub_ACD640(v7, 0, 0);
  v9 = (_QWORD *)a3[9];
  v27 = v8;
  v10 = sub_BCB2E0(v9);
  v11 = sub_ACD640(v10, a2, 0);
  v12 = a3[10];
  v28 = v11;
  v13 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v12 + 64LL);
  if ( v13 == sub_920540 )
  {
    if ( sub_BCEA30(v4) || *(_BYTE *)v25 > 0x15u || v29 != sub_240D5B0(&v27, (__int64)v29) )
      goto LABEL_8;
    LOBYTE(v36) = 0;
    v14 = sub_AD9FD0(v4, (unsigned __int8 *)v25, (__int64 *)&v27, 2, 3u, (__int64)&v32, 0);
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
    v14 = v13(v12, v4, (_BYTE *)v25, &v27, 2, 3);
  }
  if ( v14 )
    return v14;
LABEL_8:
  v36 = 257;
  v14 = (__int64)sub_BD2C40(88, 3u);
  if ( !v14 )
    goto LABEL_11;
  v16 = *(_QWORD *)(v25 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 > 1 )
  {
    v21 = *((_QWORD *)v27 + 1);
    v22 = *(unsigned __int8 *)(v21 + 8);
    if ( v22 != 17 )
    {
      if ( v22 == 18 )
      {
LABEL_19:
        v24 = 1;
LABEL_21:
        BYTE4(v26) = v24;
        LODWORD(v26) = *(_DWORD *)(v21 + 32);
        v16 = sub_BCE1B0((__int64 *)v16, v26);
        goto LABEL_10;
      }
      v21 = *(_QWORD *)(v28 + 8);
      v23 = *(unsigned __int8 *)(v21 + 8);
      if ( v23 != 17 )
      {
        if ( v23 != 18 )
          goto LABEL_10;
        goto LABEL_19;
      }
    }
    v24 = 0;
    goto LABEL_21;
  }
LABEL_10:
  sub_B44260(v14, v16, 34, 3u, 0, 0);
  *(_QWORD *)(v14 + 72) = v4;
  *(_QWORD *)(v14 + 80) = sub_B4DC50(v4, (__int64)&v27, 2);
  sub_B4D9A0(v14, v25, (__int64 *)&v27, 2, (__int64)&v32);
LABEL_11:
  sub_B4DDE0(v14, 3);
  (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)a3[11] + 16LL))(
    a3[11],
    v14,
    v29,
    a3[7],
    a3[8]);
  v17 = *a3;
  v18 = *a3 + 16LL * *((unsigned int *)a3 + 2);
  while ( v18 != v17 )
  {
    v19 = *(_QWORD *)(v17 + 8);
    v20 = *(_DWORD *)v17;
    v17 += 16;
    sub_B99FD0(v14, v20, v19);
  }
  return v14;
}
