// Function: sub_24DBB60
// Address: 0x24dbb60
//
__int64 __fastcall sub_24DBB60(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4, unsigned int a5, __int64 a6)
{
  __int64 v10; // rax
  _BYTE *v11; // rax
  _QWORD *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 (__fastcall *v16)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v17; // r9
  _BYTE **v18; // rcx
  __int64 v19; // r13
  __int64 v21; // r10
  __int64 v22; // r12
  __int64 v23; // rbx
  __int64 v24; // r12
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 v27; // rax
  int v28; // edx
  int v29; // edx
  char v30; // dl
  __int64 v32; // [rsp+18h] [rbp-78h]
  _BYTE *v33; // [rsp+20h] [rbp-70h] BYREF
  __int64 v34; // [rsp+28h] [rbp-68h]
  unsigned __int64 v35; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v36; // [rsp+38h] [rbp-58h]
  unsigned __int64 v37; // [rsp+40h] [rbp-50h]
  unsigned int v38; // [rsp+48h] [rbp-48h]
  __int16 v39; // [rsp+50h] [rbp-40h]

  v10 = sub_BCB2D0((_QWORD *)a1[9]);
  v11 = (_BYTE *)sub_ACD640(v10, a4, 0);
  v12 = (_QWORD *)a1[9];
  v33 = v11;
  v13 = sub_BCB2D0(v12);
  v14 = sub_ACD640(v13, a5, 0);
  v15 = a1[10];
  v34 = v14;
  v16 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v15 + 64LL);
  if ( v16 == sub_920540 )
  {
    if ( sub_BCEA30(a2) )
      goto LABEL_8;
    if ( *(_BYTE *)a3 > 0x15u )
      goto LABEL_8;
    v17 = sub_24DBAA0(&v33, (__int64)&v35);
    if ( v17 != v18 )
      goto LABEL_8;
    LOBYTE(v39) = 0;
    v19 = sub_AD9FD0(a2, (unsigned __int8 *)a3, (__int64 *)&v33, 2, 3u, (__int64)v17, 0);
    if ( (_BYTE)v39 )
    {
      LOBYTE(v39) = 0;
      if ( v38 > 0x40 && v37 )
        j_j___libc_free_0_0(v37);
      if ( v36 > 0x40 && v35 )
        j_j___libc_free_0_0(v35);
    }
  }
  else
  {
    v19 = v16(v15, a2, (_BYTE *)a3, &v33, 2, 3);
  }
  if ( v19 )
    return v19;
LABEL_8:
  v39 = 257;
  v19 = (__int64)sub_BD2C40(88, 3u);
  if ( !v19 )
    goto LABEL_11;
  v21 = *(_QWORD *)(a3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 > 1 )
  {
    v27 = *((_QWORD *)v33 + 1);
    v28 = *(unsigned __int8 *)(v27 + 8);
    if ( v28 != 17 )
    {
      if ( v28 == 18 )
      {
LABEL_19:
        v30 = 1;
LABEL_21:
        BYTE4(v32) = v30;
        LODWORD(v32) = *(_DWORD *)(v27 + 32);
        v21 = sub_BCE1B0((__int64 *)v21, v32);
        goto LABEL_10;
      }
      v27 = *(_QWORD *)(v34 + 8);
      v29 = *(unsigned __int8 *)(v27 + 8);
      if ( v29 != 17 )
      {
        if ( v29 != 18 )
          goto LABEL_10;
        goto LABEL_19;
      }
    }
    v30 = 0;
    goto LABEL_21;
  }
LABEL_10:
  sub_B44260(v19, v21, 34, 3u, 0, 0);
  *(_QWORD *)(v19 + 72) = a2;
  *(_QWORD *)(v19 + 80) = sub_B4DC50(a2, (__int64)&v33, 2);
  sub_B4D9A0(v19, a3, (__int64 *)&v33, 2, (__int64)&v35);
LABEL_11:
  sub_B4DDE0(v19, 3);
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v19,
    a6,
    a1[7],
    a1[8]);
  v22 = 16LL * *((unsigned int *)a1 + 2);
  v23 = *a1;
  v24 = v23 + v22;
  while ( v24 != v23 )
  {
    v25 = *(_QWORD *)(v23 + 8);
    v26 = *(_DWORD *)v23;
    v23 += 16;
    sub_B99FD0(v19, v26, v25);
  }
  return v19;
}
