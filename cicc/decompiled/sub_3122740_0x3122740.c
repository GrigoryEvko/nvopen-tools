// Function: sub_3122740
// Address: 0x3122740
//
__int64 __fastcall sub_3122740(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  __int64 v19; // r15
  __int64 v21; // r11
  __int64 v22; // rbx
  __int64 v23; // r12
  __int64 v24; // rdx
  unsigned int v25; // esi
  __int64 v26; // rax
  int v27; // edx
  int v28; // edx
  char v29; // dl
  __int64 v31; // [rsp+18h] [rbp-78h]
  _BYTE *v32; // [rsp+20h] [rbp-70h] BYREF
  __int64 v33; // [rsp+28h] [rbp-68h]
  unsigned __int64 v34; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v35; // [rsp+38h] [rbp-58h]
  unsigned __int64 v36; // [rsp+40h] [rbp-50h]
  unsigned int v37; // [rsp+48h] [rbp-48h]
  __int16 v38; // [rsp+50h] [rbp-40h]

  v10 = sub_BCB2E0((_QWORD *)a1[9]);
  v11 = (_BYTE *)sub_ACD640(v10, a4, 0);
  v12 = (_QWORD *)a1[9];
  v32 = v11;
  v13 = sub_BCB2E0(v12);
  v14 = sub_ACD640(v13, a5, 0);
  v15 = a1[10];
  v33 = v14;
  v16 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v15 + 64LL);
  if ( v16 == sub_920540 )
  {
    if ( sub_BCEA30(a2) )
      goto LABEL_8;
    if ( *(_BYTE *)a3 > 0x15u )
      goto LABEL_8;
    v17 = sub_3120B80(&v32, (__int64)&v34);
    if ( v17 != v18 )
      goto LABEL_8;
    LOBYTE(v38) = 0;
    v19 = sub_AD9FD0(a2, (unsigned __int8 *)a3, (__int64 *)&v32, 2, 3u, (__int64)v17, 0);
    if ( (_BYTE)v38 )
    {
      LOBYTE(v38) = 0;
      if ( v37 > 0x40 && v36 )
        j_j___libc_free_0_0(v36);
      if ( v35 > 0x40 && v34 )
        j_j___libc_free_0_0(v34);
    }
  }
  else
  {
    v19 = v16(v15, a2, (_BYTE *)a3, &v32, 2, 3);
  }
  if ( v19 )
    return v19;
LABEL_8:
  v38 = 257;
  v19 = (__int64)sub_BD2C40(88, 3u);
  if ( !v19 )
    goto LABEL_11;
  v21 = *(_QWORD *)(a3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v21 + 8) - 17 > 1 )
  {
    v26 = *((_QWORD *)v32 + 1);
    v27 = *(unsigned __int8 *)(v26 + 8);
    if ( v27 != 17 )
    {
      if ( v27 == 18 )
      {
LABEL_19:
        v29 = 1;
LABEL_21:
        BYTE4(v31) = v29;
        LODWORD(v31) = *(_DWORD *)(v26 + 32);
        v21 = sub_BCE1B0((__int64 *)v21, v31);
        goto LABEL_10;
      }
      v26 = *(_QWORD *)(v33 + 8);
      v28 = *(unsigned __int8 *)(v26 + 8);
      if ( v28 != 17 )
      {
        if ( v28 != 18 )
          goto LABEL_10;
        goto LABEL_19;
      }
    }
    v29 = 0;
    goto LABEL_21;
  }
LABEL_10:
  sub_B44260(v19, v21, 34, 3u, 0, 0);
  *(_QWORD *)(v19 + 72) = a2;
  *(_QWORD *)(v19 + 80) = sub_B4DC50(a2, (__int64)&v32, 2);
  sub_B4D9A0(v19, a3, (__int64 *)&v32, 2, (__int64)&v34);
LABEL_11:
  sub_B4DDE0(v19, 3);
  (*(void (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v19,
    a6,
    a1[7],
    a1[8]);
  v22 = *a1;
  v23 = *a1 + 16LL * *((unsigned int *)a1 + 2);
  while ( v23 != v22 )
  {
    v24 = *(_QWORD *)(v22 + 8);
    v25 = *(_DWORD *)v22;
    v22 += 16;
    sub_B99FD0(v19, v25, v24);
  }
  return v19;
}
