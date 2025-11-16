// Function: sub_9213A0
// Address: 0x9213a0
//
__int64 __fastcall sub_9213A0(
        unsigned int **a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        unsigned int a5,
        __int64 a6,
        unsigned int a7)
{
  __int64 v11; // rax
  _BYTE *v12; // rax
  unsigned int *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int *v16; // rdi
  __int64 (__fastcall *v17)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v18; // rax
  _BYTE **v19; // rcx
  __int64 v20; // r13
  __int64 v22; // r10
  __int64 v23; // r12
  unsigned int *v24; // rbx
  unsigned int *v25; // r12
  __int64 v26; // rdx
  __int64 v27; // rsi
  __int64 v28; // rax
  int v29; // edx
  int v30; // edx
  char v31; // dl
  __int64 v33; // [rsp+18h] [rbp-78h]
  _BYTE *v34; // [rsp+20h] [rbp-70h] BYREF
  __int64 v35; // [rsp+28h] [rbp-68h]
  __int64 v36; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v37; // [rsp+38h] [rbp-58h]
  __int64 v38; // [rsp+40h] [rbp-50h]
  unsigned int v39; // [rsp+48h] [rbp-48h]
  __int16 v40; // [rsp+50h] [rbp-40h]

  v11 = sub_BCB2D0(a1[9]);
  v12 = (_BYTE *)sub_ACD640(v11, a4, 0);
  v13 = a1[9];
  v34 = v12;
  v14 = sub_BCB2D0(v13);
  v15 = sub_ACD640(v14, a5, 0);
  v16 = a1[10];
  v35 = v15;
  v17 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v16 + 64LL);
  if ( v17 == sub_920540 )
  {
    if ( (unsigned __int8)sub_BCEA30(a2) )
      goto LABEL_8;
    if ( *(_BYTE *)a3 > 0x15u )
      goto LABEL_8;
    v18 = sub_920370(&v34, (__int64)&v36);
    if ( v18 != v19 )
      goto LABEL_8;
    LOBYTE(v40) = 0;
    v20 = sub_AD9FD0(a2, a3, (unsigned int)&v34, 2, a7, (_DWORD)v18, 0);
    if ( (_BYTE)v40 )
    {
      LOBYTE(v40) = 0;
      if ( v39 > 0x40 && v38 )
        j_j___libc_free_0_0(v38);
      if ( v37 > 0x40 && v36 )
        j_j___libc_free_0_0(v36);
    }
  }
  else
  {
    v20 = v17((__int64)v16, a2, (_BYTE *)a3, &v34, 2, a7);
  }
  if ( v20 )
    return v20;
LABEL_8:
  v40 = 257;
  v20 = sub_BD2C40(88, 3);
  if ( !v20 )
    goto LABEL_11;
  v22 = *(_QWORD *)(a3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 > 1 )
  {
    v28 = *((_QWORD *)v34 + 1);
    v29 = *(unsigned __int8 *)(v28 + 8);
    if ( v29 != 17 )
    {
      if ( v29 == 18 )
      {
LABEL_19:
        v31 = 1;
LABEL_21:
        BYTE4(v33) = v31;
        LODWORD(v33) = *(_DWORD *)(v28 + 32);
        v22 = sub_BCE1B0(v22, v33);
        goto LABEL_10;
      }
      v28 = *(_QWORD *)(v35 + 8);
      v30 = *(unsigned __int8 *)(v28 + 8);
      if ( v30 != 17 )
      {
        if ( v30 != 18 )
          goto LABEL_10;
        goto LABEL_19;
      }
    }
    v31 = 0;
    goto LABEL_21;
  }
LABEL_10:
  sub_B44260(v20, v22, 34, 3, 0, 0);
  *(_QWORD *)(v20 + 72) = a2;
  *(_QWORD *)(v20 + 80) = sub_B4DC50(a2, &v34, 2);
  sub_B4D9A0(v20, a3, &v34, 2, &v36);
LABEL_11:
  sub_B4DDE0(v20, a7);
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v20,
    a6,
    a1[7],
    a1[8]);
  v23 = 4LL * *((unsigned int *)a1 + 2);
  v24 = *a1;
  v25 = &v24[v23];
  while ( v25 != v24 )
  {
    v26 = *((_QWORD *)v24 + 1);
    v27 = *v24;
    v24 += 4;
    sub_B99FD0(v20, v27, v26);
  }
  return v20;
}
