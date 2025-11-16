// Function: sub_94B060
// Address: 0x94b060
//
__int64 __fastcall sub_94B060(unsigned int **a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5)
{
  __int64 v8; // rax
  _BYTE *v9; // rax
  unsigned int *v10; // rdi
  __int64 (__fastcall *v11)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v12; // rax
  _BYTE **v13; // rcx
  __int64 v14; // r15
  __int64 v16; // r11
  unsigned int *v17; // rbx
  __int64 v18; // r12
  __int64 v19; // rdx
  __int64 v20; // rsi
  __int64 v21; // rdx
  int v22; // eax
  char v23; // al
  int v24; // edx
  _BYTE *v26; // [rsp+10h] [rbp-70h] BYREF
  __int64 v27; // [rsp+18h] [rbp-68h] BYREF
  __int64 v28; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v29; // [rsp+28h] [rbp-58h]
  __int64 v30; // [rsp+30h] [rbp-50h]
  unsigned int v31; // [rsp+38h] [rbp-48h]
  __int16 v32; // [rsp+40h] [rbp-40h]

  v8 = sub_BCB2D0(a1[9]);
  v9 = (_BYTE *)sub_ACD640(v8, a4, 0);
  v10 = a1[10];
  v26 = v9;
  v11 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v10 + 64LL);
  if ( v11 == sub_920540 )
  {
    if ( (unsigned __int8)sub_BCEA30(a2) )
      goto LABEL_8;
    if ( *(_BYTE *)a3 > 0x15u )
      goto LABEL_8;
    v12 = sub_9485E0(&v26, (__int64)&v27);
    if ( v13 != v12 )
      goto LABEL_8;
    LOBYTE(v32) = 0;
    v14 = sub_AD9FD0(a2, a3, (unsigned int)&v26, 1, 0, (unsigned int)&v28, 0);
    if ( (_BYTE)v32 )
    {
      LOBYTE(v32) = 0;
      if ( v31 > 0x40 && v30 )
        j_j___libc_free_0_0(v30);
      if ( v29 > 0x40 && v28 )
        j_j___libc_free_0_0(v28);
    }
  }
  else
  {
    v14 = v11((__int64)v10, a2, (_BYTE *)a3, &v26, 1, 0);
  }
  if ( v14 )
    return v14;
LABEL_8:
  v32 = 257;
  v14 = sub_BD2C40(88, 2);
  if ( !v14 )
    goto LABEL_11;
  v16 = *(_QWORD *)(a3 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v16 + 8) - 17 > 1 )
  {
    v21 = *((_QWORD *)v26 + 1);
    v22 = *(unsigned __int8 *)(v21 + 8);
    if ( v22 == 17 )
    {
      v23 = 0;
    }
    else
    {
      if ( v22 != 18 )
        goto LABEL_10;
      v23 = 1;
    }
    v24 = *(_DWORD *)(v21 + 32);
    BYTE4(v27) = v23;
    LODWORD(v27) = v24;
    v16 = sub_BCE1B0(v16, v27);
  }
LABEL_10:
  sub_B44260(v14, v16, 34, 2, 0, 0);
  *(_QWORD *)(v14 + 72) = a2;
  *(_QWORD *)(v14 + 80) = sub_B4DC50(a2, &v26, 1);
  sub_B4D9A0(v14, a3, &v26, 1, &v28);
LABEL_11:
  (*(void (__fastcall **)(unsigned int *, __int64, __int64, unsigned int *, unsigned int *))(*(_QWORD *)a1[11] + 16LL))(
    a1[11],
    v14,
    a5,
    a1[7],
    a1[8]);
  v17 = *a1;
  v18 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  while ( (unsigned int *)v18 != v17 )
  {
    v19 = *((_QWORD *)v17 + 1);
    v20 = *v17;
    v17 += 4;
    sub_B99FD0(v14, v20, v19);
  }
  return v14;
}
