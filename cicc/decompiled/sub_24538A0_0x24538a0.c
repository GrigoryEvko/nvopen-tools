// Function: sub_24538A0
// Address: 0x24538a0
//
__int64 __fastcall sub_24538A0(__int64 *a1)
{
  __int64 v1; // rax
  __int64 *v2; // r13
  _QWORD *v3; // r12
  bool v4; // zf
  char v5; // al
  char v6; // al
  _BYTE *v7; // rsi
  __int64 v8; // r14
  unsigned __int64 v9; // rax
  __int64 v10; // r14
  char v11; // al
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rax
  _QWORD *v19; // rax
  __int64 v20; // r15
  unsigned __int64 v21; // r13
  const char *v22; // r12
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // r12
  _QWORD *v26; // rax
  __int64 v27; // r13
  unsigned __int64 v28; // r15
  const char *v29; // r12
  __int64 v30; // rdx
  unsigned int v31; // esi
  _BYTE *v32; // rsi
  char v34; // al
  __int64 v35; // r15
  char *v36; // rax
  size_t v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rcx
  char v41; // [rsp+8h] [rbp-148h]
  _BYTE v42[32]; // [rsp+30h] [rbp-120h] BYREF
  __int16 v43; // [rsp+50h] [rbp-100h]
  __int64 v44[4]; // [rsp+60h] [rbp-F0h] BYREF
  __int16 v45; // [rsp+80h] [rbp-D0h]
  const char *v46; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v47; // [rsp+98h] [rbp-B8h]
  _BYTE v48[16]; // [rsp+A0h] [rbp-B0h] BYREF
  __int16 v49; // [rsp+B0h] [rbp-A0h]
  __int64 v50; // [rsp+C0h] [rbp-90h]
  __int64 v51; // [rsp+C8h] [rbp-88h]
  __int64 v52; // [rsp+D0h] [rbp-80h]
  __int64 v53; // [rsp+D8h] [rbp-78h]
  void **v54; // [rsp+E0h] [rbp-70h]
  void **v55; // [rsp+E8h] [rbp-68h]
  __int64 v56; // [rsp+F0h] [rbp-60h]
  int v57; // [rsp+F8h] [rbp-58h]
  __int16 v58; // [rsp+FCh] [rbp-54h]
  char v59; // [rsp+FEh] [rbp-52h]
  __int64 v60; // [rsp+100h] [rbp-50h]
  __int64 v61; // [rsp+108h] [rbp-48h]
  void *v62; // [rsp+110h] [rbp-40h] BYREF
  void *v63; // [rsp+118h] [rbp-38h] BYREF

  v1 = sub_BCB2D0(*(_QWORD **)*a1);
  v49 = 261;
  v2 = (__int64 *)v1;
  v47 = 22;
  v46 = "__llvm_profile_runtime";
  BYTE4(v44[0]) = 0;
  v3 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v3 )
    sub_B30000((__int64)v3, *a1, v2, 0, 0, 0, (__int64)&v46, 0, 0, v44[0], 0);
  v4 = !sub_ED1700(*a1);
  v5 = *((_BYTE *)v3 + 32);
  if ( v4 )
  {
    v34 = v5 & 0xCF | 0x10;
    *((_BYTE *)v3 + 32) = v34;
    if ( (v34 & 0xF) == 9 )
      goto LABEL_6;
    goto LABEL_5;
  }
  v6 = v5 & 0xCF | 0x20;
  *((_BYTE *)v3 + 32) = v6;
  if ( (v6 & 0xF) != 9 )
LABEL_5:
    *((_BYTE *)v3 + 33) |= 0x40u;
LABEL_6:
  if ( *((_DWORD *)a1 + 25) != 3
    || *((_DWORD *)a1 + 20) == 39 && *((_DWORD *)a1 + 22) == 3 && (unsigned int)(*((_DWORD *)a1 + 23) - 24) <= 1 )
  {
    v8 = *a1;
    v49 = 261;
    v46 = "__llvm_profile_runtime_user";
    v47 = 27;
    v9 = sub_BCF640(v2, 0);
    v10 = sub_B2C660(v9, 3, (__int64)&v46, v8);
    sub_B2CD30(v10, 31);
    if ( *((_BYTE *)a1 + 8) )
      sub_B2CD30(v10, 35);
    v11 = *(_BYTE *)(v10 + 32) & 0xCF | 0x10;
    *(_BYTE *)(v10 + 32) = v11;
    if ( (v11 & 0xF) != 9 )
      *(_BYTE *)(v10 + 33) |= 0x40u;
    v12 = *((unsigned int *)a1 + 25);
    if ( (unsigned int)v12 > 8 || (v13 = 292, !_bittest64(&v13, v12)) )
    {
      v35 = *a1;
      v36 = (char *)sub_BD5D20(v10);
      v38 = sub_BAA410(v35, v36, v37);
      sub_B2F990(v10, v38, v39, v40);
    }
    v14 = (__int64 *)*a1;
    v45 = 257;
    v15 = *v14;
    v16 = sub_22077B0(0x50u);
    if ( v16 )
      sub_AA4D50(v16, v15, (__int64)v44, v10, 0);
    v17 = sub_AA48A0(v16);
    v59 = 7;
    v53 = v17;
    v54 = &v62;
    v55 = &v63;
    v46 = v48;
    v62 = &unk_49DA100;
    v58 = 512;
    v47 = 0x200000000LL;
    v63 = &unk_49DA0B0;
    v43 = 257;
    LOWORD(v52) = 0;
    v50 = v16;
    v56 = 0;
    v57 = 0;
    v60 = 0;
    v61 = 0;
    v51 = v16 + 48;
    v18 = sub_AA4E30(v16);
    v41 = sub_AE5020(v18, (__int64)v2);
    v45 = 257;
    v19 = sub_BD2C40(80, unk_3F10A14);
    v20 = (__int64)v19;
    if ( v19 )
      sub_B4D190((__int64)v19, (__int64)v2, (__int64)v3, (__int64)v44, 0, v41, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _BYTE *, __int64, __int64))*v55 + 2))(v55, v20, v42, v51, v52);
    v21 = (unsigned __int64)v46;
    v22 = &v46[16 * (unsigned int)v47];
    if ( v46 != v22 )
    {
      do
      {
        v23 = *(_QWORD *)(v21 + 8);
        v24 = *(_DWORD *)v21;
        v21 += 16LL;
        sub_B99FD0(v20, v24, v23);
      }
      while ( v22 != (const char *)v21 );
    }
    v45 = 257;
    v25 = v53;
    v26 = sub_BD2C40(72, v20 != 0);
    v27 = (__int64)v26;
    if ( v26 )
      sub_B4BB80((__int64)v26, v25, v20, v20 != 0, 0, 0);
    (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v55 + 2))(v55, v27, v44, v51, v52);
    v28 = (unsigned __int64)v46;
    v29 = &v46[16 * (unsigned int)v47];
    if ( v46 != v29 )
    {
      do
      {
        v30 = *(_QWORD *)(v28 + 8);
        v31 = *(_DWORD *)v28;
        v28 += 16LL;
        sub_B99FD0(v27, v31, v30);
      }
      while ( v29 != (const char *)v28 );
    }
    v44[0] = v10;
    v32 = (_BYTE *)a1[32];
    if ( v32 == (_BYTE *)a1[33] )
    {
      sub_E48660((__int64)(a1 + 31), v32, v44);
    }
    else
    {
      if ( v32 )
      {
        *(_QWORD *)v32 = v10;
        v32 = (_BYTE *)a1[32];
      }
      a1[32] = (__int64)(v32 + 8);
    }
    nullsub_61();
    v62 = &unk_49DA100;
    nullsub_63();
    if ( v46 != v48 )
      _libc_free((unsigned __int64)v46);
  }
  else
  {
    v46 = (const char *)v3;
    v7 = (_BYTE *)a1[32];
    if ( v7 == (_BYTE *)a1[33] )
    {
      sub_E48660((__int64)(a1 + 31), v7, &v46);
    }
    else
    {
      if ( v7 )
      {
        *(_QWORD *)v7 = v3;
        v7 = (_BYTE *)a1[32];
      }
      a1[32] = (__int64)(v7 + 8);
    }
  }
  return 1;
}
