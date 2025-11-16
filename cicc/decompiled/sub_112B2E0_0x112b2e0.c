// Function: sub_112B2E0
// Address: 0x112b2e0
//
void *__fastcall sub_112B2E0(__int64 a1, int a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // rax
  void *v6; // r12
  bool v8; // al
  __int64 v9; // rcx
  __int64 v10; // rcx
  unsigned int v11; // edx
  const void *v12; // r8
  bool v13; // al
  bool v14; // r12
  unsigned int v15; // r15d
  const void *v16; // r13
  bool v17; // r12
  __int64 v18; // r13
  __int64 v19; // rax
  char *v20; // r12
  unsigned __int8 v21; // al
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rdi
  __int16 v27; // ax
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int16 v30; // si
  __int64 v31; // rdi
  __int64 v32; // rsi
  __int16 v33; // si
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int16 v36; // ax
  int v37; // eax
  __int64 v38; // r13
  __int64 v39; // rdx
  __int64 v40; // rdx
  __int64 v41; // rax
  const void *v42; // [rsp+0h] [rbp-F0h]
  unsigned int v43; // [rsp+Ch] [rbp-E4h]
  char v44; // [rsp+1Fh] [rbp-D1h] BYREF
  const void *v45; // [rsp+20h] [rbp-D0h] BYREF
  unsigned int v46; // [rsp+28h] [rbp-C8h]
  __int64 v47; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v48; // [rsp+38h] [rbp-B8h]
  __int64 v49; // [rsp+40h] [rbp-B0h]
  unsigned int v50; // [rsp+48h] [rbp-A8h]
  const void *v51; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v52; // [rsp+58h] [rbp-98h]
  const void *v53; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v54; // [rsp+68h] [rbp-88h]
  const void *v55; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v56; // [rsp+78h] [rbp-78h]
  const void *v57; // [rsp+80h] [rbp-70h] BYREF
  unsigned int v58; // [rsp+88h] [rbp-68h]
  const void *v59; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v60; // [rsp+98h] [rbp-58h]
  __int16 v61; // [rsp+B0h] [rbp-40h]

  sub_AB1A50((__int64)&v47, a2, a3);
  sub_AB2160((__int64)&v51, (__int64)&v47, *(_QWORD *)a1, 0);
  sub_ABB6C0((__int64)&v55, (__int64)&v47, *(_QWORD *)a1);
  if ( sub_AAF7D0((__int64)&v51) )
  {
    v4 = *(_QWORD *)(a1 + 16);
    v5 = sub_ACD720(*(__int64 **)(*(_QWORD *)(v4 + 32) + 72LL));
LABEL_3:
    v6 = sub_F162A0(v4, *(_QWORD *)(a1 + 8), v5);
    goto LABEL_4;
  }
  if ( sub_AAF7D0((__int64)&v55) )
  {
    v4 = *(_QWORD *)(a1 + 16);
    v5 = sub_ACD6D0(*(__int64 **)(*(_QWORD *)(v4 + 32) + 72LL));
    goto LABEL_3;
  }
  v8 = sub_9893F0(**(_DWORD **)(a1 + 24), **(_QWORD **)(a1 + 32), &v44);
  v9 = *(_QWORD *)(a1 + 8);
  if ( (*(_WORD *)(v9 + 2) & 0x3Fu) - 32 <= 1 )
    goto LABEL_78;
  v10 = *(_QWORD *)(v9 + 16);
  if ( v8 )
  {
    if ( v10 )
    {
      v19 = v10;
      while ( **(_BYTE **)(v19 + 24) != 31 )
      {
        v19 = *(_QWORD *)(v19 + 8);
        if ( !v19 )
          goto LABEL_54;
      }
      goto LABEL_78;
    }
    goto LABEL_28;
  }
  if ( !v10 )
    goto LABEL_28;
LABEL_54:
  if ( *(_QWORD *)(v10 + 8) )
    goto LABEL_28;
  v20 = *(char **)(v10 + 24);
  v21 = *v20;
  if ( (unsigned __int8)*v20 <= 0x1Cu )
    goto LABEL_28;
  if ( v21 == 85 )
  {
    v39 = *((_QWORD *)v20 - 4);
    if ( !v39 )
      goto LABEL_91;
    if ( !*(_BYTE *)v39
      && *(_QWORD *)(v39 + 24) == *((_QWORD *)v20 + 10)
      && (*(_BYTE *)(v39 + 33) & 0x20) != 0
      && *(_DWORD *)(v39 + 36) == 329 )
    {
      goto LABEL_78;
    }
    goto LABEL_88;
  }
  if ( v21 != 86 )
    goto LABEL_106;
  v22 = *((_QWORD *)v20 - 12);
  if ( *(_BYTE *)v22 != 82 )
    goto LABEL_28;
  v23 = *((_QWORD *)v20 - 8);
  v24 = *(_QWORD *)(v22 - 64);
  v25 = *((_QWORD *)v20 - 4);
  v26 = *(_QWORD *)(v22 - 32);
  if ( v23 != v24 || v25 != v26 )
  {
    if ( v23 != v26 || v25 != v24 )
      goto LABEL_65;
    v27 = *(_WORD *)(v22 + 2);
    if ( v23 == v24 )
      goto LABEL_62;
    if ( (unsigned int)sub_B52870(v27 & 0x3F) - 38 <= 1 )
      goto LABEL_78;
    v21 = *v20;
    if ( (unsigned __int8)*v20 <= 0x1Cu )
      goto LABEL_28;
    if ( v21 != 85 )
    {
      if ( v21 != 86 )
        goto LABEL_106;
      goto LABEL_63;
    }
    v39 = *((_QWORD *)v20 - 4);
LABEL_88:
    if ( v39
      && !*(_BYTE *)v39
      && *(_QWORD *)(v39 + 24) == *((_QWORD *)v20 + 10)
      && (*(_BYTE *)(v39 + 33) & 0x20) != 0
      && *(_DWORD *)(v39 + 36) == 330 )
    {
      goto LABEL_78;
    }
    goto LABEL_91;
  }
  v27 = *(_WORD *)(v22 + 2);
LABEL_62:
  if ( (v27 & 0x3Fu) - 38 <= 1 )
    goto LABEL_78;
LABEL_63:
  v22 = *((_QWORD *)v20 - 12);
  if ( *(_BYTE *)v22 != 82 )
    goto LABEL_28;
  v25 = *((_QWORD *)v20 - 4);
  v23 = *((_QWORD *)v20 - 8);
LABEL_65:
  v28 = *(_QWORD *)(v22 - 64);
  v29 = *(_QWORD *)(v22 - 32);
  if ( v23 == v28 && v25 == v29 )
  {
    v30 = *(_WORD *)(v22 + 2);
    goto LABEL_68;
  }
  if ( v23 == v29 && v25 == v28 )
  {
    v30 = *(_WORD *)(v22 + 2);
    if ( v23 == v28 )
    {
LABEL_68:
      if ( (v30 & 0x3Fu) - 40 <= 1 )
        goto LABEL_78;
      goto LABEL_69;
    }
    if ( (unsigned int)sub_B52870(*(_WORD *)(v22 + 2) & 0x3F) - 40 <= 1 )
      goto LABEL_78;
    v21 = *v20;
    if ( (unsigned __int8)*v20 <= 0x1Cu )
      goto LABEL_28;
LABEL_106:
    if ( v21 != 85 )
    {
      if ( v21 != 86 )
        goto LABEL_94;
      v22 = *((_QWORD *)v20 - 12);
      if ( *(_BYTE *)v22 != 82 )
        goto LABEL_28;
      v25 = *((_QWORD *)v20 - 4);
      v23 = *((_QWORD *)v20 - 8);
      goto LABEL_69;
    }
LABEL_91:
    v40 = *((_QWORD *)v20 - 4);
    if ( v40
      && !*(_BYTE *)v40
      && *(_QWORD *)(v40 + 24) == *((_QWORD *)v20 + 10)
      && (*(_BYTE *)(v40 + 33) & 0x20) != 0
      && *(_DWORD *)(v40 + 36) == 365 )
    {
      goto LABEL_78;
    }
    goto LABEL_94;
  }
LABEL_69:
  v31 = *(_QWORD *)(v22 - 64);
  v32 = *(_QWORD *)(v22 - 32);
  if ( v23 == v31 && v25 == v32 )
  {
    v33 = *(_WORD *)(v22 + 2);
LABEL_72:
    if ( (v33 & 0x3Fu) - 34 <= 1 )
      goto LABEL_78;
    goto LABEL_73;
  }
  if ( v23 != v32 || v25 != v31 )
    goto LABEL_73;
  v33 = *(_WORD *)(v22 + 2);
  if ( v23 == v31 )
    goto LABEL_72;
  if ( (unsigned int)sub_B52870(*(_WORD *)(v22 + 2) & 0x3F) - 34 <= 1 )
    goto LABEL_78;
  v21 = *v20;
  if ( (unsigned __int8)*v20 <= 0x1Cu )
    goto LABEL_28;
LABEL_94:
  if ( v21 == 85 )
  {
    v41 = *((_QWORD *)v20 - 4);
    if ( v41
      && !*(_BYTE *)v41
      && *(_QWORD *)(v41 + 24) == *((_QWORD *)v20 + 10)
      && (*(_BYTE *)(v41 + 33) & 0x20) != 0
      && *(_DWORD *)(v41 + 36) == 366 )
    {
      goto LABEL_78;
    }
    goto LABEL_28;
  }
  if ( v21 != 86 )
    goto LABEL_28;
  v22 = *((_QWORD *)v20 - 12);
  if ( *(_BYTE *)v22 != 82 )
    goto LABEL_28;
  v25 = *((_QWORD *)v20 - 4);
  v23 = *((_QWORD *)v20 - 8);
LABEL_73:
  v34 = *(_QWORD *)(v22 - 64);
  v35 = *(_QWORD *)(v22 - 32);
  if ( v23 == v34 && v25 == v35 )
  {
    v36 = *(_WORD *)(v22 + 2);
LABEL_76:
    v37 = v36 & 0x3F;
    goto LABEL_77;
  }
  if ( v23 != v35 || v25 != v34 )
    goto LABEL_28;
  v36 = *(_WORD *)(v22 + 2);
  if ( v23 == v34 )
    goto LABEL_76;
  v37 = sub_B52870(v36 & 0x3F);
LABEL_77:
  if ( (unsigned int)(v37 - 36) <= 1 )
    goto LABEL_78;
LABEL_28:
  v46 = v52;
  if ( v52 > 0x40 )
    sub_C43780((__int64)&v45, &v51);
  else
    v45 = v51;
  sub_C46A40((__int64)&v45, 1);
  v11 = v46;
  v12 = v45;
  v46 = 0;
  v60 = v11;
  v59 = v45;
  if ( v54 <= 0x40 )
  {
    v14 = v53 == v45;
  }
  else
  {
    v42 = v45;
    v43 = v11;
    v13 = sub_C43C50((__int64)&v53, &v59);
    v12 = v42;
    v11 = v43;
    v14 = v13;
  }
  if ( v11 > 0x40 )
  {
    if ( v12 )
    {
      j_j___libc_free_0_0(v12);
      if ( v46 > 0x40 )
      {
        if ( v45 )
          j_j___libc_free_0_0(v45);
      }
    }
  }
  if ( !v14 )
  {
    v46 = v56;
    if ( v56 > 0x40 )
      sub_C43780((__int64)&v45, &v55);
    else
      v45 = v55;
    sub_C46A40((__int64)&v45, 1);
    v15 = v46;
    v16 = v45;
    v46 = 0;
    v60 = v15;
    v59 = v45;
    if ( v58 <= 0x40 )
      v17 = v57 == v45;
    else
      v17 = sub_C43C50((__int64)&v57, &v59);
    if ( v15 > 0x40 )
    {
      if ( v16 )
      {
        j_j___libc_free_0_0(v16);
        if ( v46 > 0x40 )
        {
          if ( v45 )
            j_j___libc_free_0_0(v45);
        }
      }
    }
    if ( v17 )
    {
      v18 = sub_ACCFD0(*(__int64 **)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL) + 72LL), (__int64)&v55);
      v61 = 257;
      v6 = sub_BD2C40(72, unk_3F10FD0);
      if ( v6 )
        sub_1113300((__int64)v6, 33, **(_QWORD **)(a1 + 40), v18, (__int64)&v59);
      goto LABEL_4;
    }
LABEL_78:
    v6 = 0;
    goto LABEL_4;
  }
  v38 = sub_ACCFD0(*(__int64 **)(*(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL) + 72LL), (__int64)&v51);
  v61 = 257;
  v6 = sub_BD2C40(72, unk_3F10FD0);
  if ( v6 )
    sub_1113300((__int64)v6, 32, **(_QWORD **)(a1 + 40), v38, (__int64)&v59);
LABEL_4:
  if ( v58 > 0x40 && v57 )
    j_j___libc_free_0_0(v57);
  if ( v56 > 0x40 && v55 )
    j_j___libc_free_0_0(v55);
  if ( v54 > 0x40 && v53 )
    j_j___libc_free_0_0(v53);
  if ( v52 > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( v48 > 0x40 && v47 )
    j_j___libc_free_0_0(v47);
  return v6;
}
