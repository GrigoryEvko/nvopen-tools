// Function: sub_399D1D0
// Address: 0x399d1d0
//
void __fastcall sub_399D1D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  const char *v6; // rbx
  const char *v7; // r12
  unsigned int v8; // esi
  const char *v9; // r13
  int v10; // r11d
  const char **v11; // r14
  unsigned int v12; // ecx
  const char **v13; // rax
  const char *v14; // r10
  __int64 v15; // rdi
  char *v16; // rdi
  _BYTE *v17; // rsi
  unsigned __int64 *v18; // r13
  unsigned int v19; // eax
  int v20; // ecx
  const char *v21; // rdx
  int v22; // r9d
  const char **v23; // r8
  char *v24; // rsi
  unsigned __int64 v25; // rax
  __int64 v26; // r13
  __int64 **v27; // rbx
  __int64 v28; // rdi
  void (*v29)(); // rax
  __int64 v30; // rdi
  void (*v31)(); // rax
  __int64 v32; // rdi
  void (*v33)(); // rax
  __int64 v34; // r12
  __int64 v35; // r14
  __int64 *v36; // rdi
  void (*v37)(); // rax
  __int64 *v38; // rdi
  void (*v39)(); // rax
  __int64 v40; // rdi
  void (*v41)(); // rax
  __int64 v42; // rdi
  void (*v43)(); // rax
  char *v44; // rbx
  unsigned __int64 v45; // r12
  unsigned __int64 v46; // rdi
  __int64 **v47; // rdx
  __int64 v48; // rdi
  void (*v49)(); // rax
  __int64 *v50; // rdi
  void (*v51)(); // rax
  int v52; // r9d
  unsigned int v53; // edx
  const char *v54; // rax
  char *v55; // [rsp+0h] [rbp-C0h]
  __int64 v56; // [rsp+8h] [rbp-B8h]
  char *v57; // [rsp+10h] [rbp-B0h]
  __int64 **v58; // [rsp+18h] [rbp-A8h]
  unsigned int v59; // [rsp+20h] [rbp-A0h]
  char v60; // [rsp+25h] [rbp-9Bh]
  unsigned __int16 v61; // [rsp+26h] [rbp-9Ah]
  const char *v63; // [rsp+30h] [rbp-90h] BYREF
  unsigned __int64 v64; // [rsp+38h] [rbp-88h]
  __int64 v65; // [rsp+40h] [rbp-80h]
  __int64 v66; // [rsp+48h] [rbp-78h]
  __int64 v67; // [rsp+50h] [rbp-70h] BYREF
  unsigned __int64 v68; // [rsp+58h] [rbp-68h]
  __int64 v69; // [rsp+60h] [rbp-60h]
  unsigned int v70; // [rsp+68h] [rbp-58h]
  char *v71; // [rsp+70h] [rbp-50h] BYREF
  char *v72; // [rsp+78h] [rbp-48h]
  char *v73; // [rsp+80h] [rbp-40h]

  v61 = sub_398C0A0(*(_QWORD *)(a2 + 200));
  (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 256) + 176LL))(
    *(_QWORD *)(a1 + 256),
    *(_QWORD *)a3,
    0);
  v5 = *(_QWORD *)(a1 + 240);
  v68 = 0;
  v69 = 0;
  v6 = *(const char **)(a3 + 8);
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  LODWORD(v5) = *(_DWORD *)(v5 + 8);
  v67 = 0;
  v59 = v5;
  v7 = &v6[16 * *(unsigned int *)(a3 + 16)];
  if ( v6 == v7 )
    goto LABEL_69;
  do
  {
    while ( 1 )
    {
      v18 = *(unsigned __int64 **)v6;
      if ( (**(_QWORD **)v6 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v9 = *(const char **)((**(_QWORD **)v6 & 0xFFFFFFFFFFFFFFF8LL) + 24);
      }
      else
      {
        if ( (*((_BYTE *)v18 + 9) & 0xC) != 8
          || (*((_BYTE *)v18 + 8) |= 4u, v25 = (unsigned __int64)sub_38CE440(v18[3]), *v18 = v25 | *v18 & 7, !v25) )
        {
          v8 = v70;
          v9 = 0;
          if ( !v70 )
            goto LABEL_14;
          goto LABEL_5;
        }
        v9 = *(const char **)(v25 + 24);
      }
      v8 = v70;
      if ( !v70 )
      {
LABEL_14:
        ++v67;
        goto LABEL_15;
      }
LABEL_5:
      v10 = 1;
      v11 = 0;
      v12 = (v8 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v13 = (const char **)(v68 + 16LL * v12);
      v14 = *v13;
      if ( v9 == *v13 )
      {
LABEL_6:
        v15 = *((unsigned int *)v13 + 2);
        goto LABEL_7;
      }
      while ( v14 != (const char *)-8LL )
      {
        if ( !v11 && v14 == (const char *)-16LL )
          v11 = v13;
        v12 = (v8 - 1) & (v10 + v12);
        v13 = (const char **)(v68 + 16LL * v12);
        v14 = *v13;
        if ( v9 == *v13 )
          goto LABEL_6;
        ++v10;
      }
      if ( !v11 )
        v11 = v13;
      ++v67;
      v20 = v69 + 1;
      if ( 4 * ((int)v69 + 1) < 3 * v8 )
      {
        if ( v8 - HIDWORD(v69) - v20 > v8 >> 3 )
          goto LABEL_32;
        sub_3915FD0((__int64)&v67, v8);
        if ( !v70 )
        {
LABEL_112:
          LODWORD(v69) = v69 + 1;
          BUG();
        }
        v23 = 0;
        v52 = 1;
        v53 = (v70 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v20 = v69 + 1;
        v11 = (const char **)(v68 + 16LL * v53);
        v54 = *v11;
        if ( v9 == *v11 )
          goto LABEL_32;
        while ( v54 != (const char *)-8LL )
        {
          if ( v54 == (const char *)-16LL && !v23 )
            v23 = v11;
          v53 = (v70 - 1) & (v53 + v52);
          v11 = (const char **)(v68 + 16LL * v53);
          v54 = *v11;
          if ( v9 == *v11 )
            goto LABEL_32;
          ++v52;
        }
        goto LABEL_19;
      }
LABEL_15:
      sub_3915FD0((__int64)&v67, 2 * v8);
      if ( !v70 )
        goto LABEL_112;
      v19 = (v70 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v20 = v69 + 1;
      v11 = (const char **)(v68 + 16LL * v19);
      v21 = *v11;
      if ( v9 == *v11 )
        goto LABEL_32;
      v22 = 1;
      v23 = 0;
      while ( v21 != (const char *)-8LL )
      {
        if ( v21 == (const char *)-16LL && !v23 )
          v23 = v11;
        v19 = (v70 - 1) & (v19 + v22);
        v11 = (const char **)(v68 + 16LL * v19);
        v21 = *v11;
        if ( v9 == *v11 )
          goto LABEL_32;
        ++v22;
      }
LABEL_19:
      if ( v23 )
        v11 = v23;
LABEL_32:
      LODWORD(v69) = v20;
      if ( *v11 != (const char *)-8LL )
        --HIDWORD(v69);
      *v11 = v9;
      *((_DWORD *)v11 + 2) = 0;
      v24 = v72;
      v63 = v9;
      v64 = 0;
      v65 = 0;
      v66 = 0;
      if ( v72 == v73 )
      {
        sub_398F6D0((unsigned __int64 *)&v71, v72, &v63);
        if ( v64 )
          j_j___libc_free_0(v64);
      }
      else
      {
        if ( v72 )
        {
          *(_QWORD *)v72 = v9;
          *((_QWORD *)v24 + 1) = v64;
          *((_QWORD *)v24 + 2) = v65;
          *((_QWORD *)v24 + 3) = v66;
          v24 = v72;
        }
        v72 = v24 + 32;
      }
      v15 = (unsigned int)((v72 - v71) >> 5) - 1;
      *((_DWORD *)v11 + 2) = v15;
LABEL_7:
      v16 = &v71[32 * v15];
      v63 = v6;
      v17 = (_BYTE *)*((_QWORD *)v16 + 2);
      if ( v17 != *((_BYTE **)v16 + 3) )
        break;
      v6 += 16;
      sub_398F010((__int64)(v16 + 8), v17, &v63);
      if ( v7 == v6 )
        goto LABEL_42;
    }
    if ( v17 )
    {
      *(_QWORD *)v17 = v6;
      v17 = (_BYTE *)*((_QWORD *)v16 + 2);
    }
    v6 += 16;
    *((_QWORD *)v16 + 2) = v17 + 8;
  }
  while ( v7 != v6 );
LABEL_42:
  v55 = v72;
  v56 = *(_QWORD *)(a2 + 856);
  if ( v71 == v72 )
    goto LABEL_69;
  v57 = v71;
  v60 = 0;
  while ( 2 )
  {
    if ( v56 || (v47 = (__int64 **)*((_QWORD *)v57 + 1), *((_QWORD *)v57 + 2) - (_QWORD)v47 <= 8u) )
    {
      if ( v61 <= 4u )
        goto LABEL_46;
      goto LABEL_83;
    }
    if ( byte_5057340 )
    {
      v26 = **v47;
      if ( v61 > 4u )
      {
LABEL_88:
        v48 = *(_QWORD *)(a1 + 256);
        v49 = *(void (**)())(*(_QWORD *)v48 + 104LL);
        v63 = "DW_RLE_base_address";
        LOWORD(v65) = 259;
        if ( v49 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, const char **, __int64))v49)(v48, &v63, 1);
          v48 = *(_QWORD *)(a1 + 256);
        }
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v48 + 424LL))(v48, 5, 1);
      }
      else
      {
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 424LL))(
          *(_QWORD *)(a1 + 256),
          -1,
          v59);
      }
      v50 = *(__int64 **)(a1 + 256);
      v51 = *(void (**)())(*v50 + 104);
      v63 = "  base address";
      LOWORD(v65) = 259;
      if ( v51 != nullsub_580 )
      {
        ((void (__fastcall *)(__int64 *, const char **, __int64))v51)(v50, &v63, 1);
        v50 = *(__int64 **)(a1 + 256);
      }
      sub_38DDC80(v50, v26, v59, 0);
      v60 = 1;
    }
    else
    {
      if ( v61 > 4u )
      {
        v26 = **v47;
        goto LABEL_88;
      }
LABEL_46:
      if ( v60 )
      {
        (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 256) + 424LL))(
          *(_QWORD *)(a1 + 256),
          -1,
          v59);
        (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 256) + 424LL))(
          *(_QWORD *)(a1 + 256),
          0,
          v59);
        v60 = 0;
        v26 = v56;
        goto LABEL_48;
      }
LABEL_83:
      v26 = v56;
    }
LABEL_48:
    v27 = (__int64 **)*((_QWORD *)v57 + 1);
    v58 = (__int64 **)*((_QWORD *)v57 + 2);
    if ( v58 != v27 )
    {
      while ( 1 )
      {
        v34 = **v27;
        v35 = (*v27)[1];
        if ( v26 )
          break;
        v36 = *(__int64 **)(a1 + 256);
        if ( v61 <= 4u )
        {
          sub_38DDC80(v36, **v27, v59, 0);
          sub_38DDC80(*(__int64 **)(a1 + 256), v35, v59, 0);
LABEL_58:
          if ( v58 == ++v27 )
            goto LABEL_68;
        }
        else
        {
          v37 = *(void (**)())(*v36 + 104);
          v63 = "DW_RLE_start_length";
          LOWORD(v65) = 259;
          if ( v37 != nullsub_580 )
          {
            ((void (__fastcall *)(__int64 *, const char **, __int64))v37)(v36, &v63, 1);
            v36 = *(__int64 **)(a1 + 256);
          }
          (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v36 + 424))(v36, 7, 1);
          v38 = *(__int64 **)(a1 + 256);
          v39 = *(void (**)())(*v38 + 104);
          v63 = "  start";
          LOWORD(v65) = 259;
          if ( v39 != nullsub_580 )
          {
            ((void (__fastcall *)(__int64 *, const char **, __int64))v39)(v38, &v63, 1);
            v38 = *(__int64 **)(a1 + 256);
          }
          sub_38DDC80(v38, v34, v59, 0);
          v40 = *(_QWORD *)(a1 + 256);
          v41 = *(void (**)())(*(_QWORD *)v40 + 104LL);
          v63 = "  length";
          LOWORD(v65) = 259;
          if ( v41 != nullsub_580 )
            ((void (__fastcall *)(__int64, const char **, __int64))v41)(v40, &v63, 1);
          ++v27;
          sub_397C140(a1);
          if ( v58 == v27 )
            goto LABEL_68;
        }
      }
      if ( v61 <= 4u )
      {
        sub_396F380(a1);
        sub_396F380(a1);
      }
      else
      {
        v28 = *(_QWORD *)(a1 + 256);
        v29 = *(void (**)())(*(_QWORD *)v28 + 104LL);
        v63 = "DW_RLE_offset_pair";
        LOWORD(v65) = 259;
        if ( v29 != nullsub_580 )
        {
          ((void (__fastcall *)(__int64, const char **, __int64))v29)(v28, &v63, 1);
          v28 = *(_QWORD *)(a1 + 256);
        }
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v28 + 424LL))(v28, 4, 1);
        v30 = *(_QWORD *)(a1 + 256);
        v31 = *(void (**)())(*(_QWORD *)v30 + 104LL);
        v63 = "  starting offset";
        LOWORD(v65) = 259;
        if ( v31 != nullsub_580 )
          ((void (__fastcall *)(__int64, const char **, __int64))v31)(v30, &v63, 1);
        sub_397C140(a1);
        v32 = *(_QWORD *)(a1 + 256);
        v33 = *(void (**)())(*(_QWORD *)v32 + 104LL);
        v63 = "  ending offset";
        LOWORD(v65) = 259;
        if ( v33 != nullsub_580 )
          ((void (__fastcall *)(__int64, const char **, __int64))v33)(v32, &v63, 1);
        sub_397C140(a1);
      }
      goto LABEL_58;
    }
LABEL_68:
    v57 += 32;
    if ( v55 != v57 )
      continue;
    break;
  }
LABEL_69:
  v42 = *(_QWORD *)(a1 + 256);
  if ( v61 <= 4u )
  {
    (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v42 + 424LL))(v42, 0, v59);
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 256) + 424LL))(*(_QWORD *)(a1 + 256), 0, v59);
  }
  else
  {
    v43 = *(void (**)())(*(_QWORD *)v42 + 104LL);
    v63 = "DW_RLE_end_of_list";
    LOWORD(v65) = 259;
    if ( v43 != nullsub_580 )
    {
      ((void (__fastcall *)(__int64, const char **, __int64))v43)(v42, &v63, 1);
      v42 = *(_QWORD *)(a1 + 256);
    }
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v42 + 424LL))(v42, 0, 1);
  }
  v44 = v72;
  v45 = (unsigned __int64)v71;
  if ( v72 != v71 )
  {
    do
    {
      v46 = *(_QWORD *)(v45 + 8);
      if ( v46 )
        j_j___libc_free_0(v46);
      v45 += 32LL;
    }
    while ( v44 != (char *)v45 );
    v45 = (unsigned __int64)v71;
  }
  if ( v45 )
    j_j___libc_free_0(v45);
  j___libc_free_0(v68);
}
