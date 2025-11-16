// Function: sub_11136F0
// Address: 0x11136f0
//
_QWORD *__fastcall sub_11136F0(__int64 a1, unsigned int **a2)
{
  __int16 v3; // bx
  __int64 v4; // r14
  __int64 v5; // r15
  unsigned int v6; // r13d
  __int64 v7; // rax
  __int64 v8; // r11
  unsigned __int8 *v9; // rsi
  int v10; // eax
  __int64 v11; // rdx
  char v12; // al
  _BYTE *v13; // rdi
  _QWORD *v14; // r14
  void *v16; // r10
  __int64 v17; // rcx
  unsigned int v18; // r9d
  unsigned __int8 *v19; // rsi
  char *v20; // r10
  char *v21; // rdi
  __int64 v22; // r15
  char *v23; // rax
  unsigned int v24; // r8d
  __int64 v25; // rax
  _BYTE *v26; // r14
  __int64 v27; // r11
  unsigned int i; // r9d
  __int64 *v29; // rax
  __int64 v30; // r12
  __int64 v31; // r13
  __int64 v32; // rsi
  _QWORD *v33; // rax
  __int64 v34; // r9
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rcx
  _BYTE *v38; // rsi
  char v39; // al
  _BYTE *v40; // rdi
  int v41; // eax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // r12
  _QWORD *v45; // rax
  __int64 v46; // r9
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rdx
  __int64 v51; // rax
  int v52; // eax
  int v53; // eax
  signed __int64 v54; // rdx
  signed __int64 v55; // rax
  int v56; // eax
  __int64 *v57; // rax
  __int64 *v58; // r10
  unsigned int v59; // [rsp+8h] [rbp-158h]
  void *v60; // [rsp+10h] [rbp-150h]
  __int64 v61; // [rsp+18h] [rbp-148h]
  unsigned int v62; // [rsp+18h] [rbp-148h]
  unsigned int v63; // [rsp+18h] [rbp-148h]
  __int64 v64; // [rsp+20h] [rbp-140h]
  __int64 v65; // [rsp+20h] [rbp-140h]
  void *v66; // [rsp+20h] [rbp-140h]
  unsigned int v67; // [rsp+20h] [rbp-140h]
  void *v68; // [rsp+20h] [rbp-140h]
  unsigned int v69; // [rsp+28h] [rbp-138h]
  char v70; // [rsp+28h] [rbp-138h]
  __int64 v71; // [rsp+28h] [rbp-138h]
  char *v72; // [rsp+30h] [rbp-130h]
  __int64 v73; // [rsp+30h] [rbp-130h]
  __int64 v74; // [rsp+30h] [rbp-130h]
  __int64 v75; // [rsp+30h] [rbp-130h]
  __int64 v76; // [rsp+38h] [rbp-128h]
  __int64 v77; // [rsp+38h] [rbp-128h]
  void *v78; // [rsp+38h] [rbp-128h]
  void *v79; // [rsp+38h] [rbp-128h]
  __int64 v80; // [rsp+38h] [rbp-128h]
  char v81; // [rsp+40h] [rbp-120h]
  __int64 v82; // [rsp+40h] [rbp-120h]
  unsigned int v83; // [rsp+40h] [rbp-120h]
  __int64 v84; // [rsp+40h] [rbp-120h]
  __int64 v85; // [rsp+48h] [rbp-118h]
  __int64 v86; // [rsp+48h] [rbp-118h]
  unsigned __int64 v87; // [rsp+48h] [rbp-118h]
  __int64 *v88; // [rsp+48h] [rbp-118h]
  __int64 v89; // [rsp+48h] [rbp-118h]
  __int64 v90; // [rsp+48h] [rbp-118h]
  __int64 v91; // [rsp+58h] [rbp-108h]
  __int64 v92[2]; // [rsp+60h] [rbp-100h] BYREF
  __int64 **v93; // [rsp+70h] [rbp-F0h] BYREF
  _BYTE **v94; // [rsp+78h] [rbp-E8h]
  _BYTE *v95; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v96; // [rsp+88h] [rbp-D8h]
  _BYTE v97[16]; // [rsp+90h] [rbp-D0h] BYREF
  __int16 v98; // [rsp+A0h] [rbp-C0h]
  __int64 *v99; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v100; // [rsp+D8h] [rbp-88h]
  __int64 v101; // [rsp+E0h] [rbp-80h] BYREF
  int v102; // [rsp+E8h] [rbp-78h]
  char v103; // [rsp+ECh] [rbp-74h]
  _WORD v104[56]; // [rsp+F0h] [rbp-70h] BYREF

  v3 = *(_WORD *)(a1 + 2);
  v4 = *(_QWORD *)(a1 - 64);
  v92[0] = (__int64)a2;
  v92[1] = a1;
  v5 = *(_QWORD *)(a1 - 32);
  v6 = v3 & 0x3F;
  if ( *(_BYTE *)v4 != 85
    || (v7 = *(_QWORD *)(v4 - 32)) == 0
    || *(_BYTE *)v7
    || *(_QWORD *)(v7 + 24) != *(_QWORD *)(v4 + 80)
    || *(_DWORD *)(v7 + 36) != 402
    || !*(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)) )
  {
    if ( sub_9B7DA0((char *)v4, 0xFFFFFFFF, 0) )
    {
      v35 = *(_QWORD *)(v5 + 16);
      if ( v35 )
      {
        if ( !*(_QWORD *)(v35 + 8) && *(_BYTE *)v5 == 85 )
        {
          v36 = *(_QWORD *)(v5 - 32);
          if ( v36 )
          {
            if ( !*(_BYTE *)v36 && *(_QWORD *)(v36 + 24) == *(_QWORD *)(v5 + 80) && *(_DWORD *)(v36 + 36) == 402 )
            {
              v37 = *(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
              if ( v37 )
                return sub_1112240(v92, v3 & 0x3F, v4, v37);
            }
          }
        }
      }
    }
    goto LABEL_6;
  }
  v47 = *(_QWORD *)(v4 + 16);
  if ( *(_BYTE *)v5 != 85 )
    goto LABEL_88;
  v48 = *(_QWORD *)(v5 - 32);
  if ( !v48 )
    goto LABEL_88;
  if ( *(_BYTE *)v48 )
    goto LABEL_88;
  if ( *(_QWORD *)(v48 + 24) != *(_QWORD *)(v5 + 80) )
    goto LABEL_88;
  if ( *(_DWORD *)(v48 + 36) != 402 )
    goto LABEL_88;
  v49 = *(_QWORD *)(v5 - 32LL * (*(_DWORD *)(v5 + 4) & 0x7FFFFFF));
  if ( !v49 )
    goto LABEL_88;
  if ( !v47 )
  {
    v50 = *(_QWORD *)(v5 + 16);
    if ( !v50 )
      return 0;
    goto LABEL_99;
  }
  if ( !*(_QWORD *)(v47 + 8) )
    return sub_1112240(v92, v3 & 0x3F, *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)), v49);
  v50 = *(_QWORD *)(v5 + 16);
  if ( v50 )
  {
LABEL_99:
    if ( !*(_QWORD *)(v50 + 8) )
      return sub_1112240(v92, v3 & 0x3F, *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF)), v49);
LABEL_88:
    if ( !v47 )
      return 0;
  }
  if ( *(_QWORD *)(v47 + 8) )
    return 0;
  v89 = *(_QWORD *)(v4 - 32LL * (*(_DWORD *)(v4 + 4) & 0x7FFFFFF));
  if ( sub_9B7DA0(*(char **)(a1 - 32), 0xFFFFFFFF, 0) )
    return sub_1112240(v92, v3 & 0x3F, v89, v5);
LABEL_6:
  if ( *(_BYTE *)v4 != 92 )
    return 0;
  v8 = *(_QWORD *)(v4 - 64);
  if ( !v8 )
    return 0;
  v9 = *(unsigned __int8 **)(v4 - 32);
  v10 = *v9;
  v11 = (unsigned int)(v10 - 12);
  if ( (unsigned __int8)(v10 - 12) > 1u )
  {
    v85 = *(_QWORD *)(v4 - 64);
    if ( (unsigned __int8)(v10 - 9) > 2u )
      return 0;
    v103 = 1;
    v100 = (__int64)v104;
    v94 = &v95;
    v95 = v97;
    v99 = 0;
    v101 = 8;
    v102 = 0;
    v96 = 0x800000000LL;
    v93 = &v99;
    v12 = sub_AA8FD0(&v93, (__int64)v9);
    v8 = v85;
    v81 = v12;
    if ( v12 )
    {
      do
      {
        v13 = v95;
        if ( !(_DWORD)v96 )
        {
          v8 = v85;
          goto LABEL_15;
        }
        v9 = *(unsigned __int8 **)&v95[8 * (unsigned int)v96 - 8];
        LODWORD(v96) = v96 - 1;
      }
      while ( (unsigned __int8)sub_AA8FD0(&v93, (__int64)v9) );
      v8 = v85;
    }
    v81 = 0;
    v13 = v95;
LABEL_15:
    if ( v13 != v97 )
    {
      v86 = v8;
      _libc_free(v13, v9);
      v8 = v86;
    }
    if ( !v103 )
    {
      v90 = v8;
      _libc_free(v100, v9);
      v8 = v90;
    }
    if ( !v81 )
      return 0;
  }
  v16 = *(void **)(v4 + 72);
  v17 = *(_QWORD *)(v8 + 8);
  v87 = *(unsigned int *)(v4 + 80);
  v18 = *(_DWORD *)(v4 + 80);
  if ( *(_BYTE *)v5 != 92 )
    goto LABEL_23;
  v84 = *(_QWORD *)(v5 - 64);
  if ( !v84 )
    goto LABEL_23;
  v38 = *(_BYTE **)(v5 - 32);
  v11 = (unsigned __int8)*v38;
  if ( (unsigned __int8)(*v38 - 12) > 1u )
  {
    if ( (unsigned int)v11 <= 8 || (unsigned int)v11 > 0xB )
      goto LABEL_23;
    v65 = *(_QWORD *)(v8 + 8);
    v100 = (__int64)v104;
    v94 = &v95;
    v95 = v97;
    v59 = v18;
    v60 = v16;
    v61 = v8;
    v96 = 0x800000000LL;
    v99 = 0;
    v101 = 8;
    v102 = 0;
    v103 = 1;
    v93 = &v99;
    v39 = sub_AA8FD0(&v93, (__int64)v38);
    v17 = v65;
    v8 = v61;
    v70 = v39;
    v16 = v60;
    v18 = v59;
    if ( v39 )
    {
      do
      {
        v40 = v95;
        if ( !(_DWORD)v96 )
        {
          v17 = v65;
          v8 = v61;
          v16 = v60;
          v18 = v59;
          goto LABEL_69;
        }
        v38 = *(_BYTE **)&v95[8 * (unsigned int)v96 - 8];
        LODWORD(v96) = v96 - 1;
      }
      while ( (unsigned __int8)sub_AA8FD0(&v93, (__int64)v38) );
      v17 = v65;
      v8 = v61;
      v16 = v60;
      v18 = v59;
    }
    v70 = 0;
    v40 = v95;
LABEL_69:
    if ( v40 != v97 )
    {
      v62 = v18;
      v66 = v16;
      v73 = v8;
      v77 = v17;
      _libc_free(v40, v38);
      v18 = v62;
      v16 = v66;
      v8 = v73;
      v17 = v77;
    }
    if ( !v103 )
    {
      v63 = v18;
      v68 = v16;
      v75 = v8;
      v80 = v17;
      _libc_free(v100, v38);
      v18 = v63;
      v16 = v68;
      v8 = v75;
      v17 = v80;
    }
    if ( !v70 )
    {
LABEL_23:
      v14 = *(_QWORD **)(v4 + 16);
      goto LABEL_24;
    }
  }
  v11 = v87;
  v14 = *(_QWORD **)(v4 + 16);
  if ( v87 != *(_DWORD *)(v5 + 80) )
    goto LABEL_24;
  v11 = 4 * v87;
  if ( 4 * v87 )
  {
    v67 = v18;
    v71 = v8;
    v74 = v17;
    v78 = v16;
    v41 = memcmp(v16, *(const void **)(v5 + 72), v11);
    v16 = v78;
    v17 = v74;
    v8 = v71;
    v18 = v67;
    if ( v41 )
      goto LABEL_24;
  }
  if ( v17 != *(_QWORD *)(v84 + 8) )
  {
LABEL_24:
    if ( !v14 )
      return v14;
    goto LABEL_25;
  }
  if ( !v14 )
  {
    v42 = *(_QWORD *)(v5 + 16);
    if ( !v42 || *(_QWORD *)(v42 + 8) )
      return v14;
    goto LABEL_81;
  }
  if ( !v14[1] || (v51 = *(_QWORD *)(v5 + 16)) != 0 && !*(_QWORD *)(v51 + 8) )
  {
LABEL_81:
    v79 = v16;
    v104[0] = 257;
    if ( (v3 & 0x30) != 0 )
    {
      v43 = sub_92B530(a2, v6, v8, (_BYTE *)v84, (__int64)&v99);
    }
    else
    {
      HIDWORD(v95) = 0;
      v43 = sub_B35C90((__int64)a2, v6, v8, v84, (__int64)&v99, 0, (unsigned int)v95, 0);
    }
    v44 = v43;
    v104[0] = 257;
    v45 = sub_BD2C40(112, unk_3F1FE60);
    v14 = v45;
    if ( v45 )
      sub_B4EB40((__int64)v45, v44, v79, v87, (__int64)&v99, v46, 0);
    return v14;
  }
LABEL_25:
  v14 = (_QWORD *)v14[1];
  if ( v14 )
    return 0;
  v69 = v18;
  v72 = (char *)v16;
  v76 = v8;
  v82 = v17;
  if ( *(_BYTE *)v5 > 0x15u )
    return v14;
  v19 = sub_AD7630(v5, 1, v11);
  if ( !v19 )
    return v14;
  v20 = v72;
  v64 = 4 * v87;
  v21 = &v72[4 * v87];
  v22 = (__int64)(4 * v87) >> 4;
  if ( !v22 )
  {
    v23 = v72;
LABEL_119:
    v54 = v21 - v23;
    if ( v21 - v23 != 8 )
    {
      if ( v54 != 12 )
      {
        if ( v54 != 4 )
          return v14;
        goto LABEL_122;
      }
      if ( *(_DWORD *)v23 != -1 )
        goto LABEL_123;
      v23 += 4;
    }
    if ( *(_DWORD *)v23 != -1 )
      goto LABEL_123;
    v23 += 4;
LABEL_122:
    if ( *(_DWORD *)v23 == -1 )
      return v14;
LABEL_123:
    if ( v21 == v23 )
      return v14;
    v24 = *(_DWORD *)v23;
    if ( v22 )
      goto LABEL_37;
LABEL_125:
    v55 = v21 - v20;
    if ( v21 - v20 != 8 )
    {
      if ( v55 != 12 )
      {
        if ( v55 != 4 )
          goto LABEL_40;
        goto LABEL_128;
      }
      if ( *(_DWORD *)v20 != -1 && *(_DWORD *)v20 != v24 )
        goto LABEL_39;
      v20 += 4;
    }
    if ( *(_DWORD *)v20 != -1 && *(_DWORD *)v20 != v24 )
      goto LABEL_39;
    v20 += 4;
LABEL_128:
    if ( *(_DWORD *)v20 != v24 && *(_DWORD *)v20 != -1 )
      goto LABEL_39;
LABEL_40:
    BYTE4(v91) = *(_BYTE *)(v82 + 8) == 18;
    LODWORD(v91) = *(_DWORD *)(v82 + 32);
    v83 = v24;
    v25 = sub_AD5E10(v91, v19);
    v99 = &v101;
    v26 = (_BYTE *)v25;
    v27 = v76;
    v100 = 0x800000000LL;
    i = v69;
    if ( v87 > 8 )
    {
      sub_C8D5F0((__int64)&v99, &v101, v87, 4u, v83, v69);
      v57 = v99;
      v27 = v76;
      v58 = (__int64 *)((char *)v99 + v64);
      for ( i = v69; v58 != v57; v57 = (__int64 *)((char *)v57 + 4) )
        *(_DWORD *)v57 = v83;
    }
    else if ( v87 && (__int64 *)((char *)&v101 + v64) != &v101 )
    {
      v29 = &v101;
      do
      {
        *(_DWORD *)v29 = v83;
        v29 = (__int64 *)((char *)v29 + 4);
      }
      while ( (__int64 *)((char *)&v101 + v64) != v29 );
    }
    LODWORD(v100) = i;
    v98 = 257;
    if ( (v3 & 0x30) != 0 )
    {
      v30 = sub_92B530(a2, v6, v27, v26, (__int64)&v95);
    }
    else
    {
      HIDWORD(v93) = 0;
      v30 = sub_B35C90((__int64)a2, v6, v27, (__int64)v26, (__int64)&v95, 0, (unsigned int)v93, 0);
    }
    v98 = 257;
    v31 = (unsigned int)v100;
    v32 = unk_3F1FE60;
    v88 = v99;
    v33 = sub_BD2C40(112, unk_3F1FE60);
    v14 = v33;
    if ( v33 )
    {
      v32 = v30;
      sub_B4EB40((__int64)v33, v30, v88, v31, (__int64)&v95, v34, 0);
    }
    if ( v99 != &v101 )
      _libc_free(v99, v32);
    return v14;
  }
  v23 = v72;
  while ( *(_DWORD *)v23 == -1 )
  {
    if ( *((_DWORD *)v23 + 1) != -1 )
    {
      v23 += 4;
      break;
    }
    if ( *((_DWORD *)v23 + 2) != -1 )
    {
      v23 += 8;
      break;
    }
    if ( *((_DWORD *)v23 + 3) != -1 )
    {
      v23 += 12;
      break;
    }
    v23 += 16;
    if ( &v72[16 * ((__int64)(4 * v87) >> 4)] == v23 )
      goto LABEL_119;
  }
  if ( v23 != v21 )
  {
    v24 = *(_DWORD *)v23;
LABEL_37:
    while ( v24 == *(_DWORD *)v20 || *(_DWORD *)v20 == -1 )
    {
      v52 = *((_DWORD *)v20 + 1);
      if ( v52 != -1 && v24 != v52 )
      {
        v20 += 4;
        break;
      }
      v53 = *((_DWORD *)v20 + 2);
      if ( v24 != v53 && v53 != -1 )
      {
        v20 += 8;
        break;
      }
      v56 = *((_DWORD *)v20 + 3);
      if ( v24 != v56 && v56 != -1 )
      {
        v20 += 12;
        break;
      }
      v20 += 16;
      if ( !--v22 )
        goto LABEL_125;
    }
LABEL_39:
    if ( v21 != v20 )
      return v14;
    goto LABEL_40;
  }
  return v14;
}
