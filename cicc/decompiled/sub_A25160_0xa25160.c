// Function: sub_A25160
// Address: 0xa25160
//
__int64 __fastcall sub_A25160(__int64 *a1)
{
  __int64 *v1; // rbx
  __int64 v2; // r13
  __int64 v3; // rax
  __int64 v4; // rdi
  volatile signed __int32 *v5; // rax
  __int64 v6; // rdi
  volatile signed __int32 *v7; // rax
  volatile signed __int32 *v8; // r8
  __int64 v9; // rax
  __int64 v10; // rdi
  volatile signed __int32 *v11; // rax
  __int64 v12; // rdi
  volatile signed __int32 *v13; // rax
  volatile signed __int32 *v14; // r8
  __int64 v15; // rax
  __int64 v16; // rdi
  volatile signed __int32 *v17; // rax
  __int64 v18; // rdi
  volatile signed __int32 *v19; // rax
  volatile signed __int32 *v20; // r8
  __int64 v21; // rax
  __int64 v22; // rdi
  volatile signed __int32 *v23; // rax
  __int64 v24; // rdi
  volatile signed __int32 *v25; // rax
  volatile signed __int32 *v26; // r8
  __int64 v27; // rax
  __int64 v28; // rdi
  volatile signed __int32 *v29; // rax
  __int64 v30; // rdi
  volatile signed __int32 *v31; // rax
  volatile signed __int32 *v32; // r8
  __int64 v33; // rax
  __int64 v34; // rdi
  volatile signed __int32 *v35; // rax
  __int64 v36; // rsi
  __int64 v37; // rax
  __int64 v38; // r13
  __int64 v39; // r12
  unsigned int v40; // r15d
  unsigned int v41; // eax
  unsigned int v42; // r9d
  __int64 result; // rax
  unsigned int v44; // eax
  __int64 *v45; // rcx
  __int64 *v46; // r15
  __int64 *v47; // r12
  __int64 *v48; // rbx
  unsigned int v49; // edx
  __int64 *v50; // rax
  __int64 v51; // r11
  __int64 v52; // rcx
  __int64 v53; // rsi
  __int64 v54; // rdi
  unsigned int v55; // r12d
  unsigned int v56; // eax
  int v57; // eax
  __int64 v58; // r15
  __int64 v59; // r13
  __int64 v60; // r12
  unsigned int v61; // edx
  __int64 *v62; // rax
  __int64 v63; // r11
  __int64 v64; // rcx
  __int64 v65; // rdi
  __int64 v66; // rsi
  __int64 *v67; // rcx
  __int64 *v68; // r15
  __int64 *v69; // r12
  __int64 *v70; // rbx
  unsigned int v71; // edx
  __int64 *v72; // rax
  __int64 v73; // r11
  __int64 v74; // rcx
  __int64 v75; // rsi
  __int64 v76; // rdi
  unsigned int *v77; // r15
  unsigned int *v78; // r12
  __int64 v79; // rsi
  int v80; // eax
  int v81; // r8d
  int v82; // eax
  int v83; // r8d
  int v84; // eax
  int v85; // r8d
  unsigned int v86; // r9d
  int v87; // r15d
  __int64 v88; // rdx
  char *v89; // rax
  __int64 v90; // rdx
  __int64 v91; // [rsp+8h] [rbp-2A8h]
  __int64 v92; // [rsp+8h] [rbp-2A8h]
  __int64 v93; // [rsp+8h] [rbp-2A8h]
  unsigned int v94; // [rsp+8h] [rbp-2A8h]
  unsigned int v95; // [rsp+10h] [rbp-2A0h]
  unsigned int v96; // [rsp+14h] [rbp-29Ch]
  unsigned int v97; // [rsp+18h] [rbp-298h]
  unsigned int v98; // [rsp+1Ch] [rbp-294h]
  unsigned int v99; // [rsp+20h] [rbp-290h]
  unsigned int v100; // [rsp+24h] [rbp-28Ch]
  __int64 v101; // [rsp+30h] [rbp-280h]
  __int64 v102; // [rsp+38h] [rbp-278h]
  __int64 v103; // [rsp+50h] [rbp-260h] BYREF
  volatile signed __int32 *v104; // [rsp+58h] [rbp-258h]
  __int64 v105; // [rsp+60h] [rbp-250h] BYREF
  volatile signed __int32 *v106; // [rsp+68h] [rbp-248h]
  _BYTE *v107; // [rsp+70h] [rbp-240h] BYREF
  __int64 v108; // [rsp+78h] [rbp-238h]
  _BYTE v109[560]; // [rsp+80h] [rbp-230h] BYREF

  v1 = a1;
  sub_A19830(*a1, 0x11u, 4u);
  v107 = v109;
  v108 = 0x4000000000LL;
  v101 = (__int64)(a1 + 3);
  v2 = sub_A3FA40(a1 + 3);
  sub_A23770(&v103);
  sub_A186C0(v103, 25, 1);
  sub_A186C0(v103, 0, 1);
  v3 = v103;
  v4 = *a1;
  v103 = 0;
  v105 = v3;
  v5 = v104;
  v104 = 0;
  v106 = v5;
  v99 = sub_A1AB30(v4, &v105);
  if ( v106 )
    sub_A191D0(v106);
  sub_A23770(&v105);
  v6 = v105;
  v7 = v106;
  v105 = 0;
  v8 = v104;
  v106 = 0;
  v103 = v6;
  v104 = v7;
  if ( v8 )
  {
    sub_A191D0(v8);
    if ( v106 )
      sub_A191D0(v106);
    v6 = v103;
  }
  sub_A186C0(v6, 21, 1);
  sub_A186C0(v103, 1, 2);
  sub_A186C0(v103, 0, 6);
  sub_A186C0(v103, v2, 2);
  v9 = v103;
  v10 = *v1;
  v103 = 0;
  v105 = v9;
  v11 = v104;
  v104 = 0;
  v106 = v11;
  v98 = sub_A1AB30(v10, &v105);
  if ( v106 )
    sub_A191D0(v106);
  sub_A23770(&v105);
  v12 = v105;
  v13 = v106;
  v105 = 0;
  v14 = v104;
  v106 = 0;
  v103 = v12;
  v104 = v13;
  if ( v14 )
  {
    sub_A191D0(v14);
    if ( v106 )
      sub_A191D0(v106);
    v12 = v103;
  }
  sub_A186C0(v12, 18, 1);
  sub_A186C0(v103, 1, 2);
  sub_A186C0(v103, 0, 6);
  sub_A186C0(v103, v2, 2);
  v15 = v103;
  v16 = *v1;
  v103 = 0;
  v105 = v15;
  v17 = v104;
  v104 = 0;
  v106 = v17;
  v96 = sub_A1AB30(v16, &v105);
  if ( v106 )
    sub_A191D0(v106);
  sub_A23770(&v105);
  v18 = v105;
  v19 = v106;
  v105 = 0;
  v20 = v104;
  v106 = 0;
  v103 = v18;
  v104 = v19;
  if ( v20 )
  {
    sub_A191D0(v20);
    if ( v106 )
      sub_A191D0(v106);
    v18 = v103;
  }
  sub_A186C0(v18, 19, 1);
  sub_A186C0(v103, 0, 6);
  sub_A186C0(v103, 0, 8);
  v21 = v103;
  v22 = *v1;
  v103 = 0;
  v105 = v21;
  v23 = v104;
  v104 = 0;
  v106 = v23;
  v100 = sub_A1AB30(v22, &v105);
  if ( v106 )
    sub_A191D0(v106);
  sub_A23770(&v105);
  v24 = v105;
  v25 = v106;
  v105 = 0;
  v26 = v104;
  v106 = 0;
  v103 = v24;
  v104 = v25;
  if ( v26 )
  {
    sub_A191D0(v26);
    if ( v106 )
      sub_A191D0(v106);
    v24 = v103;
  }
  sub_A186C0(v24, 20, 1);
  sub_A186C0(v103, 1, 2);
  sub_A186C0(v103, 0, 6);
  sub_A186C0(v103, v2, 2);
  v27 = v103;
  v28 = *v1;
  v103 = 0;
  v105 = v27;
  v29 = v104;
  v104 = 0;
  v106 = v29;
  v95 = sub_A1AB30(v28, &v105);
  if ( v106 )
    sub_A191D0(v106);
  sub_A23770(&v105);
  v30 = v105;
  v31 = v106;
  v105 = 0;
  v32 = v104;
  v106 = 0;
  v103 = v30;
  v104 = v31;
  if ( v32 )
  {
    sub_A191D0(v32);
    if ( v106 )
      sub_A191D0(v106);
    v30 = v103;
  }
  sub_A186C0(v30, 11, 1);
  sub_A186C0(v103, 8, 4);
  sub_A186C0(v103, v2, 2);
  v33 = v103;
  v34 = *v1;
  v103 = 0;
  v105 = v33;
  v35 = v104;
  v104 = 0;
  v106 = v35;
  v97 = sub_A1AB30(v34, &v105);
  if ( v106 )
    sub_A191D0(v106);
  sub_A188E0((__int64)&v107, (v1[11] - v1[10]) >> 3);
  v36 = 1;
  sub_A1FB70(*v1, 1u, (__int64)&v107, 0);
  v37 = v1[11];
  v38 = v1[10];
  LODWORD(v108) = 0;
  v102 = v37;
  if ( v38 == v37 )
    goto LABEL_38;
  do
  {
    v39 = *(_QWORD *)v38;
    switch ( *(_BYTE *)(*(_QWORD *)v38 + 8LL) )
    {
      case 0:
        v42 = 0;
        v40 = 10;
        goto LABEL_37;
      case 1:
        v42 = 0;
        v40 = 23;
        goto LABEL_37;
      case 2:
        v42 = 0;
        v40 = 3;
        goto LABEL_37;
      case 3:
        v42 = 0;
        v40 = 4;
        goto LABEL_37;
      case 4:
        v42 = 0;
        v40 = 13;
        goto LABEL_37;
      case 5:
        v42 = 0;
        v40 = 14;
        goto LABEL_37;
      case 6:
        v42 = 0;
        v40 = 15;
        goto LABEL_37;
      case 7:
        v42 = 0;
        v40 = 2;
        goto LABEL_37;
      case 8:
        v42 = 0;
        v40 = 5;
        goto LABEL_37;
      case 9:
        v42 = 0;
        v40 = 16;
        goto LABEL_37;
      case 0xA:
        v42 = 0;
        v40 = 24;
        goto LABEL_37;
      case 0xB:
        v42 = 0;
        v40 = 22;
        goto LABEL_37;
      case 0xC:
        v40 = 7;
        sub_A188E0((__int64)&v107, *(_DWORD *)(v39 + 8) >> 8);
        v42 = 0;
        goto LABEL_37;
      case 0xD:
        sub_A188E0((__int64)&v107, *(_DWORD *)(v39 + 8) >> 8 != 0);
        v56 = sub_A172F0(v101, **(_QWORD **)(v39 + 16));
        sub_A188E0((__int64)&v107, v56);
        v57 = *(_DWORD *)(v39 + 12);
        if ( v57 == 1 )
          goto LABEL_84;
        v92 = v38;
        v58 = 8;
        v59 = v39;
        v60 = 8LL * (unsigned int)(v57 - 2) + 16;
        while ( 1 )
        {
          v64 = *((unsigned int *)v1 + 18);
          v65 = v1[7];
          v66 = *(_QWORD *)(*(_QWORD *)(v59 + 16) + v58);
          if ( !(_DWORD)v64 )
            goto LABEL_58;
          v61 = (v64 - 1) & (((unsigned int)v66 >> 9) ^ ((unsigned int)v66 >> 4));
          v62 = (__int64 *)(v65 + 16LL * v61);
          v63 = *v62;
          if ( v66 != *v62 )
            break;
LABEL_56:
          v58 += 8;
          sub_A188E0((__int64)&v107, (unsigned int)(*((_DWORD *)v62 + 2) - 1));
          if ( v60 == v58 )
          {
            v38 = v92;
LABEL_84:
            v42 = v98;
            v40 = 21;
            goto LABEL_37;
          }
        }
        v80 = 1;
        while ( v63 != -4096 )
        {
          v81 = v80 + 1;
          v61 = (v64 - 1) & (v80 + v61);
          v62 = (__int64 *)(v65 + 16LL * v61);
          v63 = *v62;
          if ( v66 == *v62 )
            goto LABEL_56;
          v80 = v81;
        }
LABEL_58:
        v62 = (__int64 *)(v65 + 16 * v64);
        goto LABEL_56;
      case 0xE:
        v40 = 25;
        v55 = *(_DWORD *)(v39 + 8) >> 8;
        sub_A188E0((__int64)&v107, v55);
        v42 = 0;
        if ( !v55 )
          v42 = v99;
        goto LABEL_37;
      case 0xF:
        sub_A188E0((__int64)&v107, (*(_DWORD *)(v39 + 8) >> 9) & 1);
        v45 = *(__int64 **)(v39 + 16);
        if ( &v45[*(unsigned int *)(v39 + 12)] == v45 )
          goto LABEL_86;
        v91 = v39;
        v46 = &v45[*(unsigned int *)(v39 + 12)];
        v47 = v1;
        v48 = v45;
        while ( 1 )
        {
          v52 = *((unsigned int *)v47 + 18);
          v53 = *v48;
          v54 = v47[7];
          if ( !(_DWORD)v52 )
            goto LABEL_49;
          v49 = (v52 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
          v50 = (__int64 *)(v54 + 16LL * v49);
          v51 = *v50;
          if ( v53 != *v50 )
            break;
LABEL_47:
          ++v48;
          sub_A188E0((__int64)&v107, (unsigned int)(*((_DWORD *)v50 + 2) - 1));
          if ( v46 == v48 )
          {
            v1 = v47;
            v39 = v91;
LABEL_86:
            if ( (*(_DWORD *)(v39 + 8) & 0x400) != 0 )
            {
              v42 = v96;
              v40 = 18;
            }
            else
            {
              v86 = 0;
              if ( (*(_DWORD *)(v39 + 8) & 0x100) != 0 )
                v86 = v95;
              v87 = -((*(_DWORD *)(v39 + 8) & 0x100) != 0);
              v94 = v86;
              sub_BCB490(v39);
              v42 = v94;
              v40 = (v87 & 0xE) + 6;
              if ( v88 )
              {
                v89 = (char *)sub_BCB490(v39);
                sub_A215B0(*v1, 0x13u, v89, v90, v100);
                v42 = v94;
              }
            }
            goto LABEL_37;
          }
        }
        v84 = 1;
        while ( v51 != -4096 )
        {
          v85 = v84 + 1;
          v49 = (v52 - 1) & (v84 + v49);
          v50 = (__int64 *)(v54 + 16LL * v49);
          v51 = *v50;
          if ( v53 == *v50 )
            goto LABEL_47;
          v84 = v85;
        }
LABEL_49:
        v50 = (__int64 *)(v54 + 16 * v52);
        goto LABEL_47;
      case 0x10:
        v40 = 11;
        sub_A188E0((__int64)&v107, *(_QWORD *)(v39 + 32));
        v44 = sub_A172F0(v101, *(_QWORD *)(v39 + 24));
        sub_A188E0((__int64)&v107, v44);
        v42 = v97;
        goto LABEL_37;
      case 0x11:
      case 0x12:
        v40 = 12;
        sub_A188E0((__int64)&v107, *(unsigned int *)(v39 + 32));
        v41 = sub_A172F0(v101, *(_QWORD *)(v39 + 24));
        sub_A188E0((__int64)&v107, v41);
        v42 = 0;
        if ( *(_BYTE *)(v39 + 8) == 18 )
        {
          sub_A188E0((__int64)&v107, 1);
          v42 = 0;
        }
        goto LABEL_37;
      case 0x13:
        BUG();
      case 0x14:
        sub_A215B0(*v1, 0x13u, *(char **)(v39 + 24), *(_QWORD *)(v39 + 32), v100);
        sub_A188E0((__int64)&v107, *(unsigned int *)(v39 + 12));
        v67 = *(__int64 **)(v39 + 16);
        if ( &v67[*(unsigned int *)(v39 + 12)] == v67 )
          goto LABEL_79;
        v93 = v39;
        v68 = &v67[*(unsigned int *)(v39 + 12)];
        v69 = v1;
        v70 = v67;
        break;
      default:
        v42 = 0;
        v40 = 0;
        goto LABEL_37;
    }
    do
    {
      v74 = *((unsigned int *)v69 + 18);
      v75 = *v70;
      v76 = v69[7];
      if ( (_DWORD)v74 )
      {
        v71 = (v74 - 1) & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
        v72 = (__int64 *)(v76 + 16LL * v71);
        v73 = *v72;
        if ( v75 == *v72 )
          goto LABEL_73;
        v82 = 1;
        while ( v73 != -4096 )
        {
          v83 = v82 + 1;
          v71 = (v74 - 1) & (v82 + v71);
          v72 = (__int64 *)(v76 + 16LL * v71);
          v73 = *v72;
          if ( v75 == *v72 )
            goto LABEL_73;
          v82 = v83;
        }
      }
      v72 = (__int64 *)(v76 + 16 * v74);
LABEL_73:
      ++v70;
      sub_A188E0((__int64)&v107, (unsigned int)(*((_DWORD *)v72 + 2) - 1));
    }
    while ( v68 != v70 );
    v1 = v69;
    v39 = v93;
LABEL_79:
    v77 = *(unsigned int **)(v39 + 40);
    v78 = &v77[*(_DWORD *)(v39 + 8) >> 8];
    while ( v78 != v77 )
    {
      v79 = *v77++;
      sub_A188E0((__int64)&v107, v79);
    }
    v42 = 0;
    v40 = 26;
LABEL_37:
    v36 = v40;
    v38 += 8;
    sub_A1FB70(*v1, v40, (__int64)&v107, v42);
    LODWORD(v108) = 0;
  }
  while ( v102 != v38 );
LABEL_38:
  result = sub_A192A0(*v1);
  if ( v104 )
    result = sub_A191D0(v104);
  if ( v107 != v109 )
    return _libc_free(v107, v36);
  return result;
}
