// Function: sub_323BE90
// Address: 0x323be90
//
void __fastcall sub_323BE90(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __int64 a7,
        __int64 *a8)
{
  __int64 *v8; // r15
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 *v13; // rax
  __int64 v14; // r14
  const char *v15; // rbx
  const char *v16; // r14
  const char *v17; // r13
  char v18; // cl
  __int64 *v19; // rdi
  int v20; // esi
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // rdi
  const char **v24; // rdi
  char *v25; // rsi
  __int64 *v26; // r13
  __int64 v27; // rax
  _QWORD *v28; // rax
  unsigned int v29; // esi
  unsigned int v30; // eax
  __int64 *v31; // r15
  unsigned int v32; // edx
  __int64 v33; // rcx
  unsigned __int64 v34; // rsi
  int v35; // eax
  const char **v36; // rdx
  const char **v37; // rcx
  const char **v38; // rdx
  const char *v39; // rax
  const char *v40; // rax
  const char *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r14
  __int64 **v44; // rax
  __int64 **v45; // rbx
  __int64 v46; // r13
  unsigned __int64 v47; // rdx
  __int64 v48; // rdi
  void (*v49)(); // rax
  __int64 v50; // rdi
  void (*v51)(); // rax
  __int64 *v52; // r12
  __int64 v53; // r14
  unsigned __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rdi
  void (*v57)(); // rax
  unsigned int v58; // eax
  __int64 v59; // rdi
  void (*v60)(); // rax
  __int64 *v61; // r12
  __int64 v62; // rax
  void (*v63)(); // rbx
  const char *v64; // rax
  unsigned __int64 v65; // rdx
  const char **v66; // rbx
  const char **v67; // r12
  unsigned __int64 v68; // rdi
  __int64 v69; // rbx
  _QWORD *v70; // rax
  __int64 v71; // rax
  __int64 v72; // rdi
  void (*v73)(); // rax
  __int64 v74; // r13
  void (*v75)(); // r12
  const char *v76; // rax
  unsigned __int64 v77; // rdx
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // rdi
  void (*v81)(); // rax
  void (__fastcall *v82)(__int64 *, _QWORD, _QWORD, _QWORD); // rbx
  unsigned int v83; // eax
  int v84; // r10d
  __int64 *v85; // rdi
  unsigned int v86; // edx
  const char *v87; // rsi
  __int64 *v88; // rax
  const char **v89; // r13
  __int64 *v90; // rdi
  unsigned int v91; // edx
  const char *v92; // rsi
  __int64 v93; // [rsp+10h] [rbp-3E0h]
  const char **v94; // [rsp+18h] [rbp-3D8h]
  const char **v96; // [rsp+28h] [rbp-3C8h]
  __int64 v97; // [rsp+30h] [rbp-3C0h]
  __int64 v98; // [rsp+30h] [rbp-3C0h]
  void (*v100)(); // [rsp+38h] [rbp-3B8h]
  void (*v101)(); // [rsp+38h] [rbp-3B8h]
  void (__fastcall *v102)(__int64 *, _QWORD, _QWORD, _QWORD); // [rsp+38h] [rbp-3B8h]
  unsigned int v103; // [rsp+50h] [rbp-3A0h]
  char v104; // [rsp+54h] [rbp-39Ch]
  char v105; // [rsp+55h] [rbp-39Bh]
  unsigned __int16 v106; // [rsp+56h] [rbp-39Ah]
  __int64 **v108; // [rsp+58h] [rbp-398h]
  __int64 v110; // [rsp+60h] [rbp-390h]
  const char *v111; // [rsp+70h] [rbp-380h] BYREF
  unsigned __int64 v112; // [rsp+78h] [rbp-378h]
  __int64 v113; // [rsp+80h] [rbp-370h]
  __int64 v114; // [rsp+88h] [rbp-368h]
  __int16 v115; // [rsp+90h] [rbp-360h]
  __int64 v116; // [rsp+A0h] [rbp-350h] BYREF
  __int64 v117; // [rsp+A8h] [rbp-348h]
  __int64 *v118; // [rsp+B0h] [rbp-340h] BYREF
  unsigned int v119; // [rsp+B8h] [rbp-338h]
  const char **v120; // [rsp+1B0h] [rbp-240h] BYREF
  __int64 v121; // [rsp+1B8h] [rbp-238h]
  _BYTE v122[560]; // [rsp+1C0h] [rbp-230h] BYREF

  v8 = (__int64 *)a2;
  v103 = *(_DWORD *)(*(_QWORD *)(a2 + 208) + 8LL);
  v106 = sub_3220AA0(a1);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 224) + 208LL))(*(_QWORD *)(a2 + 224), a3, 0);
  v13 = (__int64 *)&v118;
  v116 = 0;
  v117 = 1;
  do
  {
    *v13 = -4096;
    v13 += 2;
  }
  while ( v13 != (__int64 *)&v120 );
  v14 = *(_QWORD *)(a4 + 8);
  v15 = *(const char **)a4;
  v120 = (const char **)v122;
  v16 = &v15[32 * v14];
  v121 = 0x1000000000LL;
  if ( v15 == v16 )
    goto LABEL_58;
  do
  {
    while ( 1 )
    {
      v26 = *(__int64 **)v15;
      v27 = **(_QWORD **)v15;
      if ( v27 )
      {
        v17 = *(const char **)(v27 + 8);
        v18 = v117 & 1;
        if ( (v117 & 1) != 0 )
          goto LABEL_6;
      }
      else
      {
        if ( (*((_BYTE *)v26 + 9) & 0x70) != 0x20 || *((char *)v26 + 8) < 0 )
          BUG();
        *((_BYTE *)v26 + 8) |= 8u;
        v28 = sub_E807D0(v26[3]);
        *v26 = (__int64)v28;
        v17 = (const char *)v28[1];
        v18 = v117 & 1;
        if ( (v117 & 1) != 0 )
        {
LABEL_6:
          v19 = (__int64 *)&v118;
          v20 = 15;
          goto LABEL_7;
        }
      }
      v29 = v119;
      v19 = v118;
      if ( !v119 )
      {
        v30 = v117;
        ++v116;
        v31 = 0;
        v32 = ((unsigned int)v117 >> 1) + 1;
        goto LABEL_20;
      }
      v20 = v119 - 1;
LABEL_7:
      v21 = v20 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v22 = &v19[2 * v21];
      v12 = *v22;
      if ( v17 == (const char *)*v22 )
      {
LABEL_8:
        v23 = *((unsigned int *)v22 + 2);
        goto LABEL_9;
      }
      v84 = 1;
      v31 = 0;
      while ( v12 != -4096 )
      {
        if ( !v31 && v12 == -8192 )
          v31 = v22;
        v11 = (unsigned int)(v84 + 1);
        v21 = v20 & (v84 + v21);
        v22 = &v19[2 * v21];
        v12 = *v22;
        if ( v17 == (const char *)*v22 )
          goto LABEL_8;
        ++v84;
      }
      if ( !v31 )
        v31 = v22;
      v30 = v117;
      ++v116;
      v32 = ((unsigned int)v117 >> 1) + 1;
      if ( !v18 )
      {
        v29 = v119;
LABEL_20:
        if ( 3 * v29 > 4 * v32 )
          goto LABEL_21;
        goto LABEL_97;
      }
      v29 = 16;
      if ( 4 * v32 < 0x30 )
      {
LABEL_21:
        v33 = v29 - HIDWORD(v117) - v32;
        if ( (unsigned int)v33 > v29 >> 3 )
          goto LABEL_22;
        sub_323BA50((__int64)&v116, v29);
        if ( (v117 & 1) != 0 )
        {
          v90 = (__int64 *)&v118;
          v33 = 15;
        }
        else
        {
          v90 = v118;
          if ( !v119 )
          {
LABEL_137:
            LODWORD(v117) = (2 * ((unsigned int)v117 >> 1) + 2) | v117 & 1;
            BUG();
          }
          v33 = v119 - 1;
        }
        v30 = v117;
        v91 = v33 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v11 = 16LL * v91;
        v31 = (__int64 *)((char *)v90 + v11);
        v92 = *(const char **)((char *)v90 + v11);
        if ( v17 == v92 )
          goto LABEL_22;
        v12 = 1;
        v88 = 0;
        while ( v92 != (const char *)-4096LL )
        {
          if ( !v88 && v92 == (const char *)-8192LL )
            v88 = v31;
          v11 = (unsigned int)(v12 + 1);
          v91 = v33 & (v12 + v91);
          v31 = &v90[2 * v91];
          v92 = (const char *)*v31;
          if ( v17 == (const char *)*v31 )
            goto LABEL_123;
          v12 = (unsigned int)v11;
        }
        goto LABEL_121;
      }
LABEL_97:
      sub_323BA50((__int64)&v116, 2 * v29);
      if ( (v117 & 1) != 0 )
      {
        v85 = (__int64 *)&v118;
        v33 = 15;
      }
      else
      {
        v85 = v118;
        if ( !v119 )
          goto LABEL_137;
        v33 = v119 - 1;
      }
      v30 = v117;
      v86 = v33 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v11 = 16LL * v86;
      v31 = (__int64 *)((char *)v85 + v11);
      v87 = *(const char **)((char *)v85 + v11);
      if ( v17 == v87 )
        goto LABEL_22;
      v12 = 1;
      v88 = 0;
      while ( v87 != (const char *)-4096LL )
      {
        if ( v87 == (const char *)-8192LL && !v88 )
          v88 = v31;
        v11 = (unsigned int)(v12 + 1);
        v86 = v33 & (v12 + v86);
        v31 = &v85[2 * v86];
        v87 = (const char *)*v31;
        if ( v17 == (const char *)*v31 )
          goto LABEL_123;
        v12 = (unsigned int)v11;
      }
LABEL_121:
      if ( v88 )
        v31 = v88;
LABEL_123:
      v30 = v117;
LABEL_22:
      LODWORD(v117) = (2 * (v30 >> 1) + 2) | v30 & 1;
      if ( *v31 != -4096 )
        --HIDWORD(v117);
      *v31 = (__int64)v17;
      *((_DWORD *)v31 + 2) = 0;
      v23 = (unsigned int)v121;
      v111 = v17;
      v34 = (unsigned int)v121 + 1LL;
      v35 = v121;
      v112 = 0;
      v113 = 0;
      v114 = 0;
      if ( v34 > HIDWORD(v121) )
      {
        v89 = v120;
        if ( v120 > &v111 || &v111 >= &v120[4 * (unsigned int)v121] )
        {
          sub_322DF10((__int64)&v120, v34, HIDWORD(v121), v33, v11, v12);
          v23 = (unsigned int)v121;
          v36 = v120;
          v37 = &v111;
          v35 = v121;
        }
        else
        {
          sub_322DF10((__int64)&v120, v34, HIDWORD(v121), v33, v11, v12);
          v36 = v120;
          v23 = (unsigned int)v121;
          v37 = (const char **)((char *)v120 + (char *)&v111 - (char *)v89);
          v35 = v121;
        }
      }
      else
      {
        v36 = v120;
        v37 = &v111;
      }
      v38 = &v36[4 * v23];
      if ( v38 )
      {
        *v38 = *v37;
        v39 = v37[1];
        v37[1] = 0;
        v38[1] = v39;
        v40 = v37[2];
        v37[2] = 0;
        v38[2] = v40;
        v41 = v37[3];
        v37[3] = 0;
        v38[3] = v41;
        v23 = (unsigned int)v121;
        v12 = v112;
        v35 = v121;
        LODWORD(v121) = v121 + 1;
        if ( v112 )
        {
          j_j___libc_free_0(v112);
          v23 = (unsigned int)(v121 - 1);
          v35 = v121 - 1;
        }
      }
      else
      {
        LODWORD(v121) = v35 + 1;
      }
      *((_DWORD *)v31 + 2) = v35;
LABEL_9:
      v24 = &v120[4 * v23];
      v111 = v15;
      v25 = (char *)v24[2];
      if ( v25 != v24[3] )
        break;
      v15 += 32;
      sub_32274F0((__int64)(v24 + 1), v25, &v111);
      if ( v16 == v15 )
        goto LABEL_31;
    }
    if ( v25 )
    {
      *(_QWORD *)v25 = v15;
      v25 = (char *)v24[2];
    }
    v15 += 32;
    v24[2] = v25 + 8;
  }
  while ( v16 != v15 );
LABEL_31:
  v8 = (__int64 *)a2;
  v42 = 4LL * (unsigned int)v121;
  v93 = *(_QWORD *)(a5 + 520);
  v94 = &v120[v42];
  if ( v120 == &v120[v42] )
    goto LABEL_58;
  v96 = v120;
  v104 = 0;
  v105 = a6 & (*(_QWORD *)(a5 + 520) == 0);
  while ( 2 )
  {
    if ( v105 )
    {
      v69 = **(_QWORD **)v96[1];
      v70 = *(_QWORD **)v69;
      if ( !*(_QWORD *)v69 )
      {
        if ( (*(_BYTE *)(v69 + 9) & 0x70) != 0x20 || *(char *)(v69 + 8) < 0 )
          BUG();
        *(_BYTE *)(v69 + 8) |= 8u;
        v70 = sub_E807D0(*(_QWORD *)(v69 + 24));
        *(_QWORD *)v69 = v70;
      }
      v71 = sub_3222A80(a1, v70[1]);
      v43 = v71;
      if ( v106 > 4u )
      {
        if ( v69 == v71 )
        {
          v44 = (__int64 **)v96[1];
          v108 = (__int64 **)v96[2];
          if ( (unsigned __int64)((char *)v108 - (char *)v44) <= 8 )
          {
            v43 = 0;
            goto LABEL_37;
          }
        }
        v74 = v8[28];
        v75 = *(void (**)())(*(_QWORD *)v74 + 120LL);
        v76 = sub_E0C970(1);
        v115 = 261;
        v111 = v76;
        v112 = v77;
        if ( v75 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v75)(v74, &v111, 1);
        sub_31DC9D0((__int64)v8, 1);
        v80 = v8[28];
        v81 = *(void (**)())(*(_QWORD *)v80 + 120LL);
        v111 = "  base address index";
        v115 = 259;
        if ( v81 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v81)(v80, &v111, 1);
        v82 = *(void (__fastcall **)(__int64 *, _QWORD, _QWORD, _QWORD))(*v8 + 424);
        v83 = sub_37291A0(a1 + 4840, v43, 0, v78, v79);
        v82(v8, v83, 0, 0);
      }
      else
      {
        (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v8[28] + 536LL))(v8[28], -1, v103);
        v72 = v8[28];
        v73 = *(void (**)())(*(_QWORD *)v72 + 120LL);
        v111 = "  base address";
        v115 = 259;
        if ( v73 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, const char **, __int64))v73)(v72, &v111, 1);
          v72 = v8[28];
        }
        sub_E9A500(v72, v43, v103, 0);
      }
      v44 = (__int64 **)v96[1];
      v108 = (__int64 **)v96[2];
      v104 = v105;
      goto LABEL_37;
    }
    if ( v106 <= 4u && v104 )
    {
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v8[28] + 536LL))(v8[28], -1, v103);
      (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v8[28] + 536LL))(v8[28], 0, v103);
      v104 = 0;
      v43 = v93;
      v44 = (__int64 **)v96[1];
      v108 = (__int64 **)v96[2];
    }
    else
    {
      v43 = v93;
      v44 = (__int64 **)v96[1];
      v108 = (__int64 **)v96[2];
    }
LABEL_37:
    v45 = v44;
    v46 = v43;
    while ( v108 != v45 )
    {
      v52 = *v45;
      v53 = **v45;
      v110 = (*v45)[1];
      if ( v46 )
      {
        if ( v106 <= 4u )
        {
          sub_31DCA50((__int64)v8);
          sub_31DCA50((__int64)v8);
          goto LABEL_47;
        }
        v97 = v8[28];
        v100 = *(void (**)())(*(_QWORD *)v97 + 120LL);
        v111 = sub_E0C970(4);
        v115 = 261;
        v112 = v47;
        if ( v100 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v100)(v97, &v111, 1);
        sub_31DC9D0((__int64)v8, 4);
        v48 = v8[28];
        v49 = *(void (**)())(*(_QWORD *)v48 + 120LL);
        v111 = "  starting offset";
        v115 = 259;
        if ( v49 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v49)(v48, &v111, 1);
        sub_31DCA60((__int64)v8);
        v50 = v8[28];
        v51 = *(void (**)())(*(_QWORD *)v50 + 120LL);
        v111 = "  ending offset";
        v115 = 259;
        if ( v51 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v51)(v50, &v111, 1);
      }
      else
      {
        if ( v106 <= 4u )
        {
          sub_E9A500(v8[28], v53, v103, 0);
          sub_E9A500(v8[28], v110, v103, 0);
          goto LABEL_47;
        }
        v98 = v8[28];
        v101 = *(void (**)())(*(_QWORD *)v98 + 120LL);
        v111 = sub_E0C970(3);
        v115 = 261;
        v112 = v54;
        if ( v101 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v101)(v98, &v111, 1);
        sub_31DC9D0((__int64)v8, 3);
        v56 = v8[28];
        v57 = *(void (**)())(*(_QWORD *)v56 + 120LL);
        v111 = "  start index";
        v115 = 259;
        if ( v57 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v57)(v56, &v111, 1);
        v102 = *(void (__fastcall **)(__int64 *, _QWORD, _QWORD, _QWORD))(*v8 + 424);
        v58 = sub_37291A0(a1 + 4840, v53, 0, v55, v102);
        v102(v8, v58, 0, 0);
        v59 = v8[28];
        v60 = *(void (**)())(*(_QWORD *)v59 + 120LL);
        v111 = "  length";
        v115 = 259;
        if ( v60 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v60)(v59, &v111, 1);
      }
      sub_31DCA60((__int64)v8);
LABEL_47:
      ++v45;
      sub_3220AC0(a7, v52, *a8);
    }
    v96 += 4;
    if ( v94 != v96 )
      continue;
    break;
  }
LABEL_58:
  v61 = (__int64 *)v8[28];
  v62 = *v61;
  if ( v106 <= 4u )
  {
    (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(v62 + 536))(v8[28], 0, v103);
    (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v8[28] + 536LL))(v8[28], 0, v103);
  }
  else
  {
    v63 = *(void (**)())(v62 + 120);
    v64 = sub_E0C970(0);
    v115 = 261;
    v111 = v64;
    v112 = v65;
    if ( v63 != nullsub_98 )
      ((void (__fastcall *)(__int64 *, const char **, __int64))v63)(v61, &v111, 1);
    sub_31DC9D0((__int64)v8, 0);
  }
  v66 = v120;
  v67 = &v120[4 * (unsigned int)v121];
  if ( v120 != v67 )
  {
    do
    {
      v68 = (unsigned __int64)*(v67 - 3);
      v67 -= 4;
      if ( v68 )
        j_j___libc_free_0(v68);
    }
    while ( v66 != v67 );
    v67 = v120;
  }
  if ( v67 != (const char **)v122 )
    _libc_free((unsigned __int64)v67);
  if ( (v117 & 1) == 0 )
    sub_C7D6A0((__int64)v118, 16LL * v119, 8);
}
