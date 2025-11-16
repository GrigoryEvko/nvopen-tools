// Function: sub_323D190
// Address: 0x323d190
//
void __fastcall sub_323D190(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6)
{
  __int64 *v6; // r15
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 *v11; // rax
  __int64 v12; // r14
  const char *v13; // rbx
  const char *v14; // r14
  const char *v15; // r13
  char v16; // cl
  __int64 *v17; // rdi
  int v18; // esi
  unsigned int v19; // edx
  __int64 *v20; // rax
  __int64 v21; // rdi
  const char **v22; // rdi
  char *v23; // rsi
  __int64 *v24; // r13
  __int64 v25; // rax
  _QWORD *v26; // rax
  unsigned int v27; // esi
  unsigned int v28; // eax
  __int64 *v29; // r15
  unsigned int v30; // edx
  __int64 v31; // rcx
  unsigned __int64 v32; // rsi
  int v33; // eax
  const char **v34; // rdx
  const char **v35; // rcx
  const char **v36; // rdx
  const char *v37; // rax
  const char *v38; // rax
  const char *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // r12
  __int64 **v42; // rbx
  unsigned __int64 v43; // rdx
  __int64 v44; // rdi
  void (*v45)(); // rax
  __int64 v46; // rdi
  void (*v47)(); // rax
  __int64 v48; // r14
  __int64 v49; // r13
  unsigned __int64 v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rdi
  void (*v53)(); // rax
  unsigned int v54; // eax
  __int64 v55; // rdi
  void (*v56)(); // rax
  __int64 *v57; // r12
  __int64 v58; // rax
  void (*v59)(); // rbx
  const char *v60; // rax
  unsigned __int64 v61; // rdx
  const char **v62; // rbx
  const char **v63; // r12
  unsigned __int64 v64; // rdi
  __int64 v65; // rbx
  _QWORD *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rdi
  void (*v69)(); // rax
  __int64 v70; // r13
  void (*v71)(); // r14
  const char *v72; // rax
  unsigned __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // rdi
  void (*v77)(); // rax
  void (__fastcall *v78)(__int64 *, _QWORD, _QWORD, _QWORD); // rbx
  unsigned int v79; // eax
  int v80; // r10d
  __int64 *v81; // rdi
  unsigned int v82; // edx
  const char *v83; // rsi
  __int64 *v84; // rax
  const char **v85; // r13
  __int64 *v86; // rdi
  unsigned int v87; // edx
  const char *v88; // rsi
  __int64 v89; // [rsp+8h] [rbp-3C8h]
  const char **v90; // [rsp+10h] [rbp-3C0h]
  const char **v92; // [rsp+20h] [rbp-3B0h]
  __int64 v94; // [rsp+28h] [rbp-3A8h]
  __int64 v95; // [rsp+28h] [rbp-3A8h]
  void (*v97)(); // [rsp+30h] [rbp-3A0h]
  void (*v98)(); // [rsp+30h] [rbp-3A0h]
  void (__fastcall *v99)(__int64 *, _QWORD, _QWORD, _QWORD); // [rsp+30h] [rbp-3A0h]
  unsigned int v100; // [rsp+38h] [rbp-398h]
  char v101; // [rsp+3Ch] [rbp-394h]
  char v102; // [rsp+3Dh] [rbp-393h]
  unsigned __int16 v103; // [rsp+3Eh] [rbp-392h]
  __int64 **v105; // [rsp+40h] [rbp-390h]
  const char *v106; // [rsp+50h] [rbp-380h] BYREF
  unsigned __int64 v107; // [rsp+58h] [rbp-378h]
  __int64 v108; // [rsp+60h] [rbp-370h]
  __int64 v109; // [rsp+68h] [rbp-368h]
  __int16 v110; // [rsp+70h] [rbp-360h]
  __int64 v111; // [rsp+80h] [rbp-350h] BYREF
  __int64 v112; // [rsp+88h] [rbp-348h]
  __int64 *v113; // [rsp+90h] [rbp-340h] BYREF
  unsigned int v114; // [rsp+98h] [rbp-338h]
  const char **v115; // [rsp+190h] [rbp-240h] BYREF
  __int64 v116; // [rsp+198h] [rbp-238h]
  _BYTE v117[560]; // [rsp+1A0h] [rbp-230h] BYREF

  v6 = (__int64 *)a2;
  v100 = *(_DWORD *)(*(_QWORD *)(a2 + 208) + 8LL);
  v103 = sub_3220AA0(a1);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 224) + 208LL))(*(_QWORD *)(a2 + 224), a3, 0);
  v11 = (__int64 *)&v113;
  v111 = 0;
  v112 = 1;
  do
  {
    *v11 = -4096;
    v11 += 2;
  }
  while ( v11 != (__int64 *)&v115 );
  v12 = *(unsigned int *)(a4 + 8);
  v13 = *(const char **)a4;
  v115 = (const char **)v117;
  v14 = &v13[16 * v12];
  v116 = 0x1000000000LL;
  if ( v13 == v14 )
    goto LABEL_58;
  do
  {
    while ( 1 )
    {
      v24 = *(__int64 **)v13;
      v25 = **(_QWORD **)v13;
      if ( v25 )
      {
        v15 = *(const char **)(v25 + 8);
        v16 = v112 & 1;
        if ( (v112 & 1) != 0 )
          goto LABEL_6;
      }
      else
      {
        if ( (*((_BYTE *)v24 + 9) & 0x70) != 0x20 || *((char *)v24 + 8) < 0 )
          BUG();
        *((_BYTE *)v24 + 8) |= 8u;
        v26 = sub_E807D0(v24[3]);
        *v24 = (__int64)v26;
        v15 = (const char *)v26[1];
        v16 = v112 & 1;
        if ( (v112 & 1) != 0 )
        {
LABEL_6:
          v17 = (__int64 *)&v113;
          v18 = 15;
          goto LABEL_7;
        }
      }
      v27 = v114;
      v17 = v113;
      if ( !v114 )
      {
        v28 = v112;
        ++v111;
        v29 = 0;
        v30 = ((unsigned int)v112 >> 1) + 1;
        goto LABEL_20;
      }
      v18 = v114 - 1;
LABEL_7:
      v19 = v18 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v20 = &v17[2 * v19];
      v10 = *v20;
      if ( v15 == (const char *)*v20 )
      {
LABEL_8:
        v21 = *((unsigned int *)v20 + 2);
        goto LABEL_9;
      }
      v80 = 1;
      v29 = 0;
      while ( v10 != -4096 )
      {
        if ( !v29 && v10 == -8192 )
          v29 = v20;
        v9 = (unsigned int)(v80 + 1);
        v19 = v18 & (v80 + v19);
        v20 = &v17[2 * v19];
        v10 = *v20;
        if ( v15 == (const char *)*v20 )
          goto LABEL_8;
        ++v80;
      }
      if ( !v29 )
        v29 = v20;
      v28 = v112;
      ++v111;
      v30 = ((unsigned int)v112 >> 1) + 1;
      if ( !v16 )
      {
        v27 = v114;
LABEL_20:
        if ( 3 * v27 > 4 * v30 )
          goto LABEL_21;
        goto LABEL_97;
      }
      v27 = 16;
      if ( 4 * v30 < 0x30 )
      {
LABEL_21:
        v31 = v27 - HIDWORD(v112) - v30;
        if ( (unsigned int)v31 > v27 >> 3 )
          goto LABEL_22;
        sub_323BA50((__int64)&v111, v27);
        if ( (v112 & 1) != 0 )
        {
          v86 = (__int64 *)&v113;
          v31 = 15;
        }
        else
        {
          v86 = v113;
          if ( !v114 )
          {
LABEL_137:
            LODWORD(v112) = (2 * ((unsigned int)v112 >> 1) + 2) | v112 & 1;
            BUG();
          }
          v31 = v114 - 1;
        }
        v28 = v112;
        v87 = v31 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v9 = 16LL * v87;
        v29 = (__int64 *)((char *)v86 + v9);
        v88 = *(const char **)((char *)v86 + v9);
        if ( v15 == v88 )
          goto LABEL_22;
        v10 = 1;
        v84 = 0;
        while ( v88 != (const char *)-4096LL )
        {
          if ( !v84 && v88 == (const char *)-8192LL )
            v84 = v29;
          v9 = (unsigned int)(v10 + 1);
          v87 = v31 & (v10 + v87);
          v29 = &v86[2 * v87];
          v88 = (const char *)*v29;
          if ( v15 == (const char *)*v29 )
            goto LABEL_123;
          v10 = (unsigned int)v9;
        }
        goto LABEL_121;
      }
LABEL_97:
      sub_323BA50((__int64)&v111, 2 * v27);
      if ( (v112 & 1) != 0 )
      {
        v81 = (__int64 *)&v113;
        v31 = 15;
      }
      else
      {
        v81 = v113;
        if ( !v114 )
          goto LABEL_137;
        v31 = v114 - 1;
      }
      v28 = v112;
      v82 = v31 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v9 = 16LL * v82;
      v29 = (__int64 *)((char *)v81 + v9);
      v83 = *(const char **)((char *)v81 + v9);
      if ( v15 == v83 )
        goto LABEL_22;
      v10 = 1;
      v84 = 0;
      while ( v83 != (const char *)-4096LL )
      {
        if ( v83 == (const char *)-8192LL && !v84 )
          v84 = v29;
        v9 = (unsigned int)(v10 + 1);
        v82 = v31 & (v10 + v82);
        v29 = &v81[2 * v82];
        v83 = (const char *)*v29;
        if ( v15 == (const char *)*v29 )
          goto LABEL_123;
        v10 = (unsigned int)v9;
      }
LABEL_121:
      if ( v84 )
        v29 = v84;
LABEL_123:
      v28 = v112;
LABEL_22:
      LODWORD(v112) = (2 * (v28 >> 1) + 2) | v28 & 1;
      if ( *v29 != -4096 )
        --HIDWORD(v112);
      *v29 = (__int64)v15;
      *((_DWORD *)v29 + 2) = 0;
      v21 = (unsigned int)v116;
      v106 = v15;
      v32 = (unsigned int)v116 + 1LL;
      v33 = v116;
      v107 = 0;
      v108 = 0;
      v109 = 0;
      if ( v32 > HIDWORD(v116) )
      {
        v85 = v115;
        if ( v115 > &v106 || &v106 >= &v115[4 * (unsigned int)v116] )
        {
          sub_322E010((__int64)&v115, v32, HIDWORD(v116), v31, v9, v10);
          v21 = (unsigned int)v116;
          v34 = v115;
          v35 = &v106;
          v33 = v116;
        }
        else
        {
          sub_322E010((__int64)&v115, v32, HIDWORD(v116), v31, v9, v10);
          v34 = v115;
          v21 = (unsigned int)v116;
          v35 = (const char **)((char *)v115 + (char *)&v106 - (char *)v85);
          v33 = v116;
        }
      }
      else
      {
        v34 = v115;
        v35 = &v106;
      }
      v36 = &v34[4 * v21];
      if ( v36 )
      {
        *v36 = *v35;
        v37 = v35[1];
        v35[1] = 0;
        v36[1] = v37;
        v38 = v35[2];
        v35[2] = 0;
        v36[2] = v38;
        v39 = v35[3];
        v35[3] = 0;
        v36[3] = v39;
        v21 = (unsigned int)v116;
        v10 = v107;
        v33 = v116;
        LODWORD(v116) = v116 + 1;
        if ( v107 )
        {
          j_j___libc_free_0(v107);
          v21 = (unsigned int)(v116 - 1);
          v33 = v116 - 1;
        }
      }
      else
      {
        LODWORD(v116) = v33 + 1;
      }
      *((_DWORD *)v29 + 2) = v33;
LABEL_9:
      v22 = &v115[4 * v21];
      v106 = v13;
      v23 = (char *)v22[2];
      if ( v23 != v22[3] )
        break;
      v13 += 16;
      sub_3227680((__int64)(v22 + 1), v23, &v106);
      if ( v14 == v13 )
        goto LABEL_31;
    }
    if ( v23 )
    {
      *(_QWORD *)v23 = v13;
      v23 = (char *)v22[2];
    }
    v13 += 16;
    v22[2] = v23 + 8;
  }
  while ( v14 != v13 );
LABEL_31:
  v6 = (__int64 *)a2;
  v40 = 4LL * (unsigned int)v116;
  v89 = *(_QWORD *)(a5 + 520);
  v90 = &v115[v40];
  if ( v115 == &v115[v40] )
    goto LABEL_58;
  v92 = v115;
  v101 = 0;
  v102 = a6 & (*(_QWORD *)(a5 + 520) == 0);
  while ( 2 )
  {
    if ( v102 )
    {
      v65 = **(_QWORD **)v92[1];
      v66 = *(_QWORD **)v65;
      if ( !*(_QWORD *)v65 )
      {
        if ( (*(_BYTE *)(v65 + 9) & 0x70) != 0x20 || *(char *)(v65 + 8) < 0 )
          BUG();
        *(_BYTE *)(v65 + 8) |= 8u;
        v66 = sub_E807D0(*(_QWORD *)(v65 + 24));
        *(_QWORD *)v65 = v66;
      }
      v67 = sub_3222A80(a1, v66[1]);
      v41 = v67;
      if ( v103 > 4u )
      {
        if ( v65 == v67 )
        {
          v42 = (__int64 **)v92[1];
          v105 = (__int64 **)v92[2];
          if ( (unsigned __int64)((char *)v105 - (char *)v42) <= 8 )
          {
            v41 = 0;
            goto LABEL_37;
          }
        }
        v70 = v6[28];
        v71 = *(void (**)())(*(_QWORD *)v70 + 120LL);
        v72 = sub_E0C8C0(1);
        v110 = 261;
        v106 = v72;
        v107 = v73;
        if ( v71 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v71)(v70, &v106, 1);
        sub_31DC9D0((__int64)v6, 1);
        v76 = v6[28];
        v77 = *(void (**)())(*(_QWORD *)v76 + 120LL);
        v106 = "  base address index";
        v110 = 259;
        if ( v77 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v77)(v76, &v106, 1);
        v78 = *(void (__fastcall **)(__int64 *, _QWORD, _QWORD, _QWORD))(*v6 + 424);
        v79 = sub_37291A0(a1 + 4840, v41, 0, v74, v75);
        v78(v6, v79, 0, 0);
      }
      else
      {
        (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v6[28] + 536LL))(v6[28], -1, v100);
        v68 = v6[28];
        v69 = *(void (**)())(*(_QWORD *)v68 + 120LL);
        v106 = "  base address";
        v110 = 259;
        if ( v69 != nullsub_98 )
        {
          ((void (__fastcall *)(__int64, const char **, __int64))v69)(v68, &v106, 1);
          v68 = v6[28];
        }
        sub_E9A500(v68, v41, v100, 0);
      }
      v42 = (__int64 **)v92[1];
      v105 = (__int64 **)v92[2];
      v101 = v102;
      goto LABEL_37;
    }
    if ( v103 <= 4u && v101 )
    {
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v6[28] + 536LL))(v6[28], -1, v100);
      (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v6[28] + 536LL))(v6[28], 0, v100);
      v101 = 0;
      v41 = v89;
      v42 = (__int64 **)v92[1];
      v105 = (__int64 **)v92[2];
    }
    else
    {
      v41 = v89;
      v42 = (__int64 **)v92[1];
      v105 = (__int64 **)v92[2];
    }
LABEL_37:
    if ( v42 != v105 )
    {
      while ( 1 )
      {
        v48 = **v42;
        v49 = (*v42)[1];
        if ( v41 )
          break;
        if ( v103 <= 4u )
        {
          sub_E9A500(v6[28], **v42, v100, 0);
          sub_E9A500(v6[28], v49, v100, 0);
LABEL_47:
          if ( v105 == ++v42 )
            goto LABEL_57;
        }
        else
        {
          v95 = v6[28];
          v98 = *(void (**)())(*(_QWORD *)v95 + 120LL);
          v106 = sub_E0C8C0(3);
          v110 = 261;
          v107 = v50;
          if ( v98 != nullsub_98 )
            ((void (__fastcall *)(__int64, const char **, __int64))v98)(v95, &v106, 1);
          sub_31DC9D0((__int64)v6, 3);
          v52 = v6[28];
          v53 = *(void (**)())(*(_QWORD *)v52 + 120LL);
          v106 = "  start index";
          v110 = 259;
          if ( v53 != nullsub_98 )
            ((void (__fastcall *)(__int64, const char **, __int64))v53)(v52, &v106, 1);
          v99 = *(void (__fastcall **)(__int64 *, _QWORD, _QWORD, _QWORD))(*v6 + 424);
          v54 = sub_37291A0(a1 + 4840, v48, 0, v51, v99);
          v99(v6, v54, 0, 0);
          v55 = v6[28];
          v56 = *(void (**)())(*(_QWORD *)v55 + 120LL);
          v106 = "  length";
          v110 = 259;
          if ( v56 != nullsub_98 )
            ((void (__fastcall *)(__int64, const char **, __int64))v56)(v55, &v106, 1);
          ++v42;
          sub_31DCA60((__int64)v6);
          if ( v105 == v42 )
            goto LABEL_57;
        }
      }
      if ( v103 <= 4u )
      {
        sub_31DCA50((__int64)v6);
        sub_31DCA50((__int64)v6);
      }
      else
      {
        v94 = v6[28];
        v97 = *(void (**)())(*(_QWORD *)v94 + 120LL);
        v106 = sub_E0C8C0(4);
        v110 = 261;
        v107 = v43;
        if ( v97 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v97)(v94, &v106, 1);
        sub_31DC9D0((__int64)v6, 4);
        v44 = v6[28];
        v45 = *(void (**)())(*(_QWORD *)v44 + 120LL);
        v106 = "  starting offset";
        v110 = 259;
        if ( v45 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v45)(v44, &v106, 1);
        sub_31DCA60((__int64)v6);
        v46 = v6[28];
        v47 = *(void (**)())(*(_QWORD *)v46 + 120LL);
        v106 = "  ending offset";
        v110 = 259;
        if ( v47 != nullsub_98 )
          ((void (__fastcall *)(__int64, const char **, __int64))v47)(v46, &v106, 1);
        sub_31DCA60((__int64)v6);
      }
      goto LABEL_47;
    }
LABEL_57:
    v92 += 4;
    if ( v90 != v92 )
      continue;
    break;
  }
LABEL_58:
  v57 = (__int64 *)v6[28];
  v58 = *v57;
  if ( v103 <= 4u )
  {
    (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(v58 + 536))(v6[28], 0, v100);
    (*(void (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)v6[28] + 536LL))(v6[28], 0, v100);
  }
  else
  {
    v59 = *(void (**)())(v58 + 120);
    v60 = sub_E0C8C0(0);
    v110 = 261;
    v106 = v60;
    v107 = v61;
    if ( v59 != nullsub_98 )
      ((void (__fastcall *)(__int64 *, const char **, __int64))v59)(v57, &v106, 1);
    sub_31DC9D0((__int64)v6, 0);
  }
  v62 = v115;
  v63 = &v115[4 * (unsigned int)v116];
  if ( v115 != v63 )
  {
    do
    {
      v64 = (unsigned __int64)*(v63 - 3);
      v63 -= 4;
      if ( v64 )
        j_j___libc_free_0(v64);
    }
    while ( v62 != v63 );
    v63 = v115;
  }
  if ( v63 != (const char **)v117 )
    _libc_free((unsigned __int64)v63);
  if ( (v112 & 1) == 0 )
    sub_C7D6A0((__int64)v113, 16LL * v114, 8);
}
