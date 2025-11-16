// Function: sub_1C62760
// Address: 0x1c62760
//
__int64 __fastcall sub_1C62760(_QWORD *a1, char *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  const void *v6; // rax
  const void *v7; // rsi
  signed __int64 v8; // r12
  __int64 v10; // rax
  char *v11; // r15
  __int64 v12; // r12
  unsigned int v13; // r11d
  __int64 *v14; // r14
  __int64 v15; // rcx
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  int v20; // esi
  __int64 v21; // r15
  char *v22; // r8
  int v23; // edi
  unsigned int v24; // r15d
  char *v25; // r11
  int v26; // esi
  char *v27; // rax
  __int64 v28; // rdx
  char **v29; // rdi
  char *v30; // r10
  unsigned int v31; // ebx
  char *v32; // r14
  __int64 *v33; // r13
  __int64 v34; // r12
  unsigned int v35; // r9d
  __int64 *v36; // rdi
  __int64 v37; // r8
  __int64 v38; // rcx
  __int64 v39; // rdx
  char *v40; // rsi
  __int64 v41; // rbx
  __int64 v42; // rbx
  char v43; // al
  unsigned int v44; // r11d
  unsigned int v45; // esi
  char *v46; // rcx
  __int64 v47; // r10
  __int64 v48; // rdx
  char **v49; // rax
  char *v50; // rdi
  char *v51; // r14
  _QWORD *v52; // rax
  _QWORD *v53; // r12
  _QWORD *v54; // r14
  __int64 *v55; // r13
  unsigned int v56; // r15d
  __int64 v57; // rax
  __int64 **v58; // rdx
  _QWORD *v59; // rdi
  _BYTE *v60; // rsi
  unsigned int v61; // r12d
  _QWORD *v62; // rax
  unsigned int v63; // ebx
  unsigned int v64; // r14d
  _QWORD *v65; // r15
  __int64 v66; // rdx
  __int64 v67; // rax
  _QWORD *v68; // rdi
  _BYTE *v69; // rsi
  __int64 v70; // rdx
  unsigned __int64 v71; // rsi
  __int64 *v72; // rcx
  int v73; // eax
  __int64 v74; // rdx
  __int64 v75; // rdi
  int v76; // r11d
  __int64 *v77; // r9
  int v78; // r11d
  __int64 v79; // rdx
  __int64 v80; // rdi
  __int64 *v81; // r11
  int v82; // r12d
  char **v83; // rcx
  int v84; // edi
  char *v85; // rax
  _BYTE *v86; // rsi
  unsigned int v87; // esi
  _QWORD *v88; // rdx
  __int64 v89; // r8
  unsigned int v90; // eax
  __int64 *v91; // rcx
  _QWORD *v92; // rdi
  _QWORD *v93; // rax
  __int64 v94; // rdi
  __int64 v95; // rbx
  __int64 v96; // rbx
  char v97; // r8
  __int64 v98; // rax
  __int64 v99; // rax
  unsigned int v100; // esi
  char *v101; // rcx
  __int64 v102; // r8
  __int64 v103; // rdx
  char **v104; // rax
  char *v105; // rdi
  __int64 result; // rax
  int v107; // r10d
  __int64 *v108; // r11
  int v109; // eax
  int v110; // ecx
  char **v111; // rbx
  __int64 v112; // rbx
  int v113; // edx
  __int64 v114; // rbx
  int v115; // edx
  int v116; // r11d
  char **v117; // r10
  int v118; // ebx
  char **v119; // r12
  _QWORD *v120; // rbx
  char *v121; // [rsp+10h] [rbp-100h]
  int v122; // [rsp+10h] [rbp-100h]
  signed __int64 v124; // [rsp+20h] [rbp-F0h]
  char *v125; // [rsp+28h] [rbp-E8h]
  __int64 *v126; // [rsp+28h] [rbp-E8h]
  int v128; // [rsp+38h] [rbp-D8h]
  int v129; // [rsp+40h] [rbp-D0h]
  unsigned int v131; // [rsp+58h] [rbp-B8h]
  char **v133; // [rsp+68h] [rbp-A8h]
  __int64 v134; // [rsp+68h] [rbp-A8h]
  unsigned int v135; // [rsp+68h] [rbp-A8h]
  __int64 v136; // [rsp+70h] [rbp-A0h]
  char *v137; // [rsp+70h] [rbp-A0h]
  unsigned int v138; // [rsp+70h] [rbp-A0h]
  unsigned int v139; // [rsp+70h] [rbp-A0h]
  char *v140; // [rsp+70h] [rbp-A0h]
  char *v141; // [rsp+78h] [rbp-98h] BYREF
  char v142; // [rsp+87h] [rbp-89h] BYREF
  _QWORD *v143; // [rsp+88h] [rbp-88h] BYREF
  char *v144; // [rsp+90h] [rbp-80h] BYREF
  char *v145; // [rsp+98h] [rbp-78h] BYREF
  __int64 v146; // [rsp+A0h] [rbp-70h] BYREF
  __int64 v147; // [rsp+A8h] [rbp-68h] BYREF
  char *v148; // [rsp+B0h] [rbp-60h] BYREF
  __int64 *v149; // [rsp+B8h] [rbp-58h] BYREF
  __int64 v150; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v151; // [rsp+C8h] [rbp-48h]
  __int64 v152; // [rsp+D0h] [rbp-40h]
  unsigned int v153; // [rsp+D8h] [rbp-38h]

  v6 = *(const void **)(a3 + 8);
  v141 = a2;
  v7 = *(const void **)a3;
  v8 = (signed __int64)v6 - *(_QWORD *)a3;
  v124 = v8;
  if ( v6 == *(const void **)a3 )
  {
    v11 = 0;
  }
  else
  {
    if ( (unsigned __int64)v8 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, v7, a3);
    v10 = sub_22077B0(v8);
    v7 = *(const void **)a3;
    v11 = (char *)v10;
    v6 = *(const void **)(a3 + 8);
    v8 = (signed __int64)v6 - *(_QWORD *)a3;
  }
  if ( v7 != v6 )
    memmove(v11, v7, v8);
  v12 = v8 >> 3;
  v150 = 0;
  v151 = 0;
  v13 = v12;
  v152 = 0;
  v153 = 0;
  v128 = v12;
  if ( !(_DWORD)v12 )
    goto LABEL_31;
  v133 = (char **)v11;
  v14 = (__int64 *)v11;
  v125 = v11;
  v136 = (__int64)&v11[8 * (unsigned int)(v12 - 1) + 8];
  do
  {
    v19 = *v14;
    v147 = 0;
    v146 = v19;
    v148 = 0;
    sub_1C620D0(a1, *(unsigned int **)v19, *(__int64 ***)(v19 + 8), &v147, (unsigned __int64 *)&v148, a6, 0);
    v20 = v153;
    v21 = v147;
    v22 = v148;
    if ( !v153 )
    {
      ++v150;
LABEL_12:
      v121 = v148;
      v20 = 2 * v153;
      goto LABEL_13;
    }
    v15 = v146;
    v16 = (v153 - 1) & (((unsigned int)v146 >> 9) ^ ((unsigned int)v146 >> 4));
    v17 = (__int64 *)(v151 + 24LL * v16);
    v18 = *v17;
    if ( *v17 == v146 )
      goto LABEL_9;
    v122 = 1;
    v81 = 0;
    while ( v18 != -8 )
    {
      if ( !v81 && v18 == -16 )
        v81 = v17;
      v16 = (v153 - 1) & (v122 + v16);
      v17 = (__int64 *)(v151 + 24LL * v16);
      v18 = *v17;
      if ( v146 == *v17 )
        goto LABEL_9;
      ++v122;
    }
    if ( v81 )
      v17 = v81;
    ++v150;
    v23 = v152 + 1;
    if ( 4 * ((int)v152 + 1) >= 3 * v153 )
      goto LABEL_12;
    if ( v153 - HIDWORD(v152) - v23 > v153 >> 3 )
      goto LABEL_14;
    v121 = v148;
LABEL_13:
    sub_1C52E00((__int64)&v150, v20);
    sub_1C50190((__int64)&v150, &v146, &v149);
    v17 = v149;
    v15 = v146;
    v22 = v121;
    v23 = v152 + 1;
LABEL_14:
    LODWORD(v152) = v23;
    if ( *v17 != -8 )
      --HIDWORD(v152);
    *v17 = v15;
    v17[1] = 0;
    v17[2] = 0;
LABEL_9:
    v17[1] = v21;
    ++v14;
    v17[2] = (__int64)v22;
  }
  while ( (__int64 *)v136 != v14 );
  v131 = 0;
  v24 = v12;
  v25 = v125;
  while ( 2 )
  {
    v26 = v153;
    v27 = *v133;
    v148 = *v133;
    if ( v153 )
    {
      LODWORD(v28) = (v153 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v29 = (char **)(v151 + 24LL * (unsigned int)v28);
      v30 = *v29;
      if ( v27 == *v29 )
      {
        v137 = v29[1];
        goto LABEL_21;
      }
      v82 = 1;
      v83 = 0;
      while ( v30 != (char *)-8LL )
      {
        if ( v83 || v30 != (char *)-16LL )
          v29 = v83;
        v28 = (v153 - 1) & ((_DWORD)v28 + v82);
        v111 = (char **)(v151 + 24 * v28);
        v30 = *v111;
        if ( v27 == *v111 )
        {
          v137 = v111[1];
          goto LABEL_21;
        }
        v83 = v29;
        ++v82;
        v29 = (char **)(v151 + 24 * v28);
      }
      if ( !v83 )
        v83 = v29;
      ++v150;
      v84 = v152 + 1;
      if ( 4 * ((int)v152 + 1) < 3 * v153 )
      {
        if ( v153 - HIDWORD(v152) - v84 > v153 >> 3 )
          goto LABEL_99;
        v140 = v25;
        goto LABEL_134;
      }
    }
    else
    {
      ++v150;
    }
    v140 = v25;
    v26 = 2 * v153;
LABEL_134:
    sub_1C52E00((__int64)&v150, v26);
    sub_1C50190((__int64)&v150, (__int64 *)&v148, &v149);
    v83 = (char **)v149;
    v27 = v148;
    v25 = v140;
    v84 = v152 + 1;
LABEL_99:
    LODWORD(v152) = v84;
    if ( *v83 != (char *)-8LL )
      --HIDWORD(v152);
    *v83 = v27;
    v83[1] = 0;
    v83[2] = 0;
    v27 = v148;
    v137 = 0;
LABEL_21:
    v31 = ++v131;
    if ( v24 == v131 )
      goto LABEL_30;
LABEL_22:
    v32 = v25;
LABEL_24:
    v33 = (__int64 *)&v32[8 * v31];
    v34 = *v33;
    if ( *(_QWORD *)(*v33 + 8) != *(_QWORD *)v27 )
      goto LABEL_23;
    if ( v153 )
    {
      v35 = (v153 - 1) & (((unsigned int)v34 >> 4) ^ ((unsigned int)v34 >> 9));
      v36 = (__int64 *)(v151 + 24LL * v35);
      v37 = *v36;
      if ( v34 == *v36 )
      {
        v38 = v36[1];
        v39 = v36[2];
        goto LABEL_28;
      }
      v129 = 1;
      v72 = 0;
      while ( v37 != -8 )
      {
        if ( v37 != -16 || v72 )
          v36 = v72;
        v35 = (v153 - 1) & (v129 + v35);
        v126 = (__int64 *)(v151 + 24LL * v35);
        v37 = *v126;
        if ( v34 == *v126 )
        {
          v38 = v126[1];
          v39 = v126[2];
          goto LABEL_28;
        }
        ++v129;
        v72 = v36;
        v36 = (__int64 *)(v151 + 24LL * v35);
      }
      if ( !v72 )
        v72 = v36;
      ++v150;
      v73 = v152 + 1;
      if ( 4 * ((int)v152 + 1) < 3 * v153 )
      {
        if ( v153 - HIDWORD(v152) - v73 <= v153 >> 3 )
        {
          sub_1C52E00((__int64)&v150, v153);
          if ( !v153 )
          {
LABEL_219:
            LODWORD(v152) = v152 + 1;
            BUG();
          }
          v77 = 0;
          v78 = 1;
          LODWORD(v79) = (v153 - 1) & (((unsigned int)v34 >> 4) ^ ((unsigned int)v34 >> 9));
          v72 = (__int64 *)(v151 + 24LL * (unsigned int)v79);
          v80 = *v72;
          v73 = v152 + 1;
          if ( v34 != *v72 )
          {
            while ( v80 != -8 )
            {
              if ( v80 == -16 && !v77 )
                v77 = v72;
              v79 = (v153 - 1) & ((_DWORD)v79 + v78);
              v72 = (__int64 *)(v151 + 24 * v79);
              v80 = *v72;
              if ( v34 == *v72 )
                goto LABEL_67;
              ++v78;
            }
            goto LABEL_75;
          }
        }
        goto LABEL_67;
      }
    }
    else
    {
      ++v150;
    }
    sub_1C52E00((__int64)&v150, 2 * v153);
    if ( !v153 )
      goto LABEL_219;
    LODWORD(v74) = (v153 - 1) & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
    v72 = (__int64 *)(v151 + 24LL * (unsigned int)v74);
    v75 = *v72;
    v73 = v152 + 1;
    if ( v34 != *v72 )
    {
      v76 = 1;
      v77 = 0;
      while ( v75 != -8 )
      {
        if ( v75 == -16 && !v77 )
          v77 = v72;
        v74 = (v153 - 1) & ((_DWORD)v74 + v76);
        v72 = (__int64 *)(v151 + 24 * v74);
        v75 = *v72;
        if ( v34 == *v72 )
          goto LABEL_67;
        ++v76;
      }
LABEL_75:
      if ( v77 )
        v72 = v77;
    }
LABEL_67:
    LODWORD(v152) = v73;
    if ( *v72 != -8 )
      --HIDWORD(v152);
    *v72 = v34;
    v39 = 0;
    v72[1] = 0;
    v72[2] = 0;
    v38 = 0;
    v27 = v148;
LABEL_28:
    if ( v137 != (char *)v39 )
    {
LABEL_23:
      if ( v24 == ++v31 )
      {
        v25 = v32;
        *v133++ = v27;
        continue;
      }
      goto LABEL_24;
    }
    break;
  }
  v31 = v131;
  *v33 = (__int64)v27;
  v25 = v32;
  v27 = (char *)v34;
  v148 = (char *)v34;
  v137 = (char *)v38;
  if ( v24 != v131 )
    goto LABEL_22;
LABEL_30:
  v40 = v25;
  v13 = v24;
  v11 = v40;
  *v133 = v27;
LABEL_31:
  v138 = v13;
  v41 = *(unsigned int *)(a4 + 24);
  v148 = v141;
  v42 = *(_QWORD *)(a4 + 8) + 16 * v41;
  v43 = sub_1C507A0(a4, (__int64 *)&v148, &v149);
  v44 = v138;
  if ( !v43 )
  {
    if ( v42 == *(_QWORD *)(a4 + 8) + 16LL * *(unsigned int *)(a4 + 24) )
      goto LABEL_103;
LABEL_33:
    v45 = *(_DWORD *)(a4 + 24);
    if ( v45 )
    {
      v46 = v141;
      v47 = *(_QWORD *)(a4 + 8);
      LODWORD(v48) = (v45 - 1) & (((unsigned int)v141 >> 9) ^ ((unsigned int)v141 >> 4));
      v49 = (char **)(v47 + 16LL * (unsigned int)v48);
      v50 = *v49;
      if ( v141 == *v49 )
      {
LABEL_35:
        v51 = v49[1];
        goto LABEL_36;
      }
      v118 = 1;
      v119 = 0;
      while ( v50 != (char *)-8LL )
      {
        if ( !v119 && v50 == (char *)-16LL )
          v119 = v49;
        v48 = (v45 - 1) & ((_DWORD)v48 + v118);
        v49 = (char **)(v47 + 16 * v48);
        v50 = *v49;
        if ( v141 == *v49 )
          goto LABEL_35;
        ++v118;
      }
      if ( v119 )
        v49 = v119;
      ++*(_QWORD *)a4;
      v115 = *(_DWORD *)(a4 + 16) + 1;
      if ( 4 * v115 < 3 * v45 )
      {
        v114 = a4;
        if ( v45 - *(_DWORD *)(a4 + 20) - v115 > v45 >> 3 )
          goto LABEL_174;
        v135 = v138;
LABEL_173:
        sub_1C532A0(v114, v45);
        sub_1C507A0(v114, (__int64 *)&v141, &v149);
        v49 = (char **)v149;
        v46 = v141;
        v44 = v135;
        v115 = *(_DWORD *)(v114 + 16) + 1;
LABEL_174:
        *(_DWORD *)(a4 + 16) = v115;
        if ( *v49 != (char *)-8LL )
          --*(_DWORD *)(a4 + 20);
        *v49 = v46;
        v51 = 0;
        v49[1] = 0;
        goto LABEL_36;
      }
    }
    else
    {
      ++*(_QWORD *)a4;
    }
    v135 = v138;
    v114 = a4;
    v45 *= 2;
    goto LABEL_173;
  }
  if ( (__int64 *)v42 != v149 )
    goto LABEL_33;
LABEL_103:
  v85 = (char *)sub_22077B0(24);
  v44 = v138;
  v51 = v85;
  if ( v85 )
  {
    *(_QWORD *)v85 = 0;
    *((_QWORD *)v85 + 1) = 0;
    *((_QWORD *)v85 + 2) = 0;
  }
LABEL_36:
  v139 = v44;
  v52 = (_QWORD *)sub_22077B0(24);
  v53 = v52;
  if ( v52 )
  {
    *v52 = 0;
    v52[1] = 0;
    v52[2] = 0;
    v143 = v52;
    if ( !v128 )
      goto LABEL_116;
  }
  else
  {
    v143 = 0;
    if ( !v128 )
      goto LABEL_119;
  }
  v134 = (__int64)v51;
  v54 = a1;
  v55 = (__int64 *)v11;
  v56 = v139;
  while ( 2 )
  {
    v57 = *v55;
    v144 = 0;
    v145 = 0;
    v146 = v57;
    v58 = *(__int64 ***)(v57 + 8);
    v142 = 0;
    sub_1C620D0(v54, *(unsigned int **)v57, v58, &v144, (unsigned __int64 *)&v145, a6, &v142);
    v59 = v143;
    v60 = (_BYTE *)v143[1];
    if ( v60 == (_BYTE *)v143[2] )
    {
      sub_1C50D80((__int64)v143, v60, &v146);
    }
    else
    {
      if ( v60 )
      {
        *(_QWORD *)v60 = v146;
        v60 = (_BYTE *)v59[1];
      }
      v59[1] = v60 + 8;
    }
    v61 = 0;
    if ( v56 != 1 )
    {
      v62 = v54;
      v63 = 1;
      v64 = v56;
      v65 = v62;
      while ( 1 )
      {
        while ( 1 )
        {
          v67 = v55[v63];
          v147 = v67;
          if ( *(_QWORD *)(v146 + 8) == *(_QWORD *)v67 )
          {
            sub_1C620D0(v65, *(unsigned int **)v67, *(__int64 ***)(v67 + 8), &v148, (unsigned __int64 *)&v149, a6, 0);
            v67 = v147;
            if ( v145 == v148 )
              break;
          }
          v66 = v61++;
          v55[v66] = v67;
LABEL_46:
          if ( ++v63 == v64 )
            goto LABEL_53;
        }
        v144 = v148;
        v68 = v143;
        v146 = v147;
        v69 = (_BYTE *)v143[1];
        v145 = (char *)v149;
        if ( v69 == (_BYTE *)v143[2] )
        {
          sub_1C50D80((__int64)v143, v69, &v147);
          goto LABEL_46;
        }
        if ( v69 )
        {
          *(_QWORD *)v69 = v147;
          v69 = (_BYTE *)v68[1];
        }
        ++v63;
        v68[1] = v69 + 8;
        if ( v63 == v64 )
        {
LABEL_53:
          v54 = v65;
          break;
        }
      }
    }
    v70 = *v143;
    v71 = v143[1] - *v143;
    if ( v142 )
    {
      if ( v71 <= 8 )
      {
LABEL_56:
        if ( v143[1] != v70 )
          v143[1] = v70;
        if ( !v61 )
          break;
        goto LABEL_59;
      }
    }
    else if ( v71 <= 0x10 )
    {
      goto LABEL_56;
    }
    v86 = *(_BYTE **)(v134 + 8);
    if ( v86 == *(_BYTE **)(v134 + 16) )
    {
      sub_1C50BF0(v134, v86, &v143);
    }
    else
    {
      if ( v86 )
      {
        *(_QWORD *)v86 = v143;
        v86 = *(_BYTE **)(v134 + 8);
      }
      *(_QWORD *)(v134 + 8) = v86 + 8;
    }
    v87 = *((_DWORD *)v54 + 64);
    if ( !v87 )
    {
      ++v54[29];
LABEL_146:
      v87 *= 2;
      goto LABEL_147;
    }
    v88 = v143;
    v89 = v54[30];
    v90 = (v87 - 1) & (((unsigned int)v143 >> 9) ^ ((unsigned int)v143 >> 4));
    v91 = (__int64 *)(v89 + 8LL * v90);
    v92 = (_QWORD *)*v91;
    if ( (_QWORD *)*v91 == v143 )
      goto LABEL_112;
    v107 = 1;
    v108 = 0;
    while ( v92 != (_QWORD *)-8LL )
    {
      if ( v92 != (_QWORD *)-16LL || v108 )
        v91 = v108;
      v90 = (v87 - 1) & (v107 + v90);
      v120 = (_QWORD *)(v89 + 8LL * v90);
      v92 = (_QWORD *)*v120;
      if ( v143 == (_QWORD *)*v120 )
        goto LABEL_112;
      ++v107;
      v108 = v91;
      v91 = (__int64 *)(v89 + 8LL * v90);
    }
    v109 = *((_DWORD *)v54 + 62);
    if ( !v108 )
      v108 = v91;
    ++v54[29];
    v110 = v109 + 1;
    if ( 4 * (v109 + 1) >= 3 * v87 )
      goto LABEL_146;
    if ( v87 - *((_DWORD *)v54 + 63) - v110 <= v87 >> 3 )
    {
LABEL_147:
      sub_1C52C60((__int64)(v54 + 29), v87);
      sub_1C50240((__int64)(v54 + 29), (__int64 *)&v143, &v149);
      v108 = v149;
      v88 = v143;
      v110 = *((_DWORD *)v54 + 62) + 1;
    }
    *((_DWORD *)v54 + 62) = v110;
    if ( *v108 != -8 )
      --*((_DWORD *)v54 + 63);
    *v108 = (__int64)v88;
LABEL_112:
    v93 = (_QWORD *)sub_22077B0(24);
    if ( v93 )
    {
      *v93 = 0;
      v93[1] = 0;
      v93[2] = 0;
    }
    v143 = v93;
    if ( v61 )
    {
LABEL_59:
      v56 = v61;
      continue;
    }
    break;
  }
  v53 = v143;
  v11 = (char *)v55;
  v51 = (char *)v134;
  if ( v143 )
  {
LABEL_116:
    if ( *v53 )
      j_j___libc_free_0(*v53, v53[2] - *v53);
    j_j___libc_free_0(v53, 24);
  }
LABEL_119:
  v94 = *(_QWORD *)v51;
  if ( *((_QWORD *)v51 + 1) == *(_QWORD *)v51 )
  {
    if ( v94 )
      j_j___libc_free_0(v94, *((_QWORD *)v51 + 2) - v94);
    j_j___libc_free_0(v51, 24);
  }
  else
  {
    v95 = *(unsigned int *)(a4 + 24);
    v148 = v141;
    v96 = *(_QWORD *)(a4 + 8) + 16 * v95;
    v97 = sub_1C507A0(a4, (__int64 *)&v148, &v149);
    v98 = (__int64)v149;
    if ( !v97 )
      v98 = *(_QWORD *)(a4 + 8) + 16LL * *(unsigned int *)(a4 + 24);
    if ( v96 != v98 )
    {
      v99 = a4;
      v100 = *(_DWORD *)(a4 + 24);
      if ( v100 )
        goto LABEL_124;
LABEL_166:
      ++*(_QWORD *)a4;
LABEL_167:
      v112 = a4;
      v100 *= 2;
      goto LABEL_168;
    }
    sub_1C55CE0(a5, &v141);
    v99 = a4;
    v100 = *(_DWORD *)(a4 + 24);
    if ( !v100 )
      goto LABEL_166;
LABEL_124:
    v101 = v141;
    v102 = *(_QWORD *)(v99 + 8);
    LODWORD(v103) = (v100 - 1) & (((unsigned int)v141 >> 9) ^ ((unsigned int)v141 >> 4));
    v104 = (char **)(v102 + 16LL * (unsigned int)v103);
    v105 = *v104;
    if ( *v104 != v141 )
    {
      v116 = 1;
      v117 = 0;
      while ( v105 != (char *)-8LL )
      {
        if ( !v117 && v105 == (char *)-16LL )
          v117 = v104;
        v103 = (v100 - 1) & ((_DWORD)v103 + v116);
        v104 = (char **)(v102 + 16 * v103);
        v105 = *v104;
        if ( v141 == *v104 )
          goto LABEL_125;
        ++v116;
      }
      if ( v117 )
        v104 = v117;
      ++*(_QWORD *)a4;
      v113 = *(_DWORD *)(a4 + 16) + 1;
      if ( 4 * v113 >= 3 * v100 )
        goto LABEL_167;
      v112 = a4;
      if ( v100 - *(_DWORD *)(a4 + 20) - v113 <= v100 >> 3 )
      {
LABEL_168:
        sub_1C532A0(v112, v100);
        sub_1C507A0(v112, (__int64 *)&v141, &v149);
        v104 = (char **)v149;
        v101 = v141;
        v113 = *(_DWORD *)(v112 + 16) + 1;
      }
      *(_DWORD *)(a4 + 16) = v113;
      if ( *v104 != (char *)-8LL )
        --*(_DWORD *)(a4 + 20);
      *v104 = v101;
      v104[1] = 0;
    }
LABEL_125:
    v104[1] = v51;
  }
  result = j___libc_free_0(v151);
  if ( v11 )
    return j_j___libc_free_0(v11, v124);
  return result;
}
