// Function: sub_1198360
// Address: 0x1198360
//
unsigned __int8 *__fastcall sub_1198360(
        __int64 a1,
        unsigned __int8 *a2,
        unsigned __int8 *a3,
        __int64 a4,
        __int64 a5,
        __int64 *a6)
{
  unsigned __int8 *v8; // r12
  char v10; // r10
  __int64 v11; // rdx
  __int64 v12; // r14
  unsigned int v13; // eax
  int v14; // eax
  __int64 v15; // rdx
  unsigned __int8 *v16; // r14
  _QWORD *v17; // rsi
  unsigned __int8 *v18; // r14
  _QWORD *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rax
  _BYTE *v23; // rax
  _BYTE *v24; // rax
  unsigned __int8 *v25; // r15
  _BYTE *v26; // r14
  char v27; // al
  void *v28; // r11
  __int64 *v29; // rdi
  int v30; // esi
  __int64 v31; // rax
  _BOOL4 v32; // esi
  __int64 v33; // rdi
  __int64 v34; // rdx
  int v35; // eax
  int v36; // eax
  unsigned int v37; // eax
  unsigned __int64 v38; // rcx
  unsigned __int64 v39; // rsi
  unsigned __int64 v40; // rcx
  unsigned int v41; // eax
  __int64 v42; // rax
  __int64 v43; // r8
  __int64 v44; // rdx
  unsigned int v45; // esi
  __int64 v46; // rax
  __int64 *v47; // rbx
  _QWORD *v48; // r13
  int v49; // edi
  __int64 v50; // rax
  char v51; // al
  unsigned __int8 *v52; // rdi
  __int64 v53; // rdx
  __int64 v54; // rax
  _BYTE *v55; // r8
  __int64 v56; // rcx
  size_t v57; // r14
  int v58; // r14d
  __int64 v59; // rax
  __int64 *v60; // rdi
  __int64 v61; // rdx
  int v62; // esi
  __int64 v63; // rax
  __int64 v64; // rdx
  __int64 *v65; // rdi
  int v66; // esi
  unsigned __int8 *v67; // r12
  char v68; // si
  char v69; // si
  __int64 v70; // rdx
  _BYTE *v71; // rax
  _BYTE *v72; // rax
  unsigned __int8 v73; // al
  void **v74; // rdx
  void *v75; // r11
  _QWORD *v76; // r10
  __int64 v77; // rcx
  _BYTE *v78; // r10
  unsigned __int8 *v79; // r11
  __int64 v80; // rax
  _BYTE *v81; // rax
  __int64 v82; // rdi
  __int64 *v83; // rdi
  int v84; // esi
  __int64 v85; // rax
  __int64 *v86; // rdi
  __int64 v87; // r14
  int v88; // esi
  __int64 v89; // rax
  __int64 *v90; // rdi
  __int64 v91; // rbx
  __int64 v92; // rax
  __int64 v93; // r12
  unsigned __int8 *v94; // rax
  __int64 v95; // rax
  __int64 v96; // rax
  __int64 v97; // rax
  __int64 v98; // rax
  __int64 v99; // rax
  __int64 v100; // rax
  _QWORD **v101; // rdx
  int v102; // ecx
  __int64 *v103; // rax
  __int64 v104; // rax
  __int16 v105; // r10
  __int64 v106; // r8
  __int64 v107; // rcx
  __int64 v108; // rbx
  __int64 v109; // rdx
  unsigned int v110; // esi
  __int64 v111; // rdx
  _BYTE *v112; // rax
  _BYTE *v113; // r14
  _BYTE *v114; // rdx
  __int64 v115; // rdi
  int v116; // esi
  __int64 *v117; // rdi
  __int64 v118; // rax
  __int64 *v119; // rdi
  int v120; // esi
  __int64 v121; // rax
  __int64 *v122; // rdi
  __int64 v123; // r12
  __int64 v124; // rax
  __int64 v125; // rdx
  _BYTE *v126; // rax
  const char *v127; // rax
  char *v128; // rdi
  int v129; // [rsp+10h] [rbp-D0h]
  __int64 v130; // [rsp+10h] [rbp-D0h]
  __int64 v131; // [rsp+10h] [rbp-D0h]
  __int64 v132; // [rsp+10h] [rbp-D0h]
  __int64 v133; // [rsp+10h] [rbp-D0h]
  char v134; // [rsp+18h] [rbp-C8h]
  _QWORD *v135; // [rsp+18h] [rbp-C8h]
  __int64 v136; // [rsp+18h] [rbp-C8h]
  __int64 v137; // [rsp+18h] [rbp-C8h]
  unsigned int v138; // [rsp+18h] [rbp-C8h]
  __int64 v139; // [rsp+18h] [rbp-C8h]
  __int64 v140; // [rsp+18h] [rbp-C8h]
  __int64 v141; // [rsp+18h] [rbp-C8h]
  unsigned __int8 *v142; // [rsp+18h] [rbp-C8h]
  __int64 v143; // [rsp+18h] [rbp-C8h]
  char srcg; // [rsp+20h] [rbp-C0h]
  unsigned int srch; // [rsp+20h] [rbp-C0h]
  char srci; // [rsp+20h] [rbp-C0h]
  void *srcj; // [rsp+20h] [rbp-C0h]
  unsigned int srck; // [rsp+20h] [rbp-C0h]
  char src; // [rsp+20h] [rbp-C0h]
  void *srca; // [rsp+20h] [rbp-C0h]
  char srcl; // [rsp+20h] [rbp-C0h]
  unsigned __int8 *srcb; // [rsp+20h] [rbp-C0h]
  void *srcc; // [rsp+20h] [rbp-C0h]
  void *srcm; // [rsp+20h] [rbp-C0h]
  _BYTE *srcd; // [rsp+20h] [rbp-C0h]
  void *srce; // [rsp+20h] [rbp-C0h]
  unsigned __int8 *srcn; // [rsp+20h] [rbp-C0h]
  _QWORD *srcf; // [rsp+20h] [rbp-C0h]
  _BYTE *srco; // [rsp+20h] [rbp-C0h]
  char v160; // [rsp+28h] [rbp-B8h]
  __int64 v161; // [rsp+28h] [rbp-B8h]
  unsigned __int8 v162; // [rsp+28h] [rbp-B8h]
  char v163; // [rsp+28h] [rbp-B8h]
  unsigned int v164; // [rsp+28h] [rbp-B8h]
  __int64 v165; // [rsp+28h] [rbp-B8h]
  char v166; // [rsp+28h] [rbp-B8h]
  char v167; // [rsp+28h] [rbp-B8h]
  unsigned int v168; // [rsp+28h] [rbp-B8h]
  __int64 v169; // [rsp+28h] [rbp-B8h]
  __int64 v170; // [rsp+28h] [rbp-B8h]
  __int64 v171; // [rsp+28h] [rbp-B8h]
  unsigned __int8 *v172; // [rsp+28h] [rbp-B8h]
  __int64 v173; // [rsp+28h] [rbp-B8h]
  void *v174; // [rsp+28h] [rbp-B8h]
  __int64 v175; // [rsp+28h] [rbp-B8h]
  __int64 v176; // [rsp+28h] [rbp-B8h]
  __int64 v177; // [rsp+38h] [rbp-A8h]
  const char *v178; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v179; // [rsp+48h] [rbp-98h]
  __int16 v180; // [rsp+60h] [rbp-80h]
  const char *v181; // [rsp+70h] [rbp-70h] BYREF
  const char *v182; // [rsp+78h] [rbp-68h]
  _QWORD v183[2]; // [rsp+80h] [rbp-60h] BYREF
  __int64 v184; // [rsp+90h] [rbp-50h]
  __int64 v185; // [rsp+98h] [rbp-48h]
  __int64 v186; // [rsp+A0h] [rbp-40h]

  v8 = (unsigned __int8 *)a4;
  v10 = *(_BYTE *)a4;
  if ( *(_BYTE *)a4 == *a2 )
  {
    v26 = (_BYTE *)*((_QWORD *)a2 - 8);
    if ( *v26 <= 0x15u && *v26 != 5 )
    {
      v166 = *(_BYTE *)a4;
      v27 = sub_AD6CA0(*((_QWORD *)a2 - 8));
      v10 = v166;
      if ( !v27 )
      {
        v28 = (void *)*((_QWORD *)a2 - 4);
        if ( v28 )
        {
          v29 = *(__int64 **)(a1 + 32);
          LOWORD(v184) = 257;
          v30 = *v8;
          v180 = 257;
          srcj = v28;
          v31 = sub_1194380(v29, v30 - 29, (__int64)v26, (__int64)a3, v177, 0, (__int64)&v178, 0);
          v18 = (unsigned __int8 *)sub_B504D0((unsigned int)*v8 - 29, v31, (__int64)srcj, (__int64)&v181, 0, 0);
          if ( v166 == 54 )
          {
            v68 = 0;
            if ( sub_B448F0((__int64)v8) )
              v68 = sub_B448F0((__int64)a2);
            sub_B447F0(v18, v68);
            v69 = 0;
            if ( sub_B44900((__int64)v8) )
              v69 = sub_B44900((__int64)a2);
            sub_B44850(v18, v69);
          }
          else
          {
            v32 = 0;
            if ( sub_B44E60((__int64)v8) )
              v32 = sub_B44E60((__int64)a2);
            sub_B448B0((__int64)v18, v32);
          }
          return v18;
        }
      }
    }
  }
  v11 = *a3;
  if ( v10 == 54 )
    goto LABEL_9;
  v12 = *((_QWORD *)v8 + 1);
  srcg = *a3;
  v160 = v10;
  v13 = sub_BCB060(v12);
  v10 = v160;
  a4 = (__int64)a3;
  a5 = v13;
  if ( srcg == 17 )
    goto LABEL_4;
  v164 = v13;
  v22 = *((_QWORD *)a3 + 1);
  v11 = (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17;
  if ( (unsigned int)v11 > 1 )
    goto LABEL_20;
  srci = v10;
  v24 = sub_AD7630((__int64)a3, 1, v11);
  v10 = srci;
  a5 = v164;
  a4 = (__int64)v24;
  if ( v24 )
  {
    if ( *v24 == 17 )
    {
LABEL_4:
      if ( *(_DWORD *)(a4 + 32) <= 0x40u )
      {
        v15 = *(_QWORD *)(a4 + 24);
      }
      else
      {
        v129 = *(_DWORD *)(a4 + 32);
        v134 = v10;
        srch = a5;
        v161 = a4;
        v14 = sub_C444A0(a4 + 24);
        a4 = v161;
        a5 = srch;
        v10 = v134;
        if ( (unsigned int)(v129 - v14) > 0x40 )
          goto LABEL_8;
        v15 = **(_QWORD **)(v161 + 24);
      }
      if ( (_DWORD)a5 - 1 != v15 )
        goto LABEL_8;
      if ( *a2 != 49 )
        goto LABEL_8;
      a5 = *((_QWORD *)a2 - 8);
      if ( !a5 )
        goto LABEL_8;
      v33 = *((_QWORD *)a2 - 4);
      v34 = v33 + 24;
      if ( *(_BYTE *)v33 != 17 )
      {
        v170 = *((_QWORD *)a2 - 8);
        v70 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v33 + 8) + 8LL) - 17;
        if ( (unsigned int)v70 > 1 )
          goto LABEL_8;
        if ( *(_BYTE *)v33 > 0x15u )
          goto LABEL_8;
        srcl = v10;
        v71 = sub_AD7630(v33, 0, v70);
        v10 = srcl;
        a5 = v170;
        if ( !v71 || *v71 != 17 )
          goto LABEL_8;
        v34 = (__int64)(v71 + 24);
      }
      a4 = *(unsigned int *)(v34 + 8);
      if ( (unsigned int)a4 <= 0x40 )
      {
        if ( *(_QWORD *)v34 )
        {
          a4 = (unsigned int)(a4 - 1);
          if ( *(_QWORD *)v34 != 1LL << a4 )
          {
LABEL_46:
            v37 = *(_DWORD *)(v34 + 8);
            v179 = v37;
            if ( v37 > 0x40 )
            {
              v140 = a5;
              srcc = (void *)v34;
              sub_C43780((__int64)&v178, (const void **)v34);
              v37 = v179;
              v34 = (__int64)srcc;
              a5 = v140;
              if ( v179 > 0x40 )
              {
                v141 = (__int64)srcc;
                srcm = (void *)a5;
                sub_C43D10((__int64)&v178);
                v34 = v141;
                a5 = (__int64)srcm;
LABEL_51:
                v131 = v34;
                v136 = a5;
                sub_C46250((__int64)&v178);
                v41 = v179;
                v179 = 0;
                LODWORD(v182) = v41;
                v181 = v178;
                v42 = sub_AD8D80(v12, (__int64)&v181);
                v43 = v136;
                srca = (void *)v42;
                v44 = v131;
                if ( (unsigned int)v182 > 0x40 && v181 )
                {
                  j_j___libc_free_0_0(v181);
                  v44 = v131;
                  v43 = v136;
                }
                if ( v179 > 0x40 && v178 )
                {
                  v132 = v44;
                  v137 = v43;
                  j_j___libc_free_0_0(v178);
                  v44 = v132;
                  v43 = v137;
                }
                v45 = *(_DWORD *)(v44 + 8);
                if ( v45 > 0x40 )
                  v46 = *(_QWORD *)(*(_QWORD *)v44 + 8LL * ((v45 - 1) >> 6));
                else
                  v46 = *(_QWORD *)v44;
                v47 = *(__int64 **)(a1 + 32);
                v133 = v43;
                v180 = 257;
                v138 = ((1LL << ((unsigned __int8)v45 - 1)) & v46) == 0 ? 41 : 39;
                v48 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, _QWORD, __int64, void *))(*(_QWORD *)v47[10] + 56LL))(
                                  v47[10],
                                  v138,
                                  v43,
                                  srca);
                if ( !v48 )
                {
                  LOWORD(v184) = 257;
                  v48 = sub_BD2C40(72, unk_3F10FD0);
                  if ( v48 )
                  {
                    v101 = *(_QWORD ***)(v133 + 8);
                    v102 = *((unsigned __int8 *)v101 + 8);
                    if ( (unsigned int)(v102 - 17) > 1 )
                    {
                      v104 = sub_BCB2A0(*v101);
                      v106 = v133;
                      v105 = v138;
                    }
                    else
                    {
                      BYTE4(v177) = (_BYTE)v102 == 18;
                      LODWORD(v177) = *((_DWORD *)v101 + 8);
                      v103 = (__int64 *)sub_BCB2A0(*v101);
                      v104 = sub_BCE1B0(v103, v177);
                      v105 = v138;
                      v106 = v133;
                    }
                    sub_B523C0((__int64)v48, v104, 53, v105, v106, (__int64)srca, (__int64)&v181, 0, 0, 0);
                  }
                  (*(void (__fastcall **)(__int64, _QWORD *, const char **, __int64, __int64))(*(_QWORD *)v47[11] + 16LL))(
                    v47[11],
                    v48,
                    &v178,
                    v47[7],
                    v47[8]);
                  v107 = *v47 + 16LL * *((unsigned int *)v47 + 2);
                  v108 = *v47;
                  v171 = v107;
                  while ( v171 != v108 )
                  {
                    v109 = *(_QWORD *)(v108 + 8);
                    v110 = *(_DWORD *)v108;
                    v108 += 16;
                    sub_B99FD0((__int64)v48, v110, v109);
                  }
                }
                v49 = (*v8 == 56) + 39;
                LOWORD(v184) = 257;
                return (unsigned __int8 *)sub_B51D30(v49, (__int64)v48, v12, (__int64)&v181, 0, 0);
              }
              v38 = (unsigned __int64)v178;
            }
            else
            {
              v38 = *(_QWORD *)v34;
            }
            v39 = ~v38;
            v40 = 0;
            if ( v37 )
              v40 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v37;
            v178 = (const char *)(v39 & v40);
            goto LABEL_51;
          }
        }
      }
      else
      {
        srck = *(_DWORD *)(v34 + 8);
        v167 = v10;
        v130 = a5;
        v135 = (_QWORD *)v34;
        v35 = sub_C444A0(v34);
        a4 = srck;
        v10 = v167;
        if ( srck != v35 )
        {
          v34 = (__int64)v135;
          src = v167;
          v168 = a4 - 1;
          a5 = v130;
          if ( (*(_QWORD *)(*v135 + 8LL * ((unsigned int)(a4 - 1) >> 6)) & (1LL << ((unsigned __int8)a4 - 1))) == 0 )
            goto LABEL_46;
          v36 = sub_C44590((__int64)v135);
          a4 = v168;
          v34 = (__int64)v135;
          a5 = v130;
          v10 = src;
          if ( v168 != v36 )
            goto LABEL_46;
        }
      }
    }
  }
LABEL_8:
  v11 = *a3;
LABEL_9:
  v16 = a3 + 24;
  if ( (_BYTE)v11 == 17 )
    goto LABEL_10;
  v22 = *((_QWORD *)a3 + 1);
LABEL_20:
  v163 = v10;
  if ( (unsigned int)*(unsigned __int8 *)(v22 + 8) - 17 > 1 )
    return 0;
  v23 = sub_AD7630((__int64)a3, 0, v11);
  if ( !v23 || *v23 != 17 )
    return 0;
  v10 = v163;
  v16 = v23 + 24;
LABEL_10:
  if ( *v8 == 56 )
    goto LABEL_14;
  v17 = *(_QWORD **)v16;
  if ( *((_DWORD *)v16 + 2) > 0x40u )
    v17 = (_QWORD *)*v17;
  v162 = v10 == 54;
  if ( !(unsigned __int8)sub_1194CD0((__int64)a2, (unsigned int)v17, v10 == 54, a1, (__int64)v8) )
  {
LABEL_14:
    v18 = sub_F28360(a1, v8, v11, a4, a5, a6);
    if ( v18 )
      return v18;
    v50 = *((_QWORD *)a2 + 2);
    if ( !v50 || *(_QWORD *)(v50 + 8) )
      return v18;
    v51 = *a2;
    if ( (unsigned __int8)(*a2 - 42) <= 0x11u )
    {
      v52 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
      v53 = *v52;
      if ( (_BYTE)v53 == 17 )
        goto LABEL_65;
      if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v52 + 1) + 8LL) - 17 <= 1 && (unsigned __int8)v53 <= 0x15u )
      {
        v72 = sub_AD7630((__int64)v52, 0, v53);
        if ( !v72 || *v72 != 17 )
        {
LABEL_89:
          v51 = *a2;
          goto LABEL_90;
        }
LABEL_65:
        v54 = sub_B43CA0((__int64)v8);
        v181 = (const char *)v183;
        v55 = *(_BYTE **)(v54 + 232);
        v56 = v54;
        v57 = *(_QWORD *)(v54 + 240);
        if ( &v55[v57] && !v55 )
          sub_426248((__int64)"basic_string::_M_construct null not valid");
        v178 = *(const char **)(v54 + 240);
        if ( v57 > 0xF )
        {
          srco = v55;
          v175 = v54;
          v127 = (const char *)sub_22409D0(&v181, &v178, 0);
          v56 = v175;
          v55 = srco;
          v181 = v127;
          v128 = (char *)v127;
          v183[0] = v178;
        }
        else
        {
          if ( v57 == 1 )
          {
            LOBYTE(v183[0]) = *v55;
            goto LABEL_70;
          }
          if ( !v57 )
          {
LABEL_70:
            v182 = v178;
            v178[(_QWORD)v181] = 0;
            v58 = *(_DWORD *)(v56 + 264);
            v184 = *(_QWORD *)(v56 + 264);
            v185 = *(_QWORD *)(v56 + 272);
            v186 = *(_QWORD *)(v56 + 280);
            if ( v181 != (const char *)v183 )
              j_j___libc_free_0(v181, v183[0] + 1LL);
            v59 = *((_QWORD *)v8 + 2);
            if ( (!v59
               || *(_QWORD *)(v59 + 8)
               || **(_BYTE **)(v59 + 24) != 42
               || (unsigned int)(v58 - 42) > 1
               || *v8 != 54)
              && (unsigned __int8)sub_1196680(v8, a2) )
            {
              v60 = *(__int64 **)(a1 + 32);
              v61 = *((_QWORD *)a2 - 4);
              v62 = *v8 - 29;
              LOWORD(v184) = 257;
              v63 = sub_1194380(v60, v62, v61, (__int64)a3, (int)v178, 0, (__int64)&v181, 0);
              v64 = *((_QWORD *)a2 - 8);
              v65 = *(__int64 **)(a1 + 32);
              v66 = *v8 - 29;
              v169 = v63;
              LOWORD(v184) = 257;
              v67 = (unsigned __int8 *)sub_1194380(v65, v66, v64, (__int64)a3, (int)v178, 0, (__int64)&v181, 0);
              sub_BD6B90(v67, a2);
              LODWORD(v65) = *a2;
              LOWORD(v184) = 257;
              return (unsigned __int8 *)sub_B504D0((int)v65 - 29, (__int64)v67, v169, (__int64)&v181, 0, 0);
            }
            goto LABEL_89;
          }
          v128 = (char *)v183;
        }
        v176 = v56;
        memcpy(v128, v55, v57);
        v56 = v176;
        goto LABEL_70;
      }
    }
LABEL_90:
    if ( v51 != 86 )
      return 0;
    v73 = a2[7];
    if ( (v73 & 0x40) != 0 )
    {
      v74 = (void **)*((_QWORD *)a2 - 1);
      v75 = *v74;
      if ( !*v74 )
        goto LABEL_94;
    }
    else
    {
      v74 = (void **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v75 = *v74;
      if ( !*v74 )
        goto LABEL_94;
    }
    v76 = v74[4];
    v77 = v76[2];
    if ( v77 )
    {
      if ( !*(_QWORD *)(v77 + 8) && (unsigned __int8)(*(_BYTE *)v76 - 42) <= 0x11u )
      {
        v113 = v74[8];
        if ( v113 )
        {
          if ( *v113 > 0x15u )
          {
            v114 = (_BYTE *)*(v76 - 8);
            if ( v114 )
            {
              if ( v114 == v113 )
              {
                v115 = *(v76 - 4);
                if ( *(_BYTE *)v115 == 17 )
                  goto LABEL_151;
                srcf = v76;
                v174 = v75;
                v125 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v115 + 8) + 8LL) - 17;
                if ( (unsigned int)v125 <= 1 && *(_BYTE *)v115 <= 0x15u )
                {
                  v126 = sub_AD7630(v115, 0, v125);
                  v75 = v174;
                  v76 = srcf;
                  if ( !v126 || *v126 != 17 )
                  {
LABEL_158:
                    if ( *a2 != 86 )
                      return 0;
                    v73 = a2[7];
                    goto LABEL_94;
                  }
LABEL_151:
                  srce = v75;
                  v172 = (unsigned __int8 *)v76;
                  if ( (unsigned __int8)sub_1196680(v8, v76) )
                  {
                    v116 = *v8;
                    v117 = *(__int64 **)(a1 + 32);
                    LOWORD(v184) = 257;
                    v143 = (__int64)srce;
                    srcn = v172;
                    v118 = sub_1194380(
                             v117,
                             v116 - 29,
                             *((_QWORD *)v172 - 4),
                             (__int64)a3,
                             (int)v178,
                             0,
                             (__int64)&v181,
                             0);
                    v119 = *(__int64 **)(a1 + 32);
                    v120 = *v8 - 29;
                    v173 = v118;
                    LOWORD(v184) = 257;
                    v121 = sub_1194380(v119, v120, (__int64)v113, (__int64)a3, (int)v178, 0, (__int64)&v181, 0);
                    v122 = *(__int64 **)(a1 + 32);
                    LOWORD(v184) = 257;
                    v123 = v121;
                    v124 = sub_1194380(v122, (unsigned int)*srcn - 29, v121, v173, (int)v178, 0, (__int64)&v181, 0);
                    LOWORD(v184) = 257;
                    return sub_109FEA0(v143, v124, v123, &v181, 0, 0, 0);
                  }
                  goto LABEL_158;
                }
              }
            }
          }
        }
      }
    }
LABEL_94:
    if ( (v73 & 0x40) != 0 )
    {
      v25 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
      v165 = *(_QWORD *)v25;
      if ( *(_QWORD *)v25 )
        goto LABEL_96;
    }
    else
    {
      v25 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
      v165 = *(_QWORD *)v25;
      if ( *(_QWORD *)v25 )
      {
LABEL_96:
        v78 = (_BYTE *)*((_QWORD *)v25 + 4);
        if ( v78 )
        {
          v79 = (unsigned __int8 *)*((_QWORD *)v25 + 8);
          v80 = *((_QWORD *)v79 + 2);
          if ( v80 )
          {
            v18 = *(unsigned __int8 **)(v80 + 8);
            if ( !v18 && (unsigned __int8)(*v79 - 42) <= 0x11u )
            {
              if ( *v78 > 0x15u )
              {
                v81 = (_BYTE *)*((_QWORD *)v79 - 8);
                if ( v81 )
                {
                  if ( v78 == v81 )
                  {
                    v82 = *((_QWORD *)v79 - 4);
                    if ( *(_BYTE *)v82 == 17
                      || (v142 = (unsigned __int8 *)*((_QWORD *)v25 + 8),
                          srcd = (_BYTE *)*((_QWORD *)v25 + 4),
                          v111 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v82 + 8) + 8LL) - 17,
                          (unsigned int)v111 <= 1)
                      && *(_BYTE *)v82 <= 0x15u
                      && (v112 = sub_AD7630(v82, 0, v111)) != 0
                      && (v78 = srcd, v79 = v142, *v112 == 17) )
                    {
                      v139 = (__int64)v78;
                      srcb = v79;
                      if ( (unsigned __int8)sub_1196680(v8, v79) )
                      {
                        v83 = *(__int64 **)(a1 + 32);
                        v84 = *v8;
                        LOWORD(v184) = 257;
                        v85 = sub_1194380(
                                v83,
                                v84 - 29,
                                *((_QWORD *)srcb - 4),
                                (__int64)a3,
                                (int)v178,
                                0,
                                (__int64)&v181,
                                0);
                        v86 = *(__int64 **)(a1 + 32);
                        v87 = v85;
                        v88 = *v8 - 29;
                        LOWORD(v184) = 257;
                        v89 = sub_1194380(v86, v88, v139, (__int64)a3, (int)v178, 0, (__int64)&v181, 0);
                        v90 = *(__int64 **)(a1 + 32);
                        v91 = v89;
                        LOWORD(v184) = 257;
                        v92 = sub_1194380(v90, (unsigned int)*srcb - 29, v89, v87, (int)v178, 0, (__int64)&v181, 0);
                        LOWORD(v184) = 257;
                        v93 = v92;
                        v94 = (unsigned __int8 *)sub_BD2C40(72, 3u);
                        v18 = v94;
                        if ( v94 )
                        {
                          sub_B44260((__int64)v94, *(_QWORD *)(v91 + 8), 57, 3u, 0, 0);
                          if ( *((_QWORD *)v18 - 12) )
                          {
                            v95 = *((_QWORD *)v18 - 11);
                            **((_QWORD **)v18 - 10) = v95;
                            if ( v95 )
                              *(_QWORD *)(v95 + 16) = *((_QWORD *)v18 - 10);
                          }
                          *((_QWORD *)v18 - 12) = v165;
                          v96 = *(_QWORD *)(v165 + 16);
                          *((_QWORD *)v18 - 11) = v96;
                          if ( v96 )
                            *(_QWORD *)(v96 + 16) = v18 - 88;
                          *((_QWORD *)v18 - 10) = v165 + 16;
                          *(_QWORD *)(v165 + 16) = v18 - 96;
                          if ( *((_QWORD *)v18 - 8) )
                          {
                            v97 = *((_QWORD *)v18 - 7);
                            **((_QWORD **)v18 - 6) = v97;
                            if ( v97 )
                              *(_QWORD *)(v97 + 16) = *((_QWORD *)v18 - 6);
                          }
                          *((_QWORD *)v18 - 8) = v91;
                          v98 = *(_QWORD *)(v91 + 16);
                          *((_QWORD *)v18 - 7) = v98;
                          if ( v98 )
                            *(_QWORD *)(v98 + 16) = v18 - 56;
                          *((_QWORD *)v18 - 6) = v91 + 16;
                          *(_QWORD *)(v91 + 16) = v18 - 64;
                          if ( *((_QWORD *)v18 - 4) )
                          {
                            v99 = *((_QWORD *)v18 - 3);
                            **((_QWORD **)v18 - 2) = v99;
                            if ( v99 )
                              *(_QWORD *)(v99 + 16) = *((_QWORD *)v18 - 2);
                          }
                          *((_QWORD *)v18 - 4) = v93;
                          if ( v93 )
                          {
                            v100 = *(_QWORD *)(v93 + 16);
                            *((_QWORD *)v18 - 3) = v100;
                            if ( v100 )
                              *(_QWORD *)(v100 + 16) = v18 - 24;
                            *((_QWORD *)v18 - 2) = v93 + 16;
                            *(_QWORD *)(v93 + 16) = v18 - 32;
                          }
                          sub_BD6B50(v18, &v181);
                        }
                      }
                    }
                  }
                }
              }
              return v18;
            }
          }
        }
      }
    }
    return 0;
  }
  v20 = *(_QWORD **)v16;
  if ( *((_DWORD *)v16 + 2) > 0x40u )
    v20 = (_QWORD *)*v20;
  v21 = sub_1197740((__int64)a2, (unsigned int)v20, v162, a1);
  return sub_F162A0(a1, (__int64)v8, v21);
}
