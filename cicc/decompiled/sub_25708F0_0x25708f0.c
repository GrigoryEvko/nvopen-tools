// Function: sub_25708F0
// Address: 0x25708f0
//
__int64 __fastcall sub_25708F0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // r13
  __int64 v3; // rax
  unsigned __int8 *v4; // rax
  __int64 v5; // r8
  __int64 v6; // r9
  int v7; // edx
  unsigned __int8 *v8; // r12
  unsigned __int8 *v9; // rbx
  __int64 *v10; // rax
  __int64 v11; // r14
  int v12; // edx
  __int64 *v13; // rax
  char v14; // bl
  _QWORD *v15; // r12
  _QWORD *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rax
  __int64 v22; // r13
  unsigned __int64 v23; // rdx
  __int64 *v24; // rax
  __int64 v25; // r13
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rdx
  int v31; // ecx
  int v32; // eax
  _QWORD *v33; // rdi
  __int64 *v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rax
  unsigned __int64 v37; // rbx
  __int64 v38; // r14
  __int64 v39; // rdx
  __int64 v40; // r15
  const char *v41; // rax
  __int64 v42; // r14
  __int64 v43; // rdx
  _QWORD *v44; // rax
  unsigned int v45; // r12d
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rsi
  __int64 v50; // r9
  _QWORD *v51; // rax
  __int64 v52; // r9
  _QWORD *v53; // r15
  unsigned __int64 v54; // r13
  unsigned __int8 *v55; // r12
  const char *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rax
  unsigned __int64 v62; // rdx
  unsigned __int8 **v63; // rax
  __int64 v64; // rcx
  __int64 v65; // rax
  __int16 v66; // dx
  __int64 v67; // r15
  char v68; // bl
  const char *v69; // rax
  int v70; // r12d
  __int64 v71; // rdx
  __int64 v72; // r14
  unsigned __int16 v73; // ax
  _QWORD *v74; // rbx
  __int64 v75; // r12
  __int64 v76; // r15
  int v77; // eax
  int v78; // eax
  unsigned int v79; // ecx
  __int64 v80; // rax
  __int64 v81; // rcx
  __int64 v82; // rcx
  _QWORD *v83; // r12
  _QWORD *v84; // rax
  __int64 **v85; // rdi
  __int64 v86; // r15
  int v87; // eax
  int v88; // eax
  unsigned int v89; // ecx
  __int64 v90; // rax
  __int64 v91; // rcx
  __int64 v92; // rcx
  __int64 *v93; // rax
  __int64 v94; // r12
  __int64 v95; // rdx
  _QWORD *v96; // rax
  _QWORD *v97; // rbx
  __int64 *v98; // rax
  __int64 v99; // rax
  __int64 v100; // r12
  _QWORD *v101; // rdi
  __int64 v102; // rcx
  __int64 v103; // rax
  __int64 v104; // r14
  __int64 *v105; // rax
  __int64 v106; // rbx
  int v107; // r15d
  __int64 v108; // rdx
  _QWORD *v109; // rax
  __int64 v110; // r12
  unsigned __int64 v111; // rax
  __int64 v112; // rdx
  __int64 v113; // [rsp-8h] [rbp-208h]
  const char *v114; // [rsp+20h] [rbp-1E0h]
  int v115; // [rsp+28h] [rbp-1D8h]
  __int64 v116; // [rsp+28h] [rbp-1D8h]
  unsigned __int16 v117; // [rsp+28h] [rbp-1D8h]
  __int64 v118; // [rsp+38h] [rbp-1C8h]
  char v119; // [rsp+42h] [rbp-1BEh]
  char v120; // [rsp+43h] [rbp-1BDh]
  unsigned int v121; // [rsp+44h] [rbp-1BCh]
  __int64 v122; // [rsp+48h] [rbp-1B8h]
  __int64 *v123; // [rsp+50h] [rbp-1B0h]
  __int64 v124; // [rsp+58h] [rbp-1A8h]
  __int64 *v125; // [rsp+58h] [rbp-1A8h]
  __int64 v126; // [rsp+70h] [rbp-190h]
  __int64 v128; // [rsp+80h] [rbp-180h]
  __int64 v129; // [rsp+88h] [rbp-178h]
  __int64 *v131; // [rsp+98h] [rbp-168h]
  __int64 v132; // [rsp+98h] [rbp-168h]
  __int64 v133; // [rsp+98h] [rbp-168h]
  __int64 v134; // [rsp+98h] [rbp-168h]
  __int64 v135; // [rsp+A0h] [rbp-160h]
  char v136; // [rsp+A0h] [rbp-160h]
  __int64 *v137; // [rsp+A8h] [rbp-158h]
  __int64 v138; // [rsp+A8h] [rbp-158h]
  _QWORD *v139; // [rsp+A8h] [rbp-158h]
  __int64 *v140; // [rsp+A8h] [rbp-158h]
  char v141; // [rsp+BFh] [rbp-141h] BYREF
  __int64 v142; // [rsp+C0h] [rbp-140h] BYREF
  __int64 v143; // [rsp+C8h] [rbp-138h]
  const char *v144; // [rsp+D0h] [rbp-130h] BYREF
  __int64 v145; // [rsp+D8h] [rbp-128h]
  char *v146; // [rsp+E0h] [rbp-120h]
  __int16 v147; // [rsp+F0h] [rbp-110h]
  __int64 *v148; // [rsp+100h] [rbp-100h] BYREF
  __int64 v149; // [rsp+108h] [rbp-F8h]
  _BYTE v150[48]; // [rsp+110h] [rbp-F0h] BYREF
  _BYTE *v151; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v152; // [rsp+148h] [rbp-B8h]
  _BYTE v153[48]; // [rsp+150h] [rbp-B0h] BYREF
  __int64 *v154; // [rsp+180h] [rbp-80h] BYREF
  __int64 v155; // [rsp+188h] [rbp-78h]
  _QWORD v156[2]; // [rsp+190h] [rbp-70h] BYREF
  __int16 v157; // [rsp+1A0h] [rbp-60h]

  if ( !*(_BYTE *)(a1 + 296) && !*(_DWORD *)(a1 + 256) )
    return 1;
  v141 = 0;
  v2 = sub_2509740((_QWORD *)(a1 + 72));
  if ( (unsigned __int8)sub_251BFD0(a2, v2, a1, 0, &v141, 0, 1, 0) )
    return 1;
  v128 = *(_QWORD *)(v2 - 32);
  v3 = *(_QWORD *)(v128 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  v131 = (__int64 *)(v2 + 24);
  if ( *(_DWORD *)(v3 + 8) >> 8 )
  {
    v93 = (__int64 *)sub_BD5C60(v128);
    v94 = sub_BCE3C0(v93, 0);
    v154 = (__int64 *)sub_BD5D20(v128);
    v156[0] = ".as0";
    v157 = 773;
    v155 = v95;
    v96 = sub_BD2C40(72, unk_3F10A14);
    v97 = v96;
    if ( v96 )
      sub_B51C90((__int64)v96, v128, v94, (__int64)&v154, (__int64)v131, 0);
    v128 = (__int64)v97;
  }
  v119 = *(_BYTE *)(*(_QWORD *)(v2 + 8) + 8LL);
  v122 = *(_QWORD *)(v2 + 80);
  v4 = sub_24E54B0((unsigned __int8 *)v2);
  v7 = 0;
  v8 = v4;
  LODWORD(v4) = *(_DWORD *)(v2 + 4);
  v149 = 0x600000000LL;
  v9 = (unsigned __int8 *)(v2 - 32LL * ((unsigned int)v4 & 0x7FFFFFF));
  v10 = (__int64 *)v150;
  v148 = (__int64 *)v150;
  v11 = (v8 - v9) >> 5;
  if ( (unsigned __int64)(v8 - v9) > 0xC0 )
  {
    sub_C8D5F0((__int64)&v148, v150, (v8 - v9) >> 5, 8u, v5, v6);
    v7 = v149;
    v10 = &v148[(unsigned int)v149];
    if ( v8 != v9 )
      goto LABEL_9;
  }
  else if ( v8 != v9 )
  {
    do
    {
LABEL_9:
      if ( v10 )
        *v10 = *(_QWORD *)v9;
      v9 += 32;
      ++v10;
    }
    while ( v8 != v9 );
    v7 = v149;
  }
  LODWORD(v149) = v11 + v7;
  v12 = *(_DWORD *)(a1 + 256);
  if ( v12 )
  {
    v13 = *(__int64 **)(a1 + 248);
    v120 = *(_BYTE *)(a1 + 296);
    v14 = v120 & (v12 == 1);
    if ( v14 )
    {
      v104 = *v13;
      if ( (unsigned __int8)sub_29A3A40(v2, *v13, 0) )
      {
        v45 = 0;
        sub_29A3E20(v2, v104, 0);
      }
      else
      {
        v105 = (__int64 *)sub_BD5D20(v2);
        v106 = (unsigned int)v149;
        v154 = v105;
        v157 = 261;
        v107 = v149 + 1;
        v155 = v108;
        v140 = v148;
        v109 = sub_BD2C40(88, (int)v149 + 1);
        v110 = (__int64)v109;
        if ( v109 )
        {
          sub_B44260((__int64)v109, **(_QWORD **)(v122 + 16), 56, v107 & 0x7FFFFFF, (__int64)v131, 0);
          *(_QWORD *)(v110 + 72) = 0;
          sub_B4A290(v110, v122, v104, v140, v106, (__int64)&v154, 0, 0);
        }
        if ( v119 != 7 )
        {
          v111 = sub_254C980(v2);
          sub_256F570(a2, v111, v112, (unsigned __int8 *)v110, 1u);
        }
        v45 = 0;
        sub_2570110(a2, v2);
      }
    }
    else
    {
      v137 = *(__int64 **)(a1 + 248);
      v15 = 0;
      v154 = v156;
      v155 = 0x800000000LL;
      v151 = v153;
      v152 = 0x300000000LL;
      v135 = v2;
      v123 = &v13[v12];
      v129 = a2 + 3560;
      while ( 1 )
      {
        v25 = *v137;
        if ( *(_QWORD *)(a2 + 4352)
          && (LODWORD(v144) = v12,
              !(*(unsigned __int8 (__fastcall **)(__int64, __int64, __int64, __int64, __int64, const char **))(a2 + 4360))(
                 a2 + 4336,
                 a2,
                 a1,
                 v135,
                 v25,
                 &v144)) )
        {
          sub_255C0E0((__int64)&v154, v25, v26, v27, v28, v29);
          v120 = 0;
        }
        else
        {
          v147 = 257;
          v15 = sub_BD2C40(72, unk_3F10FD0);
          if ( v15 )
          {
            v30 = *(_QWORD *)(v128 + 8);
            v31 = *(unsigned __int8 *)(v30 + 8);
            if ( (unsigned int)(v31 - 17) > 1 )
            {
              v35 = sub_BCB2A0(*(_QWORD **)v30);
            }
            else
            {
              v32 = *(_DWORD *)(v30 + 32);
              v33 = *(_QWORD **)v30;
              BYTE4(v143) = (_BYTE)v31 == 18;
              LODWORD(v143) = v32;
              v34 = (__int64 *)sub_BCB2A0(v33);
              v35 = sub_BCE1B0(v34, v143);
            }
            sub_B523C0((__int64)v15, v35, 53, 32, v128, v25, (__int64)&v144, (__int64)v131, 0, 0);
          }
          v36 = v126;
          LOWORD(v36) = 0;
          v126 = v36;
          v37 = sub_F38250((__int64)v15, v131, v36, 0, 0, 0, 0, 0);
          v38 = *(_QWORD *)(v135 + 40);
          sub_AE6EC0(v129, *(_QWORD *)(v37 + 40));
          sub_AE6EC0(v129, v131[2]);
          v39 = v15[4];
          if ( v39 == v15[5] + 48LL || !v39 )
            v124 = 0;
          else
            v124 = v39 - 24;
          if ( (__int64 *)v135 == v131 - 3 )
          {
            v47 = sub_B43CB0(v37);
            v147 = 257;
            v116 = v47;
            v132 = sub_BD5C60(v37);
            v48 = sub_22077B0(0x50u);
            v49 = v132;
            v50 = v48;
            if ( v48 )
            {
              v133 = v48;
              sub_AA4D50(v48, v49, (__int64)&v144, v116, v38);
              v50 = v133;
            }
            v134 = v50;
            sub_AE6EC0(v129, v50);
            sub_B43C20((__int64)&v144, v134);
            v114 = v144;
            v117 = v145;
            v51 = sub_BD2C40(72, 1u);
            v52 = v134;
            v53 = v51;
            if ( v51 )
            {
              sub_B4C8F0((__int64)v51, v38, 1u, (__int64)v114, v117);
              v52 = v134;
            }
            v131 = v53 + 3;
            sub_BD2ED0(v124, v38, v52);
          }
          else
          {
            sub_BD2ED0(v37, v131[2], v38);
          }
          v40 = v37 + 24;
          v142 = 0;
          if ( (unsigned __int8)sub_29A3A40(v135, v25, 0) )
          {
            v16 = (_QWORD *)sub_B47F80((_BYTE *)v135);
            v17 = v118;
            LOWORD(v17) = 0;
            v118 = v17;
            sub_B44220(v16, v37 + 24, v17);
            v18 = sub_29A3E20(v16, v25, &v142);
          }
          else
          {
            v41 = sub_BD5D20(v135);
            v42 = (unsigned int)v149;
            v145 = v43;
            v144 = v41;
            v147 = 261;
            v125 = v148;
            v115 = v149 + 1;
            v44 = sub_BD2C40(88, (int)v149 + 1);
            v18 = (__int64)v44;
            if ( v44 )
            {
              v121 = v115 & 0x7FFFFFF | v121 & 0xE0000000;
              sub_B44260((__int64)v44, **(_QWORD **)(v122 + 16), 56, v121, v40, 0);
              *(_QWORD *)(v18 + 72) = 0;
              sub_B4A290(v18, v122, v25, v125, v42, (__int64)&v144, 0, 0);
              v19 = v113;
            }
          }
          v21 = (unsigned int)v152;
          v22 = v142;
          v23 = (unsigned int)v152 + 1LL;
          if ( v23 > HIDWORD(v152) )
          {
            sub_C8D5F0((__int64)&v151, v153, v23, 0x10u, v19, v20);
            v21 = (unsigned int)v152;
          }
          v24 = (__int64 *)&v151[16 * v21];
          *v24 = v18;
          v14 = 1;
          v24[1] = v22;
          LODWORD(v152) = v152 + 1;
        }
        if ( v123 == ++v137 )
          break;
        v12 = *(_DWORD *)(a1 + 256);
      }
      v54 = v135;
      if ( v14 )
      {
        if ( v120 )
        {
          v98 = (__int64 *)sub_BD5C60((__int64)v15);
          v99 = sub_ACD6D0(v98);
          sub_BD84D0((__int64)v15, v99);
          sub_B43D60(v15);
          v100 = sub_BD5C60((__int64)(v131 - 3));
          v101 = sub_BD2C40(72, unk_3F148B8);
          if ( v101 )
            sub_B4C8A0((__int64)v101, v100, (__int64)v131, 0);
          sub_B43D60(v131 - 3);
        }
        else
        {
          v55 = (unsigned __int8 *)sub_B47F80((_BYTE *)v135);
          v56 = sub_BD5D20(v135);
          v147 = 261;
          v144 = v56;
          v145 = v57;
          sub_BD6B50(v55, &v144);
          v58 = v126;
          LOWORD(v58) = 0;
          sub_B44150(v55, v131[2], (unsigned __int64 *)v131, v58);
          v61 = (unsigned int)v152;
          v62 = (unsigned int)v152 + 1LL;
          if ( v62 > HIDWORD(v152) )
          {
            sub_C8D5F0((__int64)&v151, v153, v62, 0x10u, v59, v60);
            v61 = (unsigned int)v152;
          }
          v63 = (unsigned __int8 **)&v151[16 * v61];
          *v63 = v55;
          v63[1] = 0;
          LODWORD(v152) = v152 + 1;
          if ( *(_BYTE *)(a1 + 296) )
          {
            v144 = (const char *)sub_BD5C60((__int64)v55);
            v65 = sub_B8C880(&v144, v154, (unsigned int)v155, v64);
            sub_B99FD0((__int64)v55, 0x17u, v65);
          }
        }
        if ( v119 != 7 )
        {
          v67 = sub_AA5190(*(_QWORD *)(v135 + 40));
          if ( v67 )
          {
            v136 = v66;
            v68 = HIBYTE(v66);
          }
          else
          {
            v136 = 0;
            v68 = 0;
          }
          v69 = sub_BD5D20(v54);
          v70 = v152;
          v144 = v69;
          v147 = 773;
          v145 = v71;
          v146 = ".phi";
          v138 = *(_QWORD *)(v54 + 8);
          v72 = sub_BD2DA0(80);
          if ( v72 )
          {
            LOBYTE(v73) = v136;
            HIBYTE(v73) = v68;
            sub_B44260(v72, v138, 55, 0x8000000u, v67, v73);
            *(_DWORD *)(v72 + 72) = v70;
            sub_BD6B50((unsigned __int8 *)v72, &v144);
            sub_BD2A10(v72, *(_DWORD *)(v72 + 72), 1);
          }
          if ( &v151[16 * (unsigned int)v152] != v151 )
          {
            v139 = &v151[16 * (unsigned int)v152];
            v74 = v151;
            do
            {
              while ( 1 )
              {
                v83 = (_QWORD *)v74[1];
                v84 = (_QWORD *)*v74;
                v85 = *(__int64 ***)(v54 + 8);
                if ( !v83 )
                  v83 = (_QWORD *)*v74;
                if ( v85 != (__int64 **)v83[1] )
                  break;
                v86 = v83[5];
                v87 = *(_DWORD *)(v72 + 4) & 0x7FFFFFF;
                if ( v87 == *(_DWORD *)(v72 + 72) )
                {
                  sub_B48D90(v72);
                  v87 = *(_DWORD *)(v72 + 4) & 0x7FFFFFF;
                }
                v88 = (v87 + 1) & 0x7FFFFFF;
                v89 = v88 | *(_DWORD *)(v72 + 4) & 0xF8000000;
                v90 = *(_QWORD *)(v72 - 8) + 32LL * (unsigned int)(v88 - 1);
                *(_DWORD *)(v72 + 4) = v89;
                if ( *(_QWORD *)v90 )
                {
                  v91 = *(_QWORD *)(v90 + 8);
                  **(_QWORD **)(v90 + 16) = v91;
                  if ( v91 )
                    *(_QWORD *)(v91 + 16) = *(_QWORD *)(v90 + 16);
                }
                *(_QWORD *)v90 = v83;
                v92 = v83[2];
                *(_QWORD *)(v90 + 8) = v92;
                if ( v92 )
                  *(_QWORD *)(v92 + 16) = v90 + 8;
                *(_QWORD *)(v90 + 16) = v83 + 2;
                v74 += 2;
                v83[2] = v90;
                *(_QWORD *)(*(_QWORD *)(v72 - 8)
                          + 32LL * *(unsigned int *)(v72 + 72)
                          + 8LL * ((*(_DWORD *)(v72 + 4) & 0x7FFFFFFu) - 1)) = v86;
                if ( v139 == v74 )
                  goto LABEL_80;
              }
              if ( *(_BYTE *)(v84[1] + 8LL) != 7 )
                BUG();
              v75 = v84[5];
              v76 = sub_ACADE0(v85);
              v77 = *(_DWORD *)(v72 + 4) & 0x7FFFFFF;
              if ( v77 == *(_DWORD *)(v72 + 72) )
              {
                sub_B48D90(v72);
                v77 = *(_DWORD *)(v72 + 4) & 0x7FFFFFF;
              }
              v78 = (v77 + 1) & 0x7FFFFFF;
              v79 = v78 | *(_DWORD *)(v72 + 4) & 0xF8000000;
              v80 = *(_QWORD *)(v72 - 8) + 32LL * (unsigned int)(v78 - 1);
              *(_DWORD *)(v72 + 4) = v79;
              if ( *(_QWORD *)v80 )
              {
                v81 = *(_QWORD *)(v80 + 8);
                **(_QWORD **)(v80 + 16) = v81;
                if ( v81 )
                  *(_QWORD *)(v81 + 16) = *(_QWORD *)(v80 + 16);
              }
              *(_QWORD *)v80 = v76;
              if ( v76 )
              {
                v82 = *(_QWORD *)(v76 + 16);
                *(_QWORD *)(v80 + 8) = v82;
                if ( v82 )
                  *(_QWORD *)(v82 + 16) = v80 + 8;
                *(_QWORD *)(v80 + 16) = v76 + 16;
                *(_QWORD *)(v76 + 16) = v80;
              }
              v74 += 2;
              *(_QWORD *)(*(_QWORD *)(v72 - 8)
                        + 32LL * *(unsigned int *)(v72 + 72)
                        + 8LL * ((*(_DWORD *)(v72 + 4) & 0x7FFFFFFu) - 1)) = v75;
            }
            while ( v139 != v74 );
          }
LABEL_80:
          sub_250D230((unsigned __int64 *)&v144, v54, 3, 0);
          sub_256F570(a2, (__int64)v144, v145, (unsigned __int8 *)v72, 1u);
        }
        v45 = 0;
        sub_2570110(a2, v54);
      }
      else
      {
        v45 = 1;
        if ( *(_BYTE *)(a1 + 296) )
        {
          v45 = 0;
          v144 = (const char *)sub_BD5C60(v135);
          v103 = sub_B8C880(&v144, v154, (unsigned int)v155, v102);
          sub_B99FD0(v135, 0x17u, v103);
        }
      }
      if ( v151 != v153 )
        _libc_free((unsigned __int64)v151);
      if ( v154 != v156 )
        _libc_free((unsigned __int64)v154);
    }
  }
  else
  {
    v45 = 0;
    sub_2570610(a2, v2);
  }
  if ( v148 != (__int64 *)v150 )
    _libc_free((unsigned __int64)v148);
  return v45;
}
