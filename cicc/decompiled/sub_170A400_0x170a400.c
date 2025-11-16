// Function: sub_170A400
// Address: 0x170a400
//
__int64 __fastcall sub_170A400(__int64 a1, __int64 a2, __int64 a3, char a4, double a5, double a6, double a7)
{
  char v8; // bl
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 *v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // r15
  __int64 v20; // rax
  unsigned __int64 v21; // r12
  __int64 v22; // r13
  __int64 v23; // rbx
  __int64 v24; // rsi
  unsigned int v25; // eax
  __int64 v26; // rcx
  unsigned __int64 v27; // r8
  __int64 v28; // rax
  unsigned __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rcx
  _QWORD *v32; // rbx
  __int64 v33; // r13
  char v34; // al
  int v36; // eax
  unsigned int v37; // eax
  __int64 v38; // rdx
  unsigned __int64 v39; // r9
  _QWORD *v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rax
  const char *v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rax
  const char *v48; // rax
  __int64 v49; // rdx
  bool v50; // cc
  __int64 v51; // rsi
  __int64 *v52; // rdi
  __int64 v53; // rax
  __int64 *v54; // rax
  __int64 v55; // r13
  __int64 v56; // rdx
  __int64 v57; // rdi
  __int64 *v58; // r13
  __int64 v59; // rax
  __int64 v60; // rcx
  __int64 *v61; // rsi
  __int64 v62; // rdi
  __int64 v63; // rdx
  __int64 v64; // rsi
  __int64 v65; // rsi
  __int64 v66; // rdx
  unsigned __int8 *v67; // rsi
  __int64 v68; // rdi
  __int64 *v69; // r13
  __int64 v70; // rax
  __int64 v71; // rcx
  __int64 v72; // rsi
  __int64 v73; // rsi
  unsigned __int8 *v74; // rsi
  const char *v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rax
  __int64 v78; // rdi
  __int64 v79; // rsi
  __int64 v80; // rax
  __int64 v81; // rsi
  __int64 v82; // rsi
  __int64 v83; // rdx
  unsigned __int8 *v84; // rsi
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rax
  unsigned __int64 v88; // rax
  int v89; // eax
  _QWORD *v90; // rax
  __int64 v91; // [rsp+8h] [rbp-1B8h]
  __int64 v92; // [rsp+10h] [rbp-1B0h]
  __int64 v93; // [rsp+10h] [rbp-1B0h]
  __int64 v94; // [rsp+10h] [rbp-1B0h]
  __int64 v95; // [rsp+18h] [rbp-1A8h]
  unsigned __int64 v96; // [rsp+18h] [rbp-1A8h]
  __int64 v97; // [rsp+18h] [rbp-1A8h]
  __int64 v98; // [rsp+18h] [rbp-1A8h]
  __int64 v99; // [rsp+20h] [rbp-1A0h]
  __int64 v100; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v101; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v102; // [rsp+20h] [rbp-1A0h]
  unsigned __int64 v103; // [rsp+28h] [rbp-198h]
  __int64 v104; // [rsp+28h] [rbp-198h]
  unsigned __int64 v105; // [rsp+28h] [rbp-198h]
  __int64 *v106; // [rsp+28h] [rbp-198h]
  __int64 v107; // [rsp+28h] [rbp-198h]
  __int64 v108; // [rsp+28h] [rbp-198h]
  __int64 v109; // [rsp+28h] [rbp-198h]
  unsigned __int64 v110; // [rsp+30h] [rbp-190h]
  __int64 v112; // [rsp+40h] [rbp-180h]
  __int64 v113; // [rsp+48h] [rbp-178h]
  __int64 v116; // [rsp+60h] [rbp-160h]
  unsigned __int64 v117; // [rsp+68h] [rbp-158h]
  __int64 v118; // [rsp+68h] [rbp-158h]
  __int64 v119; // [rsp+68h] [rbp-158h]
  __int64 v120; // [rsp+68h] [rbp-158h]
  __int64 v121; // [rsp+68h] [rbp-158h]
  char v122; // [rsp+76h] [rbp-14Ah]
  char v123; // [rsp+77h] [rbp-149h]
  __int64 *v124; // [rsp+78h] [rbp-148h]
  __int64 v125; // [rsp+88h] [rbp-138h] BYREF
  __int64 v126; // [rsp+90h] [rbp-130h] BYREF
  __int64 v127; // [rsp+98h] [rbp-128h] BYREF
  _QWORD v128[2]; // [rsp+A0h] [rbp-120h] BYREF
  _QWORD v129[2]; // [rsp+B0h] [rbp-110h] BYREF
  _QWORD v130[2]; // [rsp+C0h] [rbp-100h] BYREF
  __int64 v131[2]; // [rsp+D0h] [rbp-F0h] BYREF
  __int16 v132; // [rsp+E0h] [rbp-E0h]
  __int64 v133[2]; // [rsp+F0h] [rbp-D0h] BYREF
  __int16 v134; // [rsp+100h] [rbp-C0h]
  __int64 v135[2]; // [rsp+110h] [rbp-B0h] BYREF
  __int16 v136; // [rsp+120h] [rbp-A0h]
  char v137[16]; // [rsp+130h] [rbp-90h] BYREF
  __int16 v138; // [rsp+140h] [rbp-80h]
  unsigned __int8 *v139; // [rsp+150h] [rbp-70h] BYREF
  __int64 v140; // [rsp+158h] [rbp-68h]
  __int16 v141; // [rsp+160h] [rbp-60h]
  unsigned __int8 *v142; // [rsp+170h] [rbp-50h] BYREF
  char *v143; // [rsp+178h] [rbp-48h]
  __int16 v144; // [rsp+180h] [rbp-40h]

  v8 = a4 ^ 1;
  v10 = *(_QWORD *)a3;
  v113 = sub_15A9650(a2, *(_QWORD *)a3);
  v13 = sub_15A06D0((__int64 **)v113, v10, v11, v12);
  v122 = v8 & ((*(_BYTE *)(a3 + 17) & 2) != 0);
  v14 = v113;
  if ( *(_BYTE *)(v113 + 8) == 16 )
    v14 = **(_QWORD **)(v113 + 16);
  v110 = 0xFFFFFFFFFFFFFFFFLL >> (64 - BYTE1(*(_DWORD *)(v14 + 8)));
  if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
    v15 = *(_QWORD *)(a3 - 8);
  else
    v15 = a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
  v16 = (__int64 *)(v15 + 24);
  v17 = sub_16348C0(a3) | 4;
  if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
  {
    v18 = *(_QWORD *)(a3 - 8);
    v112 = v18 + 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
  }
  else
  {
    v112 = a3;
    v18 = a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
  }
  if ( v112 != v18 + 24 )
  {
    v124 = v16;
    v19 = v18 + 48;
    while ( 1 )
    {
      v20 = v17;
      v21 = v17 & 0xFFFFFFFFFFFFFFF8LL;
      v116 = v19;
      v22 = *(_QWORD *)(v19 - 24);
      v23 = v21;
      LODWORD(v20) = (v20 >> 2) & 1;
      v123 = v20;
      if ( !(_DWORD)v20 || (v24 = v21) == 0 )
        v24 = sub_1643D30(v21, *v124);
      v25 = sub_15A9FE0(a2, v24);
      v26 = 1;
      v27 = v25;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v24 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v41 = *(_QWORD *)(v24 + 32);
            v24 = *(_QWORD *)(v24 + 24);
            v26 *= v41;
            continue;
          case 1:
            v28 = 16;
            goto LABEL_14;
          case 2:
            v28 = 32;
            goto LABEL_14;
          case 3:
          case 9:
            v28 = 64;
            goto LABEL_14;
          case 4:
            v28 = 80;
            goto LABEL_14;
          case 5:
          case 6:
            v28 = 128;
            goto LABEL_14;
          case 7:
            v103 = v27;
            v24 = 0;
            v118 = v26;
            goto LABEL_32;
          case 0xB:
            v28 = *(_DWORD *)(v24 + 8) >> 8;
            goto LABEL_14;
          case 0xD:
            v105 = v27;
            v120 = v26;
            v40 = (_QWORD *)sub_15A9930(a2, v24);
            v26 = v120;
            v27 = v105;
            v28 = 8LL * *v40;
            goto LABEL_14;
          case 0xE:
            v95 = v27;
            v99 = v26;
            v119 = *(_QWORD *)(v24 + 32);
            v104 = *(_QWORD *)(v24 + 24);
            v37 = sub_15A9FE0(a2, v104);
            v27 = v95;
            v38 = 1;
            v24 = v104;
            v26 = v99;
            v39 = v37;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v24 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v86 = *(_QWORD *)(v24 + 32);
                  v24 = *(_QWORD *)(v24 + 24);
                  v38 *= v86;
                  continue;
                case 1:
                  v85 = 16;
                  goto LABEL_93;
                case 2:
                  v85 = 32;
                  goto LABEL_93;
                case 3:
                case 9:
                  v85 = 64;
                  goto LABEL_93;
                case 4:
                  v85 = 80;
                  goto LABEL_93;
                case 5:
                case 6:
                  v85 = 128;
                  goto LABEL_93;
                case 7:
                  v93 = v95;
                  v24 = 0;
                  v97 = v99;
                  v101 = v39;
                  v108 = v38;
                  goto LABEL_98;
                case 0xB:
                  v85 = *(_DWORD *)(v24 + 8) >> 8;
                  goto LABEL_93;
                case 0xD:
                  v94 = v95;
                  v98 = v99;
                  v102 = v39;
                  v109 = v38;
                  v90 = (_QWORD *)sub_15A9930(a2, v24);
                  v38 = v109;
                  v39 = v102;
                  v26 = v98;
                  v27 = v94;
                  v85 = 8LL * *v90;
                  goto LABEL_93;
                case 0xE:
                  v87 = *(_QWORD *)(v24 + 32);
                  v91 = v95;
                  v24 = *(_QWORD *)(v24 + 24);
                  v92 = v99;
                  v96 = v39;
                  v100 = v38;
                  v107 = v87;
                  v88 = sub_12BE0A0(a2, v24);
                  v38 = v100;
                  v39 = v96;
                  v26 = v92;
                  v27 = v91;
                  v85 = 8 * v107 * v88;
                  goto LABEL_93;
                case 0xF:
                  v93 = v95;
                  v97 = v99;
                  v101 = v39;
                  v24 = *(_DWORD *)(v24 + 8) >> 8;
                  v108 = v38;
LABEL_98:
                  v89 = sub_15A9520(a2, v24);
                  v38 = v108;
                  v39 = v101;
                  v26 = v97;
                  v27 = v93;
                  v85 = (unsigned int)(8 * v89);
LABEL_93:
                  v28 = 8 * v119 * v39 * ((v39 + ((unsigned __int64)(v38 * v85 + 7) >> 3) - 1) / v39);
                  break;
              }
              goto LABEL_14;
            }
          case 0xF:
            v103 = v27;
            v118 = v26;
            v24 = *(_DWORD *)(v24 + 8) >> 8;
LABEL_32:
            v36 = sub_15A9520(a2, v24);
            v26 = v118;
            v27 = v103;
            v28 = (unsigned int)(8 * v36);
LABEL_14:
            v29 = (unsigned __int64)(v28 * v26 + 7) >> 3;
            v117 = v110 & (v27 * ((v27 + v29 - 1) / v27));
            if ( *(_BYTE *)(v22 + 16) > 0x10u )
            {
              if ( v113 != *(_QWORD *)v22 )
              {
                v129[0] = sub_1649960(v22);
                v133[0] = (__int64)v129;
                v129[1] = v42;
                v134 = 773;
                v133[1] = (__int64)".c";
                if ( v113 != *(_QWORD *)v22 )
                {
                  if ( *(_BYTE *)(v22 + 16) > 0x10u )
                  {
                    v141 = 257;
                    v22 = sub_15FE0A0((_QWORD *)v22, v113, 1, (__int64)&v139, 0);
                    v78 = *(_QWORD *)(a1 + 8);
                    if ( v78 )
                    {
                      v106 = *(__int64 **)(a1 + 16);
                      sub_157E9D0(v78 + 40, v22);
                      v79 = *v106;
                      v80 = *(_QWORD *)(v22 + 24) & 7LL;
                      *(_QWORD *)(v22 + 32) = v106;
                      v79 &= 0xFFFFFFFFFFFFFFF8LL;
                      *(_QWORD *)(v22 + 24) = v79 | v80;
                      *(_QWORD *)(v79 + 8) = v22 + 24;
                      *v106 = *v106 & 7 | (v22 + 24);
                    }
                    v61 = v133;
                    v62 = v22;
                    sub_164B780(v22, v133);
                    v126 = v22;
                    if ( !*(_QWORD *)(a1 + 80) )
LABEL_105:
                      sub_4263D6(v62, v61, v63);
                    (*(void (__fastcall **)(__int64, __int64 *))(a1 + 88))(a1 + 64, &v126);
                    v81 = *(_QWORD *)a1;
                    if ( *(_QWORD *)a1 )
                    {
                      v142 = *(unsigned __int8 **)a1;
                      sub_1623A60((__int64)&v142, v81, 2);
                      v82 = *(_QWORD *)(v22 + 48);
                      v83 = v22 + 48;
                      if ( v82 )
                      {
                        sub_161E7C0(v22 + 48, v82);
                        v83 = v22 + 48;
                      }
                      v84 = v142;
                      *(_QWORD *)(v22 + 48) = v142;
                      if ( v84 )
                        sub_1623210((__int64)&v142, v84, v83);
                    }
                  }
                  else
                  {
                    v22 = sub_15A4750((__int64 ***)v22, (__int64 **)v113, 1);
                    v43 = sub_14DBA30(v22, *(_QWORD *)(a1 + 96), 0);
                    if ( v43 )
                      v22 = v43;
                  }
                }
              }
              if ( v117 != 1 )
              {
                v44 = sub_1649960(a3);
                v144 = 773;
                v143 = ".idx";
                v139 = (unsigned __int8 *)v44;
                v140 = v45;
                v142 = (unsigned __int8 *)&v139;
                v46 = sub_15A0680(v113, v117, 0);
                if ( *(_BYTE *)(v22 + 16) > 0x10u || *(_BYTE *)(v46 + 16) > 0x10u )
                {
                  v22 = (__int64)sub_170A2B0(a1, 15, (__int64 *)v22, v46, (__int64 *)&v142, v122, 0);
                }
                else
                {
                  v22 = sub_15A2C20((__int64 *)v22, v46, v122, 0, a5, a6, a7);
                  v47 = sub_14DBA30(v22, *(_QWORD *)(a1 + 96), 0);
                  if ( v47 )
                    v22 = v47;
                }
              }
              v48 = sub_1649960(a3);
              v130[1] = v49;
              v130[0] = v48;
              v136 = 773;
              v50 = *(_BYTE *)(v22 + 16) <= 0x10u;
              v135[0] = (__int64)v130;
              v135[1] = (__int64)".offs";
              if ( v50 && *(_BYTE *)(v13 + 16) <= 0x10u )
              {
                v51 = v13;
                v52 = (__int64 *)v22;
                goto LABEL_53;
              }
              v144 = 257;
              v13 = sub_15FB440(11, (__int64 *)v22, v13, (__int64)&v142, 0);
              v57 = *(_QWORD *)(a1 + 8);
              if ( v57 )
              {
                v58 = *(__int64 **)(a1 + 16);
                sub_157E9D0(v57 + 40, v13);
                v59 = *(_QWORD *)(v13 + 24);
                v60 = *v58;
                *(_QWORD *)(v13 + 32) = v58;
                v60 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v13 + 24) = v60 | v59 & 7;
                *(_QWORD *)(v60 + 8) = v13 + 24;
                *v58 = *v58 & 7 | (v13 + 24);
              }
              v61 = v135;
              v62 = v13;
              sub_164B780(v13, v135);
              v127 = v13;
              if ( !*(_QWORD *)(a1 + 80) )
                goto LABEL_105;
              (*(void (__fastcall **)(__int64, __int64 *))(a1 + 88))(a1 + 64, &v127);
              v64 = *(_QWORD *)a1;
              if ( *(_QWORD *)a1 )
              {
                v139 = *(unsigned __int8 **)a1;
                sub_1623A60((__int64)&v139, v64, 2);
                v65 = *(_QWORD *)(v13 + 48);
                v66 = v13 + 48;
                if ( v65 )
                {
                  sub_161E7C0(v13 + 48, v65);
                  v66 = v13 + 48;
                }
                v67 = v139;
                *(_QWORD *)(v13 + 48) = v139;
                if ( v67 )
                  sub_1623210((__int64)&v139, v67, v66);
              }
            }
            else
            {
              if ( sub_1595F50(v22, v24, (v27 + v29 - 1) % v27, v29) )
                goto LABEL_55;
              if ( !v123 && v21 )
              {
                if ( *(_BYTE *)(*(_QWORD *)v22 + 8LL) == 16 )
                  v22 = sub_15A1020((_BYTE *)v22, v24, v30, v31);
                v32 = *(_QWORD **)(v22 + 24);
                if ( *(_DWORD *)(v22 + 32) > 0x40u )
                  v32 = (_QWORD *)*v32;
                v33 = *(_QWORD *)(sub_15A9930(a2, v21) + 8LL * (unsigned int)v32 + 16);
                if ( v33 )
                {
                  v75 = sub_1649960(a3);
                  v144 = 773;
                  v139 = (unsigned __int8 *)v75;
                  v140 = v76;
                  v142 = (unsigned __int8 *)&v139;
                  v143 = ".offs";
                  v77 = sub_15A0680(v113, v33, 0);
                  v13 = (__int64)sub_17094A0(a1, v13, v77, (__int64 *)&v142, 0, 0, a5, a6, a7);
                }
LABEL_24:
                v23 = sub_1643D30(v21, *v124);
                v34 = *(_BYTE *)(v23 + 8);
                if ( ((v34 - 14) & 0xFD) != 0 )
                  goto LABEL_58;
                goto LABEL_25;
              }
              v121 = sub_15A0680(v113, v117, 0);
              v54 = (__int64 *)sub_15A4750((__int64 ***)v22, (__int64 **)v113, 1);
              v55 = sub_15A2C20(v54, v121, v122, 0, a5, a6, a7);
              v128[0] = sub_1649960(a3);
              v128[1] = v56;
              v132 = 773;
              v50 = *(_BYTE *)(v13 + 16) <= 0x10u;
              v131[0] = (__int64)v128;
              v131[1] = (__int64)".offs";
              if ( v50 && *(_BYTE *)(v55 + 16) <= 0x10u )
              {
                v51 = v55;
                v52 = (__int64 *)v13;
LABEL_53:
                v13 = sub_15A2B30(v52, v51, 0, 0, a5, a6, a7);
                v53 = sub_14DBA30(v13, *(_QWORD *)(a1 + 96), 0);
                if ( v53 )
                  v13 = v53;
                goto LABEL_55;
              }
              v138 = 257;
              v13 = sub_15FB440(11, (__int64 *)v13, v55, (__int64)v137, 0);
              v68 = *(_QWORD *)(a1 + 8);
              if ( v68 )
              {
                v69 = *(__int64 **)(a1 + 16);
                sub_157E9D0(v68 + 40, v13);
                v70 = *(_QWORD *)(v13 + 24);
                v71 = *v69;
                *(_QWORD *)(v13 + 32) = v69;
                v71 &= 0xFFFFFFFFFFFFFFF8LL;
                *(_QWORD *)(v13 + 24) = v71 | v70 & 7;
                *(_QWORD *)(v71 + 8) = v13 + 24;
                *v69 = *v69 & 7 | (v13 + 24);
              }
              v61 = v131;
              v62 = v13;
              sub_164B780(v13, v131);
              v125 = v13;
              if ( !*(_QWORD *)(a1 + 80) )
                goto LABEL_105;
              (*(void (__fastcall **)(__int64, __int64 *))(a1 + 88))(a1 + 64, &v125);
              v72 = *(_QWORD *)a1;
              if ( *(_QWORD *)a1 )
              {
                v142 = *(unsigned __int8 **)a1;
                sub_1623A60((__int64)&v142, v72, 2);
                v73 = *(_QWORD *)(v13 + 48);
                if ( v73 )
                  sub_161E7C0(v13 + 48, v73);
                v74 = v142;
                *(_QWORD *)(v13 + 48) = v142;
                if ( v74 )
                  sub_1623210((__int64)&v142, v74, v13 + 48);
              }
            }
LABEL_55:
            if ( !v123 || !v21 )
              goto LABEL_24;
            v34 = *(_BYTE *)(v21 + 8);
            if ( ((v34 - 14) & 0xFD) != 0 )
            {
LABEL_58:
              v17 = 0;
              if ( v34 == 13 )
                v17 = v23;
              goto LABEL_26;
            }
LABEL_25:
            v17 = *(_QWORD *)(v23 + 24) | 4LL;
LABEL_26:
            v124 += 3;
            v19 += 24;
            if ( v116 == v112 )
              return v13;
            break;
        }
        break;
      }
    }
  }
  return v13;
}
