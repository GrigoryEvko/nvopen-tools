// Function: sub_1CEE970
// Address: 0x1cee970
//
__int64 __fastcall sub_1CEE970(unsigned int **a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // rax
  __int64 v4; // rbx
  __int64 *v5; // rbx
  __int64 *v6; // rdx
  __int64 *v7; // r15
  unsigned int v8; // r13d
  __int64 v9; // r12
  __int64 v10; // r12
  unsigned __int64 v11; // r8
  __int64 v12; // rax
  int v13; // eax
  __int64 v14; // rax
  __m128i *v15; // rdx
  __int64 v16; // rdi
  __m128i si128; // xmm0
  __int64 v18; // rax
  __m128i *v19; // rdx
  __int64 v20; // rdi
  __m128i v21; // xmm0
  __int64 v22; // rax
  __m128i *v23; // rdx
  __int64 v24; // r13
  __m128i v25; // xmm0
  char *v26; // rax
  size_t v27; // rdx
  void *v28; // rdi
  __int64 v29; // rax
  int v30; // eax
  int v31; // eax
  unsigned int v32; // eax
  __int64 v33; // r9
  unsigned __int64 v34; // r10
  _QWORD *v35; // rax
  unsigned int v36; // eax
  __int64 v37; // r9
  unsigned __int64 v38; // r8
  __int64 v39; // rax
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // r12
  unsigned int v44; // eax
  __int64 v45; // r11
  unsigned __int64 v46; // r10
  _QWORD *v47; // rax
  __int64 v49; // rax
  _QWORD *v50; // rax
  unsigned int v51; // eax
  __int64 v52; // rdx
  unsigned __int64 v53; // r11
  int v54; // eax
  __int64 v55; // rax
  __int64 v56; // rax
  _QWORD *v57; // rax
  unsigned int v58; // eax
  unsigned __int64 v59; // r12
  int v60; // eax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  unsigned __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  int v68; // eax
  __int64 v69; // rax
  unsigned __int64 v70; // rax
  _QWORD *v71; // rax
  __int64 v72; // [rsp+0h] [rbp-100h]
  __int64 v73; // [rsp+0h] [rbp-100h]
  __int64 v74; // [rsp+8h] [rbp-F8h]
  __int64 v75; // [rsp+8h] [rbp-F8h]
  __int64 v76; // [rsp+8h] [rbp-F8h]
  __int64 v77; // [rsp+8h] [rbp-F8h]
  __int64 v78; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v79; // [rsp+10h] [rbp-F0h]
  __int64 v80; // [rsp+10h] [rbp-F0h]
  __int64 v81; // [rsp+10h] [rbp-F0h]
  __int64 v82; // [rsp+10h] [rbp-F0h]
  __int64 v83; // [rsp+18h] [rbp-E8h]
  __int64 v84; // [rsp+18h] [rbp-E8h]
  __int64 v85; // [rsp+18h] [rbp-E8h]
  __int64 v86; // [rsp+18h] [rbp-E8h]
  __int64 v87; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v88; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v89; // [rsp+18h] [rbp-E8h]
  __int64 v90; // [rsp+20h] [rbp-E0h]
  __int64 v91; // [rsp+20h] [rbp-E0h]
  __int64 v92; // [rsp+20h] [rbp-E0h]
  __int64 v93; // [rsp+20h] [rbp-E0h]
  __int64 v94; // [rsp+20h] [rbp-E0h]
  __int64 v95; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v96; // [rsp+20h] [rbp-E0h]
  __int64 v97; // [rsp+20h] [rbp-E0h]
  __int64 v98; // [rsp+20h] [rbp-E0h]
  __int64 v99; // [rsp+20h] [rbp-E0h]
  __int64 v100; // [rsp+28h] [rbp-D8h]
  __int64 v101; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v102; // [rsp+28h] [rbp-D8h]
  __int64 v103; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v104; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v105; // [rsp+28h] [rbp-D8h]
  __int64 v106; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v107; // [rsp+28h] [rbp-D8h]
  __int64 v108; // [rsp+28h] [rbp-D8h]
  __int64 v109; // [rsp+28h] [rbp-D8h]
  __int64 v110; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v111; // [rsp+30h] [rbp-D0h]
  __int64 v112; // [rsp+30h] [rbp-D0h]
  unsigned __int64 v113; // [rsp+30h] [rbp-D0h]
  __int64 v114; // [rsp+30h] [rbp-D0h]
  __int64 v115; // [rsp+30h] [rbp-D0h]
  __int64 v116; // [rsp+30h] [rbp-D0h]
  __int64 v117; // [rsp+30h] [rbp-D0h]
  __int64 v118; // [rsp+30h] [rbp-D0h]
  __int64 v119; // [rsp+30h] [rbp-D0h]
  __int64 v120; // [rsp+38h] [rbp-C8h]
  unsigned int v121; // [rsp+40h] [rbp-C0h]
  unsigned __int8 v122; // [rsp+46h] [rbp-BAh]
  unsigned __int8 v123; // [rsp+47h] [rbp-B9h]
  __int64 v124; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v125; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v126; // [rsp+50h] [rbp-B0h]
  __int64 v127; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v128; // [rsp+50h] [rbp-B0h]
  __int64 v129; // [rsp+50h] [rbp-B0h]
  __int64 v130; // [rsp+50h] [rbp-B0h]
  __int64 v131; // [rsp+50h] [rbp-B0h]
  size_t v132; // [rsp+50h] [rbp-B0h]
  __int64 v133; // [rsp+58h] [rbp-A8h]
  _QWORD v134[2]; // [rsp+60h] [rbp-A0h] BYREF
  _QWORD v135[2]; // [rsp+70h] [rbp-90h] BYREF
  char *v136[2]; // [rsp+80h] [rbp-80h] BYREF
  __int64 v137; // [rsp+90h] [rbp-70h] BYREF
  void *v138; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v139; // [rsp+A8h] [rbp-58h]
  __int64 v140; // [rsp+B0h] [rbp-50h]
  __int64 v141; // [rsp+B8h] [rbp-48h]
  int v142; // [rsp+C0h] [rbp-40h]
  _QWORD *v143; // [rsp+C8h] [rbp-38h]

  v122 = 0;
  v2 = sub_1632FA0(a2);
  v120 = a2 + 24;
  v121 = **a1;
  v133 = *(_QWORD *)(a2 + 32);
  if ( v133 != a2 + 24 )
  {
    do
    {
      v3 = 0;
      if ( v133 )
        v3 = v133 - 56;
      v124 = v3;
      v4 = v3;
      v123 = sub_1C2F070(v3);
      if ( v123 )
      {
        if ( (*(_BYTE *)(v4 + 18) & 1) != 0 )
        {
          sub_15E08E0(v124, a2);
          v5 = *(__int64 **)(v124 + 88);
          if ( (*(_BYTE *)(v124 + 18) & 1) != 0 )
            sub_15E08E0(v124, a2);
          v6 = *(__int64 **)(v124 + 88);
        }
        else
        {
          v5 = *(__int64 **)(v124 + 88);
          v6 = v5;
        }
        v7 = &v6[5 * *(_QWORD *)(v124 + 96)];
        if ( v7 != v5 )
        {
          v8 = 0;
          while ( 2 )
          {
            v9 = *v5;
            if ( (unsigned __int8)sub_15E0450((__int64)v5) )
            {
              if ( *(_BYTE *)(v9 + 8) == 15 )
              {
                a2 = *(_QWORD *)(v9 + 24);
                v10 = 1;
                v11 = (unsigned int)sub_15A9FE0(v2, a2);
                while ( 2 )
                {
                  switch ( *(_BYTE *)(a2 + 8) )
                  {
                    case 0:
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v29 = *(_QWORD *)(a2 + 32);
                      a2 = *(_QWORD *)(a2 + 24);
                      v10 *= v29;
                      continue;
                    case 1:
                      v12 = 16;
                      goto LABEL_16;
                    case 2:
                      v12 = 32;
                      goto LABEL_16;
                    case 3:
                    case 9:
                      v12 = 64;
                      goto LABEL_16;
                    case 4:
                      v12 = 80;
                      goto LABEL_16;
                    case 5:
                    case 6:
                      v12 = 128;
                      goto LABEL_16;
                    case 7:
                      a2 = 0;
                      v125 = v11;
                      v30 = sub_15A9520(v2, 0);
                      v11 = v125;
                      v12 = (unsigned int)(8 * v30);
                      goto LABEL_16;
                    case 0xB:
                      v12 = *(_DWORD *)(a2 + 8) >> 8;
                      goto LABEL_16;
                    case 0xD:
                      v128 = v11;
                      v35 = (_QWORD *)sub_15A9930(v2, a2);
                      v11 = v128;
                      v12 = 8LL * *v35;
                      goto LABEL_16;
                    case 0xE:
                      v100 = v11;
                      v110 = *(_QWORD *)(a2 + 24);
                      v127 = *(_QWORD *)(a2 + 32);
                      v32 = sub_15A9FE0(v2, v110);
                      v11 = v100;
                      a2 = v110;
                      v33 = 1;
                      v34 = v32;
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(a2 + 8) )
                        {
                          case 0:
                          case 8:
                          case 0xA:
                          case 0xC:
                          case 0x10:
                            v55 = *(_QWORD *)(a2 + 32);
                            a2 = *(_QWORD *)(a2 + 24);
                            v33 *= v55;
                            continue;
                          case 1:
                            v49 = 16;
                            goto LABEL_67;
                          case 2:
                            v49 = 32;
                            goto LABEL_67;
                          case 3:
                          case 9:
                            v49 = 64;
                            goto LABEL_67;
                          case 4:
                            v49 = 80;
                            goto LABEL_67;
                          case 5:
                          case 6:
                            v49 = 128;
                            goto LABEL_67;
                          case 7:
                            v92 = v100;
                            a2 = 0;
                            v104 = v34;
                            v116 = v33;
                            goto LABEL_77;
                          case 0xB:
                            v49 = *(_DWORD *)(a2 + 8) >> 8;
                            goto LABEL_67;
                          case 0xD:
                            v90 = v100;
                            v102 = v34;
                            v114 = v33;
                            v50 = (_QWORD *)sub_15A9930(v2, a2);
                            v33 = v114;
                            v34 = v102;
                            v11 = v90;
                            v49 = 8LL * *v50;
                            goto LABEL_67;
                          case 0xE:
                            v78 = v100;
                            v83 = v34;
                            v91 = v33;
                            v103 = *(_QWORD *)(a2 + 24);
                            v115 = *(_QWORD *)(a2 + 32);
                            v51 = sub_15A9FE0(v2, v103);
                            v11 = v78;
                            v34 = v83;
                            v52 = 1;
                            a2 = v103;
                            v33 = v91;
                            v53 = v51;
                            while ( 2 )
                            {
                              switch ( *(_BYTE *)(a2 + 8) )
                              {
                                case 0:
                                case 8:
                                case 0xA:
                                case 0xC:
                                case 0x10:
                                  v65 = *(_QWORD *)(a2 + 32);
                                  a2 = *(_QWORD *)(a2 + 24);
                                  v52 *= v65;
                                  continue;
                                case 1:
                                  v62 = 16;
                                  goto LABEL_99;
                                case 2:
                                  v62 = 32;
                                  goto LABEL_99;
                                case 3:
                                case 9:
                                  v62 = 64;
                                  goto LABEL_99;
                                case 4:
                                  v62 = 80;
                                  goto LABEL_99;
                                case 5:
                                case 6:
                                  v62 = 128;
                                  goto LABEL_99;
                                case 7:
                                  v75 = v78;
                                  a2 = 0;
                                  v80 = v83;
                                  v87 = v91;
                                  v96 = v53;
                                  v108 = v52;
                                  goto LABEL_104;
                                case 0xB:
                                  v62 = *(_DWORD *)(a2 + 8) >> 8;
                                  goto LABEL_99;
                                case 0xD:
                                  v75 = v78;
                                  v80 = v83;
                                  v87 = v91;
                                  v96 = v53;
                                  v108 = v52;
                                  v62 = 8LL * *(_QWORD *)sub_15A9930(v2, a2);
                                  goto LABEL_105;
                                case 0xE:
                                  v63 = *(_QWORD *)(a2 + 32);
                                  a2 = *(_QWORD *)(a2 + 24);
                                  v73 = v78;
                                  v76 = v83;
                                  v81 = v91;
                                  v88 = v53;
                                  v97 = v52;
                                  v109 = v63;
                                  v64 = sub_12BE0A0(v2, a2);
                                  v52 = v97;
                                  v53 = v88;
                                  v33 = v81;
                                  v34 = v76;
                                  v11 = v73;
                                  v62 = 8 * v109 * v64;
                                  goto LABEL_99;
                                case 0xF:
                                  v75 = v78;
                                  v80 = v83;
                                  v87 = v91;
                                  a2 = *(_DWORD *)(a2 + 8) >> 8;
                                  v96 = v53;
                                  v108 = v52;
LABEL_104:
                                  v62 = 8 * (unsigned int)sub_15A9520(v2, a2);
LABEL_105:
                                  v52 = v108;
                                  v53 = v96;
                                  v33 = v87;
                                  v34 = v80;
                                  v11 = v75;
LABEL_99:
                                  v49 = 8 * v115 * v53 * ((v53 + ((unsigned __int64)(v62 * v52 + 7) >> 3) - 1) / v53);
                                  goto LABEL_67;
                                default:
                                  goto LABEL_126;
                              }
                            }
                          case 0xF:
                            v92 = v100;
                            v104 = v34;
                            v116 = v33;
                            a2 = *(_DWORD *)(a2 + 8) >> 8;
LABEL_77:
                            v54 = sub_15A9520(v2, a2);
                            v33 = v116;
                            v34 = v104;
                            v11 = v92;
                            v49 = (unsigned int)(8 * v54);
LABEL_67:
                            v12 = 8 * v127 * v34 * ((v34 + ((unsigned __int64)(v49 * v33 + 7) >> 3) - 1) / v34);
                            goto LABEL_16;
                          default:
                            goto LABEL_126;
                        }
                      }
                    case 0xF:
                      v126 = v11;
                      a2 = *(_DWORD *)(a2 + 8) >> 8;
                      v31 = sub_15A9520(v2, a2);
                      v11 = v126;
                      v12 = (unsigned int)(8 * v31);
LABEL_16:
                      v13 = v11 * ((v11 + ((unsigned __int64)(v12 * v10 + 7) >> 3) - 1) / v11);
                      goto LABEL_17;
                    default:
                      goto LABEL_126;
                  }
                }
              }
LABEL_126:
              BUG();
            }
            a2 = v9;
            v36 = sub_15A9FE0(v2, v9);
            v37 = 1;
            v38 = v36;
LABEL_47:
            switch ( *(_BYTE *)(v9 + 8) )
            {
              case 1:
                v39 = 16;
                goto LABEL_49;
              case 2:
                v39 = 32;
                goto LABEL_49;
              case 3:
              case 9:
                v39 = 64;
                goto LABEL_49;
              case 4:
                v39 = 80;
                goto LABEL_49;
              case 5:
              case 6:
                v39 = 128;
                goto LABEL_49;
              case 7:
                v111 = v38;
                a2 = 0;
                v129 = v37;
                goto LABEL_54;
              case 0xB:
                v39 = *(_DWORD *)(v9 + 8) >> 8;
                goto LABEL_49;
              case 0xD:
                a2 = v9;
                v113 = v38;
                v131 = v37;
                v47 = (_QWORD *)sub_15A9930(v2, v9);
                v37 = v131;
                v38 = v113;
                v39 = 8LL * *v47;
                goto LABEL_49;
              case 0xE:
                v42 = *(_QWORD *)(v9 + 32);
                v43 = *(_QWORD *)(v9 + 24);
                v101 = v38;
                v112 = v37;
                a2 = v43;
                v130 = v42;
                v44 = sub_15A9FE0(v2, v43);
                v38 = v101;
                v37 = v112;
                v45 = 1;
                v46 = v44;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v43 + 8) )
                  {
                    case 1:
                      v56 = 16;
                      goto LABEL_85;
                    case 2:
                      v56 = 32;
                      goto LABEL_85;
                    case 3:
                    case 9:
                      v56 = 64;
                      goto LABEL_85;
                    case 4:
                      v56 = 80;
                      goto LABEL_85;
                    case 5:
                    case 6:
                      v56 = 128;
                      goto LABEL_85;
                    case 7:
                      v86 = v101;
                      a2 = 0;
                      v95 = v112;
                      v107 = v46;
                      v119 = v45;
                      goto LABEL_93;
                    case 0xB:
                      v56 = *(_DWORD *)(v43 + 8) >> 8;
                      goto LABEL_85;
                    case 0xD:
                      a2 = v43;
                      v84 = v101;
                      v93 = v112;
                      v105 = v46;
                      v117 = v45;
                      v57 = (_QWORD *)sub_15A9930(v2, v43);
                      v45 = v117;
                      v46 = v105;
                      v37 = v93;
                      v38 = v84;
                      v56 = 8LL * *v57;
                      goto LABEL_85;
                    case 0xE:
                      v72 = v101;
                      v74 = v112;
                      v79 = v46;
                      v85 = v45;
                      v94 = *(_QWORD *)(v43 + 24);
                      v106 = *(_QWORD *)(v43 + 32);
                      v58 = sub_15A9FE0(v2, v94);
                      v38 = v72;
                      v118 = 1;
                      v37 = v74;
                      v46 = v79;
                      v59 = v58;
                      a2 = v94;
                      v45 = v85;
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(a2 + 8) )
                        {
                          case 1:
                            v66 = 16;
                            goto LABEL_113;
                          case 2:
                            v66 = 32;
                            goto LABEL_113;
                          case 3:
                          case 9:
                            v66 = 64;
                            goto LABEL_113;
                          case 4:
                            v66 = 80;
                            goto LABEL_113;
                          case 5:
                          case 6:
                            v66 = 128;
                            goto LABEL_113;
                          case 7:
                            v77 = v72;
                            a2 = 0;
                            v82 = v37;
                            v89 = v46;
                            v98 = v45;
                            goto LABEL_117;
                          case 0xB:
                            v66 = *(_DWORD *)(a2 + 8) >> 8;
                            goto LABEL_113;
                          case 0xD:
                            v71 = (_QWORD *)sub_15A9930(v2, a2);
                            v45 = v85;
                            v46 = v79;
                            v37 = v74;
                            v38 = v72;
                            v66 = 8LL * *v71;
                            goto LABEL_113;
                          case 0xE:
                            v69 = *(_QWORD *)(a2 + 32);
                            a2 = *(_QWORD *)(a2 + 24);
                            v99 = v69;
                            v70 = sub_12BE0A0(v2, a2);
                            v45 = v85;
                            v46 = v79;
                            v37 = v74;
                            v38 = v72;
                            v66 = 8 * v99 * v70;
                            goto LABEL_113;
                          case 0xF:
                            v77 = v72;
                            v82 = v37;
                            v89 = v46;
                            a2 = *(_DWORD *)(a2 + 8) >> 8;
                            v98 = v45;
LABEL_117:
                            v68 = sub_15A9520(v2, a2);
                            v45 = v98;
                            v46 = v89;
                            v37 = v82;
                            v38 = v77;
                            v66 = (unsigned int)(8 * v68);
LABEL_113:
                            v56 = 8 * v106 * v59 * ((v59 + ((unsigned __int64)(v118 * v66 + 7) >> 3) - 1) / v59);
                            goto LABEL_85;
                          case 0x10:
                            v67 = v118 * *(_QWORD *)(a2 + 32);
                            a2 = *(_QWORD *)(a2 + 24);
                            v118 = v67;
                            continue;
                          default:
                            goto LABEL_126;
                        }
                      }
                    case 0xF:
                      v86 = v101;
                      v95 = v112;
                      v107 = v46;
                      a2 = *(_DWORD *)(v43 + 8) >> 8;
                      v119 = v45;
LABEL_93:
                      v60 = sub_15A9520(v2, a2);
                      v45 = v119;
                      v46 = v107;
                      v37 = v95;
                      v38 = v86;
                      v56 = (unsigned int)(8 * v60);
LABEL_85:
                      v39 = 8 * v130 * v46 * ((v46 + ((unsigned __int64)(v56 * v45 + 7) >> 3) - 1) / v46);
                      goto LABEL_49;
                    case 0x10:
                      v61 = *(_QWORD *)(v43 + 32);
                      v43 = *(_QWORD *)(v43 + 24);
                      v45 *= v61;
                      continue;
                    default:
                      goto LABEL_126;
                  }
                }
              case 0xF:
                v111 = v38;
                v129 = v37;
                a2 = *(_DWORD *)(v9 + 8) >> 8;
LABEL_54:
                v40 = sub_15A9520(v2, a2);
                v37 = v129;
                v38 = v111;
                v39 = (unsigned int)(8 * v40);
LABEL_49:
                v13 = v38 * ((v38 + ((unsigned __int64)(v39 * v37 + 7) >> 3) - 1) / v38);
LABEL_17:
                v5 += 5;
                v8 += v13;
                if ( v5 != v7 )
                  continue;
                if ( v121 < v8 )
                {
                  v143 = v134;
                  v134[0] = v135;
                  v138 = &unk_49EFBE0;
                  v134[1] = 0;
                  LOBYTE(v135[0]) = 0;
                  v142 = 1;
                  v141 = 0;
                  v140 = 0;
                  v139 = 0;
                  sub_1C30B20((__int64)v136, v124);
                  v14 = sub_16E7EE0((__int64)&v138, v136[0], (size_t)v136[1]);
                  v15 = *(__m128i **)(v14 + 24);
                  v16 = v14;
                  if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 0x2Bu )
                  {
                    v16 = sub_16E7EE0(v14, ": Error: Formal parameter space overflowed (", 0x2Cu);
                  }
                  else
                  {
                    si128 = _mm_load_si128((const __m128i *)&xmmword_42E1EE0);
                    qmemcpy(&v15[2], "overflowed (", 12);
                    *v15 = si128;
                    v15[1] = _mm_load_si128((const __m128i *)&xmmword_42E1EF0);
                    *(_QWORD *)(v14 + 24) += 44LL;
                  }
                  v18 = sub_16E7A90(v16, v8);
                  v19 = *(__m128i **)(v18 + 24);
                  v20 = v18;
                  if ( *(_QWORD *)(v18 + 16) - (_QWORD)v19 <= 0x14u )
                  {
                    v20 = sub_16E7EE0(v18, " bytes required, max ", 0x15u);
                  }
                  else
                  {
                    v21 = _mm_load_si128((const __m128i *)&xmmword_42E1F00);
                    v19[1].m128i_i32[0] = 2019650848;
                    v19[1].m128i_i8[4] = 32;
                    *v19 = v21;
                    *(_QWORD *)(v18 + 24) += 21LL;
                  }
                  v22 = sub_16E7A90(v20, v121);
                  v23 = *(__m128i **)(v22 + 24);
                  v24 = v22;
                  if ( *(_QWORD *)(v22 + 16) - (_QWORD)v23 <= 0x1Bu )
                  {
                    v24 = sub_16E7EE0(v22, " bytes allowed) in function ", 0x1Cu);
                  }
                  else
                  {
                    v25 = _mm_load_si128((const __m128i *)&xmmword_42E1F10);
                    qmemcpy(&v23[1], "in function ", 12);
                    *v23 = v25;
                    *(_QWORD *)(v22 + 24) += 28LL;
                  }
                  v26 = (char *)sub_1649960(v124);
                  v28 = *(void **)(v24 + 24);
                  if ( v27 > *(_QWORD *)(v24 + 16) - (_QWORD)v28 )
                  {
                    sub_16E7EE0(v24, v26, v27);
                  }
                  else if ( v27 )
                  {
                    v132 = v27;
                    memcpy(v28, v26, v27);
                    *(_QWORD *)(v24 + 24) += v132;
                  }
                  if ( (__int64 *)v136[0] != &v137 )
                    j_j___libc_free_0(v136[0], v137 + 1);
                  if ( v141 != v139 )
                    sub_16E7BA0((__int64 *)&v138);
                  a2 = 1;
                  sub_1C3EFD0((__int64)v143, 1);
                  sub_16E7BC0((__int64 *)&v138);
                  if ( (_QWORD *)v134[0] != v135 )
                  {
                    a2 = v135[0] + 1LL;
                    j_j___libc_free_0(v134[0], v135[0] + 1LL);
                  }
                  v122 = v123;
                }
                break;
              case 0x10:
                v41 = *(_QWORD *)(v9 + 32);
                v9 = *(_QWORD *)(v9 + 24);
                v37 *= v41;
                goto LABEL_47;
              default:
                goto LABEL_126;
            }
            break;
          }
        }
      }
      v133 = *(_QWORD *)(v133 + 8);
    }
    while ( v120 != v133 );
  }
  return v122;
}
