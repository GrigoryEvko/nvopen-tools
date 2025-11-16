// Function: sub_135EAB0
// Address: 0x135eab0
//
__int64 __fastcall sub_135EAB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int8 v5; // al
  int v6; // edx
  __int64 v7; // r15
  __int64 *v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  _QWORD **v15; // rbx
  __int64 v16; // rax
  unsigned int v17; // r13d
  _QWORD *v18; // rbx
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 *v21; // r15
  char v22; // r14
  _QWORD *v23; // r13
  __int64 v24; // rax
  unsigned __int64 v25; // r12
  __int64 v26; // r10
  __int64 v27; // rbx
  __int64 v28; // rax
  _QWORD *v29; // rdx
  char v30; // al
  __int64 v31; // rax
  unsigned __int64 v32; // rsi
  __int64 v33; // r14
  unsigned int v34; // eax
  __int64 v35; // r10
  unsigned __int64 v36; // rcx
  int v37; // eax
  unsigned __int64 v38; // rsi
  unsigned int v39; // eax
  __int64 v40; // r10
  __int64 v41; // r8
  unsigned __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  unsigned __int64 v46; // r14
  unsigned int v47; // eax
  int v48; // edx
  __int64 v49; // rax
  __int64 v50; // rcx
  unsigned __int32 v51; // edx
  __int64 v52; // r8
  __int64 v53; // rax
  __int64 v54; // rax
  unsigned __int64 v55; // r14
  __int64 v56; // rcx
  _QWORD *v57; // rsi
  __int64 v58; // r10
  _QWORD *v59; // rdi
  char *v60; // rdx
  __int64 v61; // r14
  __int64 v62; // rsi
  int v63; // eax
  __int64 v64; // rax
  unsigned int v65; // eax
  __int64 v66; // r9
  __int64 v67; // rsi
  unsigned __int64 v68; // r8
  _QWORD *v69; // rax
  __int64 v70; // rax
  unsigned int v71; // esi
  __int64 *v72; // rdx
  unsigned __int64 v73; // rax
  __int64 v74; // rcx
  unsigned __int64 v75; // rdi
  __int64 v76; // rax
  _QWORD *v77; // rax
  __int64 v78; // rsi
  int v79; // eax
  unsigned int v80; // eax
  __int64 v81; // rsi
  __int64 v82; // rax
  __m128i *v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rsi
  int v87; // eax
  __int64 v88; // rax
  _QWORD *v89; // rax
  __int64 v90; // rax
  __int64 v91; // rax
  __int64 v92; // rsi
  int v93; // eax
  __int64 v94; // rax
  _QWORD *v95; // rax
  __int64 v96; // [rsp+8h] [rbp-118h]
  __int64 v97; // [rsp+10h] [rbp-110h]
  __int64 v98; // [rsp+10h] [rbp-110h]
  __int64 v99; // [rsp+18h] [rbp-108h]
  __int64 v100; // [rsp+18h] [rbp-108h]
  __int64 v101; // [rsp+18h] [rbp-108h]
  __int64 v102; // [rsp+20h] [rbp-100h]
  unsigned __int64 v103; // [rsp+20h] [rbp-100h]
  __int64 v104; // [rsp+20h] [rbp-100h]
  __int64 v105; // [rsp+20h] [rbp-100h]
  __int64 v106; // [rsp+28h] [rbp-F8h]
  __int64 v107; // [rsp+28h] [rbp-F8h]
  __int64 v108; // [rsp+28h] [rbp-F8h]
  __int64 v109; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v110; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v111; // [rsp+28h] [rbp-F8h]
  __int64 v112; // [rsp+30h] [rbp-F0h]
  __int64 v113; // [rsp+30h] [rbp-F0h]
  __int64 v114; // [rsp+30h] [rbp-F0h]
  __int64 v115; // [rsp+30h] [rbp-F0h]
  __int64 v116; // [rsp+30h] [rbp-F0h]
  __int64 v117; // [rsp+30h] [rbp-F0h]
  __int64 v118; // [rsp+30h] [rbp-F0h]
  __int64 v119; // [rsp+38h] [rbp-E8h]
  __int64 v120; // [rsp+38h] [rbp-E8h]
  __int64 v121; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v122; // [rsp+38h] [rbp-E8h]
  __int64 v123; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v124; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v125; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v126; // [rsp+38h] [rbp-E8h]
  __int64 v129; // [rsp+50h] [rbp-D0h]
  int v130; // [rsp+50h] [rbp-D0h]
  __int64 v131; // [rsp+50h] [rbp-D0h]
  __int64 v132; // [rsp+50h] [rbp-D0h]
  unsigned __int64 v133; // [rsp+50h] [rbp-D0h]
  unsigned __int64 v134; // [rsp+50h] [rbp-D0h]
  __int64 v135; // [rsp+50h] [rbp-D0h]
  __int64 v136; // [rsp+50h] [rbp-D0h]
  __int64 v137; // [rsp+50h] [rbp-D0h]
  __int64 v138; // [rsp+50h] [rbp-D0h]
  __int64 v139; // [rsp+58h] [rbp-C8h]
  __int64 v140; // [rsp+58h] [rbp-C8h]
  __int64 v141; // [rsp+58h] [rbp-C8h]
  __int64 v142; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v143; // [rsp+58h] [rbp-C8h]
  __int64 v144; // [rsp+58h] [rbp-C8h]
  unsigned __int64 v145; // [rsp+58h] [rbp-C8h]
  __int64 v146; // [rsp+58h] [rbp-C8h]
  __int64 v147; // [rsp+58h] [rbp-C8h]
  __int64 v148; // [rsp+58h] [rbp-C8h]
  unsigned int v149; // [rsp+58h] [rbp-C8h]
  __int64 v150; // [rsp+58h] [rbp-C8h]
  int v151; // [rsp+60h] [rbp-C0h]
  unsigned int v152; // [rsp+64h] [rbp-BCh]
  __int64 *v154; // [rsp+70h] [rbp-B0h]
  char v156; // [rsp+80h] [rbp-A0h]
  char v158; // [rsp+96h] [rbp-8Ah] BYREF
  char v159; // [rsp+97h] [rbp-89h] BYREF
  __int64 v160; // [rsp+98h] [rbp-88h] BYREF
  __int64 v161; // [rsp+A0h] [rbp-80h] BYREF
  unsigned __int32 v162; // [rsp+A8h] [rbp-78h]
  __int64 *v163; // [rsp+B0h] [rbp-70h] BYREF
  unsigned int v164; // [rsp+B8h] [rbp-68h]
  __m128i v165; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v166; // [rsp+D0h] [rbp-50h]
  __int64 v167; // [rsp+D8h] [rbp-48h]
  __int64 v168; // [rsp+E0h] [rbp-40h]

  *(_QWORD *)(a2 + 8) = 0;
  *(_QWORD *)(a2 + 16) = 0;
  *(_DWORD *)(a2 + 32) = 0;
  v151 = 6;
  do
  {
    v5 = *(_BYTE *)(a1 + 16);
    if ( v5 > 0x17u )
    {
      if ( (unsigned int)v5 - 71 <= 1 )
        goto LABEL_11;
      switch ( v5 )
      {
        case 0x38u:
          goto LABEL_22;
        case 0x4Eu:
          v75 = a1 | 4;
          break;
        case 0x1Du:
          v75 = a1 & 0xFFFFFFFFFFFFFFFBLL;
          break;
        default:
          goto LABEL_20;
      }
      if ( (v75 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
      {
        v10 = sub_14AD130();
        if ( v10 )
          goto LABEL_21;
        if ( *(_BYTE *)(a1 + 16) <= 0x17u )
        {
          v7 = a1;
          goto LABEL_9;
        }
      }
LABEL_20:
      v165 = (__m128i)(unsigned __int64)a3;
      v166 = 0;
      v167 = 0;
      v168 = 0;
      v10 = sub_13E3350(a1, &v165, 0, 1);
      if ( !v10 )
      {
LABEL_8:
        v7 = a1;
LABEL_9:
        *(_QWORD *)a2 = v7;
        return 0;
      }
LABEL_21:
      a1 = v10;
      goto LABEL_14;
    }
    if ( v5 != 5 )
    {
      if ( v5 == 1 )
        __asm { jmp     rax }
      goto LABEL_8;
    }
    v6 = *(unsigned __int16 *)(a1 + 18);
    if ( (unsigned int)(v6 - 47) <= 1 )
      goto LABEL_11;
    if ( (_WORD)v6 != 32 )
      goto LABEL_8;
LABEL_22:
    v11 = sub_16348C0(a1);
    v12 = *(unsigned __int8 *)(v11 + 8);
    if ( (unsigned __int8)v12 <= 0xFu && (v13 = 35454, _bittest64(&v13, v12)) )
    {
      v14 = a1;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        goto LABEL_25;
    }
    else
    {
      if ( (unsigned int)(v12 - 13) > 1 && (_DWORD)v12 != 16 || !(unsigned __int8)sub_16435F0(v11, 0) )
        goto LABEL_8;
      v14 = a1;
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      {
LABEL_25:
        v15 = *(_QWORD ***)(v14 - 8);
        v16 = **v15;
        if ( *(_BYTE *)(v16 + 8) == 16 )
          v16 = **(_QWORD **)(v16 + 16);
        v17 = *(_DWORD *)(v16 + 8) >> 8;
        goto LABEL_28;
      }
    }
    v31 = **(_QWORD **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    if ( *(_BYTE *)(v31 + 8) == 16 )
      v31 = **(_QWORD **)(v31 + 16);
    v15 = (_QWORD **)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    v17 = *(_DWORD *)(v31 + 8) >> 8;
LABEL_28:
    v18 = v15 + 3;
    v19 = sub_16348C0(a1) | 4;
    v152 = 8 * sub_15A9520(a3, v17);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
    {
      v20 = *(_QWORD *)(a1 - 8);
      v154 = (__int64 *)(v20 + 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    }
    else
    {
      v154 = (__int64 *)a1;
      v20 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    }
    v21 = (__int64 *)(v20 + 24);
    v22 = 1;
    v23 = v18;
    if ( (__int64 *)(v20 + 24) == v154 )
    {
LABEL_46:
      *(_QWORD *)(a2 + 8) = (__int64)(*(_QWORD *)(a2 + 8) << (64 - (unsigned __int8)v152)) >> (64 - (unsigned __int8)v152);
      *(_QWORD *)(a2 + 16) = (__int64)(*(_QWORD *)(a2 + 16) << (64 - (unsigned __int8)v152)) >> (64
                                                                                               - (unsigned __int8)v152);
      goto LABEL_11;
    }
    do
    {
      while ( 1 )
      {
        v24 = v19;
        v25 = v19 & 0xFFFFFFFFFFFFFFF8LL;
        v26 = *v21;
        v27 = v25;
        v28 = (v24 >> 2) & 1;
        v156 = v28;
        if ( (_DWORD)v28 )
        {
          if ( *(_BYTE *)(v26 + 16) == 13 )
            goto LABEL_59;
          v32 = v25;
          if ( v25 )
            goto LABEL_57;
        }
        else
        {
          if ( v25 )
          {
            v29 = *(_QWORD **)(v26 + 24);
            if ( *(_DWORD *)(v26 + 32) > 0x40u )
              v29 = (_QWORD *)*v29;
            if ( (_DWORD)v29 )
              *(_QWORD *)(a2 + 8) += *(_QWORD *)(sub_15A9930(a3, v25) + 8LL * (unsigned int)v29 + 16);
            goto LABEL_39;
          }
          if ( *(_BYTE *)(v26 + 16) == 13 )
          {
LABEL_59:
            if ( *(_DWORD *)(v26 + 32) <= 0x40u )
            {
              if ( !*(_QWORD *)(v26 + 24) )
                goto LABEL_39;
              if ( (_BYTE)v28 )
                goto LABEL_62;
            }
            else
            {
              v130 = *(_DWORD *)(v26 + 32);
              v139 = *v21;
              v37 = sub_16A57B0(v26 + 24);
              v26 = v139;
              if ( v130 == v37 )
                goto LABEL_39;
              if ( v156 )
              {
LABEL_62:
                v38 = v25;
                if ( v25 )
                {
LABEL_63:
                  v131 = v26;
                  v39 = sub_15A9FE0(a3, v38);
                  v40 = v131;
                  v41 = 1;
                  v42 = v39;
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v38 + 8) )
                    {
                      case 1:
                        v70 = 16;
                        goto LABEL_110;
                      case 2:
                        v70 = 32;
                        goto LABEL_110;
                      case 3:
                      case 9:
                        v70 = 64;
                        goto LABEL_110;
                      case 4:
                        v70 = 80;
                        goto LABEL_110;
                      case 5:
                      case 6:
                        v70 = 128;
                        goto LABEL_110;
                      case 7:
                        v121 = v131;
                        v78 = 0;
                        v134 = v42;
                        v147 = v41;
                        goto LABEL_125;
                      case 0xB:
                        v70 = *(_DWORD *)(v38 + 8) >> 8;
                        goto LABEL_110;
                      case 0xD:
                        v120 = v131;
                        v133 = v42;
                        v146 = v41;
                        v77 = (_QWORD *)sub_15A9930(a3, v38);
                        v41 = v146;
                        v42 = v133;
                        v40 = v120;
                        v70 = 8LL * *v77;
                        goto LABEL_110;
                      case 0xE:
                        v99 = v131;
                        v102 = v42;
                        v106 = v41;
                        v113 = *(_QWORD *)(v38 + 24);
                        v135 = *(_QWORD *)(v38 + 32);
                        v80 = sub_15A9FE0(a3, v113);
                        v40 = v99;
                        v148 = 1;
                        v42 = v102;
                        v81 = v113;
                        v122 = v80;
                        v41 = v106;
                        while ( 2 )
                        {
                          switch ( *(_BYTE *)(v81 + 8) )
                          {
                            case 1:
                              v90 = 16;
                              goto LABEL_155;
                            case 2:
                              v90 = 32;
                              goto LABEL_155;
                            case 3:
                            case 9:
                              v90 = 64;
                              goto LABEL_155;
                            case 4:
                              v90 = 80;
                              goto LABEL_155;
                            case 5:
                            case 6:
                              v90 = 128;
                              goto LABEL_155;
                            case 7:
                              v104 = v99;
                              v92 = 0;
                              v110 = v42;
                              v117 = v41;
                              goto LABEL_160;
                            case 0xB:
                              v90 = *(_DWORD *)(v81 + 8) >> 8;
                              goto LABEL_155;
                            case 0xD:
                              v95 = (_QWORD *)sub_15A9930(a3, v81);
                              v41 = v106;
                              v42 = v102;
                              v40 = v99;
                              v90 = 8LL * *v95;
                              goto LABEL_155;
                            case 0xE:
                              v96 = v99;
                              v98 = v102;
                              v101 = v106;
                              v105 = *(_QWORD *)(v81 + 24);
                              v118 = *(_QWORD *)(v81 + 32);
                              v111 = (unsigned int)sub_15A9FE0(a3, v105);
                              v94 = sub_127FA20(a3, v105);
                              v41 = v101;
                              v42 = v98;
                              v40 = v96;
                              v90 = 8 * v118 * v111 * ((v111 + ((unsigned __int64)(v94 + 7) >> 3) - 1) / v111);
                              goto LABEL_155;
                            case 0xF:
                              v104 = v99;
                              v110 = v42;
                              v117 = v41;
                              v92 = *(_DWORD *)(v81 + 8) >> 8;
LABEL_160:
                              v93 = sub_15A9520(a3, v92);
                              v41 = v117;
                              v42 = v110;
                              v40 = v104;
                              v90 = (unsigned int)(8 * v93);
LABEL_155:
                              v70 = 8 * v122 * v135 * ((v122 + ((unsigned __int64)(v148 * v90 + 7) >> 3) - 1) / v122);
                              goto LABEL_110;
                            case 0x10:
                              v91 = v148 * *(_QWORD *)(v81 + 32);
                              v81 = *(_QWORD *)(v81 + 24);
                              v148 = v91;
                              continue;
                            default:
                              goto LABEL_168;
                          }
                        }
                      case 0xF:
                        v121 = v131;
                        v134 = v42;
                        v147 = v41;
                        v78 = *(_DWORD *)(v38 + 8) >> 8;
LABEL_125:
                        v79 = sub_15A9520(a3, v78);
                        v41 = v147;
                        v42 = v134;
                        v40 = v121;
                        v70 = (unsigned int)(8 * v79);
LABEL_110:
                        v71 = *(_DWORD *)(v40 + 32);
                        v72 = *(__int64 **)(v40 + 24);
                        v73 = v42 * ((v42 + ((unsigned __int64)(v41 * v70 + 7) >> 3) - 1) / v42);
                        if ( v71 > 0x40 )
                          v74 = *v72;
                        else
                          v74 = (__int64)((_QWORD)v72 << (64 - (unsigned __int8)v71)) >> (64 - (unsigned __int8)v71);
                        *(_QWORD *)(a2 + 16) += v73 * v74;
                        goto LABEL_39;
                      case 0x10:
                        v76 = *(_QWORD *)(v38 + 32);
                        v38 = *(_QWORD *)(v38 + 24);
                        v41 *= v76;
                        continue;
                      default:
LABEL_168:
                        BUG();
                    }
                  }
                }
              }
            }
            v141 = v26;
            v44 = sub_1643D30(v25, *v23);
            v26 = v141;
            v38 = v44;
            goto LABEL_63;
          }
        }
        v140 = *v21;
        v43 = sub_1643D30(0, *v23);
        v26 = v140;
        v32 = v43;
LABEL_57:
        v129 = v26;
        v33 = 1;
        v34 = sub_15A9FE0(a3, v32);
        v35 = v129;
        v36 = v34;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v32 + 8) )
          {
            case 1:
              v45 = 16;
              goto LABEL_72;
            case 2:
              v45 = 32;
              goto LABEL_72;
            case 3:
            case 9:
              v45 = 64;
              goto LABEL_72;
            case 4:
              v45 = 80;
              goto LABEL_72;
            case 5:
            case 6:
              v45 = 128;
              goto LABEL_72;
            case 7:
              v62 = 0;
              v143 = v36;
              goto LABEL_101;
            case 0xB:
              v45 = *(_DWORD *)(v32 + 8) >> 8;
              goto LABEL_72;
            case 0xD:
              v145 = v36;
              v69 = (_QWORD *)sub_15A9930(a3, v32);
              v36 = v145;
              v35 = v129;
              v45 = 8LL * *v69;
              goto LABEL_72;
            case 0xE:
              v112 = v129;
              v119 = v36;
              v144 = *(_QWORD *)(v32 + 32);
              v132 = *(_QWORD *)(v32 + 24);
              v65 = sub_15A9FE0(a3, v132);
              v35 = v112;
              v36 = v119;
              v66 = 1;
              v67 = v132;
              v68 = v65;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v67 + 8) )
                {
                  case 1:
                    v84 = 16;
                    goto LABEL_142;
                  case 2:
                    v84 = 32;
                    goto LABEL_142;
                  case 3:
                  case 9:
                    v84 = 64;
                    goto LABEL_142;
                  case 4:
                    v84 = 80;
                    goto LABEL_142;
                  case 5:
                  case 6:
                    v84 = 128;
                    goto LABEL_142;
                  case 7:
                    v107 = v112;
                    v86 = 0;
                    v114 = v119;
                    v124 = v68;
                    v136 = v66;
                    goto LABEL_148;
                  case 0xB:
                    v84 = *(_DWORD *)(v67 + 8) >> 8;
                    goto LABEL_142;
                  case 0xD:
                    v109 = v112;
                    v116 = v119;
                    v126 = v68;
                    v138 = v66;
                    v89 = (_QWORD *)sub_15A9930(a3, v67);
                    v66 = v138;
                    v68 = v126;
                    v36 = v116;
                    v35 = v109;
                    v84 = 8LL * *v89;
                    goto LABEL_142;
                  case 0xE:
                    v97 = v112;
                    v100 = v119;
                    v103 = v68;
                    v108 = v66;
                    v115 = *(_QWORD *)(v67 + 24);
                    v137 = *(_QWORD *)(v67 + 32);
                    v125 = (unsigned int)sub_15A9FE0(a3, v115);
                    v88 = sub_127FA20(a3, v115);
                    v66 = v108;
                    v68 = v103;
                    v36 = v100;
                    v35 = v97;
                    v84 = 8 * v137 * v125 * ((v125 + ((unsigned __int64)(v88 + 7) >> 3) - 1) / v125);
                    goto LABEL_142;
                  case 0xF:
                    v107 = v112;
                    v114 = v119;
                    v124 = v68;
                    v86 = *(_DWORD *)(v67 + 8) >> 8;
                    v136 = v66;
LABEL_148:
                    v87 = sub_15A9520(a3, v86);
                    v66 = v136;
                    v68 = v124;
                    v36 = v114;
                    v35 = v107;
                    v84 = (unsigned int)(8 * v87);
LABEL_142:
                    v45 = 8 * v144 * v68 * ((v68 + ((unsigned __int64)(v84 * v66 + 7) >> 3) - 1) / v68);
                    goto LABEL_72;
                  case 0x10:
                    v85 = *(_QWORD *)(v67 + 32);
                    v67 = *(_QWORD *)(v67 + 24);
                    v66 *= v85;
                    continue;
                  default:
                    goto LABEL_168;
                }
              }
            case 0xF:
              v143 = v36;
              v62 = *(_DWORD *)(v32 + 8) >> 8;
LABEL_101:
              v63 = sub_15A9520(a3, v62);
              v36 = v143;
              v35 = v129;
              v45 = (unsigned int)(8 * v63);
LABEL_72:
              v160 = 0;
              v46 = v36 * ((v36 + ((unsigned __int64)(v45 * v33 + 7) >> 3) - 1) / v36);
              v47 = *(_DWORD *)(*(_QWORD *)v35 + 8LL) >> 8;
              v48 = v152 - v47;
              v162 = v47;
              if ( v152 <= v47 )
                v48 = 0;
              HIDWORD(v160) = v48;
              if ( v47 > 0x40 )
              {
                v123 = v35;
                v149 = v47;
                sub_16A4EF0(&v161, 0, 0);
                v164 = v149;
                sub_16A4EF0(&v163, 0, 0);
                v35 = v123;
              }
              else
              {
                v161 = 0;
                v163 = 0;
                v164 = v47;
              }
              v158 = 1;
              v159 = 1;
              v49 = sub_135E160(
                      v35,
                      &v161,
                      (unsigned int *)&v163,
                      (int *)&v160,
                      (int *)&v160 + 1,
                      a3,
                      0,
                      a4,
                      a5,
                      &v158,
                      &v159);
              v51 = v162;
              v52 = v49;
              if ( v162 > 0x40 )
              {
                v150 = v49;
                sub_16A5D70(&v165, &v161, 64, v50, v49);
                v52 = v150;
                if ( v162 > 0x40 && v161 )
                {
                  j_j___libc_free_0_0(v161);
                  v52 = v150;
                }
                v51 = v165.m128i_u32[2];
                v161 = v165.m128i_i64[0];
                v162 = v165.m128i_u32[2];
              }
              if ( v164 > 0x40 )
                v53 = *v163;
              else
                v53 = (__int64)((_QWORD)v163 << (64 - (unsigned __int8)v164)) >> (64 - (unsigned __int8)v164);
              *(_QWORD *)(a2 + 16) += v46 * v53;
              if ( v51 > 0x40 )
                v54 = *(_QWORD *)v161;
              else
                v54 = v161 << (64 - (unsigned __int8)v51) >> (64 - (unsigned __int8)v51);
              v55 = v54 * v46;
              v56 = *(unsigned int *)(a2 + 32);
              if ( !(_DWORD)v56 )
                goto LABEL_89;
              v57 = *(_QWORD **)(a2 + 24);
              v58 = (unsigned int)(v56 - 1);
              v59 = v57;
              break;
            case 0x10:
              v64 = *(_QWORD *)(v32 + 32);
              v32 = *(_QWORD *)(v32 + 24);
              v33 *= v64;
              continue;
            default:
              goto LABEL_168;
          }
          break;
        }
        while ( *v59 != v52 || v59[1] != v160 )
        {
          v59 += 3;
          if ( &v57[3 * v58 + 3] == v59 )
            goto LABEL_89;
        }
        v55 += v59[2];
        v60 = (char *)&v57[3 * v56];
        if ( v60 != (char *)(v59 + 3) )
        {
          v142 = v52;
          memmove(v59, v59 + 3, v60 - (char *)(v59 + 3));
          v52 = v142;
          LODWORD(v58) = *(_DWORD *)(a2 + 32) - 1;
        }
        *(_DWORD *)(a2 + 32) = v58;
LABEL_89:
        v61 = (__int64)(v55 << (64 - (unsigned __int8)v152)) >> (64 - (unsigned __int8)v152);
        if ( v61 )
        {
          v165.m128i_i64[0] = v52;
          v166 = v61;
          v165.m128i_i64[1] = v160;
          v82 = *(unsigned int *)(a2 + 32);
          if ( (unsigned int)v82 >= *(_DWORD *)(a2 + 36) )
          {
            sub_16CD150(a2 + 24, a2 + 40, 0, 24);
            v82 = *(unsigned int *)(a2 + 32);
          }
          v83 = (__m128i *)(*(_QWORD *)(a2 + 24) + 24 * v82);
          *v83 = _mm_loadu_si128(&v165);
          v83[1].m128i_i64[0] = v166;
          ++*(_DWORD *)(a2 + 32);
        }
        if ( v164 > 0x40 && v163 )
          j_j___libc_free_0_0(v163);
        if ( v162 > 0x40 && v161 )
          j_j___libc_free_0_0(v161);
        v22 = 0;
LABEL_39:
        v21 += 3;
        if ( !v156 || !v25 )
          v27 = sub_1643D30(v25, *v23);
        v30 = *(_BYTE *)(v27 + 8);
        if ( ((v30 - 14) & 0xFD) != 0 )
          break;
        v23 += 3;
        v19 = *(_QWORD *)(v27 + 24) | 4LL;
        if ( v21 == v154 )
          goto LABEL_45;
      }
      v19 = 0;
      if ( v30 == 13 )
        v19 = v27;
      v23 += 3;
    }
    while ( v21 != v154 );
LABEL_45:
    if ( v22 )
      goto LABEL_46;
LABEL_11:
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v9 = *(__int64 **)(a1 - 8);
    else
      v9 = (__int64 *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
    a1 = *v9;
LABEL_14:
    --v151;
  }
  while ( v151 );
  *(_QWORD *)a2 = a1;
  return 1;
}
