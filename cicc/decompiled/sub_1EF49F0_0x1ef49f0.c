// Function: sub_1EF49F0
// Address: 0x1ef49f0
//
__int64 __fastcall sub_1EF49F0(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  _QWORD *v7; // rdi
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 *v11; // r13
  __int64 v12; // r12
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // r14
  int v16; // edx
  __int64 v17; // rdi
  __int64 v18; // rsi
  __int64 v19; // r14
  unsigned int v20; // r14d
  __int64 *v21; // rax
  char v22; // dl
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  _QWORD *v27; // rdx
  __int64 **v29; // r14
  __int64 *v30; // rsi
  __int64 *v31; // rcx
  __int64 v32; // rdx
  unsigned int v33; // eax
  __int64 v34; // rax
  unsigned int v35; // eax
  __int64 v36; // rdi
  __int64 v37; // rsi
  __int64 v38; // r8
  unsigned __int64 v39; // r9
  __int64 v40; // rax
  unsigned int v41; // eax
  __int64 v42; // rdi
  __int64 v43; // rsi
  __int64 v44; // r9
  unsigned __int64 v45; // r8
  unsigned __int64 v46; // r14
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // rax
  __int64 **v52; // rbx
  __int64 **v53; // r12
  _QWORD *v54; // rdi
  unsigned __int64 v55; // r9
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rax
  int v68; // eax
  int v69; // eax
  __int64 v70; // rax
  _QWORD *v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  _QWORD *v74; // rax
  int v75; // eax
  __int64 v76; // rax
  int v77; // eax
  unsigned __int64 v78; // r9
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rax
  unsigned __int64 v84; // [rsp+0h] [rbp-190h]
  unsigned __int64 v85; // [rsp+0h] [rbp-190h]
  __int64 v86; // [rsp+8h] [rbp-188h]
  __int64 v87; // [rsp+8h] [rbp-188h]
  __int64 v88; // [rsp+10h] [rbp-180h]
  __int64 v89; // [rsp+10h] [rbp-180h]
  __int64 v90; // [rsp+10h] [rbp-180h]
  __int64 v91; // [rsp+10h] [rbp-180h]
  unsigned __int64 v92; // [rsp+18h] [rbp-178h]
  __int64 v93; // [rsp+18h] [rbp-178h]
  __int64 v94; // [rsp+18h] [rbp-178h]
  __int64 v95; // [rsp+18h] [rbp-178h]
  __int64 v96; // [rsp+18h] [rbp-178h]
  __int64 v97; // [rsp+20h] [rbp-170h]
  __int64 v98; // [rsp+20h] [rbp-170h]
  __int64 v99; // [rsp+20h] [rbp-170h]
  unsigned __int64 v100; // [rsp+20h] [rbp-170h]
  unsigned __int64 v101; // [rsp+20h] [rbp-170h]
  unsigned __int64 v102; // [rsp+20h] [rbp-170h]
  unsigned __int64 v103; // [rsp+20h] [rbp-170h]
  unsigned __int64 v104; // [rsp+20h] [rbp-170h]
  unsigned __int64 v105; // [rsp+20h] [rbp-170h]
  unsigned __int64 v106; // [rsp+20h] [rbp-170h]
  unsigned __int64 v107; // [rsp+20h] [rbp-170h]
  __int64 v108; // [rsp+28h] [rbp-168h]
  __int64 v109; // [rsp+28h] [rbp-168h]
  __int64 v110; // [rsp+28h] [rbp-168h]
  int v111; // [rsp+28h] [rbp-168h]
  __int64 v112; // [rsp+28h] [rbp-168h]
  __int64 v113; // [rsp+28h] [rbp-168h]
  int v114; // [rsp+28h] [rbp-168h]
  __int64 v115; // [rsp+28h] [rbp-168h]
  __int64 v116; // [rsp+28h] [rbp-168h]
  __int64 v117; // [rsp+28h] [rbp-168h]
  __int64 v118; // [rsp+28h] [rbp-168h]
  __int64 v119; // [rsp+28h] [rbp-168h]
  __int64 v120; // [rsp+28h] [rbp-168h]
  __int64 v121; // [rsp+28h] [rbp-168h]
  __int64 v122; // [rsp+28h] [rbp-168h]
  __int64 v123; // [rsp+30h] [rbp-160h]
  __int64 v124; // [rsp+30h] [rbp-160h]
  unsigned __int64 v125; // [rsp+30h] [rbp-160h]
  __int64 *v126; // [rsp+38h] [rbp-158h]
  __int64 v128; // [rsp+50h] [rbp-140h] BYREF
  __int64 v129; // [rsp+58h] [rbp-138h] BYREF
  _QWORD *v130; // [rsp+60h] [rbp-130h] BYREF
  __int64 v131; // [rsp+68h] [rbp-128h]
  _QWORD v132[8]; // [rsp+70h] [rbp-120h] BYREF
  __int64 v133; // [rsp+B0h] [rbp-E0h] BYREF
  __int64 *v134; // [rsp+B8h] [rbp-D8h]
  __int64 *v135; // [rsp+C0h] [rbp-D0h]
  __int64 v136; // [rsp+C8h] [rbp-C8h]
  int v137; // [rsp+D0h] [rbp-C0h]
  _BYTE v138[184]; // [rsp+D8h] [rbp-B8h] BYREF

  v7 = v132;
  v133 = 0;
  v136 = 16;
  v137 = 0;
  v130 = v132;
  v132[0] = a2;
  v134 = (__int64 *)v138;
  v135 = (__int64 *)v138;
  v131 = 0x800000001LL;
  v8 = 1;
  while ( 1 )
  {
    v9 = v8--;
    v10 = v7[v9 - 1];
    LODWORD(v131) = v8;
    v11 = *(__int64 **)(v10 + 8);
    v126 = (__int64 *)v10;
    if ( v11 )
      break;
LABEL_14:
    if ( !v8 )
    {
      v20 = 1;
      goto LABEL_36;
    }
  }
  v12 = a3;
  while ( 1 )
  {
    v15 = (__int64)sub_1648700((__int64)v11);
    v16 = *(unsigned __int8 *)(v15 + 16);
    if ( v16 == 55 )
      break;
    if ( (unsigned int)(v16 - 24) > 0x1F )
    {
      if ( v16 == 78 )
      {
        v24 = v15 | 4;
LABEL_23:
        v128 = v24;
        if ( (_BYTE)v16 != 78 )
          goto LABEL_72;
        v25 = *(_QWORD *)(v15 - 24);
        if ( *(_BYTE *)(v25 + 16) || (*(_BYTE *)(v25 + 33) & 0x20) == 0 )
          goto LABEL_72;
        if ( (unsigned int)(*(_DWORD *)(v25 + 36) - 116) <= 1 )
          goto LABEL_12;
        if ( (*(_BYTE *)(v25 + 33) & 0x20) == 0
          || (unsigned int)(*(_DWORD *)(v25 + 36) - 133) > 4
          || ((1LL << (*(_BYTE *)(v25 + 36) + 123)) & 0x15) == 0 )
        {
LABEL_72:
          v46 = v128 & 0xFFFFFFFFFFFFFFF8LL;
          v125 = (v128 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v128 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
          if ( (v128 & 4) != 0 )
          {
            if ( *(char *)((v128 & 0xFFFFFFFFFFFFFFF8LL) + 23) < 0 )
            {
              v47 = sub_1648A40(v128 & 0xFFFFFFFFFFFFFFF8LL);
              v110 = v48 + v47;
              if ( *(char *)(v46 + 23) >= 0 )
              {
                if ( (unsigned int)(v110 >> 4) )
                  goto LABEL_144;
              }
              else if ( (unsigned int)((v110 - sub_1648A40(v46)) >> 4) )
              {
                if ( *(char *)(v46 + 23) < 0 )
                {
                  v111 = *(_DWORD *)(sub_1648A40(v46) + 8);
                  if ( *(char *)(v46 + 23) >= 0 )
                    BUG();
                  v49 = sub_1648A40(v46);
                  v51 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v49 + v50 - 4) - v111);
                  goto LABEL_81;
                }
LABEL_144:
                BUG();
              }
            }
            v51 = -24;
            goto LABEL_81;
          }
          if ( *(char *)(v46 + 23) < 0 )
          {
            v62 = sub_1648A40(v128 & 0xFFFFFFFFFFFFFFF8LL);
            v113 = v63 + v62;
            if ( *(char *)(v46 + 23) >= 0 )
            {
              if ( (unsigned int)(v113 >> 4) )
                goto LABEL_142;
            }
            else if ( (unsigned int)((v113 - sub_1648A40(v46)) >> 4) )
            {
              if ( *(char *)(v46 + 23) < 0 )
              {
                v114 = *(_DWORD *)(sub_1648A40(v46) + 8);
                if ( *(char *)(v46 + 23) >= 0 )
LABEL_143:
                  BUG();
                v64 = sub_1648A40(v46);
                v51 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v64 + v65 - 4) - v114);
LABEL_81:
                if ( v46 + v51 == v125 )
                  goto LABEL_12;
                v112 = a1;
                v52 = (__int64 **)v125;
                v99 = v12;
                v53 = (__int64 **)(v46 + v51);
                while ( 2 )
                {
                  if ( v126 == *v52 )
                  {
                    if ( !(unsigned __int8)sub_134FA60(
                                             &v128,
                                             -1431655765 * (unsigned int)((__int64)((__int64)v52 - v125) >> 3) + 1,
                                             22) )
                      goto LABEL_35;
                    if ( !(unsigned __int8)sub_134FA60(
                                             &v128,
                                             -1431655765 * (unsigned int)((__int64)((__int64)v52 - v125) >> 3) + 1,
                                             36) )
                    {
                      v92 = v128 & 0xFFFFFFFFFFFFFFF8LL;
                      v54 = (_QWORD *)((v128 & 0xFFFFFFFFFFFFFFF8LL) + 56);
                      if ( (v128 & 4) != 0 )
                      {
                        if ( !(unsigned __int8)sub_1560260(v54, -1, 36) )
                        {
                          v55 = v92;
                          if ( *(char *)(v92 + 23) < 0 )
                          {
                            v88 = v92;
                            v56 = sub_1648A40(v92);
                            v55 = v92;
                            v58 = v57 + v56;
                            v59 = 0;
                            v93 = v58;
                            if ( *(char *)(v55 + 23) < 0 )
                            {
                              v60 = sub_1648A40(v88);
                              v55 = v88;
                              v59 = v60;
                            }
                            if ( (unsigned int)((v93 - v59) >> 4) )
                              goto LABEL_35;
                          }
                          v61 = *(_QWORD *)(v55 - 24);
                          if ( *(_BYTE *)(v61 + 16) )
                            goto LABEL_35;
                          goto LABEL_94;
                        }
                      }
                      else if ( !(unsigned __int8)sub_1560260(v54, -1, 36) )
                      {
                        v78 = v92;
                        if ( *(char *)(v92 + 23) < 0 )
                        {
                          v91 = v92;
                          v79 = sub_1648A40(v92);
                          v78 = v92;
                          v81 = v80 + v79;
                          v82 = 0;
                          v96 = v81;
                          if ( *(char *)(v78 + 23) < 0 )
                          {
                            v83 = sub_1648A40(v91);
                            v78 = v91;
                            v82 = v83;
                          }
                          if ( (unsigned int)((v96 - v82) >> 4) )
                            goto LABEL_35;
                        }
                        v61 = *(_QWORD *)(v78 - 72);
                        if ( *(_BYTE *)(v61 + 16) )
                          goto LABEL_35;
LABEL_94:
                        v129 = *(_QWORD *)(v61 + 112);
                        if ( !(unsigned __int8)sub_1560260(&v129, -1, 36) )
                          goto LABEL_35;
                      }
                    }
                  }
                  v52 += 3;
                  if ( v52 == v53 )
                  {
                    a1 = v112;
                    v12 = v99;
                    goto LABEL_12;
                  }
                  continue;
                }
              }
LABEL_142:
              BUG();
            }
          }
          v51 = -72;
          goto LABEL_81;
        }
        if ( *v11 == *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF)) )
        {
          v26 = *(_QWORD *)(v15 + 24 * (2LL - (*(_DWORD *)(v15 + 20) & 0xFFFFFFF)));
          if ( *(_BYTE *)(v26 + 16) != 13 )
            goto LABEL_35;
          v27 = *(_QWORD **)(v26 + 24);
          if ( *(_DWORD *)(v26 + 32) > 0x40u )
            v27 = (_QWORD *)*v27;
          if ( !(unsigned __int8)sub_1EF4660(a1, *v11, (__int64)v27, a2, v12, a4, a5) )
            goto LABEL_35;
        }
        goto LABEL_12;
      }
      if ( v16 == 82 )
        goto LABEL_12;
    }
    else
    {
      switch ( v16 )
      {
        case 29:
          v24 = v15 & 0xFFFFFFFFFFFFFFFBLL;
          goto LABEL_23;
        case 54:
          v17 = *(_QWORD *)(a1 + 16);
          v18 = *(_QWORD *)v15;
          v19 = 1;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v18 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v34 = *(_QWORD *)(v18 + 32);
                v18 = *(_QWORD *)(v18 + 24);
                v19 *= v34;
                continue;
              case 1:
                goto LABEL_62;
              case 2:
                goto LABEL_65;
              case 3:
              case 9:
                goto LABEL_55;
              case 4:
                goto LABEL_61;
              case 5:
              case 6:
                goto LABEL_58;
              case 7:
                goto LABEL_59;
              case 0xB:
                goto LABEL_60;
              case 0xD:
                goto LABEL_63;
              case 0xE:
                v108 = *(_QWORD *)(a1 + 16);
                v97 = *(_QWORD *)(v18 + 24);
                v123 = *(_QWORD *)(v18 + 32);
                v35 = sub_15A9FE0(v17, v97);
                v36 = v108;
                v37 = v97;
                v38 = 1;
                v39 = v35;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v37 + 8) )
                  {
                    case 0:
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v72 = *(_QWORD *)(v37 + 32);
                      v37 = *(_QWORD *)(v37 + 24);
                      v38 *= v72;
                      continue;
                    case 1:
                      v66 = 16;
                      goto LABEL_109;
                    case 2:
                      v66 = 32;
                      goto LABEL_109;
                    case 3:
                    case 9:
                      v66 = 64;
                      goto LABEL_109;
                    case 4:
                      v66 = 80;
                      goto LABEL_109;
                    case 5:
                    case 6:
                      v66 = 128;
                      goto LABEL_109;
                    case 7:
                      v100 = v39;
                      v115 = v38;
                      v68 = sub_15A9520(v36, 0);
                      v38 = v115;
                      v39 = v100;
                      v66 = (unsigned int)(8 * v68);
                      goto LABEL_109;
                    case 0xB:
                      v66 = *(_DWORD *)(v37 + 8) >> 8;
                      goto LABEL_109;
                    case 0xD:
                      v103 = v39;
                      v118 = v38;
                      v71 = (_QWORD *)sub_15A9930(v36, v37);
                      v38 = v118;
                      v39 = v103;
                      v66 = 8LL * *v71;
                      goto LABEL_109;
                    case 0xE:
                      v84 = v39;
                      v86 = v38;
                      v89 = *(_QWORD *)(v37 + 24);
                      v94 = v108;
                      v117 = *(_QWORD *)(v37 + 32);
                      v102 = (unsigned int)sub_15A9FE0(v36, v89);
                      v70 = sub_127FA20(v94, v89);
                      v38 = v86;
                      v39 = v84;
                      v66 = 8 * v117 * v102 * ((v102 + ((unsigned __int64)(v70 + 7) >> 3) - 1) / v102);
                      goto LABEL_109;
                    case 0xF:
                      v101 = v39;
                      v116 = v38;
                      v69 = sub_15A9520(v36, *(_DWORD *)(v37 + 8) >> 8);
                      v38 = v116;
                      v39 = v101;
                      v66 = (unsigned int)(8 * v69);
LABEL_109:
                      v32 = 8 * v39 * v123 * ((v39 + ((unsigned __int64)(v66 * v38 + 7) >> 3) - 1) / v39);
                      goto LABEL_56;
                    default:
                      goto LABEL_143;
                  }
                }
              case 0xF:
                goto LABEL_64;
              default:
                goto LABEL_143;
            }
          }
        case 25:
          goto LABEL_35;
      }
    }
    v21 = v134;
    if ( v135 != v134 )
      goto LABEL_18;
    v30 = &v134[HIDWORD(v136)];
    if ( v134 != v30 )
    {
      v31 = 0;
      while ( v15 != *v21 )
      {
        if ( *v21 == -2 )
          v31 = v21;
        if ( v30 == ++v21 )
        {
          if ( !v31 )
            goto LABEL_106;
          *v31 = v15;
          --v137;
          ++v133;
          goto LABEL_19;
        }
      }
      goto LABEL_12;
    }
LABEL_106:
    if ( HIDWORD(v136) < (unsigned int)v136 )
    {
      ++HIDWORD(v136);
      *v30 = v15;
      ++v133;
    }
    else
    {
LABEL_18:
      sub_16CCBA0((__int64)&v133, v15);
      if ( !v22 )
        goto LABEL_12;
    }
LABEL_19:
    v23 = (unsigned int)v131;
    if ( (unsigned int)v131 >= HIDWORD(v131) )
    {
      sub_16CD150((__int64)&v130, v132, 0, 8, v13, v14);
      v23 = (unsigned int)v131;
    }
    v130[v23] = v15;
    LODWORD(v131) = v131 + 1;
LABEL_12:
    v11 = (__int64 *)v11[1];
    if ( !v11 )
    {
      v8 = v131;
      v7 = v130;
      goto LABEL_14;
    }
  }
  if ( (*(_BYTE *)(v15 + 23) & 0x40) != 0 )
    v29 = *(__int64 ***)(v15 - 8);
  else
    v29 = (__int64 **)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
  if ( v126 == *v29 )
  {
LABEL_35:
    v7 = v130;
    v20 = 0;
  }
  else
  {
    v17 = *(_QWORD *)(a1 + 16);
    v18 = **v29;
    v19 = 1;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v18 + 8) )
      {
        case 1:
LABEL_62:
          v32 = 16;
          goto LABEL_56;
        case 2:
LABEL_65:
          v32 = 32;
          goto LABEL_56;
        case 3:
        case 9:
LABEL_55:
          v32 = 64;
          goto LABEL_56;
        case 4:
LABEL_61:
          v32 = 80;
          goto LABEL_56;
        case 5:
        case 6:
LABEL_58:
          v32 = 128;
          goto LABEL_56;
        case 7:
LABEL_59:
          v32 = 8 * (unsigned int)sub_15A9520(v17, 0);
          goto LABEL_56;
        case 0xB:
LABEL_60:
          v32 = *(_DWORD *)(v18 + 8) >> 8;
          goto LABEL_56;
        case 0xD:
LABEL_63:
          v32 = 8LL * *(_QWORD *)sub_15A9930(v17, v18);
          goto LABEL_56;
        case 0xE:
          v109 = *(_QWORD *)(a1 + 16);
          v98 = *(_QWORD *)(v18 + 24);
          v124 = *(_QWORD *)(v18 + 32);
          v41 = sub_15A9FE0(v17, v98);
          v42 = v109;
          v43 = v98;
          v44 = 1;
          v45 = v41;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v43 + 8) )
            {
              case 1:
                v67 = 16;
                goto LABEL_112;
              case 2:
                v67 = 32;
                goto LABEL_112;
              case 3:
              case 9:
                v67 = 64;
                goto LABEL_112;
              case 4:
                v67 = 80;
                goto LABEL_112;
              case 5:
              case 6:
                v67 = 128;
                goto LABEL_112;
              case 7:
                v106 = v45;
                v121 = v44;
                v75 = sub_15A9520(v42, 0);
                v44 = v121;
                v45 = v106;
                v67 = (unsigned int)(8 * v75);
                goto LABEL_112;
              case 0xB:
                v67 = *(_DWORD *)(v43 + 8) >> 8;
                goto LABEL_112;
              case 0xD:
                v105 = v45;
                v120 = v44;
                v74 = (_QWORD *)sub_15A9930(v42, v43);
                v44 = v120;
                v45 = v105;
                v67 = 8LL * *v74;
                goto LABEL_112;
              case 0xE:
                v85 = v45;
                v87 = v44;
                v90 = *(_QWORD *)(v43 + 24);
                v95 = v109;
                v119 = *(_QWORD *)(v43 + 32);
                v104 = (unsigned int)sub_15A9FE0(v42, v90);
                v73 = sub_127FA20(v95, v90);
                v44 = v87;
                v45 = v85;
                v67 = 8 * v119 * v104 * ((v104 + ((unsigned __int64)(v73 + 7) >> 3) - 1) / v104);
                goto LABEL_112;
              case 0xF:
                v107 = v45;
                v122 = v44;
                v77 = sub_15A9520(v42, *(_DWORD *)(v43 + 8) >> 8);
                v44 = v122;
                v45 = v107;
                v67 = (unsigned int)(8 * v77);
LABEL_112:
                v32 = 8 * v45 * v124 * ((v45 + ((unsigned __int64)(v67 * v44 + 7) >> 3) - 1) / v45);
                goto LABEL_56;
              case 0x10:
                v76 = *(_QWORD *)(v43 + 32);
                v43 = *(_QWORD *)(v43 + 24);
                v44 *= v76;
                continue;
              default:
                goto LABEL_143;
            }
          }
        case 0xF:
LABEL_64:
          v32 = 8 * (unsigned int)sub_15A9520(v17, *(_DWORD *)(v18 + 8) >> 8);
LABEL_56:
          v33 = sub_1EF4660(a1, *v11, (unsigned __int64)(v32 * v19 + 7) >> 3, a2, v12, a4, a5);
          if ( (_BYTE)v33 )
            goto LABEL_12;
          v7 = v130;
          v20 = v33;
          break;
        case 0x10:
          v40 = *(_QWORD *)(v18 + 32);
          v18 = *(_QWORD *)(v18 + 24);
          v19 *= v40;
          continue;
        default:
          goto LABEL_143;
      }
      break;
    }
  }
LABEL_36:
  if ( v7 != v132 )
    _libc_free((unsigned __int64)v7);
  if ( v135 != v134 )
    _libc_free((unsigned __int64)v135);
  return v20;
}
