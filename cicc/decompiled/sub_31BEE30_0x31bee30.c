// Function: sub_31BEE30
// Address: 0x31bee30
//
__int64 __fastcall sub_31BEE30(__int64 a1, __int64 *a2, unsigned __int64 a3, char a4)
{
  char *v4; // r14
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 *v8; // r13
  __int64 *v9; // rbx
  _QWORD *v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  _QWORD *v13; // r13
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  char *v16; // rbx
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rsi
  int v19; // eax
  _QWORD *v20; // rdx
  __int64 v21; // rax
  __int64 result; // rax
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rcx
  unsigned __int64 v26; // r8
  __int64 v27; // r9
  __int64 *v28; // rdx
  __int64 v29; // rdx
  __int64 *v30; // r15
  __int64 v31; // rax
  __int64 *v32; // r13
  _QWORD *v33; // rax
  __int64 v34; // r9
  __int64 v35; // rdx
  unsigned __int64 v36; // rdi
  char *v37; // rcx
  unsigned __int64 v38; // rsi
  __int64 v39; // r8
  _QWORD *v40; // rdx
  __int64 *v41; // r13
  char v42; // di
  __int64 v43; // rsi
  char *v44; // rax
  __int64 v45; // rax
  unsigned int *v46; // rax
  __int64 v47; // r9
  unsigned __int64 v48; // r14
  __int64 v49; // rcx
  unsigned __int64 v50; // rdi
  char *v51; // rdx
  unsigned __int64 v52; // rsi
  __int64 v53; // r8
  int v54; // eax
  _QWORD *v55; // rcx
  __int64 v56; // rax
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 *v59; // rax
  __int64 v60; // rdx
  __int64 *v61; // rsi
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // rdx
  __int64 v65; // r8
  __int64 v66; // r9
  unsigned __int64 v67; // r14
  __int64 v68; // rax
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 v71; // r15
  __int64 v72; // rcx
  unsigned __int64 v73; // rdx
  unsigned __int64 v74; // rsi
  int v75; // eax
  __int64 v76; // rdx
  __int64 *v77; // r14
  __int64 *v78; // rdx
  __int64 v79; // rax
  unsigned __int64 v80; // rdi
  __int64 v81; // rdi
  char *v82; // rbx
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // r8
  __int64 v88; // rax
  __int64 v89; // r9
  unsigned __int64 v90; // r14
  unsigned __int64 v91; // rsi
  unsigned __int64 v92; // rdi
  unsigned __int64 *v93; // rdx
  unsigned __int64 v94; // rcx
  __int64 v95; // r8
  int v96; // eax
  unsigned __int64 *v97; // rcx
  __int64 v98; // rax
  __int64 v99; // rdi
  char *v100; // rbx
  __int64 v101; // rax
  __int64 v102; // rdx
  __int64 v103; // rcx
  __int64 v104; // r8
  _DWORD *v105; // rax
  _BYTE *v106; // rcx
  int v107; // edx
  __int64 v108; // rax
  __int64 v109; // rcx
  __int64 v110; // rdx
  char *v111; // rbx
  int v112; // ebx
  __int64 v113; // rax
  __int64 v114; // r8
  __int64 v115; // r9
  __int64 v116; // rcx
  unsigned __int64 v117; // rdx
  __int64 v118; // rdx
  unsigned __int64 *v119; // rbx
  _QWORD *v120; // rax
  __int64 v121; // r8
  __int64 v122; // r9
  _QWORD *v123; // r14
  __int64 v124; // rcx
  unsigned __int64 v125; // rdx
  unsigned __int64 v126; // rsi
  int v127; // eax
  __int64 v128; // rdx
  unsigned __int64 *v129; // rbx
  unsigned __int64 *v130; // rdx
  __int64 v131; // rax
  char *v132; // rbx
  unsigned __int64 v133; // r15
  __int64 v134; // rdi
  unsigned __int64 v135; // rax
  __int64 v136; // rdi
  unsigned __int64 v137; // rdx
  __int64 v138; // rdi
  __int64 *v139; // [rsp+0h] [rbp-1A0h]
  __int64 v142; // [rsp+10h] [rbp-190h]
  __int64 *v143; // [rsp+18h] [rbp-188h]
  __int64 v144; // [rsp+18h] [rbp-188h]
  __int64 v145; // [rsp+18h] [rbp-188h]
  __int64 v146; // [rsp+18h] [rbp-188h]
  __int64 v147; // [rsp+18h] [rbp-188h]
  __int64 v148; // [rsp+18h] [rbp-188h]
  __int64 v149; // [rsp+18h] [rbp-188h]
  __int64 v150; // [rsp+28h] [rbp-178h] BYREF
  unsigned __int64 v151; // [rsp+30h] [rbp-170h] BYREF
  _BYTE *v152; // [rsp+38h] [rbp-168h] BYREF
  __int64 v153; // [rsp+40h] [rbp-160h]
  _BYTE v154[40]; // [rsp+48h] [rbp-158h] BYREF
  __int64 *v155; // [rsp+70h] [rbp-130h] BYREF
  unsigned int v156; // [rsp+78h] [rbp-128h]
  char v157; // [rsp+80h] [rbp-120h] BYREF
  unsigned int *v158; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v159; // [rsp+C8h] [rbp-D8h] BYREF
  unsigned int v160; // [rsp+D0h] [rbp-D0h] BYREF
  char v161; // [rsp+D8h] [rbp-C8h] BYREF
  char v162; // [rsp+F8h] [rbp-A8h]
  _QWORD *v163; // [rsp+110h] [rbp-90h] BYREF
  char *v164; // [rsp+118h] [rbp-88h]
  __int64 v165; // [rsp+120h] [rbp-80h]
  int v166; // [rsp+128h] [rbp-78h]
  char v167; // [rsp+12Ch] [rbp-74h]
  char v168; // [rsp+130h] [rbp-70h] BYREF

  v4 = (char *)a2;
  v143 = &a2[a3];
  v6 = (__int64)(8 * a3) >> 5;
  v7 = (__int64)(8 * a3) >> 3;
  if ( v6 <= 0 )
  {
    v8 = a2;
LABEL_19:
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
          goto LABEL_23;
LABEL_22:
        if ( sub_318B630(*v8) )
          goto LABEL_23;
        goto LABEL_8;
      }
      if ( !sub_318B630(*v8) )
      {
LABEL_8:
        if ( v143 == v8 )
          goto LABEL_23;
        goto LABEL_9;
      }
      ++v8;
    }
    if ( sub_318B630(*v8) )
    {
      ++v8;
      goto LABEL_22;
    }
    goto LABEL_8;
  }
  v8 = a2;
  v9 = &a2[4 * v6];
  while ( 1 )
  {
    if ( !sub_318B630(*v8) )
      goto LABEL_8;
    if ( !sub_318B630(v8[1]) )
    {
      if ( v143 == v8 + 1 )
        goto LABEL_23;
      goto LABEL_9;
    }
    if ( !sub_318B630(v8[2]) )
      break;
    if ( !sub_318B630(v8[3]) )
    {
      if ( v143 != v8 + 3 )
        goto LABEL_9;
LABEL_23:
      v23 = sub_318B4F0(*a2);
      v24 = sub_31BC590((__int64)a2, a3, 1);
      v139 = v28;
      v29 = (__int64)v28 - v24;
      v30 = (__int64 *)v24;
      v31 = v29 >> 3;
      if ( v29 >> 5 <= 0 )
        goto LABEL_37;
      v32 = &v30[4 * (v29 >> 5)];
      while ( v23 == sub_318B4F0(*v30) )
      {
        if ( v23 == sub_318B4F0(v30[1]) )
        {
          if ( v23 == sub_318B4F0(v30[2]) )
          {
            if ( v23 == sub_318B4F0(v30[3]) )
            {
              v30 += 4;
              if ( v30 != v32 )
                continue;
              v31 = v139 - v30;
LABEL_37:
              switch ( v31 )
              {
                case 2LL:
LABEL_108:
                  if ( v23 == sub_318B4F0(*v30) )
                  {
                    ++v30;
                    goto LABEL_110;
                  }
                  break;
                case 3LL:
                  if ( v23 == sub_318B4F0(*v30) )
                  {
                    ++v30;
                    goto LABEL_108;
                  }
                  break;
                case 1LL:
LABEL_110:
                  if ( v23 == sub_318B4F0(*v30) )
                  {
LABEL_40:
                    v165 = 8;
                    v163 = 0;
                    v164 = &v168;
                    v166 = 0;
                    v167 = 1;
                    if ( a2 != v143 )
                    {
                      v41 = a2;
                      v42 = 1;
                      while ( 1 )
                      {
                        while ( 1 )
                        {
                          v43 = *v41;
                          if ( v42 )
                            break;
LABEL_60:
                          ++v41;
                          sub_C8CC70((__int64)&v163, v43, v29, v25, v26, v27);
                          v42 = v167;
                          if ( v143 == v41 )
                            goto LABEL_48;
                        }
                        v44 = v164;
                        v25 = HIDWORD(v165);
                        v29 = (__int64)&v164[8 * HIDWORD(v165)];
                        if ( v164 == (char *)v29 )
                        {
LABEL_62:
                          if ( HIDWORD(v165) >= (unsigned int)v165 )
                            goto LABEL_60;
                          v25 = (unsigned int)(HIDWORD(v165) + 1);
                          ++v41;
                          ++HIDWORD(v165);
                          *(_QWORD *)v29 = v43;
                          v42 = v167;
                          v163 = (_QWORD *)((char *)v163 + 1);
                          if ( v143 == v41 )
                            goto LABEL_48;
                        }
                        else
                        {
                          while ( v43 != *(_QWORD *)v44 )
                          {
                            v44 += 8;
                            if ( (char *)v29 == v44 )
                              goto LABEL_62;
                          }
                          if ( v143 == ++v41 )
                          {
LABEL_48:
                            v45 = (unsigned int)(HIDWORD(v165) - v166);
                            goto LABEL_49;
                          }
                        }
                      }
                    }
                    v45 = 0;
LABEL_49:
                    if ( a3 != v45 )
                    {
                      v46 = (unsigned int *)sub_22077B0(0x10u);
                      v48 = (unsigned __int64)v46;
                      if ( v46 )
                      {
                        *((_QWORD *)v46 + 1) = 0x600000000LL;
                        *(_QWORD *)v46 = &unk_4A34890;
                      }
                      v49 = *(unsigned int *)(a1 + 280);
                      v50 = *(unsigned int *)(a1 + 284);
                      v158 = v46;
                      v51 = (char *)&v158;
                      v52 = *(_QWORD *)(a1 + 272);
                      v53 = v49 + 1;
                      v54 = v49;
                      if ( v49 + 1 > v50 )
                      {
                        if ( v52 > (unsigned __int64)&v158 || (unsigned __int64)&v158 >= v52 + 8 * v49 )
                        {
                          sub_31B2A20(a1 + 272, v49 + 1, (__int64)&v158, v49, v53, v47);
                          v49 = *(unsigned int *)(a1 + 280);
                          v52 = *(_QWORD *)(a1 + 272);
                          v51 = (char *)&v158;
                          v54 = *(_DWORD *)(a1 + 280);
                        }
                        else
                        {
                          v111 = (char *)&v158 - v52;
                          sub_31B2A20(a1 + 272, v49 + 1, (__int64)&v158, v49, v53, v47);
                          v52 = *(_QWORD *)(a1 + 272);
                          v49 = *(unsigned int *)(a1 + 280);
                          v51 = &v111[v52];
                          v54 = *(_DWORD *)(a1 + 280);
                        }
                      }
                      v55 = (_QWORD *)(v52 + 8 * v49);
                      if ( v55 )
                      {
                        *v55 = *(_QWORD *)v51;
                        *(_QWORD *)v51 = 0;
                        v48 = (unsigned __int64)v158;
                        v54 = *(_DWORD *)(a1 + 280);
                      }
                      v56 = (unsigned int)(v54 + 1);
                      *(_DWORD *)(a1 + 280) = v56;
                      if ( v48 )
                      {
                        (*(void (__fastcall **)(unsigned __int64, unsigned __int64, char *))(*(_QWORD *)v48 + 8LL))(
                          v48,
                          v52,
                          v51);
                        v56 = *(unsigned int *)(a1 + 280);
                      }
                      result = *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8 * v56 - 8);
LABEL_58:
                      if ( !v167 )
                      {
                        v144 = result;
                        _libc_free((unsigned __int64)v164);
                        return v144;
                      }
                      return result;
                    }
                    sub_31BDE20(&v155, a1, (unsigned __int64)v4, a3, v26, v27);
                    v59 = v155;
                    v60 = 16LL * v156;
                    v61 = &v155[(unsigned __int64)v60 / 8];
                    v62 = v60 >> 4;
                    v63 = v60 >> 6;
                    if ( v63 )
                    {
                      v63 = (__int64)&v155[8 * v63];
                      while ( (*(_BYTE *)v59 & 4) != 0 )
                      {
                        if ( (v59[2] & 4) == 0 )
                        {
                          v59 += 2;
                          break;
                        }
                        if ( (v59[4] & 4) == 0 )
                        {
                          v59 += 4;
                          break;
                        }
                        if ( (v59[6] & 4) == 0 )
                        {
                          v59 += 6;
                          break;
                        }
                        v59 += 8;
                        if ( v59 == (__int64 *)v63 )
                        {
                          v62 = ((char *)v61 - (char *)v59) >> 4;
                          goto LABEL_113;
                        }
                      }
LABEL_72:
                      if ( v61 != v59 )
                      {
                        sub_31BC5A0((__int64)&v158, &v155, v63, v62, v57, v58);
                        if ( v162 )
                        {
                          v67 = (unsigned __int64)v158;
                          v152 = v154;
                          v151 = (unsigned __int64)v158;
                          v153 = 0x800000000LL;
                          if ( v160 )
                          {
                            sub_31BC350((__int64)&v152, (__int64)&v159, v64, v160, v65, v66);
                            v105 = v152;
                            v106 = &v152[4 * (unsigned int)v153];
                            if ( v152 == v106 )
                            {
LABEL_167:
                              v67 = v151;
                              goto LABEL_75;
                            }
                            v107 = 0;
                            while ( *v105 == v107 )
                            {
                              ++v105;
                              ++v107;
                              if ( v106 == (_BYTE *)v105 )
                                goto LABEL_167;
                            }
                            v108 = sub_22077B0(0x48u);
                            v71 = v108;
                            if ( v108 )
                            {
                              *(_QWORD *)(v108 + 16) = v151;
                              v110 = (unsigned int)v153;
                              *(_DWORD *)(v108 + 8) = 3;
                              *(_QWORD *)v108 = &unk_4A34A70;
                              *(_QWORD *)(v108 + 24) = v108 + 40;
                              *(_QWORD *)(v108 + 32) = 0x800000000LL;
                              if ( (_DWORD)v110 )
                                sub_31BC350(v108 + 24, (__int64)&v152, v110, v109, v69, v70);
                            }
                          }
                          else
                          {
LABEL_75:
                            v68 = sub_22077B0(0x18u);
                            v71 = v68;
                            if ( v68 )
                            {
                              *(_DWORD *)(v68 + 8) = 2;
                              *(_QWORD *)(v68 + 16) = v67;
                              *(_QWORD *)v68 = &unk_4A34A50;
                            }
                          }
                          v72 = *(unsigned int *)(a1 + 280);
                          v73 = *(unsigned int *)(a1 + 284);
                          v150 = v71;
                          v74 = v72 + 1;
                          v75 = v72;
                          if ( v72 + 1 > v73 )
                          {
                            v135 = *(_QWORD *)(a1 + 272);
                            v77 = &v150;
                            v136 = a1 + 272;
                            if ( v135 > (unsigned __int64)&v150
                              || (v73 = v135 + 8 * v72, v148 = *(_QWORD *)(a1 + 272), (unsigned __int64)&v150 >= v73) )
                            {
                              sub_31B2A20(v136, v74, v73, v72, v69, v70);
                              v72 = *(unsigned int *)(a1 + 280);
                              v76 = *(_QWORD *)(a1 + 272);
                              v75 = *(_DWORD *)(a1 + 280);
                            }
                            else
                            {
                              sub_31B2A20(v136, v74, v73, v72, v69, v70);
                              v76 = *(_QWORD *)(a1 + 272);
                              v72 = *(unsigned int *)(a1 + 280);
                              v77 = (__int64 *)((char *)&v150 + v76 - v148);
                              v75 = *(_DWORD *)(a1 + 280);
                            }
                          }
                          else
                          {
                            v76 = *(_QWORD *)(a1 + 272);
                            v77 = &v150;
                          }
                          v78 = (__int64 *)(v76 + 8 * v72);
                          if ( v78 )
                          {
                            *v78 = *v77;
                            *v77 = 0;
                            v71 = v150;
                            v75 = *(_DWORD *)(a1 + 280);
                          }
                          v79 = (unsigned int)(v75 + 1);
                          *(_DWORD *)(a1 + 280) = v79;
                          if ( v71 )
                          {
                            (*(void (__fastcall **)(__int64))(*(_QWORD *)v71 + 8LL))(v71);
                            v79 = *(unsigned int *)(a1 + 280);
                          }
                          result = *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8 * v79 - 8);
                          if ( v152 != v154 )
                          {
                            v145 = result;
                            _libc_free((unsigned __int64)v152);
                            result = v145;
                          }
                          if ( v162 )
                          {
                            v80 = v159;
                            if ( (char *)v159 != &v161 )
                            {
LABEL_87:
                              v146 = result;
                              _libc_free(v80);
                              result = v146;
                            }
                          }
LABEL_88:
                          if ( v155 != (__int64 *)&v157 )
                          {
                            v147 = result;
                            _libc_free((unsigned __int64)v155);
                            result = v147;
                          }
                          goto LABEL_58;
                        }
                        v101 = sub_22077B0(0x60u);
                        v90 = v101;
                        if ( v101 )
                        {
                          *(_DWORD *)(v101 + 8) = 4;
                          *(_QWORD *)v101 = &unk_4A34A90;
                          *(_QWORD *)(v101 + 16) = v101 + 32;
                          *(_QWORD *)(v101 + 24) = 0x400000000LL;
                          if ( v156 )
                            sub_31BC1F0(v101 + 16, (char **)&v155, v102, v103, v104, v89);
                        }
                        goto LABEL_131;
                      }
LABEL_117:
                      v158 = (unsigned int *)sub_31BC800(a1, v4, a3);
                      if ( BYTE4(v158) )
                      {
                        v112 = (int)v158;
                        v113 = sub_22077B0(0x10u);
                        v90 = v113;
                        if ( v113 )
                        {
                          *(_DWORD *)(v113 + 8) = 0;
                          *(_DWORD *)(v113 + 12) = v112;
                          *(_QWORD *)v113 = &unk_4A34890;
                        }
                        v116 = *(unsigned int *)(a1 + 280);
                        v117 = *(unsigned int *)(a1 + 284);
                        v151 = v113;
                        v91 = v116 + 1;
                        v96 = v116;
                        if ( v116 + 1 > v117 )
                        {
                          v133 = *(_QWORD *)(a1 + 272);
                          v119 = &v151;
                          v134 = a1 + 272;
                          if ( v133 > (unsigned __int64)&v151 || (unsigned __int64)&v151 >= v133 + 8 * v116 )
                          {
                            sub_31B2A20(v134, v91, v117, v116, v114, v115);
                            v116 = *(unsigned int *)(a1 + 280);
                            v118 = *(_QWORD *)(a1 + 272);
                            v96 = *(_DWORD *)(a1 + 280);
                          }
                          else
                          {
                            sub_31B2A20(v134, v91, v117, v116, v114, v115);
                            v118 = *(_QWORD *)(a1 + 272);
                            v116 = *(unsigned int *)(a1 + 280);
                            v119 = (unsigned __int64 *)((char *)&v151 + v118 - v133);
                            v96 = *(_DWORD *)(a1 + 280);
                          }
                        }
                        else
                        {
                          v118 = *(_QWORD *)(a1 + 272);
                          v119 = &v151;
                        }
                        v93 = (unsigned __int64 *)(v118 + 8 * v116);
                        if ( v93 )
                        {
                          *v93 = *v119;
                          *v119 = 0;
                          v90 = v151;
                          v96 = *(_DWORD *)(a1 + 280);
                        }
                      }
                      else
                      {
                        if ( !a4 )
                        {
                          v85 = 0;
                          v159 = 0x800000000LL;
                          v86 = 0;
                          v158 = &v160;
                          if ( a3 > 8 )
                          {
                            sub_C8D5F0((__int64)&v158, &v160, a3, 8u, v83, v84);
                            v85 = (unsigned int)v159;
                            v86 = (unsigned int)v159;
                          }
                          if ( v4 != (char *)v143 )
                          {
                            do
                            {
                              v87 = *(_QWORD *)v4;
                              if ( v86 + 1 > (unsigned __int64)HIDWORD(v159) )
                              {
                                v142 = *(_QWORD *)v4;
                                sub_C8D5F0((__int64)&v158, &v160, v86 + 1, 8u, v87, v84);
                                v86 = (unsigned int)v159;
                                v87 = v142;
                              }
                              v4 += 8;
                              *(_QWORD *)&v158[2 * v86] = v87;
                              v86 = (unsigned int)(v159 + 1);
                              LODWORD(v159) = v159 + 1;
                            }
                            while ( v143 != (__int64 *)v4 );
                            v85 = (unsigned int)v86;
                          }
                          if ( !(unsigned __int8)sub_31C2000(a1, v158, v85) )
                          {
                            v120 = (_QWORD *)sub_22077B0(0x10u);
                            v123 = v120;
                            if ( v120 )
                            {
                              v120[1] = 0x800000000LL;
                              *v120 = &unk_4A34890;
                            }
                            v124 = *(unsigned int *)(a1 + 280);
                            v125 = *(unsigned int *)(a1 + 284);
                            v151 = (unsigned __int64)v120;
                            v126 = v124 + 1;
                            v127 = v124;
                            if ( v124 + 1 > v125 )
                            {
                              v137 = *(_QWORD *)(a1 + 272);
                              v129 = &v151;
                              v138 = a1 + 272;
                              if ( v137 > (unsigned __int64)&v151
                                || (v149 = *(_QWORD *)(a1 + 272), (unsigned __int64)&v151 >= v137 + 8 * v124) )
                              {
                                sub_31B2A20(v138, v126, v137, v124, v121, v122);
                                v124 = *(unsigned int *)(a1 + 280);
                                v128 = *(_QWORD *)(a1 + 272);
                                v127 = *(_DWORD *)(a1 + 280);
                              }
                              else
                              {
                                sub_31B2A20(v138, v126, v137, v124, v121, v122);
                                v124 = *(unsigned int *)(a1 + 280);
                                v128 = *(_QWORD *)(a1 + 272);
                                v127 = *(_DWORD *)(a1 + 280);
                                v129 = (unsigned __int64 *)((char *)&v151 + v128 - v149);
                              }
                            }
                            else
                            {
                              v128 = *(_QWORD *)(a1 + 272);
                              v129 = &v151;
                            }
                            v130 = (unsigned __int64 *)(v128 + 8 * v124);
                            if ( v130 )
                            {
                              *v130 = *v129;
                              *v129 = 0;
                              v123 = (_QWORD *)v151;
                              v127 = *(_DWORD *)(a1 + 280);
                            }
                            v131 = (unsigned int)(v127 + 1);
                            *(_DWORD *)(a1 + 280) = v131;
                            if ( v123 )
                            {
                              (*(void (__fastcall **)(_QWORD *))(*v123 + 8LL))(v123);
                              v131 = *(unsigned int *)(a1 + 280);
                            }
                            v80 = (unsigned __int64)v158;
                            result = *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8 * v131 - 8);
                            if ( v158 != &v160 )
                              goto LABEL_87;
                            goto LABEL_88;
                          }
                          if ( v158 != &v160 )
                            _libc_free((unsigned __int64)v158);
                        }
                        v88 = sub_22077B0(0x10u);
                        v90 = v88;
                        if ( v88 )
                        {
                          *(_DWORD *)(v88 + 8) = 1;
                          *(_QWORD *)v88 = &unk_4A34A30;
                        }
LABEL_131:
                        v91 = *(unsigned int *)(a1 + 280);
                        v92 = *(unsigned int *)(a1 + 284);
                        v93 = (unsigned __int64 *)&v158;
                        v158 = (unsigned int *)v90;
                        v94 = *(_QWORD *)(a1 + 272);
                        v95 = v91 + 1;
                        v96 = v91;
                        if ( v91 + 1 > v92 )
                        {
                          if ( v94 > (unsigned __int64)&v158 || (unsigned __int64)&v158 >= v94 + 8 * v91 )
                          {
                            sub_31B2A20(a1 + 272, v91 + 1, (__int64)&v158, v94, v95, v89);
                            v91 = *(unsigned int *)(a1 + 280);
                            v94 = *(_QWORD *)(a1 + 272);
                            v93 = (unsigned __int64 *)&v158;
                            v96 = *(_DWORD *)(a1 + 280);
                          }
                          else
                          {
                            v132 = (char *)&v158 - v94;
                            sub_31B2A20(a1 + 272, v91 + 1, (__int64)&v158, v94, v95, v89);
                            v94 = *(_QWORD *)(a1 + 272);
                            v91 = *(unsigned int *)(a1 + 280);
                            v93 = (unsigned __int64 *)&v132[v94];
                            v96 = *(_DWORD *)(a1 + 280);
                          }
                        }
                        v97 = (unsigned __int64 *)(v94 + 8 * v91);
                        if ( v97 )
                        {
                          *v97 = *v93;
                          *v93 = 0;
                          v90 = (unsigned __int64)v158;
                          v96 = *(_DWORD *)(a1 + 280);
                        }
                      }
                      v98 = (unsigned int)(v96 + 1);
                      *(_DWORD *)(a1 + 280) = v98;
                      if ( v90 )
                      {
                        (*(void (__fastcall **)(unsigned __int64, unsigned __int64, unsigned __int64 *))(*(_QWORD *)v90 + 8LL))(
                          v90,
                          v91,
                          v93);
                        v98 = *(unsigned int *)(a1 + 280);
                      }
                      result = *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8 * v98 - 8);
                      goto LABEL_88;
                    }
LABEL_113:
                    if ( v62 != 2 )
                    {
                      if ( v62 != 3 )
                      {
                        if ( v62 != 1 )
                          goto LABEL_117;
                        goto LABEL_116;
                      }
                      if ( (*(_BYTE *)v59 & 4) == 0 )
                        goto LABEL_72;
                      v59 += 2;
                    }
                    if ( (*(_BYTE *)v59 & 4) == 0 )
                      goto LABEL_72;
                    v59 += 2;
LABEL_116:
                    if ( (*(_BYTE *)v59 & 4) == 0 )
                      goto LABEL_72;
                    goto LABEL_117;
                  }
                  goto LABEL_30;
                default:
                  goto LABEL_40;
              }
            }
            else
            {
              v30 += 3;
            }
          }
          else
          {
            v30 += 2;
          }
        }
        else
        {
          ++v30;
        }
        break;
      }
LABEL_30:
      if ( v139 == v30 )
        goto LABEL_40;
      v33 = (_QWORD *)sub_22077B0(0x10u);
      v13 = v33;
      if ( v33 )
      {
        v33[1] = 0x500000000LL;
        *v33 = &unk_4A34890;
      }
      v35 = *(unsigned int *)(a1 + 280);
      v36 = *(unsigned int *)(a1 + 284);
      v163 = v33;
      v37 = (char *)&v163;
      v38 = *(_QWORD *)(a1 + 272);
      v39 = v35 + 1;
      v19 = v35;
      if ( v35 + 1 > v36 )
      {
        v99 = a1 + 272;
        if ( v38 > (unsigned __int64)&v163 || (unsigned __int64)&v163 >= v38 + 8 * v35 )
        {
          sub_31B2A20(v99, v35 + 1, v35, (__int64)&v163, v39, v34);
          v35 = *(unsigned int *)(a1 + 280);
          v38 = *(_QWORD *)(a1 + 272);
          v37 = (char *)&v163;
          v19 = *(_DWORD *)(a1 + 280);
        }
        else
        {
          v100 = (char *)&v163 - v38;
          sub_31B2A20(v99, v35 + 1, v35, (__int64)&v163 - v38, v39, v34);
          v38 = *(_QWORD *)(a1 + 272);
          v35 = *(unsigned int *)(a1 + 280);
          v37 = &v100[v38];
          v19 = *(_DWORD *)(a1 + 280);
        }
      }
      v40 = (_QWORD *)(v38 + 8 * v35);
      if ( v40 )
      {
        *v40 = *(_QWORD *)v37;
        *(_QWORD *)v37 = 0;
        v13 = v163;
        v19 = *(_DWORD *)(a1 + 280);
      }
      goto LABEL_14;
    }
    v8 += 4;
    if ( v8 == v9 )
    {
      v7 = v143 - v8;
      goto LABEL_19;
    }
  }
  if ( v143 == v8 + 2 )
    goto LABEL_23;
LABEL_9:
  v10 = (_QWORD *)sub_22077B0(0x10u);
  v13 = v10;
  if ( v10 )
  {
    v10[1] = 0;
    *v10 = &unk_4A34890;
  }
  v14 = *(unsigned int *)(a1 + 280);
  v15 = *(unsigned int *)(a1 + 284);
  v163 = v10;
  v16 = (char *)&v163;
  v17 = *(_QWORD *)(a1 + 272);
  v18 = v14 + 1;
  v19 = v14;
  if ( v14 + 1 > v15 )
  {
    v81 = a1 + 272;
    if ( v17 > (unsigned __int64)&v163 || (unsigned __int64)&v163 >= v17 + 8 * v14 )
    {
      sub_31B2A20(v81, v18, v14, v17, v11, v12);
      v14 = *(unsigned int *)(a1 + 280);
      v17 = *(_QWORD *)(a1 + 272);
      v19 = *(_DWORD *)(a1 + 280);
    }
    else
    {
      v82 = (char *)&v163 - v17;
      sub_31B2A20(v81, v18, v14, v17, v11, v12);
      v17 = *(_QWORD *)(a1 + 272);
      v14 = *(unsigned int *)(a1 + 280);
      v16 = &v82[v17];
      v19 = *(_DWORD *)(a1 + 280);
    }
  }
  v20 = (_QWORD *)(v17 + 8 * v14);
  if ( v20 )
  {
    *v20 = *(_QWORD *)v16;
    *(_QWORD *)v16 = 0;
    v13 = v163;
    v19 = *(_DWORD *)(a1 + 280);
  }
LABEL_14:
  v21 = (unsigned int)(v19 + 1);
  *(_DWORD *)(a1 + 280) = v21;
  if ( v13 )
  {
    (*(void (__fastcall **)(_QWORD *))(*v13 + 8LL))(v13);
    v21 = *(unsigned int *)(a1 + 280);
  }
  return *(_QWORD *)(*(_QWORD *)(a1 + 272) + 8 * v21 - 8);
}
