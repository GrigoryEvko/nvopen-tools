// Function: sub_285F850
// Address: 0x285f850
//
void __fastcall sub_285F850(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 **v4; // r14
  __int64 *v5; // r12
  __int64 v6; // rax
  __int64 *v7; // rdi
  __int64 *v8; // rax
  __int64 *v9; // rbx
  _BYTE *v10; // rsi
  __int64 v11; // r15
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 **v14; // rcx
  __int64 v15; // rdx
  char **v16; // rbx
  __int64 *v17; // r14
  __int64 v18; // r9
  _QWORD *v19; // r12
  _QWORD *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // r15
  unsigned int v23; // r8d
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // r15
  int v27; // r12d
  int v28; // eax
  unsigned int v29; // r11d
  __int64 v30; // rcx
  __int64 v31; // r15
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  unsigned int v36; // eax
  __int64 v37; // rdx
  __int64 v38; // r8
  _QWORD *v39; // rax
  __int64 *v40; // r12
  _QWORD *v41; // rax
  __int64 v42; // r9
  __int64 v43; // r12
  __int64 v44; // rax
  unsigned int v45; // r8d
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rax
  unsigned int v49; // edx
  __int64 **v50; // rax
  __int64 *v51; // rax
  __int64 *v52; // rdi
  __int64 *v53; // r12
  __int64 *v54; // rdi
  _QWORD *v55; // rdx
  unsigned __int64 v56; // rax
  int v57; // edx
  unsigned __int64 v58; // rax
  bool v59; // cf
  char *v60; // rdx
  unsigned int v61; // eax
  __int64 v62; // rdx
  __int64 v63; // r8
  _QWORD *v64; // rax
  _QWORD *v65; // rax
  __int64 v66; // r11
  __int64 v67; // r9
  __int64 v68; // r12
  __int64 v69; // rax
  __int64 v70; // r12
  int v71; // eax
  unsigned int v72; // edx
  unsigned __int8 v73; // cl
  int v74; // r12d
  __int64 v75; // r12
  __int64 **v76; // r12
  __int64 v77; // rbx
  __int64 v78; // rdx
  unsigned int v79; // esi
  __int64 v80; // rax
  __int64 v81; // rdx
  __int64 v82; // r14
  __int64 v83; // r12
  __int64 v84; // rbx
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // rdi
  __int64 v88; // rcx
  __int64 v89; // rax
  __int64 v90; // rax
  _BYTE *v91; // rsi
  __int64 v92; // rdi
  __int64 *v93; // r15
  __int64 v94; // rsi
  __int64 v95; // r15
  _QWORD *v96; // rdx
  unsigned __int64 v97; // rax
  int v98; // edx
  __int64 v99; // rsi
  __int64 v100; // rax
  __int64 v101; // r9
  __int64 v102; // r10
  __int64 **v103; // rax
  int v104; // esi
  __int64 **v105; // rdx
  _BYTE *v106; // rax
  __int64 v107; // r15
  __int64 v108; // rdx
  __int64 v109; // rbx
  __int64 v110; // r15
  __int64 v111; // rdx
  unsigned int v112; // esi
  __int64 v113; // rdx
  __int64 v114; // rcx
  __int64 v115; // r8
  __int64 v116; // r9
  __int64 **v117; // rax
  __int64 v118; // rdx
  __int64 v119; // rbx
  __int64 v120; // rax
  __int64 v121; // r14
  __int64 v122; // r13
  __int64 v123; // r12
  __int64 v124; // r15
  __int64 v125; // rdx
  unsigned int v126; // esi
  __int64 v127; // rax
  unsigned __int64 v128; // rax
  __int64 v129; // [rsp+0h] [rbp-210h]
  __int64 v130; // [rsp+8h] [rbp-208h]
  __int64 v131; // [rsp+10h] [rbp-200h]
  __int64 v132; // [rsp+20h] [rbp-1F0h]
  __int64 v133; // [rsp+28h] [rbp-1E8h]
  __int64 v135; // [rsp+38h] [rbp-1D8h]
  __int64 v136; // [rsp+40h] [rbp-1D0h]
  __int64 v138; // [rsp+50h] [rbp-1C0h]
  char **v139; // [rsp+68h] [rbp-1A8h]
  __int64 v140; // [rsp+68h] [rbp-1A8h]
  __int64 v141; // [rsp+68h] [rbp-1A8h]
  unsigned __int64 v142; // [rsp+68h] [rbp-1A8h]
  __int64 v143; // [rsp+70h] [rbp-1A0h]
  __int64 v144; // [rsp+78h] [rbp-198h]
  int v145; // [rsp+78h] [rbp-198h]
  __int64 *v146; // [rsp+78h] [rbp-198h]
  __int64 *v147; // [rsp+80h] [rbp-190h]
  char *v148; // [rsp+88h] [rbp-188h]
  unsigned __int64 v149; // [rsp+90h] [rbp-180h]
  __int64 *v150; // [rsp+90h] [rbp-180h]
  __int64 *v151; // [rsp+90h] [rbp-180h]
  __int64 *v152; // [rsp+98h] [rbp-178h]
  char **v153; // [rsp+98h] [rbp-178h]
  __int64 **v154; // [rsp+98h] [rbp-178h]
  __int64 **v155; // [rsp+98h] [rbp-178h]
  __int64 v156; // [rsp+98h] [rbp-178h]
  _BYTE *v157; // [rsp+A8h] [rbp-168h] BYREF
  _QWORD v158[4]; // [rsp+B0h] [rbp-160h] BYREF
  char v159; // [rsp+D0h] [rbp-140h]
  char v160; // [rsp+D1h] [rbp-13Fh]
  __int64 v161[4]; // [rsp+E0h] [rbp-130h] BYREF
  __int16 v162; // [rsp+100h] [rbp-110h]
  _QWORD *v163; // [rsp+110h] [rbp-100h] BYREF
  unsigned int v164; // [rsp+118h] [rbp-F8h]
  unsigned int v165; // [rsp+11Ch] [rbp-F4h]
  _QWORD v166[6]; // [rsp+120h] [rbp-F0h] BYREF
  __int64 **v167; // [rsp+150h] [rbp-C0h] BYREF
  __int64 v168; // [rsp+158h] [rbp-B8h]
  __int64 *v169; // [rsp+160h] [rbp-B0h] BYREF
  __int64 *v170; // [rsp+168h] [rbp-A8h]
  __int64 v171; // [rsp+180h] [rbp-90h]
  __int64 v172; // [rsp+188h] [rbp-88h]
  __int64 v173; // [rsp+190h] [rbp-80h]
  __int64 v174; // [rsp+198h] [rbp-78h]
  void **v175; // [rsp+1A0h] [rbp-70h]
  _QWORD *v176; // [rsp+1A8h] [rbp-68h]
  __int64 v177; // [rsp+1B0h] [rbp-60h]
  int v178; // [rsp+1B8h] [rbp-58h]
  __int16 v179; // [rsp+1BCh] [rbp-54h]
  char v180; // [rsp+1BEh] [rbp-52h]
  __int64 v181; // [rsp+1C0h] [rbp-50h]
  __int64 v182; // [rsp+1C8h] [rbp-48h]
  void *v183; // [rsp+1D0h] [rbp-40h] BYREF
  _QWORD v184[7]; // [rsp+1D8h] [rbp-38h] BYREF

  v4 = *(__int64 ***)a2;
  v5 = **(__int64 ***)a2;
  v6 = 32LL * (*((_DWORD *)v5 + 1) & 0x7FFFFFF);
  if ( (*((_BYTE *)v5 + 7) & 0x40) != 0 )
  {
    v7 = (__int64 *)*(v5 - 1);
    v5 = &v7[(unsigned __int64)v6 / 8];
  }
  else
  {
    v7 = &v5[v6 / 0xFFFFFFFFFFFFFFF8LL];
  }
  while ( 1 )
  {
    v8 = (__int64 *)sub_284F450((_BYTE **)v7, (_BYTE **)v5, *(_QWORD *)(a1 + 56), *(_QWORD *)(a1 + 8));
    v9 = v8;
    if ( v8 == v5 )
      return;
    v10 = (_BYTE *)*v8;
    v11 = *v8;
    if ( *(_BYTE *)*v8 == 67 )
      break;
    if ( v4[2] == sub_DD8400(*(_QWORD *)(a1 + 8), (__int64)v10) )
      goto LABEL_9;
LABEL_4:
    if ( v4[2] == sub_DD8400(*(_QWORD *)(a1 + 8), v11) )
      goto LABEL_9;
    v7 = v9 + 4;
  }
  v11 = *((_QWORD *)v10 - 4);
  if ( v4[2] != sub_DD8400(*(_QWORD *)(a1 + 8), (__int64)v10) )
    goto LABEL_4;
LABEL_9:
  v136 = v11;
  v143 = *(_QWORD *)(v11 + 8);
  v135 = sub_D97090(*(_QWORD *)(a1 + 8), v143);
  v12 = sub_DA2C50(*(_QWORD *)(a1 + 8), v135, 0, 0);
  v165 = 3;
  v166[0] = v12;
  v13 = *(unsigned int *)(a2 + 8);
  v14 = *(__int64 ***)a2;
  v163 = v166;
  v166[1] = v11;
  v164 = 1;
  v15 = 3 * v13;
  v139 = (char **)&v14[v15];
  if ( &v14[v15] != v14 + 3 )
  {
    v147 = 0;
    v16 = (char **)(v14 + 3);
    v17 = v12;
    while ( 1 )
    {
      v148 = *v16;
      if ( **v16 == 84 )
      {
        v55 = (_QWORD *)(sub_D47930(*(_QWORD *)(a1 + 56)) + 48);
        v56 = *v55 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (_QWORD *)v56 == v55 )
        {
          v148 = 0;
        }
        else
        {
          if ( !v56 )
LABEL_163:
            BUG();
          v57 = *(unsigned __int8 *)(v56 - 24);
          v58 = v56 - 24;
          v59 = (unsigned int)(v57 - 30) < 0xB;
          v60 = 0;
          if ( v59 )
            v60 = (char *)v58;
          v148 = v60;
        }
      }
      if ( !sub_D968A0((__int64)v16[2]) )
      {
        v51 = sub_DD2D10(*(_QWORD *)(a1 + 8), (__int64)v16[2], v135);
        v52 = *(__int64 **)(a1 + 8);
        v53 = v51;
        v169 = v17;
        v168 = 0x200000002LL;
        v167 = &v169;
        v170 = v51;
        v17 = sub_DC7EB0(v52, (__int64)&v167, 0, 0);
        if ( v167 != &v169 )
          _libc_free((unsigned __int64)v167);
        if ( v147 )
        {
          v54 = *(__int64 **)(a1 + 8);
          v169 = v147;
          v167 = &v169;
          v170 = v53;
          v168 = 0x200000002LL;
          v147 = sub_DC7EB0(v54, (__int64)&v167, 0, 0);
          if ( v167 != &v169 )
            _libc_free((unsigned __int64)v167);
        }
        else
        {
          v147 = v53;
        }
      }
      v18 = 2LL * v164;
      v149 = (unsigned __int64)v163;
      v19 = &v163[v18];
      if ( v163 != &v163[v18] )
      {
        while ( 1 )
        {
          v152 = (__int64 *)*(v19 - 1);
          v20 = sub_DCC810(*(__int64 **)(a1 + 8), (__int64)v17, *(v19 - 2), 0, 0);
          v21 = (__int64)*v16;
          v22 = (__int64)v20;
          if ( sub_2858AB0((__int64)v20, *v16, (__int64)v16[1], *(__int64 **)(a1 + 48), v23) )
            break;
          v19 -= 2;
          if ( (_QWORD *)v149 == v19 )
            goto LABEL_34;
        }
        if ( sub_D968A0(v22) )
          goto LABEL_18;
        ++*(_QWORD *)(a1 + 496);
        if ( *(_BYTE *)(a1 + 524) )
        {
LABEL_65:
          *(_QWORD *)(a1 + 516) = 0;
        }
        else
        {
          v61 = 4 * (*(_DWORD *)(a1 + 516) - *(_DWORD *)(a1 + 520));
          v62 = *(unsigned int *)(a1 + 512);
          if ( v61 < 0x20 )
            v61 = 32;
          if ( (unsigned int)v62 <= v61 )
          {
            memset(*(void **)(a1 + 504), -1, 8 * v62);
            goto LABEL_65;
          }
          sub_C8C990(a1 + 496, v21);
        }
        sub_F817B0(a1 + 176);
        v63 = v133;
        LOWORD(v63) = 0;
        v133 = v63;
        v64 = sub_F8DB90(a1 + 80, v22, v135, (__int64)(v148 + 24), 0);
        v146 = *(__int64 **)(a1 + 8);
        v151 = sub_DA3860(v146, (__int64)v64);
        v169 = sub_DA3860(*(_QWORD **)(a1 + 8), (__int64)v152);
        v170 = v151;
        v167 = &v169;
        v168 = 0x200000002LL;
        v65 = sub_DC7EB0(v146, (__int64)&v167, 0, 0);
        v66 = a1 + 80;
        v67 = (__int64)(v148 + 24);
        v68 = (__int64)v65;
        if ( v167 != &v169 )
        {
          _libc_free((unsigned __int64)v167);
          v67 = (__int64)(v148 + 24);
          v66 = a1 + 80;
        }
        v69 = v132;
        LOWORD(v69) = 0;
        v132 = v69;
        v152 = sub_F8DB90(v66, v68, v143, v67, 0);
        goto LABEL_18;
      }
LABEL_34:
      if ( v147 )
      {
        v152 = (__int64 *)v136;
        if ( !sub_D968A0((__int64)v147) )
        {
          ++*(_QWORD *)(a1 + 496);
          if ( *(_BYTE *)(a1 + 524) )
            goto LABEL_41;
          v36 = 4 * (*(_DWORD *)(a1 + 516) - *(_DWORD *)(a1 + 520));
          v37 = *(unsigned int *)(a1 + 512);
          if ( v36 < 0x20 )
            v36 = 32;
          if ( v36 < (unsigned int)v37 )
          {
            sub_C8C990(a1 + 496, v136);
          }
          else
          {
            memset(*(void **)(a1 + 504), -1, 8 * v37);
LABEL_41:
            *(_QWORD *)(a1 + 516) = 0;
          }
          sub_F817B0(a1 + 176);
          v38 = v131;
          LOWORD(v38) = 0;
          v131 = v38;
          v39 = sub_F8DB90(a1 + 80, (__int64)v147, v135, (__int64)(v148 + 24), 0);
          v40 = *(__int64 **)(a1 + 8);
          v150 = sub_DA3860(v40, (__int64)v39);
          v169 = sub_DA3860(*(_QWORD **)(a1 + 8), v136);
          v170 = v150;
          v167 = &v169;
          v168 = 0x200000002LL;
          v41 = sub_DC7EB0(v40, (__int64)&v167, 0, 0);
          v42 = (__int64)(v148 + 24);
          v43 = (__int64)v41;
          if ( v167 != &v169 )
          {
            _libc_free((unsigned __int64)v167);
            v42 = (__int64)(v148 + 24);
          }
          v44 = v130;
          LOWORD(v44) = 0;
          v130 = v44;
          v152 = sub_F8DB90(a1 + 80, v43, v143, v42, 0);
          if ( !sub_2858AB0((__int64)v147, *v16, (__int64)v16[1], *(__int64 **)(a1 + 48), v45) )
          {
            v48 = v164;
            v49 = v164;
            if ( v164 >= (unsigned __int64)v165 )
            {
              if ( v165 < (unsigned __int64)v164 + 1 )
              {
                sub_C8D5F0((__int64)&v163, v166, v164 + 1LL, 0x10u, v46, v47);
                v48 = v164;
              }
              v117 = (__int64 **)&v163[2 * v48];
              v147 = 0;
              *v117 = v17;
              v117[1] = v152;
              v136 = (__int64)v152;
              ++v164;
            }
            else
            {
              v50 = (__int64 **)&v163[2 * v164];
              if ( v50 )
              {
                *v50 = v17;
                v50[1] = v152;
                v49 = v164;
              }
              v147 = 0;
              v164 = v49 + 1;
              v136 = (__int64)v152;
            }
          }
        }
      }
      else
      {
        v152 = (__int64 *)v136;
      }
LABEL_18:
      v24 = (__int64)v16[1];
      v144 = *(_QWORD *)(v24 + 8);
      if ( v143 != v144 )
      {
        v25 = sub_BD5C60((__int64)v148);
        v180 = 7;
        v174 = v25;
        v175 = &v183;
        v176 = v184;
        v179 = 512;
        LOWORD(v173) = 0;
        v168 = 0x200000000LL;
        v183 = &unk_49DA100;
        v167 = &v169;
        v184[0] = &unk_49DA0B0;
        v177 = 0;
        v178 = 0;
        v181 = 0;
        v182 = 0;
        v171 = 0;
        v172 = 0;
        sub_D5F1F0((__int64)&v167, (__int64)v148);
        v160 = 1;
        v158[0] = "lsr.chain";
        v159 = 3;
        v26 = v152[1];
        v27 = sub_BCB060(v26);
        v28 = sub_BCB060(v144);
        v29 = 49;
        if ( v27 != v28 )
          v29 = 38;
        if ( v144 == v26 )
        {
          v31 = (__int64)v152;
        }
        else
        {
          v138 = v144;
          v30 = v144;
          v145 = v29;
          v31 = (*((__int64 (__fastcall **)(void **, _QWORD, __int64 *, __int64))*v175 + 15))(v175, v29, v152, v30);
          if ( !v31 )
          {
            v162 = 257;
            v31 = sub_B51D30(v145, (__int64)v152, v138, (__int64)v161, 0, 0);
            if ( *(_BYTE *)v31 > 0x1Cu )
            {
              switch ( *(_BYTE *)v31 )
              {
                case ')':
                case '+':
                case '-':
                case '/':
                case '2':
                case '5':
                case 'J':
                case 'K':
                case 'S':
                  goto LABEL_77;
                case 'T':
                case 'U':
                case 'V':
                  v70 = *(_QWORD *)(v31 + 8);
                  v71 = *(unsigned __int8 *)(v70 + 8);
                  v72 = v71 - 17;
                  v73 = *(_BYTE *)(v70 + 8);
                  if ( (unsigned int)(v71 - 17) <= 1 )
                    v73 = *(_BYTE *)(**(_QWORD **)(v70 + 16) + 8LL);
                  if ( v73 <= 3u || v73 == 5 || (v73 & 0xFD) == 4 )
                    goto LABEL_77;
                  if ( (_BYTE)v71 == 15 )
                  {
                    if ( (*(_BYTE *)(v70 + 9) & 4) == 0 || !sub_BCB420(*(_QWORD *)(v31 + 8)) )
                      break;
                    v70 = **(_QWORD **)(v70 + 16);
                    v71 = *(unsigned __int8 *)(v70 + 8);
                    v72 = v71 - 17;
                  }
                  else if ( (_BYTE)v71 == 16 )
                  {
                    do
                    {
                      v70 = *(_QWORD *)(v70 + 24);
                      LOBYTE(v71) = *(_BYTE *)(v70 + 8);
                    }
                    while ( (_BYTE)v71 == 16 );
                    v72 = (unsigned __int8)v71 - 17;
                  }
                  if ( v72 <= 1 )
                    LOBYTE(v71) = *(_BYTE *)(**(_QWORD **)(v70 + 16) + 8LL);
                  if ( (unsigned __int8)v71 <= 3u || (_BYTE)v71 == 5 || (v71 & 0xFD) == 4 )
                  {
LABEL_77:
                    v74 = v178;
                    if ( v177 )
                      sub_B99FD0(v31, 3u, v177);
                    sub_B45150(v31, v74);
                  }
                  break;
                default:
                  break;
              }
            }
            (*(void (__fastcall **)(_QWORD *, __int64, _QWORD *, __int64, __int64))(*v176 + 16LL))(
              v176,
              v31,
              v158,
              v172,
              v173);
            v75 = 2LL * (unsigned int)v168;
            if ( v167 != &v167[v75] )
            {
              v153 = v16;
              v76 = &v167[v75];
              v77 = (__int64)v167;
              do
              {
                v78 = *(_QWORD *)(v77 + 8);
                v79 = *(_DWORD *)v77;
                v77 += 16;
                sub_B99FD0(v31, v79, v78);
              }
              while ( v76 != (__int64 **)v77 );
              v16 = v153;
            }
          }
        }
        nullsub_61();
        v183 = &unk_49DA100;
        nullsub_63();
        if ( v167 != &v169 )
          _libc_free((unsigned __int64)v167);
        v152 = (__int64 *)v31;
        v24 = (__int64)v16[1];
      }
      sub_BD2ED0((__int64)*v16, v24, (__int64)v152);
      if ( (unsigned __int8)*v16[1] > 0x1Cu )
      {
        v167 = (__int64 **)v16[1];
        sub_2855140(a3, (__int64 *)&v167, v32, v33, v34, v35);
      }
      v16 += 3;
      if ( v139 == v16 )
      {
        v14 = *(__int64 ***)a2;
        v15 = 3LL * *(unsigned int *)(a2 + 8);
        break;
      }
    }
  }
  if ( *(_BYTE *)v14[v15 - 3] == 84 )
  {
    v80 = sub_AA5930(**(_QWORD **)(*(_QWORD *)(a1 + 56) + 32LL));
    v82 = v81;
    v83 = v80;
    if ( v80 != v81 )
    {
      v84 = v136;
      while ( 2 )
      {
        if ( *(_QWORD *)(v84 + 8) != *(_QWORD *)(v83 + 8) )
          goto LABEL_98;
        v86 = sub_D47930(*(_QWORD *)(a1 + 56));
        v87 = *(_QWORD *)(v83 - 8);
        v88 = v86;
        if ( (*(_DWORD *)(v83 + 4) & 0x7FFFFFF) != 0 )
        {
          v89 = 0;
          while ( v88 != *(_QWORD *)(v87 + 32LL * *(unsigned int *)(v83 + 72) + 8 * v89) )
          {
            if ( (*(_DWORD *)(v83 + 4) & 0x7FFFFFF) == (_DWORD)++v89 )
              goto LABEL_134;
          }
          v90 = 32 * v89;
        }
        else
        {
LABEL_134:
          v90 = 0x1FFFFFFFE0LL;
        }
        v91 = *(_BYTE **)(v87 + v90);
        if ( !v91 )
          BUG();
        if ( *v91 <= 0x1Cu )
          goto LABEL_98;
        v92 = *(_QWORD *)(a1 + 8);
        v157 = v91;
        v93 = sub_DD8400(v92, (__int64)v91);
        if ( v93 != sub_DD8400(*(_QWORD *)(a1 + 8), v84) )
          goto LABEL_98;
        v94 = (__int64)v157;
        v95 = *((_QWORD *)v157 + 1);
        if ( v143 != v95 )
        {
          v96 = (_QWORD *)(sub_D47930(*(_QWORD *)(a1 + 56)) + 48);
          v97 = *v96 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (_QWORD *)v97 == v96 )
          {
            v99 = 0;
          }
          else
          {
            if ( !v97 )
              goto LABEL_163;
            v98 = *(unsigned __int8 *)(v97 - 24);
            v99 = 0;
            v100 = v97 - 24;
            if ( (unsigned int)(v98 - 30) < 0xB )
              v99 = v100;
          }
          v174 = sub_BD5C60(v99);
          v175 = &v183;
          v176 = v184;
          v183 = &unk_49DA100;
          v167 = &v169;
          v184[0] = &unk_49DA0B0;
          v168 = 0x200000000LL;
          v177 = 0;
          v178 = 0;
          v179 = 512;
          v180 = 7;
          v181 = 0;
          v182 = 0;
          v171 = 0;
          v172 = 0;
          LOWORD(v173) = 0;
          sub_D5F1F0((__int64)&v167, v99);
          v161[0] = *((_QWORD *)v157 + 6);
          if ( v161[0] && (sub_B96E90((__int64)v161, v161[0], 1), (v102 = v161[0]) != 0) )
          {
            v103 = v167;
            v104 = v168;
            v105 = &v167[2 * (unsigned int)v168];
            if ( v167 != v105 )
            {
              while ( 1 )
              {
                v101 = *(unsigned int *)v103;
                if ( !(_DWORD)v101 )
                  break;
                v103 += 2;
                if ( v105 == v103 )
                  goto LABEL_150;
              }
              v103[1] = (__int64 *)v161[0];
LABEL_123:
              sub_B91220((__int64)v161, v102);
LABEL_124:
              v160 = 1;
              v158[0] = "lsr.chain";
              v159 = 3;
              if ( v95 == *(_QWORD *)(v84 + 8) )
              {
                v107 = v84;
              }
              else if ( *(_BYTE *)v84 > 0x15u )
              {
                v162 = 257;
                v107 = sub_B52210(v84, v95, (__int64)v161, 0, 0);
                (*(void (__fastcall **)(_QWORD *, __int64, _QWORD *, __int64, __int64))(*v176 + 16LL))(
                  v176,
                  v107,
                  v158,
                  v172,
                  v173);
                v118 = 2LL * (unsigned int)v168;
                v155 = &v167[v118];
                if ( v167 != &v167[v118] )
                {
                  v141 = v84;
                  v119 = (__int64)v167;
                  v120 = v82;
                  v121 = a1;
                  v122 = v83;
                  v123 = v107;
                  v124 = v120;
                  do
                  {
                    v125 = *(_QWORD *)(v119 + 8);
                    v126 = *(_DWORD *)v119;
                    v119 += 16;
                    sub_B99FD0(v123, v126, v125);
                  }
                  while ( v155 != (__int64 **)v119 );
                  v127 = v124;
                  v84 = v141;
                  v107 = v123;
                  v83 = v122;
                  a1 = v121;
                  v82 = v127;
                }
              }
              else
              {
                v106 = (_BYTE *)(*((__int64 (__fastcall **)(void **, __int64, __int64))*v175 + 17))(v175, v84, v95);
                v107 = (__int64)v106;
                if ( *v106 > 0x1Cu )
                {
                  (*(void (__fastcall **)(_QWORD *, _BYTE *, _QWORD *, __int64, __int64))(*v176 + 16LL))(
                    v176,
                    v106,
                    v158,
                    v172,
                    v173);
                  v108 = 2LL * (unsigned int)v168;
                  v154 = &v167[v108];
                  if ( v167 != &v167[v108] )
                  {
                    v140 = v84;
                    v109 = v107;
                    v110 = (__int64)v167;
                    do
                    {
                      v111 = *(_QWORD *)(v110 + 8);
                      v112 = *(_DWORD *)v110;
                      v110 += 16;
                      sub_B99FD0(v109, v112, v111);
                    }
                    while ( v154 != (__int64 **)v110 );
                    v107 = v109;
                    v84 = v140;
                  }
                }
              }
              nullsub_61();
              v183 = &unk_49DA100;
              nullsub_63();
              if ( v167 != &v169 )
                _libc_free((unsigned __int64)v167);
              v94 = (__int64)v157;
LABEL_138:
              sub_BD2ED0(v83, v94, v107);
              sub_2855140(a3, (__int64 *)&v157, v113, v114, v115, v116);
LABEL_98:
              v85 = *(_QWORD *)(v83 + 32);
              if ( !v85 )
                goto LABEL_163;
              v83 = 0;
              if ( *(_BYTE *)(v85 - 24) == 84 )
                v83 = v85 - 24;
              if ( v82 == v83 )
                goto LABEL_31;
              continue;
            }
LABEL_150:
            if ( (unsigned int)v168 >= (unsigned __int64)HIDWORD(v168) )
            {
              v128 = v129 & 0xFFFFFFFF00000000LL;
              v129 &= 0xFFFFFFFF00000000LL;
              if ( HIDWORD(v168) < (unsigned __int64)(unsigned int)v168 + 1 )
              {
                v142 = v128;
                v156 = v161[0];
                sub_C8D5F0((__int64)&v167, &v169, (unsigned int)v168 + 1LL, 0x10u, (__int64)&v167, v101);
                v128 = v142;
                v102 = v156;
                v105 = &v167[2 * (unsigned int)v168];
              }
              *v105 = (__int64 *)v128;
              v105[1] = (__int64 *)v102;
              v102 = v161[0];
              LODWORD(v168) = v168 + 1;
            }
            else
            {
              if ( v105 )
              {
                *(_DWORD *)v105 = 0;
                v105[1] = (__int64 *)v102;
                v104 = v168;
                v102 = v161[0];
              }
              LODWORD(v168) = v104 + 1;
            }
          }
          else
          {
            sub_93FB40((__int64)&v167, 0);
            v102 = v161[0];
          }
          if ( v102 )
            goto LABEL_123;
          goto LABEL_124;
        }
        break;
      }
      v107 = v84;
      goto LABEL_138;
    }
  }
LABEL_31:
  if ( v163 != v166 )
    _libc_free((unsigned __int64)v163);
}
