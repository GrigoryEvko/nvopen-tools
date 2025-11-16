// Function: sub_23D89F0
// Address: 0x23d89f0
//
void __fastcall sub_23D89F0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 *v3; // rbx
  unsigned __int8 *v4; // r15
  __int64 v5; // rax
  const char *v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  const char *v9; // r12
  unsigned int *v10; // rax
  int v11; // ecx
  unsigned int *v12; // rdx
  unsigned int v13; // r12d
  unsigned __int64 *v14; // rcx
  unsigned __int8 *v15; // rdx
  unsigned __int8 *v16; // r10
  __int64 (__fastcall *v17)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *); // rax
  __int64 v18; // rax
  __int64 v19; // r13
  int v20; // r12d
  __int64 v21; // r12
  unsigned int *v22; // rsi
  unsigned int *v23; // r12
  unsigned int *v24; // rbx
  __int64 v25; // rdx
  unsigned int v26; // esi
  int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // rdx
  int v30; // ecx
  __int64 *v31; // rcx
  __int64 v32; // rax
  __int64 *v33; // rsi
  __int64 v34; // r9
  __int64 v35; // rsi
  __int64 *v36; // rdi
  int v37; // ecx
  __int64 *v38; // rdx
  __int64 v39; // r8
  __int64 *v40; // rax
  __int64 v41; // r13
  unsigned __int64 **v42; // r14
  unsigned __int64 *v43; // r13
  unsigned __int64 *v44; // rcx
  __int64 v45; // rbx
  __int64 v46; // rax
  __int64 v47; // r9
  unsigned __int64 *v48; // rbx
  __int64 v49; // rax
  unsigned __int64 *v50; // r12
  __int64 v51; // r15
  unsigned __int64 *v52; // r14
  __int64 v53; // r12
  unsigned __int64 *v54; // r15
  unsigned __int64 v55; // rbx
  __int64 v56; // rdx
  int v57; // eax
  int v58; // eax
  unsigned int v59; // esi
  __int64 v60; // rax
  __int64 v61; // rsi
  __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // rdi
  unsigned __int8 *v65; // r12
  __int64 v66; // r13
  unsigned __int64 v67; // rbx
  __int64 v68; // rax
  _QWORD *v69; // r14
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // r8
  int v73; // r9d
  unsigned int v74; // ecx
  __int64 v75; // rdx
  _QWORD *v76; // rsi
  __int64 v77; // rcx
  __int64 v78; // rax
  __int64 v79; // rdx
  char *v80; // rax
  char *v81; // rsi
  unsigned int v82; // r15d
  __int64 *v83; // rdi
  __int64 v84; // r11
  __int64 v85; // rsi
  _QWORD *v86; // rdi
  __int64 v87; // rsi
  unsigned __int64 v88; // r8
  __int64 v89; // rdx
  __m128i v90; // xmm0
  __int64 v91; // rdx
  unsigned __int64 v92; // rax
  __int64 v93; // r12
  __int64 i; // rbx
  _QWORD *v95; // rdi
  unsigned __int64 *v96; // rcx
  _BYTE *v97; // r12
  unsigned __int8 *v98; // rdx
  unsigned __int8 *v99; // r10
  __int64 (__fastcall *v100)(__int64, _BYTE *, unsigned __int8 *); // rax
  __int64 v101; // rax
  _QWORD *v102; // rax
  unsigned int *v103; // r12
  unsigned int *v104; // rbx
  __int64 v105; // rdx
  unsigned int v106; // esi
  unsigned __int8 *v107; // rdx
  __int64 v108; // r12
  __int64 v109; // r13
  unsigned __int8 *v110; // rdx
  __int64 v111; // rax
  unsigned __int64 *v112; // rcx
  _BYTE *v113; // r12
  unsigned __int8 *v114; // rdx
  _BYTE *v115; // r9
  unsigned __int8 *v116; // rdx
  unsigned __int8 *v117; // r10
  __int64 (__fastcall *v118)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v119; // rax
  __int64 v120; // rax
  _QWORD *v121; // rax
  __int64 v122; // r9
  unsigned int *v123; // r12
  unsigned int *v124; // rbx
  __int64 v125; // rdx
  unsigned int v126; // esi
  __int64 v127; // rcx
  int v128; // r12d
  __int64 v129; // rsi
  int v130; // edx
  __int64 v131; // rax
  __int64 v132; // r10
  __int64 v133; // r8
  __int64 v134; // r9
  __int64 v135; // r12
  unsigned int *v136; // r12
  unsigned int *v137; // rbx
  __int64 v138; // rdx
  unsigned int v139; // esi
  __int64 v140; // rax
  unsigned __int64 v141; // rdx
  unsigned __int8 **v142; // rax
  _QWORD *v143; // rcx
  _QWORD *v144; // rsi
  _QWORD *v145; // rdx
  unsigned __int64 v146; // rdi
  int v147; // edx
  int v148; // edi
  unsigned __int8 *v149; // rcx
  int v150; // edi
  int v151; // r12d
  unsigned __int64 v152; // rsi
  unsigned __int64 v153; // r13
  unsigned __int64 v154; // rsi
  __int64 v155; // rax
  __int64 v156; // rax
  __int64 v157; // rax
  __int64 v158; // [rsp-10h] [rbp-1E0h]
  __int64 v159; // [rsp+8h] [rbp-1C8h]
  unsigned __int8 *v160; // [rsp+10h] [rbp-1C0h]
  unsigned __int8 *v161; // [rsp+10h] [rbp-1C0h]
  __int64 v162; // [rsp+10h] [rbp-1C0h]
  unsigned __int8 *v163; // [rsp+10h] [rbp-1C0h]
  unsigned __int8 *v164; // [rsp+10h] [rbp-1C0h]
  __int64 v165; // [rsp+18h] [rbp-1B8h]
  unsigned __int8 *v166; // [rsp+30h] [rbp-1A0h]
  __int64 *v167; // [rsp+30h] [rbp-1A0h]
  unsigned __int8 *v168; // [rsp+30h] [rbp-1A0h]
  __int64 v169; // [rsp+30h] [rbp-1A0h]
  __int64 *v170; // [rsp+30h] [rbp-1A0h]
  _BYTE *v171; // [rsp+30h] [rbp-1A0h]
  __int64 *v172; // [rsp+30h] [rbp-1A0h]
  __int64 v173; // [rsp+30h] [rbp-1A0h]
  __int64 *v174; // [rsp+30h] [rbp-1A0h]
  __int64 v175; // [rsp+30h] [rbp-1A0h]
  _BYTE *v176; // [rsp+30h] [rbp-1A0h]
  unsigned __int8 *v177; // [rsp+30h] [rbp-1A0h]
  __int64 *v178; // [rsp+40h] [rbp-190h]
  unsigned __int64 **v179; // [rsp+48h] [rbp-188h]
  unsigned __int64 **v180; // [rsp+50h] [rbp-180h]
  unsigned __int64 *v181; // [rsp+58h] [rbp-178h]
  int v182; // [rsp+58h] [rbp-178h]
  unsigned __int64 **v184; // [rsp+60h] [rbp-170h]
  __int64 v186; // [rsp+70h] [rbp-160h]
  __int64 v187; // [rsp+78h] [rbp-158h]
  _BYTE v188[32]; // [rsp+80h] [rbp-150h] BYREF
  __int16 v189; // [rsp+A0h] [rbp-130h]
  const char *v190[4]; // [rsp+B0h] [rbp-120h] BYREF
  __int16 v191; // [rsp+D0h] [rbp-100h]
  unsigned __int64 **v192; // [rsp+E0h] [rbp-F0h] BYREF
  __int64 v193; // [rsp+E8h] [rbp-E8h]
  _BYTE v194[32]; // [rsp+F0h] [rbp-E0h] BYREF
  unsigned int *v195; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v196; // [rsp+118h] [rbp-B8h]
  _BYTE v197[32]; // [rsp+120h] [rbp-B0h] BYREF
  __int64 v198; // [rsp+140h] [rbp-90h]
  unsigned __int8 *v199; // [rsp+148h] [rbp-88h]
  __int64 v200; // [rsp+150h] [rbp-80h]
  __int64 v201; // [rsp+158h] [rbp-78h]
  void **v202; // [rsp+160h] [rbp-70h]
  void **v203; // [rsp+168h] [rbp-68h]
  __int64 v204; // [rsp+170h] [rbp-60h]
  int v205; // [rsp+178h] [rbp-58h]
  __int16 v206; // [rsp+17Ch] [rbp-54h]
  char v207; // [rsp+17Eh] [rbp-52h]
  __int64 v208; // [rsp+180h] [rbp-50h]
  __int64 v209; // [rsp+188h] [rbp-48h]
  void *v210; // [rsp+190h] [rbp-40h] BYREF
  void *v211; // [rsp+198h] [rbp-38h] BYREF

  v2 = *(unsigned int *)(a1 + 128);
  v3 = *(__int64 **)(a1 + 120);
  v192 = (unsigned __int64 **)v194;
  v193 = 0x200000000LL;
  v178 = &v3[3 * v2];
  if ( v3 == v178 )
    goto LABEL_77;
  while ( 2 )
  {
    v4 = (unsigned __int8 *)*v3;
    v5 = sub_BD5C60(*v3);
    v203 = &v211;
    v201 = v5;
    v204 = 0;
    v202 = &v210;
    v195 = (unsigned int *)v197;
    v210 = &unk_49DA100;
    v196 = 0x200000000LL;
    v205 = 0;
    v198 = 0;
    v199 = 0;
    v206 = 512;
    v207 = 7;
    v208 = 0;
    v209 = 0;
    LOWORD(v200) = 0;
    v211 = &unk_49DA0B0;
    v198 = *((_QWORD *)v4 + 5);
    v199 = v4 + 24;
    v6 = *(const char **)sub_B46C60((__int64)v4);
    v190[0] = v6;
    if ( v6 && (sub_B96E90((__int64)v190, (__int64)v6, 1), (v9 = v190[0]) != 0) )
    {
      v10 = v195;
      v11 = v196;
      v12 = &v195[4 * (unsigned int)v196];
      if ( v195 != v12 )
      {
        while ( 1 )
        {
          v7 = *v10;
          if ( !(_DWORD)v7 )
            break;
          v10 += 4;
          if ( v12 == v10 )
            goto LABEL_161;
        }
        *((const char **)v10 + 1) = v190[0];
        goto LABEL_9;
      }
LABEL_161:
      if ( (unsigned int)v196 >= (unsigned __int64)HIDWORD(v196) )
      {
        v152 = (unsigned int)v196 + 1LL;
        v153 = v165 & 0xFFFFFFFF00000000LL;
        v165 &= 0xFFFFFFFF00000000LL;
        if ( HIDWORD(v196) < v152 )
        {
          sub_C8D5F0((__int64)&v195, v197, v152, 0x10u, v7, v8);
          v12 = &v195[4 * (unsigned int)v196];
        }
        *(_QWORD *)v12 = v153;
        *((_QWORD *)v12 + 1) = v9;
        v9 = v190[0];
        LODWORD(v196) = v196 + 1;
      }
      else
      {
        if ( v12 )
        {
          *v12 = 0;
          *((_QWORD *)v12 + 1) = v9;
          v11 = v196;
          v9 = v190[0];
        }
        LODWORD(v196) = v11 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v195, 0);
      v9 = v190[0];
    }
    if ( v9 )
LABEL_9:
      sub_B91220((__int64)v190, (__int64)v9);
    v13 = *v4 - 29;
    switch ( *v4 )
    {
      case '*':
      case ',':
      case '.':
      case '0':
      case '3':
      case '6':
      case '7':
      case '8':
      case '9':
      case ':':
      case ';':
        if ( (v4[7] & 0x40) != 0 )
          v14 = (unsigned __int64 *)*((_QWORD *)v4 - 1);
        else
          v14 = (unsigned __int64 *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        v166 = (unsigned __int8 *)sub_23D88E0(a1, *v14, a2);
        if ( (v4[7] & 0x40) != 0 )
          v15 = (unsigned __int8 *)*((_QWORD *)v4 - 1);
        else
          v15 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        v16 = (unsigned __int8 *)sub_23D88E0(a1, *((_QWORD *)v15 + 4), a2);
        v189 = 257;
        v17 = (__int64 (__fastcall *)(__int64, unsigned int, unsigned __int8 *, unsigned __int8 *))*((_QWORD *)*v202 + 2);
        if ( v17 == sub_9202E0 )
        {
          if ( *v166 > 0x15u || *v16 > 0x15u )
            goto LABEL_22;
          v160 = v16;
          if ( (unsigned __int8)sub_AC47B0(v13) )
            v18 = sub_AD5570(v13, (__int64)v166, v160, 0, 0);
          else
            v18 = sub_AABE40(v13, v166, v160);
          v16 = v160;
          v19 = v18;
        }
        else
        {
          v164 = v16;
          v157 = v17((__int64)v202, v13, v166, v16);
          v16 = v164;
          v19 = v157;
        }
        if ( v19 )
          goto LABEL_30;
LABEL_22:
        v191 = 257;
        v19 = sub_B504D0(v13, (__int64)v166, (__int64)v16, (__int64)v190, 0, 0);
        if ( (unsigned __int8)sub_920620(v19) )
        {
          v20 = v205;
          if ( v204 )
            sub_B99FD0(v19, 3u, v204);
          sub_B45150(v19, v20);
        }
        (*((void (__fastcall **)(void **, __int64, _BYTE *, unsigned __int8 *, __int64))*v203 + 2))(
          v203,
          v19,
          v188,
          v199,
          v200);
        v21 = 4LL * (unsigned int)v196;
        v22 = &v195[v21];
        if ( v195 != &v195[v21] )
        {
          v167 = v3;
          v23 = v195;
          v24 = v22;
          do
          {
            v25 = *((_QWORD *)v23 + 1);
            v26 = *v23;
            v23 += 4;
            sub_B99FD0(v19, v26, v25);
          }
          while ( v24 != v23 );
          v3 = v167;
        }
LABEL_30:
        v27 = *v4;
        if ( ((unsigned int)(v27 - 48) <= 1 || (unsigned __int8)(v27 - 55) <= 1u) && *(_BYTE *)v19 > 0x1Cu )
          sub_B448B0(v19, (v4[1] & 2) != 0);
        goto LABEL_52;
      case 'C':
      case 'D':
      case 'E':
        v28 = *((_QWORD *)v4 + 1);
        v29 = (__int64)a2;
        v30 = *(unsigned __int8 *)(v28 + 8);
        if ( (unsigned int)(v30 - 17) <= 1 )
        {
          BYTE4(v186) = (_BYTE)v30 == 18;
          LODWORD(v186) = *(_DWORD *)(v28 + 32);
          v29 = sub_BCE1B0(a2, v186);
        }
        if ( (v4[7] & 0x40) != 0 )
        {
          v31 = (__int64 *)*((_QWORD *)v4 - 1);
          v32 = *v31;
          if ( *(_QWORD *)(*v31 + 8) == v29 )
            goto LABEL_190;
        }
        else
        {
          v149 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
          v32 = *(_QWORD *)v149;
          if ( *(_QWORD *)(*(_QWORD *)v149 + 8LL) == v29 )
          {
LABEL_190:
            v3[2] = v32;
            goto LABEL_54;
          }
        }
        v191 = 257;
        if ( (v4[7] & 0x40) != 0 )
          v33 = (__int64 *)*((_QWORD *)v4 - 1);
        else
          v33 = (__int64 *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        v19 = sub_921630(&v195, *v33, v29, v13 == 40, (__int64)v190);
        v35 = *(unsigned int *)(a1 + 40);
        v36 = *(__int64 **)(a1 + 32);
        v37 = *(_DWORD *)(a1 + 40);
        v38 = &v36[v35];
        v39 = (8 * v35) >> 3;
        if ( (8 * v35) >> 5 )
        {
          v40 = &v36[4 * ((8 * v35) >> 5)];
          while ( v4 != (unsigned __int8 *)*v36 )
          {
            if ( v4 == (unsigned __int8 *)v36[1] )
            {
              ++v36;
              break;
            }
            if ( v4 == (unsigned __int8 *)v36[2] )
            {
              v36 += 2;
              break;
            }
            if ( v4 == (unsigned __int8 *)v36[3] )
            {
              v36 += 3;
              break;
            }
            v36 += 4;
            if ( v40 == v36 )
            {
              v39 = v38 - v36;
              goto LABEL_208;
            }
          }
LABEL_47:
          if ( v38 != v36 )
          {
            if ( *(_BYTE *)v19 == 67 )
            {
              *v36 = v19;
            }
            else
            {
              if ( v38 != v36 + 1 )
              {
                memmove(v36, v36 + 1, (char *)v38 - (char *)(v36 + 1));
                v37 = *(_DWORD *)(a1 + 40);
              }
              *(_DWORD *)(a1 + 40) = v37 - 1;
            }
            goto LABEL_52;
          }
LABEL_211:
          if ( *(_BYTE *)v19 != 67 )
            goto LABEL_52;
          goto LABEL_212;
        }
LABEL_208:
        switch ( v39 )
        {
          case 2LL:
            goto LABEL_223;
          case 3LL:
            if ( v4 == (unsigned __int8 *)*v36 )
              goto LABEL_47;
            ++v36;
LABEL_223:
            if ( v4 == (unsigned __int8 *)*v36 )
              goto LABEL_47;
            ++v36;
            break;
          case 1LL:
            break;
          default:
            goto LABEL_211;
        }
        if ( v4 == (unsigned __int8 *)*v36 )
          goto LABEL_47;
        if ( *(_BYTE *)v19 != 67 )
          goto LABEL_52;
LABEL_212:
        v154 = v35 + 1;
        if ( v154 > *(unsigned int *)(a1 + 44) )
        {
          sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v154, 8u, a1 + 48, v34);
          v38 = (__int64 *)(*(_QWORD *)(a1 + 32) + 8LL * *(unsigned int *)(a1 + 40));
        }
        *v38 = v19;
        ++*(_DWORD *)(a1 + 40);
LABEL_52:
        v3[2] = v19;
        if ( *(_BYTE *)v19 > 0x1Cu )
          sub_BD6B90((unsigned __int8 *)v19, v4);
LABEL_54:
        nullsub_61();
        v210 = &unk_49DA100;
        nullsub_63();
        if ( v195 != (unsigned int *)v197 )
          _libc_free((unsigned __int64)v195);
        v3 += 3;
        if ( v178 != v3 )
          continue;
        v41 = 2LL * (unsigned int)v193;
        v180 = &v192[v41];
        if ( &v192[v41] != v192 )
        {
          v42 = v192;
          do
          {
            v43 = *v42;
            v44 = (unsigned __int64 *)*(*v42 - 1);
            v45 = 4LL * *((unsigned int *)*v42 + 18);
            v46 = *((_DWORD *)*v42 + 1) & 0x7FFFFFF;
            v47 = v45 * 8 + 8 * v46;
            v48 = &v44[v45];
            v49 = 32 * v46;
            v50 = (unsigned __int64 *)((char *)v44 + v47);
            if ( (*((_BYTE *)*v42 + 7) & 0x40) != 0 )
              v43 = &v44[(unsigned __int64)v49 / 8];
            else
              v44 = &v43[v49 / 0xFFFFFFFFFFFFFFF8LL];
            if ( v50 != v48 && v43 != v44 )
            {
              v51 = (__int64)v42[1];
              v181 = v50;
              v179 = v42;
              v52 = v48;
              v53 = v51;
              v54 = v44;
              do
              {
                v55 = *v52;
                v56 = sub_23D88E0(a1, *v54, a2);
                v57 = *(_DWORD *)(v53 + 4) & 0x7FFFFFF;
                if ( v57 == *(_DWORD *)(v53 + 72) )
                {
                  v175 = v56;
                  sub_B48D90(v53);
                  v56 = v175;
                  v57 = *(_DWORD *)(v53 + 4) & 0x7FFFFFF;
                }
                v58 = (v57 + 1) & 0x7FFFFFF;
                v59 = v58 | *(_DWORD *)(v53 + 4) & 0xF8000000;
                v60 = *(_QWORD *)(v53 - 8) + 32LL * (unsigned int)(v58 - 1);
                *(_DWORD *)(v53 + 4) = v59;
                if ( *(_QWORD *)v60 )
                {
                  v61 = *(_QWORD *)(v60 + 8);
                  **(_QWORD **)(v60 + 16) = v61;
                  if ( v61 )
                    *(_QWORD *)(v61 + 16) = *(_QWORD *)(v60 + 16);
                }
                *(_QWORD *)v60 = v56;
                if ( v56 )
                {
                  v62 = *(_QWORD *)(v56 + 16);
                  *(_QWORD *)(v60 + 8) = v62;
                  if ( v62 )
                    *(_QWORD *)(v62 + 16) = v60 + 8;
                  *(_QWORD *)(v60 + 16) = v56 + 16;
                  *(_QWORD *)(v56 + 16) = v60;
                }
                v54 += 4;
                ++v52;
                *(_QWORD *)(*(_QWORD *)(v53 - 8)
                          + 32LL * *(unsigned int *)(v53 + 72)
                          + 8LL * ((*(_DWORD *)(v53 + 4) & 0x7FFFFFFu) - 1)) = v55;
              }
              while ( v54 != v43 && v181 != v52 );
              v42 = v179;
            }
            v42 += 2;
          }
          while ( v180 != v42 );
        }
        break;
      case 'T':
        v189 = 257;
        v127 = *((_QWORD *)v4 + 1);
        v128 = *((_DWORD *)v4 + 1) & 0x7FFFFFF;
        v129 = (__int64)a2;
        v130 = *(unsigned __int8 *)(v127 + 8);
        if ( (unsigned int)(v130 - 17) <= 1 )
        {
          BYTE4(v187) = (_BYTE)v130 == 18;
          LODWORD(v187) = *(_DWORD *)(v127 + 32);
          v129 = sub_BCE1B0(a2, v187);
        }
        v191 = 257;
        v131 = sub_BD2DA0(80);
        v19 = v131;
        if ( v131 )
        {
          v173 = v131;
          sub_B44260(v131, v129, 55, 0x8000000u, 0, 0);
          *(_DWORD *)(v19 + 72) = v128;
          sub_BD6B50((unsigned __int8 *)v19, v190);
          sub_BD2A10(v19, *(_DWORD *)(v19 + 72), 1);
          v132 = v173;
        }
        else
        {
          v132 = 0;
        }
        if ( (unsigned __int8)sub_920620(v132) )
        {
          v151 = v205;
          if ( v204 )
            sub_B99FD0(v19, 3u, v204);
          sub_B45150(v19, v151);
        }
        (*((void (__fastcall **)(void **, __int64, _BYTE *, unsigned __int8 *, __int64))*v203 + 2))(
          v203,
          v19,
          v188,
          v199,
          v200);
        v135 = 4LL * (unsigned int)v196;
        if ( v195 != &v195[v135] )
        {
          v174 = v3;
          v136 = &v195[v135];
          v137 = v195;
          do
          {
            v138 = *((_QWORD *)v137 + 1);
            v139 = *v137;
            v137 += 4;
            sub_B99FD0(v19, v139, v138);
          }
          while ( v136 != v137 );
          v3 = v174;
        }
        v140 = (unsigned int)v193;
        v141 = (unsigned int)v193 + 1LL;
        if ( v141 > HIDWORD(v193) )
        {
          sub_C8D5F0((__int64)&v192, v194, v141, 0x10u, v133, v134);
          v140 = (unsigned int)v193;
        }
        v142 = (unsigned __int8 **)&v192[2 * v140];
        *v142 = v4;
        v142[1] = (unsigned __int8 *)v19;
        LODWORD(v193) = v193 + 1;
        goto LABEL_52;
      case 'V':
        if ( (v4[7] & 0x40) != 0 )
          v107 = (unsigned __int8 *)*((_QWORD *)v4 - 1);
        else
          v107 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        v108 = *(_QWORD *)v107;
        v109 = sub_23D88E0(a1, *((_QWORD *)v107 + 4), a2);
        if ( (v4[7] & 0x40) != 0 )
          v110 = (unsigned __int8 *)*((_QWORD *)v4 - 1);
        else
          v110 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        v111 = sub_23D88E0(a1, *((_QWORD *)v110 + 8), a2);
        v191 = 257;
        v19 = sub_B36550(&v195, v108, v109, v111, (__int64)v190, 0);
        goto LABEL_52;
      case 'Z':
        if ( (v4[7] & 0x40) != 0 )
          v96 = (unsigned __int64 *)*((_QWORD *)v4 - 1);
        else
          v96 = (unsigned __int64 *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        v97 = (_BYTE *)sub_23D88E0(a1, *v96, a2);
        if ( (v4[7] & 0x40) != 0 )
          v98 = (unsigned __int8 *)*((_QWORD *)v4 - 1);
        else
          v98 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        v99 = (unsigned __int8 *)*((_QWORD *)v98 + 4);
        v189 = 257;
        v100 = (__int64 (__fastcall *)(__int64, _BYTE *, unsigned __int8 *))*((_QWORD *)*v202 + 12);
        if ( v100 == sub_948070 )
        {
          if ( *v97 > 0x15u || *v99 > 0x15u )
            goto LABEL_114;
          v168 = v99;
          v101 = sub_AD5840((__int64)v97, v99, 0);
          v99 = v168;
          v19 = v101;
        }
        else
        {
          v177 = v99;
          v156 = v100((__int64)v202, v97, v99);
          v99 = v177;
          v19 = v156;
        }
        if ( v19 )
          goto LABEL_52;
LABEL_114:
        v169 = (__int64)v99;
        v191 = 257;
        v102 = sub_BD2C40(72, 2u);
        v19 = (__int64)v102;
        if ( v102 )
          sub_B4DE80((__int64)v102, (__int64)v97, v169, (__int64)v190, 0, 0);
        (*((void (__fastcall **)(void **, __int64, _BYTE *, unsigned __int8 *, __int64))*v203 + 2))(
          v203,
          v19,
          v188,
          v199,
          v200);
        v103 = v195;
        if ( v195 != &v195[4 * (unsigned int)v196] )
        {
          v170 = v3;
          v104 = &v195[4 * (unsigned int)v196];
          do
          {
            v105 = *((_QWORD *)v103 + 1);
            v106 = *v103;
            v103 += 4;
            sub_B99FD0(v19, v106, v105);
          }
          while ( v104 != v103 );
          v3 = v170;
        }
        goto LABEL_52;
      case '[':
        if ( (v4[7] & 0x40) != 0 )
          v112 = (unsigned __int64 *)*((_QWORD *)v4 - 1);
        else
          v112 = (unsigned __int64 *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        v113 = (_BYTE *)sub_23D88E0(a1, *v112, a2);
        if ( (v4[7] & 0x40) != 0 )
          v114 = (unsigned __int8 *)*((_QWORD *)v4 - 1);
        else
          v114 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        v115 = (_BYTE *)sub_23D88E0(a1, *((_QWORD *)v114 + 4), a2);
        if ( (v4[7] & 0x40) != 0 )
          v116 = (unsigned __int8 *)*((_QWORD *)v4 - 1);
        else
          v116 = &v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
        v117 = (unsigned __int8 *)*((_QWORD *)v116 + 8);
        v189 = 257;
        v118 = (__int64 (__fastcall *)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))*((_QWORD *)*v202 + 13);
        if ( v118 == sub_948040 )
        {
          v119 = 0;
          if ( *v113 <= 0x15u )
            v119 = (__int64)v113;
          if ( *v115 > 0x15u || *v117 > 0x15u || !v119 )
            goto LABEL_139;
          v161 = v117;
          v171 = v115;
          v120 = sub_AD5A90(v119, v115, v117, 0);
          v115 = v171;
          v117 = v161;
          v19 = v120;
        }
        else
        {
          v163 = v117;
          v176 = v115;
          v155 = v118((__int64)v202, v113, v115, v117);
          v117 = v163;
          v115 = v176;
          v19 = v155;
        }
        if ( v19 )
          goto LABEL_52;
LABEL_139:
        v159 = (__int64)v117;
        v162 = (__int64)v115;
        v191 = 257;
        v121 = sub_BD2C40(72, 3u);
        v122 = v162;
        v19 = (__int64)v121;
        if ( v121 )
        {
          sub_B4DFA0((__int64)v121, (__int64)v113, v162, v159, (__int64)v190, v162, 0, 0);
          v122 = v158;
        }
        (*((void (__fastcall **)(void **, __int64, _BYTE *, unsigned __int8 *, __int64, __int64))*v203 + 2))(
          v203,
          v19,
          v188,
          v199,
          v200,
          v122);
        v123 = v195;
        if ( v195 != &v195[4 * (unsigned int)v196] )
        {
          v172 = v3;
          v124 = &v195[4 * (unsigned int)v196];
          do
          {
            v125 = *((_QWORD *)v123 + 1);
            v126 = *v123;
            v123 += 4;
            sub_B99FD0(v19, v126, v125);
          }
          while ( v124 != v123 );
          v3 = v172;
        }
        goto LABEL_52;
      default:
        BUG();
    }
    break;
  }
LABEL_77:
  v63 = sub_23D88E0(a1, *(_QWORD *)(*(_QWORD *)(a1 + 80) - 32LL), a2);
  v64 = *(_QWORD *)(a1 + 80);
  v65 = (unsigned __int8 *)v63;
  v66 = *(_QWORD *)(v64 + 8);
  if ( *(_QWORD *)(v63 + 8) != v66 )
  {
    sub_23D0AB0((__int64)&v195, *(_QWORD *)(a1 + 80), 0, 0, 0);
    v191 = 257;
    v65 = (unsigned __int8 *)sub_921630(&v195, (__int64)v65, v66, 0, (__int64)v190);
    if ( *v65 > 0x1Cu )
      sub_BD6B90(v65, *(unsigned __int8 **)(a1 + 80));
    nullsub_61();
    v210 = &unk_49DA100;
    nullsub_63();
    if ( v195 != (unsigned int *)v197 )
      _libc_free((unsigned __int64)v195);
    v64 = *(_QWORD *)(a1 + 80);
  }
  sub_BD84D0(v64, (__int64)v65);
  sub_B43D60(*(_QWORD **)(a1 + 80));
  v67 = (unsigned __int64)v192;
  v68 = 2LL * (unsigned int)v193;
  if ( &v192[v68] != v192 )
  {
    v184 = &v192[v68];
    do
    {
      v69 = *(_QWORD **)v67;
      v70 = sub_ACADE0(*(__int64 ***)(*(_QWORD *)v67 + 8LL));
      sub_BD84D0((__int64)v69, v70);
      v71 = *(unsigned int *)(a1 + 112);
      v72 = *(_QWORD *)(a1 + 96);
      if ( (_DWORD)v71 )
      {
        v73 = v71 - 1;
        v74 = (v71 - 1) & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
        v75 = v72 + 16LL * v74;
        v76 = *(_QWORD **)v75;
        if ( v69 == *(_QWORD **)v75 )
        {
LABEL_87:
          if ( v75 != v72 + 16 * v71 )
          {
            v77 = *(_QWORD *)(a1 + 120);
            v78 = 3LL * *(unsigned int *)(v75 + 8);
            v79 = *(unsigned int *)(a1 + 128);
            v80 = (char *)(v77 + 8 * v78);
            v81 = (char *)(v77 + 24 * v79);
            if ( v81 != v80 )
            {
              v82 = v73 & (((unsigned int)*(_QWORD *)v80 >> 9) ^ ((unsigned int)*(_QWORD *)v80 >> 4));
              v83 = (__int64 *)(v72 + 16LL * v82);
              v84 = *v83;
              if ( *(_QWORD *)v80 == *v83 )
              {
LABEL_90:
                *v83 = -8192;
                v85 = *(unsigned int *)(a1 + 128);
                --*(_DWORD *)(a1 + 104);
                v77 = *(_QWORD *)(a1 + 120);
                ++*(_DWORD *)(a1 + 108);
                LODWORD(v79) = v85;
                v81 = (char *)(v77 + 24 * v85);
              }
              else
              {
                v150 = 1;
                while ( v84 != -4096 )
                {
                  v82 = v73 & (v150 + v82);
                  v182 = v150 + 1;
                  v83 = (__int64 *)(v72 + 16LL * v82);
                  v84 = *v83;
                  if ( *(_QWORD *)v80 == *v83 )
                    goto LABEL_90;
                  v150 = v182;
                }
              }
              v86 = v80 + 24;
              v87 = v81 - (v80 + 24);
              v88 = 0xAAAAAAAAAAAAAAABLL * (v87 >> 3);
              if ( v87 > 0 )
              {
                do
                {
                  v89 = *v86;
                  v90 = _mm_loadu_si128((const __m128i *)(v86 + 1));
                  v86 += 3;
                  *(v86 - 6) = v89;
                  *(__m128i *)(v86 - 5) = v90;
                  --v88;
                }
                while ( v88 );
                v77 = *(_QWORD *)(a1 + 120);
                LODWORD(v79) = *(_DWORD *)(a1 + 128);
              }
              v91 = (unsigned int)(v79 - 1);
              *(_DWORD *)(a1 + 128) = v91;
              if ( v80 != (char *)(v77 + 24 * v91) )
              {
                v92 = 0xAAAAAAAAAAAAAAABLL * ((__int64)&v80[-v77] >> 3);
                if ( *(_DWORD *)(a1 + 104) )
                {
                  v143 = *(_QWORD **)(a1 + 96);
                  v144 = &v143[2 * *(unsigned int *)(a1 + 112)];
                  if ( v143 != v144 )
                  {
                    while ( 1 )
                    {
                      v145 = v143;
                      if ( *v143 != -8192 && *v143 != -4096 )
                        break;
                      v143 += 2;
                      if ( v144 == v143 )
                        goto LABEL_96;
                    }
                    if ( v143 != v144 )
                    {
                      do
                      {
                        v146 = *((unsigned int *)v145 + 2);
                        if ( v92 < v146 )
                          *((_DWORD *)v145 + 2) = v146 - 1;
                        v145 += 2;
                        if ( v145 == v144 )
                          break;
                        while ( *v145 == -4096 || *v145 == -8192 )
                        {
                          v145 += 2;
                          if ( v144 == v145 )
                            goto LABEL_96;
                        }
                      }
                      while ( v144 != v145 );
                    }
                  }
                }
              }
            }
          }
        }
        else
        {
          v147 = 1;
          while ( v76 != (_QWORD *)-4096LL )
          {
            v148 = v147 + 1;
            v74 = v73 & (v147 + v74);
            v75 = v72 + 16LL * v74;
            v76 = *(_QWORD **)v75;
            if ( v69 == *(_QWORD **)v75 )
              goto LABEL_87;
            v147 = v148;
          }
        }
      }
LABEL_96:
      v67 += 16LL;
      sub_B43D60(v69);
    }
    while ( v184 != (unsigned __int64 **)v67 );
  }
  v93 = *(_QWORD *)(a1 + 120);
  for ( i = v93 + 24LL * *(unsigned int *)(a1 + 128); v93 != i; i -= 24 )
  {
    while ( 1 )
    {
      v95 = *(_QWORD **)(i - 24);
      if ( !v95[2] )
        break;
      i -= 24;
      if ( v93 == i )
        goto LABEL_102;
    }
    sub_B43D60(v95);
  }
LABEL_102:
  if ( v192 != (unsigned __int64 **)v194 )
    _libc_free((unsigned __int64)v192);
}
