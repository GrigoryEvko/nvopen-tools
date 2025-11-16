// Function: sub_1B0EA80
// Address: 0x1b0ea80
//
void __fastcall sub_1B0EA80(
        __int64 a1,
        __int64 a2,
        int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        __int64 a18,
        char a19)
{
  __int64 v19; // r12
  unsigned __int64 v20; // rax
  double v21; // xmm4_8
  double v22; // xmm5_8
  __int64 v23; // rax
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r15
  __int64 v29; // r13
  __int64 v30; // rsi
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rax
  char v34; // di
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rcx
  int v40; // eax
  __int64 v41; // rax
  int v42; // edx
  __int64 v43; // rdx
  __int64 *v44; // rax
  __int64 v45; // rsi
  unsigned __int64 v46; // rdx
  __int64 v47; // rdx
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rax
  char v51; // di
  unsigned __int64 v52; // rsi
  __int64 v53; // rdx
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rdx
  _BOOL4 v57; // eax
  int v58; // eax
  __int64 v59; // rax
  int v60; // ecx
  __int64 v61; // rcx
  __int64 *v62; // rax
  __int64 v63; // rsi
  unsigned __int64 v64; // rcx
  __int64 v65; // rcx
  __int64 v66; // rdx
  __int64 v67; // r13
  __int64 v68; // rsi
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rax
  char v74; // di
  unsigned int v75; // esi
  __int64 v76; // rdx
  __int64 v77; // rax
  __int64 v78; // rcx
  __int64 v79; // rdx
  __int64 *v80; // rax
  __int64 v81; // rcx
  unsigned __int64 v82; // rdx
  __int64 v83; // rdx
  __int64 v84; // rax
  int v85; // ecx
  int v86; // ecx
  __int64 v87; // rsi
  unsigned int v88; // edx
  __int64 *v89; // rax
  __int64 v90; // r8
  __int64 v91; // rbx
  __int64 v92; // r13
  _QWORD *v93; // rax
  __int64 v94; // rbx
  __int64 v95; // r15
  int v96; // r8d
  int v97; // r9d
  __int64 v98; // rax
  __int64 *v99; // rsi
  __int64 v100; // rdx
  __int64 v101; // rax
  unsigned __int8 *v102; // rsi
  __int64 v103; // rax
  __int64 v104; // r15
  double v105; // xmm4_8
  double v106; // xmm5_8
  __int64 v107; // r13
  int v108; // r8d
  int v109; // r9d
  __int64 v110; // rbx
  __int64 v111; // r12
  signed __int64 v112; // r12
  unsigned __int8 *v113; // rbx
  _QWORD *v114; // rax
  __int64 *v115; // rsi
  int v116; // eax
  __int64 v117; // rdx
  _QWORD *v118; // rax
  _QWORD *v119; // rbx
  unsigned __int64 *v120; // r12
  __int64 v121; // rax
  unsigned __int64 v122; // rcx
  __int64 v123; // rsi
  unsigned __int8 *v124; // rsi
  __int64 v125; // rsi
  __int64 v126; // rax
  int v127; // edi
  unsigned int v128; // r8d
  __int64 *v129; // rdx
  __int64 v130; // rcx
  __int64 *v131; // rax
  __int64 v132; // rbx
  unsigned int v133; // ecx
  __int64 *v134; // rdx
  __int64 v135; // r9
  __int64 v136; // r14
  __int64 v137; // rax
  _BYTE *v138; // rax
  __int64 v139; // rdx
  __int64 v140; // rcx
  int v141; // r8d
  int v142; // r9d
  _BYTE *v143; // rsi
  int v144; // eax
  __int64 v145; // rax
  int v146; // edx
  __int64 v147; // rdx
  __int64 *v148; // rax
  __int64 v149; // rcx
  unsigned __int64 v150; // rdx
  __int64 v151; // rdx
  __int64 v152; // rdx
  __int64 v153; // rcx
  __int64 v154; // rax
  __int64 v155; // rdi
  unsigned int v156; // r10d
  int v157; // eax
  int v158; // edi
  int v159; // edx
  int v160; // r8d
  int v161; // edx
  int v162; // r9d
  unsigned __int64 v166; // [rsp+20h] [rbp-160h]
  int v167; // [rsp+28h] [rbp-158h]
  unsigned int v168; // [rsp+2Ch] [rbp-154h]
  __int64 v170; // [rsp+38h] [rbp-148h]
  __int64 v171; // [rsp+48h] [rbp-138h]
  _QWORD *v172; // [rsp+48h] [rbp-138h]
  __int64 v174; // [rsp+58h] [rbp-128h]
  __int64 **v175; // [rsp+60h] [rbp-120h]
  __int64 v176; // [rsp+60h] [rbp-120h]
  __int64 v177; // [rsp+60h] [rbp-120h]
  __int64 v178; // [rsp+60h] [rbp-120h]
  __int64 v179; // [rsp+60h] [rbp-120h]
  __int64 v180; // [rsp+68h] [rbp-118h]
  __int64 v181; // [rsp+78h] [rbp-108h] BYREF
  __int64 v182[2]; // [rsp+80h] [rbp-100h] BYREF
  __int16 v183; // [rsp+90h] [rbp-F0h]
  __int64 *v184; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v185; // [rsp+A8h] [rbp-D8h]
  _BYTE v186[32]; // [rsp+B0h] [rbp-D0h] BYREF
  unsigned __int8 *v187; // [rsp+D0h] [rbp-B0h] BYREF
  __int64 v188; // [rsp+D8h] [rbp-A8h]
  _WORD v189[16]; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v190; // [rsp+100h] [rbp-80h] BYREF
  const char *v191; // [rsp+108h] [rbp-78h]
  unsigned __int64 *v192; // [rsp+110h] [rbp-70h]
  __int64 v193; // [rsp+118h] [rbp-68h]
  __int64 v194; // [rsp+120h] [rbp-60h]
  int v195; // [rsp+128h] [rbp-58h]
  __int64 v196; // [rsp+130h] [rbp-50h]
  __int64 v197; // [rsp+138h] [rbp-48h]

  v19 = sub_13FCB50(a1);
  v170 = sub_1B0E720(a16, v19)[2];
  v20 = sub_157EBA0(v19);
  if ( v20 )
  {
    v167 = sub_15F4D60(v20);
    v166 = sub_157EBA0(v19);
    if ( v167 )
    {
      v168 = 0;
      v180 = a1 + 56;
      while ( 1 )
      {
        v23 = sub_15F4DF0(v166, v168);
        v24 = sub_157F280(v23);
        v171 = v25;
        if ( v24 != v25 )
          break;
LABEL_72:
        if ( v167 == ++v168 )
          goto LABEL_73;
      }
      while ( 1 )
      {
        v174 = sub_157ED20(a4);
        v187 = (unsigned __int8 *)sub_1649960(v24);
        LOWORD(v192) = 773;
        v190 = (__int64)&v187;
        v188 = v26;
        v191 = ".unr";
        v175 = *(__int64 ***)v24;
        v27 = sub_1648B60(64);
        v28 = v27;
        if ( v27 )
        {
          v29 = v27;
          sub_15F1EA0(v27, (__int64)v175, 53, 0, 0, v174);
          *(_DWORD *)(v28 + 56) = 2;
          sub_164B780(v28, &v190);
          sub_1648880(v28, *(_DWORD *)(v28 + 56), 1);
        }
        else
        {
          v29 = 0;
        }
        v30 = *(_QWORD *)(v24 + 40);
        if ( !sub_1377F70(v180, v30) )
          break;
        v33 = 0x17FFFFFFE8LL;
        v34 = *(_BYTE *)(v24 + 23) & 0x40;
        v30 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
        if ( (*(_DWORD *)(v24 + 20) & 0xFFFFFFF) != 0 )
        {
          v31 = v24 - 24LL * (unsigned int)v30;
          v35 = 24LL * *(unsigned int *)(v24 + 56) + 8;
          v36 = 0;
          do
          {
            v37 = v24 - 24LL * (unsigned int)v30;
            if ( v34 )
              v37 = *(_QWORD *)(v24 - 8);
            if ( a15 == *(_QWORD *)(v37 + v35) )
            {
              v33 = 24 * v36;
              goto LABEL_15;
            }
            ++v36;
            v35 += 8;
          }
          while ( (_DWORD)v30 != (_DWORD)v36 );
          v33 = 0x17FFFFFFE8LL;
        }
LABEL_15:
        if ( v34 )
        {
          v38 = *(_QWORD *)(v24 - 8);
        }
        else
        {
          v30 = (unsigned int)v30;
          v38 = v24 - 24LL * (unsigned int)v30;
        }
        v39 = *(_QWORD *)(v38 + v33);
        v40 = *(_DWORD *)(v28 + 20) & 0xFFFFFFF;
        if ( v40 == *(_DWORD *)(v28 + 56) )
          goto LABEL_158;
LABEL_18:
        v41 = (v40 + 1) & 0xFFFFFFF;
        v42 = v41 | *(_DWORD *)(v28 + 20) & 0xF0000000;
        *(_DWORD *)(v28 + 20) = v42;
        if ( (v42 & 0x40000000) != 0 )
          v43 = *(_QWORD *)(v28 - 8);
        else
          v43 = v29 - 24 * v41;
        v44 = (__int64 *)(v43 + 24LL * (unsigned int)(v41 - 1));
        if ( *v44 )
        {
          v45 = v44[1];
          v46 = v44[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v46 = v45;
          if ( v45 )
            *(_QWORD *)(v45 + 16) = *(_QWORD *)(v45 + 16) & 3LL | v46;
        }
        *v44 = v39;
        if ( v39 )
        {
          v47 = *(_QWORD *)(v39 + 8);
          v44[1] = v47;
          if ( v47 )
          {
            v31 = (__int64)(v44 + 1);
            *(_QWORD *)(v47 + 16) = (unsigned __int64)(v44 + 1) | *(_QWORD *)(v47 + 16) & 3LL;
          }
          v44[2] = (v39 + 8) | v44[2] & 3;
          *(_QWORD *)(v39 + 8) = v44;
        }
        v48 = *(_DWORD *)(v28 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v28 + 23) & 0x40) != 0 )
          v49 = *(_QWORD *)(v28 - 8);
        else
          v49 = v29 - 24 * v48;
        *(_QWORD *)(v49 + 8LL * (unsigned int)(v48 - 1) + 24LL * *(unsigned int *)(v28 + 56) + 8) = a6;
        v50 = 0x17FFFFFFE8LL;
        v51 = *(_BYTE *)(v24 + 23) & 0x40;
        v52 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
        if ( (*(_DWORD *)(v24 + 20) & 0xFFFFFFF) != 0 )
        {
          v31 = v24 - 24LL * (unsigned int)v52;
          v53 = 24LL * *(unsigned int *)(v24 + 56) + 8;
          v54 = 0;
          do
          {
            v49 = v24 - 24LL * (unsigned int)v52;
            if ( v51 )
              v49 = *(_QWORD *)(v24 - 8);
            if ( v19 == *(_QWORD *)(v49 + v53) )
            {
              v50 = 24 * v54;
              goto LABEL_36;
            }
            ++v54;
            v53 += 8;
          }
          while ( (_DWORD)v52 != (_DWORD)v54 );
          v50 = 0x17FFFFFFE8LL;
        }
LABEL_36:
        if ( v51 )
        {
          v55 = *(_QWORD *)(v24 - 8);
        }
        else
        {
          v52 = (unsigned int)v52;
          v49 = 24LL * (unsigned int)v52;
          v55 = v24 - v49;
        }
        v56 = *(_QWORD *)(v55 + v50);
        if ( *(_BYTE *)(v56 + 16) <= 0x17u )
          goto LABEL_40;
        v52 = *(_QWORD *)(v56 + 40);
        v176 = v56;
        v57 = sub_1377F70(v180, v52);
        v56 = v176;
        if ( !v57 )
          goto LABEL_40;
        v154 = *(unsigned int *)(a16 + 24);
        if ( !(_DWORD)v154 )
          goto LABEL_177;
        v32 = (unsigned int)(v154 - 1);
        v155 = *(_QWORD *)(a16 + 8);
        v52 = (unsigned int)v32 & (((unsigned int)v176 >> 9) ^ ((unsigned int)v176 >> 4));
        v49 = v155 + (v52 << 6);
        v31 = *(_QWORD *)(v49 + 24);
        if ( v176 != v31 )
        {
          v49 = 1;
          while ( v31 != -8 )
          {
            v156 = v49 + 1;
            v52 = (unsigned int)v32 & ((_DWORD)v49 + (_DWORD)v52);
            v49 = v155 + ((unsigned __int64)(unsigned int)v52 << 6);
            v31 = *(_QWORD *)(v49 + 24);
            if ( v176 == v31 )
              goto LABEL_166;
            v49 = v156;
          }
LABEL_177:
          v56 = 0;
LABEL_40:
          v58 = *(_DWORD *)(v28 + 20) & 0xFFFFFFF;
          if ( v58 == *(_DWORD *)(v28 + 56) )
            goto LABEL_172;
          goto LABEL_41;
        }
LABEL_166:
        if ( v49 == v155 + (v154 << 6) )
          goto LABEL_177;
        v190 = 6;
        v191 = 0;
        v192 = *(unsigned __int64 **)(v49 + 56);
        v56 = (__int64)v192;
        LOBYTE(v52) = v192 + 1 != 0;
        if ( ((v192 != 0) & (unsigned __int8)v52) == 0 )
          goto LABEL_40;
        if ( v192 == (unsigned __int64 *)-16LL )
          goto LABEL_40;
        v52 = *(_QWORD *)(v49 + 40) & 0xFFFFFFFFFFFFFFF8LL;
        sub_1649AC0((unsigned __int64 *)&v190, v52);
        v56 = (__int64)v192;
        LOBYTE(v49) = v192 + 2 != 0;
        if ( ((v192 != 0) & (unsigned __int8)v49) == 0 || v192 == (unsigned __int64 *)-8LL )
          goto LABEL_40;
        v178 = (__int64)v192;
        sub_1649B30(&v190);
        v56 = v178;
        v58 = *(_DWORD *)(v28 + 20) & 0xFFFFFFF;
        if ( v58 == *(_DWORD *)(v28 + 56) )
        {
LABEL_172:
          v179 = v56;
          sub_15F55D0(v28, v52, v56, v49, v31, v32);
          v56 = v179;
          v58 = *(_DWORD *)(v28 + 20) & 0xFFFFFFF;
        }
LABEL_41:
        v59 = (v58 + 1) & 0xFFFFFFF;
        v60 = v59 | *(_DWORD *)(v28 + 20) & 0xF0000000;
        *(_DWORD *)(v28 + 20) = v60;
        if ( (v60 & 0x40000000) != 0 )
          v61 = *(_QWORD *)(v28 - 8);
        else
          v61 = v29 - 24 * v59;
        v62 = (__int64 *)(v61 + 24LL * (unsigned int)(v59 - 1));
        if ( *v62 )
        {
          v63 = v62[1];
          v64 = v62[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v64 = v63;
          if ( v63 )
            *(_QWORD *)(v63 + 16) = *(_QWORD *)(v63 + 16) & 3LL | v64;
        }
        *v62 = v56;
        if ( v56 )
        {
          v65 = *(_QWORD *)(v56 + 8);
          v62[1] = v65;
          if ( v65 )
            *(_QWORD *)(v65 + 16) = (unsigned __int64)(v62 + 1) | *(_QWORD *)(v65 + 16) & 3LL;
          v62[2] = (v56 + 8) | v62[2] & 3;
          *(_QWORD *)(v56 + 8) = v62;
        }
        v66 = *(_DWORD *)(v28 + 20) & 0xFFFFFFF;
        if ( (*(_BYTE *)(v28 + 23) & 0x40) != 0 )
          v67 = *(_QWORD *)(v28 - 8);
        else
          v67 = v29 - 24 * v66;
        *(_QWORD *)(v67 + 8LL * (unsigned int)(v66 - 1) + 24LL * *(unsigned int *)(v28 + 56) + 8) = v170;
        v68 = *(_QWORD *)(v24 + 40);
        if ( sub_1377F70(v180, v68) )
        {
          v73 = 0x17FFFFFFE8LL;
          v74 = *(_BYTE *)(v24 + 23) & 0x40;
          v75 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
          if ( v75 )
          {
            v76 = 24LL * *(unsigned int *)(v24 + 56) + 8;
            v77 = 0;
            do
            {
              v78 = v24 - 24LL * v75;
              if ( v74 )
                v78 = *(_QWORD *)(v24 - 8);
              if ( a15 == *(_QWORD *)(v78 + v76) )
              {
                v73 = 24 * v77;
                goto LABEL_60;
              }
              ++v77;
              v76 += 8;
            }
            while ( v75 != (_DWORD)v77 );
            v73 = 0x17FFFFFFE8LL;
          }
LABEL_60:
          if ( v74 )
            v79 = *(_QWORD *)(v24 - 8);
          else
            v79 = v24 - 24LL * v75;
          v80 = (__int64 *)(v79 + v73);
          if ( *v80 )
          {
            v81 = v80[1];
            v82 = v80[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v82 = v81;
            if ( v81 )
              *(_QWORD *)(v81 + 16) = *(_QWORD *)(v81 + 16) & 3LL | v82;
          }
          *v80 = v28;
          v83 = *(_QWORD *)(v28 + 8);
          v80[1] = v83;
          if ( v83 )
            *(_QWORD *)(v83 + 16) = (unsigned __int64)(v80 + 1) | *(_QWORD *)(v83 + 16) & 3LL;
          v80[2] = v80[2] & 3 | (v28 + 8);
          *(_QWORD *)(v28 + 8) = v80;
        }
        else
        {
          v144 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
          if ( v144 == *(_DWORD *)(v24 + 56) )
          {
            sub_15F55D0(v24, v68, v69, v70, v71, v72);
            v144 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
          }
          v145 = (v144 + 1) & 0xFFFFFFF;
          v146 = v145 | *(_DWORD *)(v24 + 20) & 0xF0000000;
          *(_DWORD *)(v24 + 20) = v146;
          if ( (v146 & 0x40000000) != 0 )
            v147 = *(_QWORD *)(v24 - 8);
          else
            v147 = v24 - 24 * v145;
          v148 = (__int64 *)(v147 + 24LL * (unsigned int)(v145 - 1));
          if ( *v148 )
          {
            v149 = v148[1];
            v150 = v148[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v150 = v149;
            if ( v149 )
              *(_QWORD *)(v149 + 16) = *(_QWORD *)(v149 + 16) & 3LL | v150;
          }
          *v148 = v28;
          v151 = *(_QWORD *)(v28 + 8);
          v148[1] = v151;
          if ( v151 )
            *(_QWORD *)(v151 + 16) = (unsigned __int64)(v148 + 1) | *(_QWORD *)(v151 + 16) & 3LL;
          v148[2] = (v28 + 8) | v148[2] & 3;
          *(_QWORD *)(v28 + 8) = v148;
          v152 = *(_DWORD *)(v24 + 20) & 0xFFFFFFF;
          if ( (*(_BYTE *)(v24 + 23) & 0x40) != 0 )
            v153 = *(_QWORD *)(v24 - 8);
          else
            v153 = v24 - 24 * v152;
          *(_QWORD *)(v153 + 8LL * (unsigned int)(v152 - 1) + 24LL * *(unsigned int *)(v24 + 56) + 8) = a4;
        }
        v84 = *(_QWORD *)(v24 + 32);
        if ( !v84 )
          BUG();
        v24 = 0;
        if ( *(_BYTE *)(v84 - 8) == 77 )
          v24 = v84 - 24;
        if ( v171 == v24 )
          goto LABEL_72;
      }
      v39 = sub_1599EF0(*(__int64 ***)v24);
      v40 = *(_DWORD *)(v28 + 20) & 0xFFFFFFF;
      if ( v40 != *(_DWORD *)(v28 + 56) )
        goto LABEL_18;
LABEL_158:
      v177 = v39;
      sub_15F55D0(v28, v30, v38, v39, v31, v32);
      v39 = v177;
      v40 = *(_DWORD *)(v28 + 20) & 0xFFFFFFF;
      goto LABEL_18;
    }
  }
LABEL_73:
  v184 = (__int64 *)v186;
  v185 = 0x400000000LL;
  v85 = *(_DWORD *)(a18 + 24);
  if ( v85 )
  {
    v86 = v85 - 1;
    v87 = *(_QWORD *)(a18 + 8);
    v88 = v86 & (((unsigned int)v170 >> 9) ^ ((unsigned int)v170 >> 4));
    v89 = (__int64 *)(v87 + 16LL * v88);
    v90 = *v89;
    if ( v170 == *v89 )
    {
LABEL_75:
      v91 = v89[1];
      if ( v91 )
      {
        v92 = *(_QWORD *)(a4 + 8);
        if ( v92 )
        {
          while ( 1 )
          {
            v93 = sub_1648700(v92);
            if ( (unsigned __int8)(*((_BYTE *)v93 + 16) - 25) <= 9u )
              break;
            v92 = *(_QWORD *)(v92 + 8);
            if ( !v92 )
              goto LABEL_179;
          }
          v94 = v91 + 56;
LABEL_81:
          v95 = v93[5];
          if ( sub_1377F70(v94, v95) )
          {
            v98 = (unsigned int)v185;
            if ( (unsigned int)v185 >= HIDWORD(v185) )
            {
              sub_16CD150((__int64)&v184, v186, 0, 8, v96, v97);
              v98 = (unsigned int)v185;
            }
            v184[v98] = v95;
            LODWORD(v185) = v185 + 1;
            v92 = *(_QWORD *)(v92 + 8);
            if ( v92 )
              goto LABEL_80;
          }
          else
          {
            while ( 1 )
            {
              v92 = *(_QWORD *)(v92 + 8);
              if ( !v92 )
                break;
LABEL_80:
              v93 = sub_1648700(v92);
              if ( (unsigned __int8)(*((_BYTE *)v93 + 16) - 25) <= 9u )
                goto LABEL_81;
            }
          }
          v99 = v184;
          v100 = (unsigned int)v185;
        }
        else
        {
LABEL_179:
          v99 = (__int64 *)v186;
          v100 = 0;
        }
        sub_1AAB350(a4, v99, v100, ".unr-lcssa", a17, a18, a7, a8, a9, a10, v21, v22, a13, a14, a19);
      }
    }
    else
    {
      v157 = 1;
      while ( v90 != -8 )
      {
        v158 = v157 + 1;
        v88 = v86 & (v157 + v88);
        v89 = (__int64 *)(v87 + 16LL * v88);
        v90 = *v89;
        if ( v170 == *v89 )
          goto LABEL_75;
        v157 = v158;
      }
    }
  }
  v172 = (_QWORD *)sub_157EBA0(a4);
  v101 = sub_16498A0((__int64)v172);
  v190 = 0;
  v193 = v101;
  v194 = 0;
  v195 = 0;
  v196 = 0;
  v197 = 0;
  v191 = (const char *)v172[5];
  v192 = v172 + 3;
  v102 = (unsigned __int8 *)v172[6];
  v187 = v102;
  if ( v102 )
  {
    sub_1623A60((__int64)&v187, (__int64)v102, 2);
    if ( v190 )
      sub_161E7C0((__int64)&v190, v190);
    v190 = (__int64)v187;
    if ( v187 )
      sub_1623210((__int64)&v187, v187, (__int64)&v190);
  }
  v189[0] = 257;
  v103 = sub_15A0680(*(_QWORD *)a2, (unsigned int)(a3 - 1), 0);
  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(v103 + 16) > 0x10u )
    v104 = (__int64)sub_1B0DFC0(&v190, 36, a2, v103, (__int64 *)&v187);
  else
    v104 = sub_15A37B0(0x24u, (_QWORD *)a2, (_QWORD *)v103, 0);
  v107 = *(_QWORD *)(a5 + 8);
  if ( v107 )
  {
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v107) + 16) - 25) > 9u )
    {
      v107 = *(_QWORD *)(v107 + 8);
      if ( !v107 )
        goto LABEL_181;
    }
    v110 = v107;
    v111 = 0;
    v187 = (unsigned __int8 *)v189;
    v188 = 0x400000000LL;
    while ( 1 )
    {
      v110 = *(_QWORD *)(v110 + 8);
      if ( !v110 )
        break;
      while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v110) + 16) - 25) <= 9u )
      {
        v110 = *(_QWORD *)(v110 + 8);
        ++v111;
        if ( !v110 )
          goto LABEL_101;
      }
    }
LABEL_101:
    v112 = v111 + 1;
    if ( v112 > 4 )
    {
      sub_16CD150((__int64)&v187, v189, v112, 8, v108, v109);
      v113 = &v187[8 * (unsigned int)v188];
    }
    else
    {
      v113 = (unsigned __int8 *)v189;
    }
    v114 = sub_1648700(v107);
LABEL_106:
    if ( v113 )
      *(_QWORD *)v113 = v114[5];
    while ( 1 )
    {
      v107 = *(_QWORD *)(v107 + 8);
      if ( !v107 )
        break;
      v114 = sub_1648700(v107);
      if ( (unsigned __int8)(*((_BYTE *)v114 + 16) - 25) <= 9u )
      {
        v113 += 8;
        goto LABEL_106;
      }
    }
    v115 = (__int64 *)v187;
    v116 = v112 + v188;
    v117 = (unsigned int)(v112 + v188);
  }
  else
  {
LABEL_181:
    v117 = 0;
    v116 = 0;
    HIDWORD(v188) = 4;
    v187 = (unsigned __int8 *)v189;
    v115 = (__int64 *)v189;
  }
  LODWORD(v188) = v116;
  sub_1AAB350(a5, v115, v117, ".unr-lcssa", a17, a18, a7, a8, a9, a10, v105, v106, a13, a14, a19);
  v183 = 257;
  v118 = sub_1648A60(56, 3u);
  v119 = v118;
  if ( v118 )
    sub_15F83E0((__int64)v118, a5, a15, v104, 0);
  if ( v191 )
  {
    v120 = v192;
    sub_157E9D0((__int64)(v191 + 40), (__int64)v119);
    v121 = v119[3];
    v122 = *v120;
    v119[4] = v120;
    v122 &= 0xFFFFFFFFFFFFFFF8LL;
    v119[3] = v122 | v121 & 7;
    *(_QWORD *)(v122 + 8) = v119 + 3;
    *v120 = *v120 & 7 | (unsigned __int64)(v119 + 3);
  }
  sub_164B780((__int64)v119, v182);
  if ( v190 )
  {
    v181 = v190;
    sub_1623A60((__int64)&v181, v190, 2);
    v123 = v119[6];
    if ( v123 )
      sub_161E7C0((__int64)(v119 + 6), v123);
    v124 = (unsigned __int8 *)v181;
    v119[6] = v181;
    if ( v124 )
      sub_1623210((__int64)&v181, v124, (__int64)(v119 + 6));
  }
  sub_15F20C0(v172);
  if ( a17 )
  {
    v125 = *(_QWORD *)(a17 + 32);
    v126 = *(unsigned int *)(a17 + 48);
    if ( !(_DWORD)v126 )
      goto LABEL_203;
    v127 = v126 - 1;
    v128 = (v126 - 1) & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v129 = (__int64 *)(v125 + 16LL * v128);
    v130 = *v129;
    if ( a4 == *v129 )
    {
LABEL_122:
      v131 = (__int64 *)(v125 + 16 * v126);
      if ( v129 != v131 )
      {
        v132 = v129[1];
        goto LABEL_124;
      }
    }
    else
    {
      v161 = 1;
      while ( v130 != -8 )
      {
        v162 = v161 + 1;
        v128 = v127 & (v161 + v128);
        v129 = (__int64 *)(v125 + 16LL * v128);
        v130 = *v129;
        if ( a4 == *v129 )
          goto LABEL_122;
        v161 = v162;
      }
      v131 = (__int64 *)(v125 + 16 * v126);
    }
    v132 = 0;
LABEL_124:
    v133 = v127 & (((unsigned int)a5 >> 9) ^ ((unsigned int)a5 >> 4));
    v134 = (__int64 *)(v125 + 16LL * v133);
    v135 = *v134;
    if ( a5 == *v134 )
    {
LABEL_125:
      if ( v134 != v131 )
      {
        v136 = v134[1];
        *(_BYTE *)(a17 + 72) = 0;
        v137 = *(_QWORD *)(v136 + 8);
        if ( v132 != v137 )
        {
          v182[0] = v136;
          v138 = sub_1B0DF00(*(_QWORD **)(v137 + 24), *(_QWORD *)(v137 + 32), v182);
          sub_15CDF70(*(_QWORD *)(v136 + 8) + 24LL, v138);
          *(_QWORD *)(v136 + 8) = v132;
          v182[0] = v136;
          v143 = *(_BYTE **)(v132 + 32);
          if ( v143 == *(_BYTE **)(v132 + 40) )
          {
            sub_15CE310(v132 + 24, v143, v182);
          }
          else
          {
            if ( v143 )
            {
              *(_QWORD *)v143 = v136;
              v143 = *(_BYTE **)(v132 + 32);
            }
            v143 += 8;
            *(_QWORD *)(v132 + 32) = v143;
          }
          if ( *(_DWORD *)(v136 + 16) != *(_DWORD *)(*(_QWORD *)(v136 + 8) + 16LL) + 1 )
            sub_1B0DDF0(v136, (__int64)v143, v139, v140, v141, v142);
        }
        goto LABEL_133;
      }
    }
    else
    {
      v159 = 1;
      while ( v135 != -8 )
      {
        v160 = v159 + 1;
        v133 = v127 & (v159 + v133);
        v134 = (__int64 *)(v125 + 16LL * v133);
        v135 = *v134;
        if ( a5 == *v134 )
          goto LABEL_125;
        v159 = v160;
      }
    }
LABEL_203:
    *(_BYTE *)(a17 + 72) = 0;
    BUG();
  }
LABEL_133:
  if ( v187 != (unsigned __int8 *)v189 )
    _libc_free((unsigned __int64)v187);
  if ( v190 )
    sub_161E7C0((__int64)&v190, v190);
  if ( v184 != (__int64 *)v186 )
    _libc_free((unsigned __int64)v184);
}
