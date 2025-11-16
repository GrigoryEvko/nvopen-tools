// Function: sub_385A280
// Address: 0x385a280
//
__int64 __fastcall sub_385A280(__int64 a1, __int64 a2, double a3, double a4, double a5)
{
  unsigned __int64 v6; // rbx
  _QWORD *v7; // rdi
  unsigned int v8; // r14d
  __int64 v9; // rdi
  _BYTE *v10; // r13
  unsigned __int64 v11; // rdi
  unsigned __int8 v12; // al
  unsigned int v13; // eax
  int v14; // r8d
  int v15; // r9d
  unsigned __int64 v16; // rcx
  unsigned __int64 v17; // rbx
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r15
  __int64 v22; // rax
  int v23; // r15d
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // r15
  __int64 v27; // rax
  int v28; // eax
  int v29; // ecx
  __int64 v30; // rsi
  unsigned int v31; // edx
  _QWORD *v32; // rax
  _BYTE *v33; // rdi
  __int64 v34; // rbx
  __int64 v35; // r13
  _QWORD *v36; // rdx
  _QWORD *v37; // rax
  _QWORD *v38; // r14
  __int64 v39; // rax
  __int64 v40; // rax
  __int64 v41; // r13
  __int64 *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 *v46; // rsi
  __int64 v47; // r14
  __int64 v48; // r15
  __int64 v49; // rax
  char v50; // al
  int v51; // eax
  __int64 v52; // r13
  _QWORD *v53; // rax
  __int64 v54; // rax
  unsigned __int64 v55; // rbx
  __int64 v56; // r12
  unsigned __int64 v57; // rdi
  unsigned __int64 v58; // rcx
  unsigned __int64 v59; // rdi
  unsigned __int8 v60; // dl
  __int64 v61; // rbx
  unsigned __int64 v62; // r14
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r15
  __int64 v68; // rax
  int v69; // r15d
  __int64 v70; // rax
  __int64 v71; // rdx
  unsigned __int64 v72; // rbx
  _QWORD *v73; // r13
  __int64 v74; // rax
  __int64 v75; // rdx
  __int64 v76; // r14
  __int64 v77; // rdx
  __int64 v78; // rax
  __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // r13
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rcx
  unsigned __int64 v86; // r15
  __int64 v87; // rax
  __int64 v88; // r10
  int v89; // edx
  int v90; // edx
  __int64 v91; // rsi
  unsigned int v92; // ecx
  __int64 *v93; // rax
  __int64 v94; // r11
  __int64 v95; // rax
  __int64 v96; // rdx
  __int64 v97; // r14
  __int64 v98; // rdx
  __int64 v99; // rax
  __int64 v100; // rax
  __int64 v101; // rdx
  __int64 v102; // r13
  __int64 v103; // rax
  __int64 *v104; // rsi
  unsigned int v105; // edi
  __int64 *v106; // rcx
  _QWORD *v107; // rdx
  __int64 v108; // rax
  unsigned int v109; // eax
  __int64 v110; // rdx
  __int64 v111; // rcx
  int v112; // edx
  __int64 v113; // rax
  int v114; // edx
  unsigned __int64 v115; // rax
  int v116; // eax
  int v117; // r8d
  int v118; // eax
  int v119; // edi
  _QWORD *v120; // rdx
  unsigned __int64 v121; // [rsp+0h] [rbp-2E0h]
  unsigned __int64 v122; // [rsp+0h] [rbp-2E0h]
  __int64 v123; // [rsp+0h] [rbp-2E0h]
  __int64 v124[2]; // [rsp+18h] [rbp-2C8h] BYREF
  __int64 v125; // [rsp+28h] [rbp-2B8h] BYREF
  unsigned __int64 v126; // [rsp+30h] [rbp-2B0h] BYREF
  int v128; // [rsp+3Ch] [rbp-2A4h]
  int v130; // [rsp+44h] [rbp-29Ch]
  int v132; // [rsp+4Ch] [rbp-294h]
  int v134; // [rsp+54h] [rbp-28Ch]
  int v136; // [rsp+5Ch] [rbp-284h]
  int v138; // [rsp+64h] [rbp-27Ch]
  char v140; // [rsp+6Ch] [rbp-274h]
  char v141; // [rsp+6Dh] [rbp-273h]
  __int64 *v142; // [rsp+70h] [rbp-270h] BYREF
  __int64 v143; // [rsp+78h] [rbp-268h]
  _QWORD v144[7]; // [rsp+80h] [rbp-260h] BYREF
  int v145; // [rsp+B8h] [rbp-228h]
  int v146; // [rsp+BCh] [rbp-224h]
  char v147; // [rsp+C0h] [rbp-220h]
  __int64 v148; // [rsp+C1h] [rbp-21Fh]
  char v149; // [rsp+C9h] [rbp-217h]
  __int64 v150; // [rsp+D0h] [rbp-210h]
  __int64 v151; // [rsp+D8h] [rbp-208h]
  __int64 v152; // [rsp+E0h] [rbp-200h]
  int v153; // [rsp+E8h] [rbp-1F8h]
  int v154; // [rsp+F0h] [rbp-1F0h]
  __int64 v155; // [rsp+F8h] [rbp-1E8h]
  unsigned __int64 v156; // [rsp+100h] [rbp-1E0h]
  __int64 v157; // [rsp+108h] [rbp-1D8h]
  int v158; // [rsp+110h] [rbp-1D0h]
  __int64 v159; // [rsp+118h] [rbp-1C8h]
  unsigned __int64 v160; // [rsp+120h] [rbp-1C0h]
  __int64 v161; // [rsp+128h] [rbp-1B8h]
  int v162; // [rsp+130h] [rbp-1B0h]
  __int64 v163; // [rsp+138h] [rbp-1A8h]
  unsigned __int64 v164; // [rsp+140h] [rbp-1A0h]
  __int64 v165; // [rsp+148h] [rbp-198h]
  int v166; // [rsp+150h] [rbp-190h]
  __int64 v167; // [rsp+158h] [rbp-188h]
  unsigned __int64 v168; // [rsp+160h] [rbp-180h]
  __int64 v169; // [rsp+168h] [rbp-178h]
  unsigned int v170; // [rsp+170h] [rbp-170h]
  __int64 v171; // [rsp+178h] [rbp-168h]
  unsigned __int64 v172; // [rsp+180h] [rbp-160h]
  __int64 v173; // [rsp+188h] [rbp-158h]
  __int64 v174; // [rsp+190h] [rbp-150h]
  unsigned __int64 v175; // [rsp+198h] [rbp-148h]
  __int64 v176; // [rsp+1A0h] [rbp-140h]
  __int64 v177; // [rsp+1A8h] [rbp-138h]
  __int64 v178; // [rsp+1B0h] [rbp-130h]
  unsigned __int64 v179; // [rsp+1B8h] [rbp-128h]
  __int64 v180; // [rsp+1C0h] [rbp-120h]
  int v181; // [rsp+1C8h] [rbp-118h]
  char v182; // [rsp+1D0h] [rbp-110h]
  __int64 v183; // [rsp+1D8h] [rbp-108h]
  _BYTE *v184; // [rsp+1E0h] [rbp-100h]
  _BYTE *v185; // [rsp+1E8h] [rbp-F8h]
  __int64 v186; // [rsp+1F0h] [rbp-F0h]
  int v187; // [rsp+1F8h] [rbp-E8h]
  _BYTE v188[128]; // [rsp+200h] [rbp-E0h] BYREF
  __int64 v189; // [rsp+280h] [rbp-60h]
  __int64 v190; // [rsp+288h] [rbp-58h]
  __int64 v191; // [rsp+290h] [rbp-50h]
  __int64 v192; // [rsp+298h] [rbp-48h]
  int v193; // [rsp+2A0h] [rbp-40h]

  v6 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v7 = (_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 56);
  v124[0] = a2;
  if ( (a2 & 4) != 0 )
  {
    if ( (unsigned __int8)sub_1560260(v7, -1, 39) )
      goto LABEL_3;
    v27 = *(_QWORD *)(v6 - 24);
    if ( *(_BYTE *)(v27 + 16) )
      goto LABEL_4;
  }
  else
  {
    if ( (unsigned __int8)sub_1560260(v7, -1, 39) )
      goto LABEL_3;
    v27 = *(_QWORD *)(v6 - 72);
    if ( *(_BYTE *)(v27 + 16) )
      goto LABEL_4;
  }
  v142 = *(__int64 **)(v27 + 112);
  if ( (unsigned __int8)sub_1560260(&v142, -1, 39) )
  {
LABEL_3:
    v8 = sub_1560180(*(_QWORD *)(a1 + 32) + 112LL, 39);
    if ( !(_BYTE)v8 )
    {
      *(_BYTE *)(a1 + 83) = 1;
      return v8;
    }
LABEL_4:
    v9 = v124[0];
    if ( (v124[0] & 4) == 0 )
      goto LABEL_5;
    goto LABEL_24;
  }
  v9 = v124[0];
  if ( (v124[0] & 4) == 0 )
    goto LABEL_5;
LABEL_24:
  if ( (unsigned __int8)sub_1560260((_QWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 56), -1, 24)
    || (v64 = *(_QWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) - 24), !*(_BYTE *)(v64 + 16))
    && (v142 = *(__int64 **)(v64 + 112), (unsigned __int8)sub_1560260(&v142, -1, 24)) )
  {
    *(_BYTE *)(a1 + 85) = 1;
  }
  v9 = v124[0];
  if ( (v124[0] & 4) == 0 )
  {
LABEL_5:
    v10 = *(_BYTE **)((v9 & 0xFFFFFFFFFFFFFFF8LL) - 72);
    if ( !v10[16] )
      goto LABEL_6;
LABEL_28:
    *(_DWORD *)(a1 + 76) += 5 * sub_165AFC0(v124);
    v28 = *(_DWORD *)(a1 + 160);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 144);
      v31 = (v28 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v32 = (_QWORD *)(v30 + 16LL * v31);
      v33 = (_BYTE *)*v32;
      if ( (_BYTE *)*v32 == v10 )
      {
LABEL_30:
        v34 = v32[1];
        if ( v34 && !*(_BYTE *)(v34 + 16) )
        {
          v35 = *(_QWORD *)(a1 + 96);
          v36 = *(_QWORD **)(v35 + 16);
          v37 = *(_QWORD **)(v35 + 8);
          if ( v36 == v37 )
          {
            v38 = &v37[*(unsigned int *)(v35 + 28)];
            if ( v37 == v38 )
            {
              v120 = *(_QWORD **)(v35 + 8);
            }
            else
            {
              do
              {
                if ( v34 == *v37 )
                  break;
                ++v37;
              }
              while ( v38 != v37 );
              v120 = v38;
            }
          }
          else
          {
            v38 = &v36[*(unsigned int *)(v35 + 24)];
            v37 = sub_16CC9F0(*(_QWORD *)(a1 + 96), v34);
            if ( v34 == *v37 )
            {
              v84 = *(_QWORD *)(v35 + 16);
              if ( v84 == *(_QWORD *)(v35 + 8) )
                v85 = *(unsigned int *)(v35 + 28);
              else
                v85 = *(unsigned int *)(v35 + 24);
              v120 = (_QWORD *)(v84 + 8 * v85);
            }
            else
            {
              v39 = *(_QWORD *)(v35 + 16);
              if ( v39 != *(_QWORD *)(v35 + 8) )
              {
                v37 = (_QWORD *)(v39 + 8LL * *(unsigned int *)(v35 + 24));
                goto LABEL_36;
              }
              v37 = (_QWORD *)(v39 + 8LL * *(unsigned int *)(v35 + 28));
              v120 = v37;
            }
          }
          while ( v120 != v37 && *v37 >= 0xFFFFFFFFFFFFFFFELL )
            ++v37;
LABEL_36:
          if ( v37 != v38 )
            goto LABEL_92;
          v40 = *(_QWORD *)(a1 + 64);
          if ( *(_BYTE *)(v40 + 8) )
            HIDWORD(v126) = *(_DWORD *)(v40 + 4);
          if ( *(_BYTE *)(v40 + 16) )
            v128 = *(_DWORD *)(v40 + 12);
          if ( *(_BYTE *)(v40 + 24) )
            v130 = *(_DWORD *)(v40 + 20);
          if ( *(_BYTE *)(v40 + 32) )
            v132 = *(_DWORD *)(v40 + 28);
          if ( *(_BYTE *)(v40 + 40) )
            v134 = *(_DWORD *)(v40 + 36);
          if ( *(_BYTE *)(v40 + 48) )
            v136 = *(_DWORD *)(v40 + 44);
          if ( *(_BYTE *)(v40 + 56) )
            v138 = *(_DWORD *)(v40 + 52);
          v141 = *(_BYTE *)(v40 + 61);
          if ( v141 )
            v140 = *(_BYTE *)(v40 + 60);
          LODWORD(v126) = 100;
          v41 = *(_QWORD *)(a1 + 96);
          v42 = *(__int64 **)(v41 + 8);
          if ( *(__int64 **)(v41 + 16) != v42 )
            goto LABEL_54;
          v104 = &v42[*(unsigned int *)(v41 + 28)];
          v105 = *(_DWORD *)(v41 + 28);
          if ( v42 == v104 )
          {
LABEL_222:
            if ( v105 >= *(_DWORD *)(v41 + 24) )
            {
LABEL_54:
              sub_16CCBA0(*(_QWORD *)(a1 + 96), v34);
              v41 = *(_QWORD *)(a1 + 96);
              goto LABEL_55;
            }
            *(_DWORD *)(v41 + 28) = v105 + 1;
            *v104 = v34;
            ++*(_QWORD *)v41;
            v41 = *(_QWORD *)(a1 + 96);
          }
          else
          {
            v106 = 0;
            while ( v34 != *v42 )
            {
              if ( *v42 == -2 )
                v106 = v42;
              if ( v104 == ++v42 )
              {
                if ( !v106 )
                  goto LABEL_222;
                *v106 = v34;
                --*(_DWORD *)(v41 + 32);
                ++*(_QWORD *)v41;
                v41 = *(_QWORD *)(a1 + 96);
                break;
              }
            }
          }
LABEL_55:
          v43 = *(_QWORD *)(a1 + 24);
          v44 = *(_QWORD *)(a1 + 16);
          v144[2] = v34;
          v45 = *(_QWORD *)(a1 + 8);
          v46 = *(__int64 **)a1;
          v144[0] = v44;
          v47 = *(_QWORD *)(a1 + 48);
          v142 = v46;
          v48 = v124[0];
          v143 = v45;
          v144[1] = v43;
          v49 = sub_1632FA0(*(_QWORD *)(v34 + 40));
          v144[4] = v47;
          v144[3] = v49;
          v144[6] = &v126;
          v144[5] = v48;
          v145 = v126;
          v50 = byte_5051860;
          v146 = 0;
          if ( !byte_5051860 )
            v50 = v141 | (v47 != 0);
          v147 = v50;
          v148 = 0;
          v149 = 0;
          v150 = v41;
          v151 = 0;
          v152 = 0;
          v153 = 0;
          v154 = 0;
          v155 = 0;
          v156 = 0;
          v157 = 0;
          v158 = 0;
          v159 = 0;
          v160 = 0;
          v161 = 0;
          v162 = 0;
          v163 = 0;
          v164 = 0;
          v165 = 0;
          v166 = 0;
          v167 = 0;
          v168 = 0;
          v169 = 0;
          v170 = 0;
          v171 = 0;
          v172 = 0;
          v173 = 0;
          v174 = 0;
          v175 = 0;
          v176 = 0;
          v177 = 0;
          v178 = 0;
          v179 = 0;
          v180 = 0;
          v181 = 0;
          v182 = 1;
          v183 = 0;
          v184 = v188;
          v185 = v188;
          v186 = 16;
          v187 = 0;
          v189 = 0;
          v190 = 0;
          v191 = 0;
          v192 = 0;
          v193 = 0;
          if ( (unsigned __int8)sub_38576C0((__int64)&v142, v124[0], a3, a4, a5) )
          {
            v51 = v145 - v146;
            if ( v145 - v146 < 0 )
              v51 = 0;
            *(_DWORD *)(a1 + 76) -= v51;
          }
          v52 = *(_QWORD *)(a1 + 96);
          v53 = *(_QWORD **)(v52 + 8);
          if ( *(_QWORD **)(v52 + 16) == v53 )
          {
            v107 = &v53[*(unsigned int *)(v52 + 28)];
            if ( v53 == v107 )
            {
LABEL_211:
              v53 = v107;
            }
            else
            {
              while ( v34 != *v53 )
              {
                if ( v107 == ++v53 )
                  goto LABEL_211;
              }
            }
          }
          else
          {
            v53 = sub_16CC9F0(*(_QWORD *)(a1 + 96), v34);
            if ( v34 == *v53 )
            {
              v110 = *(_QWORD *)(v52 + 16);
              if ( v110 == *(_QWORD *)(v52 + 8) )
                v111 = *(unsigned int *)(v52 + 28);
              else
                v111 = *(unsigned int *)(v52 + 24);
              v107 = (_QWORD *)(v110 + 8 * v111);
            }
            else
            {
              v54 = *(_QWORD *)(v52 + 16);
              if ( v54 != *(_QWORD *)(v52 + 8) )
              {
LABEL_64:
                if ( !(unsigned __int8)sub_1560180(v34 + 112, 36)
                  && !(unsigned __int8)sub_1560180(v34 + 112, 37)
                  && *(_BYTE *)(a1 + 352) )
                {
                  *(_DWORD *)(a1 + 76) += *(_DWORD *)(a1 + 528);
                  *(_DWORD *)(a1 + 528) = 0;
                  *(_BYTE *)(a1 + 352) = 0;
                }
                v8 = sub_384F9A0(a1, v124[0] & 0xFFFFFFFFFFFFFFF8LL);
                if ( v185 != v184 )
                  _libc_free((unsigned __int64)v185);
                j___libc_free_0(v179);
                if ( v175 )
                  j_j___libc_free_0(v175);
                j___libc_free_0(v172);
                if ( v170 )
                {
                  v55 = v168;
                  v56 = v168 + 32LL * v170;
                  do
                  {
                    if ( *(_QWORD *)v55 != -16 && *(_QWORD *)v55 != -8 && *(_DWORD *)(v55 + 24) > 0x40u )
                    {
                      v57 = *(_QWORD *)(v55 + 16);
                      if ( v57 )
                        j_j___libc_free_0_0(v57);
                    }
                    v55 += 32LL;
                  }
                  while ( v56 != v55 );
                }
                j___libc_free_0(v168);
                j___libc_free_0(v164);
                j___libc_free_0(v160);
                j___libc_free_0(v156);
                return v8;
              }
              v53 = (_QWORD *)(v54 + 8LL * *(unsigned int *)(v52 + 28));
              v107 = v53;
            }
          }
          if ( v53 != v107 )
          {
            *v53 = -2;
            ++*(_DWORD *)(v52 + 32);
          }
          goto LABEL_64;
        }
      }
      else
      {
        v116 = 1;
        while ( v33 != (_BYTE *)-8LL )
        {
          v117 = v116 + 1;
          v31 = v29 & (v116 + v31);
          v32 = (_QWORD *)(v30 + 16LL * v31);
          v33 = (_BYTE *)*v32;
          if ( (_BYTE *)*v32 == v10 )
            goto LABEL_30;
          v116 = v117;
        }
      }
    }
    v72 = v124[0] & 0xFFFFFFFFFFFFFFF8LL;
    v73 = (_QWORD *)((v124[0] & 0xFFFFFFFFFFFFFFF8LL) + 56);
    if ( (v124[0] & 4) != 0 )
    {
      if ( (unsigned __int8)sub_1560260(v73, -1, 36) )
        return (unsigned int)sub_384F9A0(a1, v124[0] & 0xFFFFFFFFFFFFFFF8LL);
      if ( *(char *)(v72 + 23) >= 0 )
        goto LABEL_236;
      v74 = sub_1648A40(v72);
      v76 = v74 + v75;
      v77 = 0;
      if ( *(char *)(v72 + 23) < 0 )
        v77 = sub_1648A40(v72);
      if ( !(unsigned int)((v76 - v77) >> 4) )
      {
LABEL_236:
        v78 = *(_QWORD *)(v72 - 24);
        if ( !*(_BYTE *)(v78 + 16) )
        {
          v142 = *(__int64 **)(v78 + 112);
          if ( (unsigned __int8)sub_1560260(&v142, -1, 36) )
            return (unsigned int)sub_384F9A0(a1, v124[0] & 0xFFFFFFFFFFFFFFF8LL);
        }
      }
      if ( (unsigned __int8)sub_1560260(v73, -1, 37) )
        return (unsigned int)sub_384F9A0(a1, v124[0] & 0xFFFFFFFFFFFFFFF8LL);
      if ( *(char *)(v72 + 23) < 0 )
      {
        v79 = sub_1648A40(v72);
        v81 = v79 + v80;
        v82 = *(char *)(v72 + 23) >= 0 ? 0LL : sub_1648A40(v72);
        if ( v82 != v81 )
        {
          while ( *(_DWORD *)(*(_QWORD *)v82 + 8LL) <= 1u )
          {
            v82 += 16;
            if ( v81 == v82 )
              goto LABEL_131;
          }
          goto LABEL_89;
        }
      }
LABEL_131:
      v83 = *(_QWORD *)(v72 - 24);
      if ( *(_BYTE *)(v83 + 16) )
      {
LABEL_89:
        if ( *(_BYTE *)(a1 + 352) )
        {
          *(_DWORD *)(a1 + 76) += *(_DWORD *)(a1 + 528);
          *(_DWORD *)(a1 + 528) = 0;
          *(_BYTE *)(a1 + 352) = 0;
        }
        return (unsigned int)sub_384F9A0(a1, v124[0] & 0xFFFFFFFFFFFFFFF8LL);
      }
    }
    else
    {
      if ( (unsigned __int8)sub_1560260(v73, -1, 36) )
        return (unsigned int)sub_384F9A0(a1, v124[0] & 0xFFFFFFFFFFFFFFF8LL);
      if ( *(char *)(v72 + 23) >= 0 )
        goto LABEL_237;
      v95 = sub_1648A40(v72);
      v97 = v95 + v96;
      v98 = 0;
      if ( *(char *)(v72 + 23) < 0 )
        v98 = sub_1648A40(v72);
      if ( !(unsigned int)((v97 - v98) >> 4) )
      {
LABEL_237:
        v99 = *(_QWORD *)(v72 - 72);
        if ( !*(_BYTE *)(v99 + 16) )
        {
          v142 = *(__int64 **)(v99 + 112);
          if ( (unsigned __int8)sub_1560260(&v142, -1, 36) )
            return (unsigned int)sub_384F9A0(a1, v124[0] & 0xFFFFFFFFFFFFFFF8LL);
        }
      }
      if ( (unsigned __int8)sub_1560260(v73, -1, 37) )
        return (unsigned int)sub_384F9A0(a1, v124[0] & 0xFFFFFFFFFFFFFFF8LL);
      if ( *(char *)(v72 + 23) < 0 )
      {
        v100 = sub_1648A40(v72);
        v102 = v100 + v101;
        v103 = *(char *)(v72 + 23) >= 0 ? 0LL : sub_1648A40(v72);
        if ( v103 != v102 )
        {
          while ( *(_DWORD *)(*(_QWORD *)v103 + 8LL) <= 1u )
          {
            v103 += 16;
            if ( v102 == v103 )
              goto LABEL_165;
          }
          goto LABEL_89;
        }
      }
LABEL_165:
      v83 = *(_QWORD *)(v72 - 72);
      if ( *(_BYTE *)(v83 + 16) )
        goto LABEL_89;
    }
    v142 = *(__int64 **)(v83 + 112);
    if ( (unsigned __int8)sub_1560260(&v142, -1, 37) )
      return (unsigned int)sub_384F9A0(a1, v124[0] & 0xFFFFFFFFFFFFFFF8LL);
    goto LABEL_89;
  }
  v10 = *(_BYTE **)((v124[0] & 0xFFFFFFFFFFFFFFF8LL) - 24);
  if ( v10[16] )
    goto LABEL_28;
LABEL_6:
  v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  v125 = v124[0];
  v12 = *(_BYTE *)(v11 + 16);
  if ( v12 <= 0x17u )
  {
    v11 = 0;
  }
  else if ( v12 == 78 )
  {
    v11 |= 4u;
  }
  else if ( v12 != 29 )
  {
    v11 = 0;
  }
  v8 = sub_14D90D0(v11, (__int64)v10);
  if ( !(_BYTE)v8 )
    goto LABEL_85;
  v142 = v144;
  v143 = 0x400000000LL;
  v13 = sub_165AFC0(&v125);
  if ( HIDWORD(v143) < v13 )
    sub_16CD150((__int64)&v142, v144, v13, 8, v14, v15);
  v16 = v125 & 0xFFFFFFFFFFFFFFF8LL;
  v17 = (v125 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v125 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  v18 = *(_BYTE *)((v125 & 0xFFFFFFFFFFFFFFF8LL) + 23);
  if ( (v125 & 4) != 0 )
  {
    if ( v18 < 0 )
    {
      v121 = v125 & 0xFFFFFFFFFFFFFFF8LL;
      v19 = sub_1648A40(v16);
      v16 = v121;
      v21 = v19 + v20;
      if ( *(char *)(v121 + 23) >= 0 )
      {
        if ( (unsigned int)(v21 >> 4) )
          goto LABEL_233;
      }
      else
      {
        v22 = sub_1648A40(v121);
        v16 = v121;
        if ( (unsigned int)((v21 - v22) >> 4) )
        {
          if ( *(char *)(v121 + 23) < 0 )
          {
            v23 = *(_DWORD *)(sub_1648A40(v121) + 8);
            if ( *(char *)(v121 + 23) >= 0 )
              BUG();
            v24 = sub_1648A40(v121);
            v16 = v121;
            v26 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v24 + v25 - 4) - v23);
            goto LABEL_139;
          }
LABEL_233:
          BUG();
        }
      }
    }
    v26 = -24;
    goto LABEL_139;
  }
  if ( v18 >= 0 )
    goto LABEL_150;
  v122 = v125 & 0xFFFFFFFFFFFFFFF8LL;
  v65 = sub_1648A40(v16);
  v16 = v122;
  v67 = v65 + v66;
  if ( *(char *)(v122 + 23) >= 0 )
  {
    if ( (unsigned int)(v67 >> 4) )
LABEL_231:
      BUG();
LABEL_150:
    v26 = -72;
    goto LABEL_139;
  }
  v68 = sub_1648A40(v122);
  v16 = v122;
  if ( !(unsigned int)((v67 - v68) >> 4) )
    goto LABEL_150;
  if ( *(char *)(v122 + 23) >= 0 )
    goto LABEL_231;
  v69 = *(_DWORD *)(sub_1648A40(v122) + 8);
  if ( *(char *)(v122 + 23) >= 0 )
    BUG();
  v70 = sub_1648A40(v122);
  v16 = v122;
  v26 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v70 + v71 - 4) - v69);
LABEL_139:
  v86 = v16 + v26;
  if ( v86 != v17 )
  {
    while ( 1 )
    {
      v88 = *(_QWORD *)v17;
      if ( *(_BYTE *)(*(_QWORD *)v17 + 16LL) <= 0x10u )
      {
        v87 = (unsigned int)v143;
        if ( (unsigned int)v143 >= HIDWORD(v143) )
          goto LABEL_148;
      }
      else
      {
        v89 = *(_DWORD *)(a1 + 160);
        if ( !v89 )
          goto LABEL_83;
        v90 = v89 - 1;
        v91 = *(_QWORD *)(a1 + 144);
        v92 = v90 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
        v93 = (__int64 *)(v91 + 16LL * v92);
        v94 = *v93;
        if ( v88 != *v93 )
        {
          v118 = 1;
          while ( v94 != -8 )
          {
            v119 = v118 + 1;
            v92 = v90 & (v118 + v92);
            v93 = (__int64 *)(v91 + 16LL * v92);
            v94 = *v93;
            if ( v88 == *v93 )
              goto LABEL_146;
            v118 = v119;
          }
          goto LABEL_83;
        }
LABEL_146:
        v88 = v93[1];
        if ( !v88 )
          goto LABEL_83;
        v87 = (unsigned int)v143;
        if ( (unsigned int)v143 >= HIDWORD(v143) )
        {
LABEL_148:
          v123 = v88;
          sub_16CD150((__int64)&v142, v144, 0, 8, v14, v15);
          v87 = (unsigned int)v143;
          v88 = v123;
        }
      }
      v17 += 24LL;
      v142[v87] = v88;
      v58 = (unsigned int)(v143 + 1);
      LODWORD(v143) = v143 + 1;
      if ( v17 == v86 )
        goto LABEL_78;
    }
  }
  v58 = (unsigned int)v143;
LABEL_78:
  v59 = 0;
  v60 = *(_BYTE *)((v125 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( v60 > 0x17u )
  {
    if ( v60 == 78 )
    {
      v59 = v125 & 0xFFFFFFFFFFFFFFF8LL | 4;
    }
    else if ( v60 == 29 )
    {
      v59 = v125 & 0xFFFFFFFFFFFFFFF8LL;
    }
  }
  v61 = sub_14DA350(v59, (__int64)v10, v142, v58, 0);
  if ( !v61 )
  {
LABEL_83:
    if ( v142 != v144 )
      _libc_free((unsigned __int64)v142);
LABEL_85:
    v62 = v124[0] & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_BYTE *)((v124[0] & 0xFFFFFFFFFFFFFFF8LL) + 16) == 78 )
    {
      v108 = *(_QWORD *)(v62 - 24);
      if ( !*(_BYTE *)(v108 + 16) && (*(_BYTE *)(v108 + 33) & 0x20) != 0 && v62 )
      {
        v109 = *(_DWORD *)(v108 + 36);
        if ( v109 > 0x89 )
        {
          if ( v109 - 213 <= 1 )
          {
            *(_BYTE *)(a1 + 89) = 1;
            return 0;
          }
        }
        else if ( v109 > 0x6B )
        {
          switch ( v109 )
          {
            case 0x6Cu:
            case 0x78u:
              *(_BYTE *)(a1 + 88) = 1;
              v8 = 0;
              break;
            case 0x76u:
              *(_DWORD *)(a1 + 76) += 15;
              v8 = 0;
              break;
            case 0x85u:
            case 0x87u:
            case 0x89u:
              v8 = *(unsigned __int8 *)(a1 + 352);
              if ( (_BYTE)v8 )
              {
                v8 = 0;
                *(_DWORD *)(a1 + 76) += *(_DWORD *)(a1 + 528);
                *(_DWORD *)(a1 + 528) = 0;
                *(_BYTE *)(a1 + 352) = 0;
              }
              break;
            default:
              goto LABEL_190;
          }
          return v8;
        }
LABEL_190:
        if ( !(unsigned __int8)sub_38514E0(v124) && !sub_14AAB80(v62) )
          goto LABEL_89;
        return (unsigned int)sub_384F9A0(a1, v124[0] & 0xFFFFFFFFFFFFFFF8LL);
      }
    }
    if ( v10 != (_BYTE *)sub_15F2060(v124[0] & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( sub_14A29D0(*(__int64 **)a1, v10) )
      {
        v112 = 5 * sub_165AFC0(v124);
        v113 = v124[0];
        v114 = *(_DWORD *)(a1 + 76) + v112;
        *(_DWORD *)(a1 + 76) = v114;
        v115 = (v113 & 4) != 0 ? (v113 & 0xFFFFFFFFFFFFFFF8LL) - 24 : (v113 & 0xFFFFFFFFFFFFFFF8LL) - 72;
        if ( *(_BYTE *)(*(_QWORD *)v115 + 16LL) != 20 )
          *(_DWORD *)(a1 + 76) = v114 + 25;
      }
      if ( (unsigned __int8)sub_38514E0(v124) )
        return (unsigned int)sub_384F9A0(a1, v124[0] & 0xFFFFFFFFFFFFFFF8LL);
      goto LABEL_89;
    }
LABEL_92:
    *(_BYTE *)(a1 + 82) = 1;
    return 0;
  }
  v126 = v125 & 0xFFFFFFFFFFFFFFF8LL;
  sub_38526A0(a1 + 136, (__int64 *)&v126)[1] = v61;
  if ( v142 != v144 )
    _libc_free((unsigned __int64)v142);
  return v8;
}
