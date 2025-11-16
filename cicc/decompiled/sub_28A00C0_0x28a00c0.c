// Function: sub_28A00C0
// Address: 0x28a00c0
//
__int64 __fastcall sub_28A00C0(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax
  __int64 *v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // r14
  unsigned __int8 v9; // al
  unsigned __int8 **v10; // rdx
  unsigned __int8 *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r8
  __int64 v14; // r10
  __int64 v15; // rdx
  __int64 v16; // r12
  __int64 v17; // rsi
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r12
  unsigned __int64 v20; // rdi
  __int64 result; // rax
  __int64 *v22; // rbx
  __int64 v23; // rax
  __int64 *v24; // r15
  __int64 *v25; // rax
  __int64 v26; // r12
  __int64 v27; // rdx
  __int64 *v28; // rsi
  __int64 v29; // rdi
  __int64 v30; // rdx
  __int64 *v31; // rdx
  int v32; // esi
  int v33; // r12d
  __int64 *v34; // r11
  __int64 v35; // rdx
  __int64 *v36; // rdi
  int v37; // eax
  __int64 v38; // r12
  __int64 v39; // rax
  unsigned __int64 v40; // rdx
  __int64 *v41; // rbx
  __int64 *v42; // r14
  __int64 v43; // r9
  __int64 v44; // r12
  int v45; // r10d
  __int64 *v46; // rsi
  __int64 v47; // rax
  unsigned int v48; // edx
  __int64 v49; // rdi
  __int64 *v50; // r12
  __int64 *v51; // rbx
  __int64 v52; // rdi
  __int64 v53; // r12
  __int64 v54; // rsi
  __int64 v55; // r13
  unsigned __int8 **v56; // rdx
  unsigned __int8 *v57; // rdi
  unsigned __int8 v58; // al
  __int64 v59; // rax
  __int64 *v60; // r12
  __int64 *v61; // r13
  unsigned int v62; // eax
  __int64 *v63; // rdi
  __int64 v64; // rcx
  int v65; // esi
  __int64 *v66; // r10
  int v67; // edx
  unsigned int v68; // r13d
  unsigned int v69; // ebx
  __int64 v70; // rax
  __int64 v71; // r13
  __int64 v72; // r13
  __int64 v73; // r13
  __int64 v74; // r13
  __int64 v75; // rbx
  __int64 v76; // r9
  __m128i *v77; // rax
  unsigned __int64 *v78; // rbx
  unsigned __int64 *v79; // r12
  unsigned __int64 v80; // rdi
  unsigned int v81; // eax
  __int64 v82; // rbx
  int v83; // r11d
  int v84; // r11d
  __int64 v85; // r12
  unsigned __int64 *v86; // [rsp+10h] [rbp-6F0h]
  __int64 *v87; // [rsp+68h] [rbp-698h]
  unsigned __int64 *v88; // [rsp+70h] [rbp-690h]
  unsigned int v89; // [rsp+9Ch] [rbp-664h]
  __int64 v90; // [rsp+A0h] [rbp-660h]
  __int64 v91; // [rsp+A8h] [rbp-658h]
  __int64 v92; // [rsp+A8h] [rbp-658h]
  __int64 v93; // [rsp+A8h] [rbp-658h]
  __int64 v94; // [rsp+A8h] [rbp-658h]
  unsigned int v95; // [rsp+E8h] [rbp-618h]
  unsigned int v96; // [rsp+ECh] [rbp-614h]
  __int64 v98; // [rsp+108h] [rbp-5F8h]
  __int64 v99; // [rsp+110h] [rbp-5F0h]
  unsigned __int64 v100; // [rsp+110h] [rbp-5F0h]
  __int64 *v101; // [rsp+118h] [rbp-5E8h]
  __int64 v102; // [rsp+128h] [rbp-5D8h] BYREF
  __int64 v103; // [rsp+130h] [rbp-5D0h] BYREF
  __int64 v104; // [rsp+138h] [rbp-5C8h]
  __int64 v105; // [rsp+140h] [rbp-5C0h]
  unsigned int v106; // [rsp+148h] [rbp-5B8h]
  __int64 v107; // [rsp+150h] [rbp-5B0h] BYREF
  __int64 v108; // [rsp+158h] [rbp-5A8h]
  __int64 v109; // [rsp+160h] [rbp-5A0h]
  unsigned int v110; // [rsp+168h] [rbp-598h]
  unsigned __int64 *v111; // [rsp+170h] [rbp-590h]
  __int64 v112; // [rsp+178h] [rbp-588h]
  __int64 *v113; // [rsp+180h] [rbp-580h] BYREF
  __int64 v114; // [rsp+188h] [rbp-578h]
  _BYTE v115[32]; // [rsp+190h] [rbp-570h] BYREF
  __int64 v116[2]; // [rsp+1B0h] [rbp-550h] BYREF
  __int64 v117; // [rsp+1C0h] [rbp-540h] BYREF
  __int64 *v118; // [rsp+1D0h] [rbp-530h]
  __int64 v119; // [rsp+1E0h] [rbp-520h] BYREF
  __int64 v120[2]; // [rsp+200h] [rbp-500h] BYREF
  _QWORD v121[2]; // [rsp+210h] [rbp-4F0h] BYREF
  _QWORD *v122; // [rsp+220h] [rbp-4E0h]
  _QWORD v123[4]; // [rsp+230h] [rbp-4D0h] BYREF
  __int64 v124[2]; // [rsp+250h] [rbp-4B0h] BYREF
  _QWORD v125[2]; // [rsp+260h] [rbp-4A0h] BYREF
  _QWORD *v126; // [rsp+270h] [rbp-490h]
  _QWORD v127[4]; // [rsp+280h] [rbp-480h] BYREF
  __int64 v128; // [rsp+2A0h] [rbp-460h] BYREF
  char *v129; // [rsp+2A8h] [rbp-458h]
  __int64 v130; // [rsp+2B0h] [rbp-450h]
  int v131; // [rsp+2B8h] [rbp-448h]
  char v132; // [rsp+2BCh] [rbp-444h]
  char v133; // [rsp+2C0h] [rbp-440h] BYREF
  __m128i v134; // [rsp+300h] [rbp-400h] BYREF
  __m128i v135; // [rsp+310h] [rbp-3F0h] BYREF
  _QWORD *v136; // [rsp+320h] [rbp-3E0h]
  void *v137; // [rsp+328h] [rbp-3D8h] BYREF
  _QWORD v138[6]; // [rsp+330h] [rbp-3D0h] BYREF
  int v139; // [rsp+360h] [rbp-3A0h]
  __int64 v140; // [rsp+368h] [rbp-398h]
  __int64 v141; // [rsp+370h] [rbp-390h]
  __int64 *v142; // [rsp+378h] [rbp-388h]
  __int64 *v143; // [rsp+380h] [rbp-380h]
  __int64 v144; // [rsp+388h] [rbp-378h]
  __int64 v145; // [rsp+390h] [rbp-370h]
  char *v146; // [rsp+398h] [rbp-368h]
  __int64 v147; // [rsp+3A0h] [rbp-360h]
  int v148; // [rsp+3A8h] [rbp-358h]
  char v149; // [rsp+3ACh] [rbp-354h]
  char v150; // [rsp+3B0h] [rbp-350h] BYREF
  __int64 v151; // [rsp+3F0h] [rbp-310h] BYREF
  __int64 v152; // [rsp+3F8h] [rbp-308h]
  __int64 v153; // [rsp+400h] [rbp-300h]
  __int64 v154; // [rsp+408h] [rbp-2F8h]
  __int64 *v155; // [rsp+410h] [rbp-2F0h] BYREF
  __int64 v156; // [rsp+418h] [rbp-2E8h]
  _BYTE v157[256]; // [rsp+420h] [rbp-2E0h] BYREF
  _BYTE v158[12]; // [rsp+520h] [rbp-1E0h] BYREF
  unsigned int v159; // [rsp+52Ch] [rbp-1D4h]
  unsigned int v160; // [rsp+530h] [rbp-1D0h]
  unsigned int v161; // [rsp+534h] [rbp-1CCh]
  unsigned int v162; // [rsp+538h] [rbp-1C8h]
  unsigned __int64 *v163; // [rsp+570h] [rbp-190h]
  unsigned int v164; // [rsp+578h] [rbp-188h]
  char v165; // [rsp+580h] [rbp-180h] BYREF

  v1 = sub_B2BE50(**(_QWORD **)(a1 + 8));
  if ( sub_B6EA50(v1)
    || (v85 = sub_B6F970(v1),
        (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v85 + 32LL))(
          v85,
          "lower-matrix-intrinsics",
          23))
    || (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v85 + 40LL))(
         v85,
         "lower-matrix-intrinsics",
         23)
    || (result = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v85 + 24LL))(
                   v85,
                   "lower-matrix-intrinsics",
                   23),
        (_BYTE)result) )
  {
    v112 = 0;
    v111 = (unsigned __int64 *)&v113;
    v2 = *(_QWORD *)a1;
    v107 = 0;
    v108 = 0;
    v3 = *(__int64 **)(v2 + 32);
    v109 = 0;
    v4 = *(unsigned int *)(v2 + 40);
    v110 = 0;
    v101 = &v3[22 * v4];
    if ( v3 != v101 )
    {
      do
      {
        if ( sub_B92180(*(_QWORD *)(a1 + 16)) )
        {
          v8 = sub_B10CD0(*v3 + 48);
          while ( v8 )
          {
            v9 = *(_BYTE *)(v8 - 16);
            if ( (v9 & 2) != 0 )
            {
LABEL_10:
              v10 = *(unsigned __int8 ***)(v8 - 32);
              goto LABEL_11;
            }
            while ( 1 )
            {
              v10 = (unsigned __int8 **)(v8 - 16 - 8LL * ((v9 >> 2) & 0xF));
LABEL_11:
              v11 = *v10;
              if ( **v10 != 18 )
                v11 = sub_AF34D0(v11);
              *(_QWORD *)v158 = v11;
              v12 = sub_289FE10((__int64)&v107, (__int64 *)v158);
              v14 = *v3;
              v15 = *(unsigned int *)(v12 + 8);
              if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
              {
                v98 = *v3;
                v99 = v12;
                sub_C8D5F0(v12, (const void *)(v12 + 16), v15 + 1, 8u, v13, v15 + 1);
                v12 = v99;
                v14 = v98;
                v15 = *(unsigned int *)(v99 + 8);
              }
              *(_QWORD *)(*(_QWORD *)v12 + 8 * v15) = v14;
              ++*(_DWORD *)(v12 + 8);
              sub_B10CB0(v158, v8);
              v8 = sub_B10D40((__int64)v158);
              if ( !*(_QWORD *)v158 )
                break;
              sub_B91220((__int64)v158, *(__int64 *)v158);
              if ( !v8 )
                goto LABEL_5;
              v9 = *(_BYTE *)(v8 - 16);
              if ( (v9 & 2) != 0 )
                goto LABEL_10;
            }
          }
        }
        else
        {
          *(_QWORD *)v158 = 0;
          v5 = sub_289FE10((__int64)&v107, (__int64 *)v158);
          sub_94F890(v5, *v3);
        }
LABEL_5:
        v3 += 22;
      }
      while ( v101 != v3 );
      v19 = v111;
      v86 = &v111[11 * (unsigned int)v112];
      if ( v86 != v111 )
      {
        v88 = v111;
        while ( 1 )
        {
          v22 = (__int64 *)v88[1];
          v23 = *((unsigned int *)v88 + 4);
          v151 = 0;
          v152 = 0;
          v153 = 0;
          v24 = &v22[v23];
          v154 = 0;
          v155 = (__int64 *)v157;
          v156 = 0x2000000000LL;
          if ( v22 == v24 )
          {
            v113 = (__int64 *)v115;
            v114 = 0x400000000LL;
LABEL_197:
            v103 = 0;
            v104 = 0;
            v105 = 0;
            v106 = 0;
LABEL_198:
            v16 = 0;
            v17 = 0;
            goto LABEL_20;
          }
          while ( 2 )
          {
            v25 = v155;
            v26 = *v22;
            v27 = 8LL * (unsigned int)v156;
            v28 = &v155[(unsigned __int64)v27 / 8];
            v29 = v27 >> 3;
            v30 = v27 >> 5;
            if ( !v30 )
              goto LABEL_89;
            v31 = &v155[4 * v30];
            do
            {
              if ( *v25 == v26 )
                goto LABEL_44;
              if ( v25[1] == v26 )
              {
                ++v25;
                goto LABEL_44;
              }
              if ( v25[2] == v26 )
              {
                v25 += 2;
                goto LABEL_44;
              }
              if ( v25[3] == v26 )
              {
                v25 += 3;
LABEL_44:
                if ( v28 == v25 )
                  goto LABEL_93;
                goto LABEL_45;
              }
              v25 += 4;
            }
            while ( v31 != v25 );
            v29 = v28 - v25;
LABEL_89:
            if ( v29 == 2 )
              goto LABEL_184;
            if ( v29 != 3 )
            {
              if ( v29 == 1 )
                goto LABEL_92;
              goto LABEL_93;
            }
            if ( *v25 == v26 )
              goto LABEL_44;
            ++v25;
LABEL_184:
            if ( *v25 == v26 )
              goto LABEL_44;
            ++v25;
LABEL_92:
            if ( *v25 == v26 )
              goto LABEL_44;
LABEL_93:
            if ( (unsigned __int64)(unsigned int)v156 + 1 > HIDWORD(v156) )
            {
              sub_C8D5F0((__int64)&v155, v157, (unsigned int)v156 + 1LL, 8u, v6, v7);
              v28 = &v155[(unsigned int)v156];
            }
            *v28 = v26;
            v59 = (unsigned int)(v156 + 1);
            LODWORD(v156) = v59;
            if ( (unsigned int)v59 > 0x20 )
            {
              v60 = v155;
              v61 = &v155[v59];
              while ( 1 )
              {
                v65 = v154;
                if ( !(_DWORD)v154 )
                  break;
                v7 = (unsigned int)(v154 - 1);
                v6 = v152;
                v62 = v7 & (((unsigned int)*v60 >> 9) ^ ((unsigned int)*v60 >> 4));
                v63 = (__int64 *)(v152 + 8LL * v62);
                v64 = *v63;
                if ( *v60 != *v63 )
                {
                  v84 = 1;
                  v66 = 0;
                  while ( v64 != -4096 )
                  {
                    if ( v64 != -8192 || v66 )
                      v63 = v66;
                    v62 = v7 & (v84 + v62);
                    v64 = *(_QWORD *)(v152 + 8LL * v62);
                    if ( *v60 == v64 )
                      goto LABEL_98;
                    ++v84;
                    v66 = v63;
                    v63 = (__int64 *)(v152 + 8LL * v62);
                  }
                  if ( !v66 )
                    v66 = v63;
                  ++v151;
                  v67 = v153 + 1;
                  *(_QWORD *)v158 = v66;
                  if ( 4 * ((int)v153 + 1) < (unsigned int)(3 * v154) )
                  {
                    if ( (int)v154 - HIDWORD(v153) - v67 <= (unsigned int)v154 >> 3 )
                    {
LABEL_102:
                      sub_CE2A30((__int64)&v151, v65);
                      sub_DA5B20((__int64)&v151, v60, v158);
                      v66 = *(__int64 **)v158;
                      v67 = v153 + 1;
                    }
                    LODWORD(v153) = v67;
                    if ( *v66 != -4096 )
                      --HIDWORD(v153);
                    *v66 = *v60;
                    goto LABEL_98;
                  }
LABEL_101:
                  v65 = 2 * v154;
                  goto LABEL_102;
                }
LABEL_98:
                if ( v61 == ++v60 )
                  goto LABEL_45;
              }
              ++v151;
              *(_QWORD *)v158 = 0;
              goto LABEL_101;
            }
LABEL_45:
            if ( v24 != ++v22 )
            {
LABEL_46:
              if ( !(_DWORD)v153 )
                continue;
              v32 = v154;
              if ( (_DWORD)v154 )
              {
                v33 = 1;
                v34 = 0;
                v7 = v152;
                LODWORD(v35) = (v154 - 1) & (((unsigned int)*v22 >> 9) ^ ((unsigned int)*v22 >> 4));
                v36 = (__int64 *)(v152 + 8LL * (unsigned int)v35);
                v6 = *v36;
                if ( *v22 == *v36 )
                  goto LABEL_45;
                while ( v6 != -4096 )
                {
                  if ( v34 || v6 != -8192 )
                    v36 = v34;
                  v35 = ((_DWORD)v154 - 1) & (unsigned int)(v35 + v33);
                  v6 = *(_QWORD *)(v152 + 8 * v35);
                  if ( *v22 == v6 )
                    goto LABEL_45;
                  ++v33;
                  v34 = v36;
                  v36 = (__int64 *)(v152 + 8 * v35);
                }
                if ( !v34 )
                  v34 = v36;
                v37 = v153 + 1;
                ++v151;
                *(_QWORD *)v158 = v34;
                if ( 4 * ((int)v153 + 1) < (unsigned int)(3 * v154) )
                {
                  if ( (int)v154 - HIDWORD(v153) - v37 > (unsigned int)v154 >> 3 )
                    goto LABEL_54;
                  goto LABEL_195;
                }
              }
              else
              {
                ++v151;
                *(_QWORD *)v158 = 0;
              }
              v32 = 2 * v154;
LABEL_195:
              sub_CE2A30((__int64)&v151, v32);
              sub_DA5B20((__int64)&v151, v22, v158);
              v34 = *(__int64 **)v158;
              v37 = v153 + 1;
LABEL_54:
              LODWORD(v153) = v37;
              if ( *v34 != -4096 )
                --HIDWORD(v153);
              v38 = *v22;
              *v34 = *v22;
              v39 = (unsigned int)v156;
              v40 = (unsigned int)v156 + 1LL;
              if ( v40 > HIDWORD(v156) )
              {
                sub_C8D5F0((__int64)&v155, v157, v40, 8u, v6, v7);
                v39 = (unsigned int)v156;
              }
              ++v22;
              v155[v39] = v38;
              LODWORD(v156) = v156 + 1;
              if ( v24 == v22 )
                break;
              goto LABEL_46;
            }
            break;
          }
          v41 = v155;
          v42 = &v155[(unsigned int)v156];
          v113 = (__int64 *)v115;
          v114 = 0x400000000LL;
          if ( v155 == v42 )
            goto LABEL_197;
          v43 = 0;
          do
          {
            v44 = *v41;
            if ( *(_BYTE *)(*(_QWORD *)(*v41 + 8) + 8LL) != 7 )
            {
              v6 = *(_QWORD *)(v44 + 16);
              if ( v6 )
              {
                v45 = v153;
                do
                {
                  v47 = *(_QWORD *)(v6 + 24);
                  *(_QWORD *)v158 = v47;
                  if ( v45 )
                  {
                    if ( (_DWORD)v154 )
                    {
                      v48 = (v154 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
                      v49 = *(_QWORD *)(v152 + 8LL * v48);
                      if ( v47 == v49 )
                      {
LABEL_69:
                        if ( v6 )
                          goto LABEL_70;
                        break;
                      }
                      v83 = 1;
                      while ( v49 != -4096 )
                      {
                        v48 = (v154 - 1) & (v83 + v48);
                        v49 = *(_QWORD *)(v152 + 8LL * v48);
                        if ( v47 == v49 )
                          goto LABEL_69;
                        ++v83;
                      }
                    }
                  }
                  else
                  {
                    v46 = &v155[(unsigned int)v156];
                    if ( v46 != sub_28946A0(v155, (__int64)v46, (__int64 *)v158) )
                      goto LABEL_69;
                  }
                  v6 = *(_QWORD *)(v6 + 8);
                }
                while ( v6 );
              }
            }
            if ( v43 + 1 > (unsigned __int64)HIDWORD(v114) )
            {
              sub_C8D5F0((__int64)&v113, v115, v43 + 1, 8u, v6, v43);
              v43 = (unsigned int)v114;
            }
            v113[v43] = v44;
            v43 = (unsigned int)(v114 + 1);
            LODWORD(v114) = v114 + 1;
LABEL_70:
            ++v41;
          }
          while ( v42 != v41 );
          v50 = v113;
          v103 = 0;
          v104 = 0;
          v105 = 0;
          v51 = &v113[v43];
          v106 = 0;
          if ( v51 == v113 )
            goto LABEL_198;
          do
          {
            v52 = *v50++;
            sub_289E060(v52, v52, (__int64)&v151, (__int64)&v103);
          }
          while ( v51 != v50 );
          v87 = &v113[(unsigned int)v114];
          if ( v87 == v113 )
          {
            v16 = v104;
            v81 = v106;
            v17 = 56LL * v106;
          }
          else
          {
            v100 = (unsigned __int64)v113;
            do
            {
              v53 = *(_QWORD *)v100;
              v54 = *(_QWORD *)(*(_QWORD *)v100 + 48LL);
              v102 = v54;
              if ( v54 )
                sub_B96E90((__int64)&v102, v54, 1);
              v55 = sub_B10CD0(v53 + 48);
              if ( v55 )
              {
                while ( 1 )
                {
                  v58 = *(_BYTE *)(v55 - 16);
                  if ( (v58 & 2) != 0 )
                    v56 = *(unsigned __int8 ***)(v55 - 32);
                  else
                    v56 = (unsigned __int8 **)(v55 - 16 - 8LL * ((v58 >> 2) & 0xF));
                  v57 = *v56;
                  if ( **v56 != 18 )
                    v57 = sub_AF34D0(v57);
                  if ( (unsigned __int8 *)*v88 == v57 )
                    break;
                  sub_B10CB0(v158, v55);
                  v55 = sub_B10D40((__int64)v158);
                  if ( *(_QWORD *)v158 )
                    sub_B91220((__int64)v158, *(__int64 *)v158);
                  if ( !v55 )
                    goto LABEL_113;
                }
                sub_B10CB0(v158, v55);
                if ( v102 )
                  sub_B91220((__int64)&v102, v102);
                v102 = *(_QWORD *)v158;
                if ( *(_QWORD *)v158 )
                  sub_B976B0((__int64)v158, *(unsigned __int8 **)v158, (__int64)&v102);
              }
LABEL_113:
              v128 = 0;
              v129 = &v133;
              v130 = 8;
              v131 = 0;
              v132 = 1;
              sub_2895EE0((__int64)v158, (__int64 *)a1, v53, (__int64)&v128, (__int64)&v151, (__int64)&v103);
              v89 = *(_DWORD *)v158;
              v90 = *(_QWORD *)&v158[4];
              v68 = v160;
              v95 = v159;
              v96 = v161;
              v69 = v162;
              v91 = *(_QWORD *)(v53 + 40);
              sub_B157E0((__int64)&v134, &v102);
              sub_B17430((__int64)v158, (__int64)"lower-matrix-intrinsics", (__int64)"matrix-lowered", 14, &v134, v91);
              sub_B18290((__int64)v158, "Lowered with ", 0xDu);
              sub_B169E0(v116, "NumStores", 9, v89);
              v92 = sub_23FD640((__int64)v158, (__int64)v116);
              sub_B18290(v92, " stores, ", 9u);
              sub_B169E0(v120, "NumLoads", 8, v90);
              v93 = sub_23FD640(v92, (__int64)v120);
              sub_B18290(v93, " loads, ", 8u);
              sub_B169E0(v124, "NumComputeOps", 13, HIDWORD(v90));
              v94 = sub_23FD640(v93, (__int64)v124);
              sub_B18290(v94, " compute ops, ", 0xEu);
              sub_B169E0(v134.m128i_i64, "NumExposedTransposes", 20, v95);
              v70 = sub_23FD640(v94, (__int64)&v134);
              sub_B18290(v70, " exposed transposes", 0x13u);
              if ( v136 != v138 )
                j_j___libc_free_0((unsigned __int64)v136);
              if ( (__m128i *)v134.m128i_i64[0] != &v135 )
                j_j___libc_free_0(v134.m128i_u64[0]);
              if ( v126 != v127 )
                j_j___libc_free_0((unsigned __int64)v126);
              if ( (_QWORD *)v124[0] != v125 )
                j_j___libc_free_0(v124[0]);
              if ( v122 != v123 )
                j_j___libc_free_0((unsigned __int64)v122);
              if ( (_QWORD *)v120[0] != v121 )
                j_j___libc_free_0(v120[0]);
              if ( v118 != &v119 )
                j_j___libc_free_0((unsigned __int64)v118);
              if ( (__int64 *)v116[0] != &v117 )
                j_j___libc_free_0(v116[0]);
              if ( v69 | v68 | v96 )
              {
                sub_B18290((__int64)v158, ",\nadditionally ", 0xFu);
                sub_B169E0(v120, "NumStores", 9, v68);
                v71 = sub_23FD640((__int64)v158, (__int64)v120);
                sub_B18290(v71, " stores, ", 9u);
                sub_B169E0(v124, "NumLoads", 8, v96);
                v72 = sub_23FD640(v71, (__int64)v124);
                sub_B18290(v72, " loads, ", 8u);
                sub_B169E0(v134.m128i_i64, "NumFPOps", 8, v69);
                v73 = sub_23FD640(v72, (__int64)&v134);
                sub_B18290(v73, " compute ops", 0xCu);
                sub_B18290(v73, " are shared with other expressions", 0x22u);
                if ( v136 != v138 )
                  j_j___libc_free_0((unsigned __int64)v136);
                if ( (__m128i *)v134.m128i_i64[0] != &v135 )
                  j_j___libc_free_0(v134.m128i_u64[0]);
                if ( v126 != v127 )
                  j_j___libc_free_0((unsigned __int64)v126);
                if ( (_QWORD *)v124[0] != v125 )
                  j_j___libc_free_0(v124[0]);
                if ( v122 != v123 )
                  j_j___libc_free_0((unsigned __int64)v122);
                if ( (_QWORD *)v120[0] != v121 )
                  j_j___libc_free_0(v120[0]);
              }
              v135.m128i_i8[8] = 0;
              v74 = *(_QWORD *)(a1 + 24);
              v75 = *(_QWORD *)a1;
              v134.m128i_i32[0] = 100;
              v135.m128i_i64[0] = 0;
              memset(v138, 0, 32);
              v134.m128i_i64[1] = (__int64)&v135.m128i_i64[1];
              v138[4] = 0x100000000LL;
              v137 = &unk_49DD210;
              v138[5] = &v134.m128i_i64[1];
              sub_CB5980((__int64)&v137, 0, 0, 0);
              v140 = v74;
              v142 = &v103;
              v139 = 0;
              v143 = &v151;
              v141 = v75;
              v146 = &v150;
              v144 = v53;
              v145 = 0;
              v147 = 8;
              v148 = 0;
              v149 = 1;
              sub_2896DF0((__int64)&v134, (unsigned __int8 *)v53, 0, 0, 0, v76);
              v124[0] = (__int64)v125;
              sub_2894EE0(v124, (_BYTE *)v134.m128i_i64[1], v134.m128i_i64[1] + v135.m128i_i64[0]);
              if ( !v149 )
                _libc_free((unsigned __int64)v146);
              v137 = &unk_49DD210;
              sub_CB5840((__int64)&v137);
              if ( (unsigned __int64 *)v134.m128i_i64[1] != &v135.m128i_u64[1] )
                j_j___libc_free_0(v134.m128i_u64[1]);
              v77 = (__m128i *)sub_2241130((unsigned __int64 *)v124, 0, 0, "\n", 1u);
              v134.m128i_i64[0] = (__int64)&v135;
              if ( (__m128i *)v77->m128i_i64[0] == &v77[1] )
              {
                v135 = _mm_loadu_si128(v77 + 1);
              }
              else
              {
                v134.m128i_i64[0] = v77->m128i_i64[0];
                v135.m128i_i64[0] = v77[1].m128i_i64[0];
              }
              v134.m128i_i64[1] = v77->m128i_i64[1];
              v77->m128i_i64[0] = (__int64)v77[1].m128i_i64;
              v77->m128i_i64[1] = 0;
              v77[1].m128i_i8[0] = 0;
              sub_B18290((__int64)v158, (__int8 *)v134.m128i_i64[0], v134.m128i_u64[1]);
              if ( (__m128i *)v134.m128i_i64[0] != &v135 )
                j_j___libc_free_0(v134.m128i_u64[0]);
              if ( (_QWORD *)v124[0] != v125 )
                j_j___libc_free_0(v124[0]);
              sub_1049740(*(__int64 **)(a1 + 8), (__int64)v158);
              v78 = v163;
              *(_QWORD *)v158 = &unk_49D9D40;
              v79 = &v163[10 * v164];
              if ( v163 != v79 )
              {
                do
                {
                  v79 -= 10;
                  v80 = v79[4];
                  if ( (unsigned __int64 *)v80 != v79 + 6 )
                    j_j___libc_free_0(v80);
                  if ( (unsigned __int64 *)*v79 != v79 + 2 )
                    j_j___libc_free_0(*v79);
                }
                while ( v78 != v79 );
                v79 = v163;
              }
              if ( v79 != (unsigned __int64 *)&v165 )
                _libc_free((unsigned __int64)v79);
              if ( !v132 )
                _libc_free((unsigned __int64)v129);
              if ( v102 )
                sub_B91220((__int64)&v102, v102);
              v100 += 8LL;
            }
            while ( v87 != (__int64 *)v100 );
            v16 = v104;
            v81 = v106;
            v17 = 56LL * v106;
          }
          if ( v81 )
          {
            v82 = v16 + v17;
            do
            {
              if ( *(_QWORD *)v16 != -4096 && *(_QWORD *)v16 != -8192 && !*(_BYTE *)(v16 + 36) )
                _libc_free(*(_QWORD *)(v16 + 16));
              v16 += 56;
            }
            while ( v82 != v16 );
            v16 = v104;
            v17 = 56LL * v106;
          }
LABEL_20:
          sub_C7D6A0(v16, v17, 8);
          if ( v113 != (__int64 *)v115 )
            _libc_free((unsigned __int64)v113);
          if ( v155 != (__int64 *)v157 )
            _libc_free((unsigned __int64)v155);
          sub_C7D6A0(v152, 8LL * (unsigned int)v154, 8);
          v88 += 11;
          if ( v86 == v88 )
          {
            v18 = v111;
            v19 = &v111[11 * (unsigned int)v112];
            if ( v111 != v19 )
            {
              do
              {
                v19 -= 11;
                v20 = v19[1];
                if ( (unsigned __int64 *)v20 != v19 + 3 )
                  _libc_free(v20);
              }
              while ( v18 != v19 );
              v19 = v111;
            }
            break;
          }
        }
      }
      if ( v19 != (unsigned __int64 *)&v113 )
        _libc_free((unsigned __int64)v19);
    }
    return sub_C7D6A0(v108, 16LL * v110, 8);
  }
  return result;
}
