// Function: sub_193C710
// Address: 0x193c710
//
__int64 __fastcall sub_193C710(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rsi
  __int64 v4; // rdx
  unsigned __int64 v5; // rdi
  unsigned __int64 v6; // rax
  bool v7; // zf
  __int64 v8; // r12
  unsigned __int8 v9; // al
  unsigned __int64 v10; // r14
  __int64 v11; // r13
  __int64 *v12; // rax
  __int64 v13; // r12
  __int64 *v14; // rax
  __int64 *v15; // rcx
  _QWORD **v16; // rbx
  _QWORD **i; // r12
  int v18; // edx
  __int64 v19; // r8
  unsigned int v20; // eax
  _QWORD *v21; // rcx
  int v22; // edx
  _QWORD *v23; // rdi
  _QWORD *v24; // rbx
  _QWORD *v25; // r12
  unsigned __int64 v26; // rdi
  int v28; // r9d
  unsigned __int64 v29; // r8
  unsigned int v30; // edi
  __int64 v31; // rbx
  unsigned __int64 v32; // rcx
  __int64 v33; // r13
  __int64 v34; // r12
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 *v37; // rax
  unsigned __int64 v38; // r9
  __int64 v39; // r14
  __int64 v40; // rax
  int v41; // ecx
  __int64 v42; // rsi
  __int64 v43; // rdi
  int v44; // ecx
  unsigned int v45; // edx
  __int64 *v46; // rax
  __int64 v47; // r10
  unsigned __int64 v48; // rax
  __int64 v49; // rbx
  __int64 v50; // rax
  _QWORD *v51; // r12
  int v52; // edx
  __int64 v53; // rsi
  int v54; // edx
  unsigned int v55; // ecx
  __int64 *v56; // rax
  __int64 v57; // r8
  __int64 *v58; // rcx
  __int64 v59; // r8
  __int64 *v60; // rbx
  __int64 v61; // rax
  __int64 v62; // r13
  __int64 v63; // r14
  _QWORD *v64; // rax
  char v65; // al
  __int64 v66; // rdi
  __int64 v67; // rax
  __int64 v68; // rdi
  int v69; // eax
  int v70; // ecx
  __int64 v71; // rax
  __int64 *v72; // rax
  __int64 *v73; // rax
  __int64 v74; // rdx
  __int64 *v75; // rax
  __int64 v76; // rsi
  unsigned __int64 v77; // rdi
  __int64 v78; // rsi
  __int64 *v79; // rax
  __int64 v80; // rdx
  int v81; // r8d
  int v82; // r9d
  __int64 *v83; // rax
  __int64 v84; // rsi
  unsigned __int64 v85; // rdi
  __int64 v86; // rsi
  __int64 v87; // rax
  __int64 v88; // r10
  unsigned int v89; // r8d
  _QWORD *v90; // rax
  __int64 v91; // rdi
  int v92; // r9d
  int v93; // eax
  int v94; // edi
  int v95; // eax
  int v96; // esi
  int v97; // ecx
  int v98; // ebx
  int v99; // r8d
  int v100; // r8d
  __int64 v101; // r9
  _QWORD *v102; // rdi
  unsigned int v103; // r12d
  int v104; // ecx
  __int64 v105; // rsi
  int v106; // r8d
  int v107; // r8d
  __int64 v108; // r9
  unsigned int v109; // ecx
  __int64 v110; // r11
  int v111; // edi
  _QWORD *v112; // rsi
  int v113; // r8d
  int v114; // eax
  int v115; // edx
  __int64 v116; // rax
  int v117; // ecx
  unsigned int v118; // r13d
  __int64 v119; // rcx
  __int64 *v120; // [rsp+8h] [rbp-238h]
  __int64 *v121; // [rsp+10h] [rbp-230h]
  __int64 v122; // [rsp+20h] [rbp-220h]
  unsigned __int8 v123; // [rsp+2Ah] [rbp-216h]
  unsigned __int8 v124; // [rsp+2Bh] [rbp-215h]
  int v125; // [rsp+2Ch] [rbp-214h]
  __int64 v126; // [rsp+30h] [rbp-210h]
  unsigned __int64 v127; // [rsp+38h] [rbp-208h]
  _QWORD *v128; // [rsp+48h] [rbp-1F8h]
  char v129; // [rsp+52h] [rbp-1EEh]
  char v130; // [rsp+53h] [rbp-1EDh]
  int v131; // [rsp+54h] [rbp-1ECh]
  char v132; // [rsp+58h] [rbp-1E8h]
  __int64 v133; // [rsp+58h] [rbp-1E8h]
  __int64 *v134; // [rsp+60h] [rbp-1E0h]
  __int64 v135; // [rsp+78h] [rbp-1C8h] BYREF
  __int64 v136; // [rsp+80h] [rbp-1C0h] BYREF
  _QWORD *v137; // [rsp+88h] [rbp-1B8h]
  __int64 v138; // [rsp+90h] [rbp-1B0h]
  unsigned int v139; // [rsp+98h] [rbp-1A8h]
  __int64 v140; // [rsp+A0h] [rbp-1A0h] BYREF
  _BYTE *v141; // [rsp+A8h] [rbp-198h]
  _BYTE *v142; // [rsp+B0h] [rbp-190h]
  __int64 v143; // [rsp+B8h] [rbp-188h]
  int v144; // [rsp+C0h] [rbp-180h]
  _BYTE v145[72]; // [rsp+C8h] [rbp-178h] BYREF
  __int64 v146; // [rsp+110h] [rbp-130h] BYREF
  __int64 *v147; // [rsp+118h] [rbp-128h]
  __int64 *v148; // [rsp+120h] [rbp-120h]
  __int64 v149; // [rsp+128h] [rbp-118h]
  int v150; // [rsp+130h] [rbp-110h]
  _QWORD v151[8]; // [rsp+138h] [rbp-108h] BYREF
  unsigned __int64 v152; // [rsp+178h] [rbp-C8h] BYREF
  unsigned __int64 v153; // [rsp+180h] [rbp-C0h]
  __int64 v154; // [rsp+188h] [rbp-B8h]
  _QWORD v155[22]; // [rsp+190h] [rbp-B0h] BYREF

  v2 = *(_QWORD *)(a1 + 24);
  v149 = 0x100000008LL;
  v3 = (unsigned __int64)v155;
  v136 = 0;
  v151[0] = v2;
  v155[0] = v2;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v147 = v151;
  v148 = v151;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v150 = 0;
  v146 = 1;
  LOBYTE(v155[2]) = 0;
  sub_13B8390(&v152, (__int64)v155);
  memset(v155, 0, 0x80u);
  v124 = 0;
  v5 = v153;
  LODWORD(v155[3]) = 8;
  v155[1] = &v155[5];
  v155[2] = &v155[5];
  v6 = v152;
LABEL_2:
  if ( v5 == v6 )
    goto LABEL_19;
  do
  {
    v7 = *(_QWORD *)(a1 + 48) == 0;
    v8 = **(_QWORD **)(v5 - 24);
    v135 = v8;
    if ( v7 )
LABEL_211:
      sub_4263D6(v5, v3, v4);
    v3 = (unsigned __int64)&v135;
    v9 = (*(__int64 (__fastcall **)(__int64, __int64 *))(a1 + 56))(a1 + 32, &v135);
    v10 = v153;
    v123 = v9;
    if ( v9 )
    {
      v3 = v139;
      if ( v139 )
      {
        v28 = v139 - 1;
        LODWORD(v29) = (_DWORD)v137;
        v30 = (v139 - 1) & (((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9));
        v31 = (__int64)&v137[11 * v30];
        v32 = *(_QWORD *)v31;
        if ( v8 == *(_QWORD *)v31 )
        {
LABEL_45:
          v33 = *(_QWORD *)(v8 + 48);
          v34 = v8 + 40;
          if ( v34 != v33 )
            goto LABEL_46;
          goto LABEL_54;
        }
        v115 = 1;
        v116 = 0;
        while ( v32 != -8 )
        {
          if ( v32 == -16 && !v116 )
            v116 = v31;
          v30 = v28 & (v30 + v115);
          v31 = (__int64)&v137[11 * v30];
          v32 = *(_QWORD *)v31;
          if ( v8 == *(_QWORD *)v31 )
            goto LABEL_45;
          ++v115;
        }
        if ( v116 )
          v31 = v116;
        ++v136;
        v114 = v138 + 1;
        if ( 4 * ((int)v138 + 1) < 3 * v139 )
        {
          v4 = v139 - HIDWORD(v138) - v114;
          if ( (unsigned int)v4 <= v139 >> 3 )
          {
            sub_193C400((__int64)&v136, v139);
            if ( v139 )
            {
              LODWORD(v29) = (_DWORD)v137;
              v117 = 1;
              v118 = (v139 - 1) & (((unsigned int)v8 >> 4) ^ ((unsigned int)v8 >> 9));
              v4 = 0;
              v31 = (__int64)&v137[11 * v118];
              v3 = *(_QWORD *)v31;
              v114 = v138 + 1;
              if ( v8 != *(_QWORD *)v31 )
              {
                while ( v3 != -8 )
                {
                  if ( !v4 && v3 == -16 )
                    v4 = v31;
                  v28 = v117 + 1;
                  v118 = (v139 - 1) & (v117 + v118);
                  v31 = (__int64)&v137[11 * v118];
                  v3 = *(_QWORD *)v31;
                  if ( v8 == *(_QWORD *)v31 )
                    goto LABEL_190;
                  ++v117;
                }
                if ( v4 )
                  v31 = v4;
              }
              goto LABEL_190;
            }
LABEL_242:
            LODWORD(v138) = v138 + 1;
            BUG();
          }
LABEL_190:
          LODWORD(v138) = v114;
          if ( *(_QWORD *)v31 != -8 )
            --HIDWORD(v138);
          *(_QWORD *)v31 = v8;
          v34 = v8 + 40;
          *(_QWORD *)(v31 + 8) = v31 + 24;
          *(_QWORD *)(v31 + 16) = 0x800000000LL;
          v33 = *(_QWORD *)(v34 + 8);
          v10 = v153;
          if ( v33 == v34 )
            goto LABEL_5;
          do
          {
LABEL_46:
            if ( !v33 )
              BUG();
            if ( *(_BYTE *)(v33 - 8) == 78 )
            {
              v35 = *(_QWORD *)(v33 - 48);
              if ( !*(_BYTE *)(v35 + 16) && *(_DWORD *)(v35 + 36) == 79 )
              {
                v36 = *(unsigned int *)(v31 + 16);
                if ( (unsigned int)v36 >= *(_DWORD *)(v31 + 20) )
                {
                  v3 = v31 + 24;
                  sub_16CD150(v31 + 8, (const void *)(v31 + 24), 0, 8, v29, v28);
                  v36 = *(unsigned int *)(v31 + 16);
                }
                *(_QWORD *)(*(_QWORD *)(v31 + 8) + 8 * v36) = v33 - 24;
                ++*(_DWORD *)(v31 + 16);
              }
            }
            v33 = *(_QWORD *)(v33 + 8);
          }
          while ( v33 != v34 );
LABEL_54:
          v37 = *(__int64 **)(v31 + 8);
          v4 = *(unsigned int *)(v31 + 16);
          v10 = v153;
          v120 = &v37[v4];
          if ( v37 == v120 )
            goto LABEL_5;
          v121 = *(__int64 **)(v31 + 8);
          v38 = v153;
LABEL_56:
          v128 = 0;
          v39 = *v121;
          v40 = *(_QWORD *)(a1 + 16);
          v41 = *(_DWORD *)(v40 + 24);
          if ( v41 )
          {
            v42 = *(_QWORD *)(v39 + 40);
            v43 = *(_QWORD *)(v40 + 8);
            v44 = v41 - 1;
            v45 = v44 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
            v46 = (__int64 *)(v43 + 16LL * v45);
            v47 = *v46;
            if ( v42 == *v46 )
            {
LABEL_58:
              v128 = (_QWORD *)v46[1];
            }
            else
            {
              v95 = 1;
              while ( v47 != -8 )
              {
                v113 = v95 + 1;
                v45 = v44 & (v95 + v45);
                v46 = (__int64 *)(v43 + 16LL * v45);
                v47 = *v46;
                if ( v42 == *v46 )
                  goto LABEL_58;
                v95 = v113;
              }
              v128 = 0;
            }
          }
          v48 = v152;
          v3 = 0xAAAAAAAAAAAAAAABLL;
          v4 = 0xAAAAAAAAAAAAAAABLL * ((__int64)(v38 - v152) >> 3);
          if ( !(_DWORD)v4 )
            goto LABEL_130;
          v127 = 0;
          v3 = (unsigned int)v4;
          v125 = v4 - 1;
          v122 = (unsigned int)v4;
          v126 = 0;
          v131 = 0;
          while ( 2 )
          {
            v5 = v127;
            v7 = *(_QWORD *)(a1 + 48) == 0;
            v4 = 3 * v127;
            v49 = **(_QWORD **)(v48 + 24 * v127);
            v140 = v49;
            if ( v7 )
              goto LABEL_211;
            v3 = (unsigned __int64)&v140;
            v129 = (*(__int64 (__fastcall **)(__int64, __int64 *))(a1 + 56))(a1 + 32, &v140);
            if ( !v129 )
              goto LABEL_108;
            v50 = *(_QWORD *)(a1 + 16);
            v51 = 0;
            v52 = *(_DWORD *)(v50 + 24);
            if ( v52 )
            {
              v53 = *(_QWORD *)(v50 + 8);
              v54 = v52 - 1;
              v55 = v54 & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
              v56 = (__int64 *)(v53 + 16LL * v55);
              v57 = *v56;
              if ( v49 == *v56 )
              {
LABEL_65:
                v51 = (_QWORD *)v56[1];
              }
              else
              {
                v93 = 1;
                while ( v57 != -8 )
                {
                  v94 = v93 + 1;
                  v55 = v54 & (v93 + v55);
                  v56 = (__int64 *)(v53 + 16LL * v55);
                  v57 = *v56;
                  if ( v49 == *v56 )
                    goto LABEL_65;
                  v93 = v94;
                }
                v51 = 0;
              }
            }
            if ( v139 )
            {
              v4 = (v139 - 1) & (((unsigned int)v49 >> 9) ^ ((unsigned int)v49 >> 4));
              v58 = &v137[11 * v4];
              v59 = *v58;
              if ( *v58 == v49 )
                goto LABEL_68;
              v70 = 1;
              while ( v59 != -8 )
              {
                v92 = v70 + 1;
                v4 = (v139 - 1) & ((_DWORD)v4 + v70);
                v58 = &v137[11 * v4];
                v59 = *v58;
                if ( v49 == *v58 )
                  goto LABEL_68;
                v70 = v92;
              }
            }
            v4 = 5LL * v139;
            v58 = &v137[11 * v139];
LABEL_68:
            v60 = (__int64 *)v58[1];
            v3 = (unsigned int)v127;
            v61 = 8LL * *((unsigned int *)v58 + 4);
            v134 = &v60[(unsigned __int64)v61 / 8];
            if ( v125 != (_DWORD)v127 )
              goto LABEL_69;
            v4 = v61 >> 3;
            v71 = v61 >> 5;
            if ( v71 )
            {
              v4 = (__int64)&v60[4 * v71];
              v72 = (__int64 *)v58[1];
              while ( 1 )
              {
                if ( v39 == *v72 )
                  goto LABEL_107;
                if ( v39 == v72[1] )
                  break;
                if ( v39 == v72[2] )
                {
                  v134 = v72 + 2;
                  goto LABEL_69;
                }
                if ( v39 == v72[3] )
                {
                  v134 = v72 + 3;
                  goto LABEL_69;
                }
                v72 += 4;
                if ( (__int64 *)v4 == v72 )
                {
                  v4 = v134 - v72;
                  goto LABEL_141;
                }
              }
              v134 = v72 + 1;
LABEL_69:
              if ( v60 == v134 )
                goto LABEL_91;
              v62 = v39;
LABEL_71:
              v63 = *v60;
              v64 = v128;
              if ( v128 == v51 )
              {
                v132 = 0;
              }
              else
              {
                if ( v51 )
                {
                  if ( !v128 )
                    goto LABEL_89;
                  while ( 1 )
                  {
                    v64 = (_QWORD *)*v64;
                    if ( v51 == v64 )
                      break;
                    if ( !v64 )
                      goto LABEL_89;
                  }
                }
                v132 = v129;
              }
              v3 = *(_QWORD *)(v62 - 24LL * (*(_DWORD *)(v62 + 20) & 0xFFFFFFF));
              v140 = 0;
              v143 = 8;
              v141 = v145;
              v142 = v145;
              v144 = 0;
              v65 = sub_1939C70((__int64 *)a1, v3, v63, (__int64)&v140);
              if ( v142 != v141 )
              {
                v130 = v65;
                _libc_free((unsigned __int64)v142);
                v65 = v130;
              }
              if ( v65 )
              {
                v3 = *(_QWORD *)(v62 - 24LL * (*(_DWORD *)(v62 + 20) & 0xFFFFFFF));
                if ( (unsigned __int8)sub_193A710(
                                        (__int64 *)a1,
                                        v3,
                                        *(_QWORD *)(v63 - 24LL * (*(_DWORD *)(v63 + 20) & 0xFFFFFFF)),
                                        0,
                                        &v140) )
                {
                  v4 = 3 - (unsigned int)(v132 == 0);
                  v69 = 3 - (v132 == 0);
                }
                else if ( v132 )
                {
                  v4 = 2;
                  v69 = 2;
                }
                else
                {
                  v66 = *(_QWORD *)(v63 + 40);
                  if ( v66 == *(_QWORD *)(v62 + 40)
                    || (v133 = *(_QWORD *)(v62 + 40), v67 = sub_157F210(v66), v4 = v133, v133 == v67) )
                  {
                    v4 = 1;
                    v69 = 1;
                  }
                  else
                  {
                    v68 = *(_QWORD *)(a1 + 8);
                    if ( !v68 )
                      goto LABEL_89;
                    v3 = *(_QWORD *)(v62 + 40);
                    v69 = sub_15CCCD0(v68, v3, *(_QWORD *)(v63 + 40));
                    v4 = (unsigned __int8)v69;
                  }
                }
                if ( v131 < (int)v4 )
                {
                  v126 = v63;
                  v131 = v69;
                }
              }
LABEL_89:
              if ( v134 == ++v60 )
              {
                v39 = v62;
LABEL_91:
                if ( ++v127 != v122 )
                {
                  v48 = v152;
                  continue;
                }
LABEL_108:
                v38 = v153;
                if ( !v131 )
                  goto LABEL_130;
                if ( (*(_BYTE *)(v126 + 23) & 0x40) != 0 )
                  v73 = *(__int64 **)(v126 - 8);
                else
                  v73 = (__int64 *)(v126 - 24LL * (*(_DWORD *)(v126 + 20) & 0xFFFFFFF));
                sub_193A710(
                  (__int64 *)a1,
                  *v73,
                  *(_QWORD *)(v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF)),
                  v126,
                  &v140);
                v74 = v140;
                v75 = (__int64 *)(v126 - 24LL * (*(_DWORD *)(v126 + 20) & 0xFFFFFFF));
                if ( *v75 )
                {
                  v76 = v75[1];
                  v77 = v75[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v77 = v76;
                  if ( v76 )
                    *(_QWORD *)(v76 + 16) = v77 | *(_QWORD *)(v76 + 16) & 3LL;
                }
                *v75 = v74;
                if ( v74 )
                {
                  v78 = *(_QWORD *)(v74 + 8);
                  v75[1] = v78;
                  if ( v78 )
                    *(_QWORD *)(v78 + 16) = (unsigned __int64)(v75 + 1) | *(_QWORD *)(v78 + 16) & 3LL;
                  v75[2] = (v74 + 8) | v75[2] & 3;
                  *(_QWORD *)(v74 + 8) = v75;
                }
                v79 = (__int64 *)sub_16498A0(v39);
                v80 = sub_159C4F0(v79);
                v83 = (__int64 *)(v39 - 24LL * (*(_DWORD *)(v39 + 20) & 0xFFFFFFF));
                if ( *v83 )
                {
                  v84 = v83[1];
                  v85 = v83[2] & 0xFFFFFFFFFFFFFFFCLL;
                  *(_QWORD *)v85 = v84;
                  if ( v84 )
                    *(_QWORD *)(v84 + 16) = v85 | *(_QWORD *)(v84 + 16) & 3LL;
                }
                *v83 = v80;
                if ( v80 )
                {
                  v86 = *(_QWORD *)(v80 + 8);
                  v83[1] = v86;
                  if ( v86 )
                  {
                    v81 = (_DWORD)v83 + 8;
                    *(_QWORD *)(v86 + 16) = (unsigned __int64)(v83 + 1) | *(_QWORD *)(v86 + 16) & 3LL;
                  }
                  v83[2] = (v80 + 8) | v83[2] & 3;
                  *(_QWORD *)(v80 + 8) = v83;
                }
                v87 = *(unsigned int *)(a1 + 72);
                if ( (unsigned int)v87 >= *(_DWORD *)(a1 + 76) )
                {
                  sub_16CD150(a1 + 64, (const void *)(a1 + 80), 0, 8, v81, v82);
                  v87 = *(unsigned int *)(a1 + 72);
                }
                v4 = *(_QWORD *)(a1 + 64);
                *(_QWORD *)(v4 + 8 * v87) = v39;
                v3 = *(unsigned int *)(a1 + 232);
                ++*(_DWORD *)(a1 + 72);
                if ( !(_DWORD)v3 )
                {
                  ++*(_QWORD *)(a1 + 208);
LABEL_178:
                  sub_1467110(a1 + 208, 2 * v3);
                  v106 = *(_DWORD *)(a1 + 232);
                  if ( v106 )
                  {
                    v107 = v106 - 1;
                    v108 = *(_QWORD *)(a1 + 216);
                    v4 = (unsigned int)(*(_DWORD *)(a1 + 224) + 1);
                    v109 = v107 & (((unsigned int)v126 >> 9) ^ ((unsigned int)v126 >> 4));
                    v90 = (_QWORD *)(v108 + 8LL * v109);
                    v110 = *v90;
                    if ( *v90 != v126 )
                    {
                      v111 = 1;
                      v112 = 0;
                      while ( v110 != -8 )
                      {
                        if ( !v112 && v110 == -16 )
                          v112 = v90;
                        v109 = v107 & (v111 + v109);
                        v90 = (_QWORD *)(v108 + 8LL * v109);
                        v110 = *v90;
                        if ( *v90 == v126 )
                          goto LABEL_168;
                        ++v111;
                      }
                      if ( v112 )
                        v90 = v112;
                    }
                    goto LABEL_168;
                  }
LABEL_244:
                  ++*(_DWORD *)(a1 + 224);
                  BUG();
                }
                v88 = *(_QWORD *)(a1 + 216);
                v89 = (v3 - 1) & (((unsigned int)v126 >> 4) ^ ((unsigned int)v126 >> 9));
                v90 = (_QWORD *)(v88 + 8LL * v89);
                v91 = *v90;
                if ( *v90 == v126 )
                  goto LABEL_129;
                v97 = 1;
                v4 = 0;
                while ( v91 != -8 )
                {
                  if ( !v4 && v91 == -16 )
                    v4 = (__int64)v90;
                  v89 = (v3 - 1) & (v97 + v89);
                  v90 = (_QWORD *)(v88 + 8LL * v89);
                  v91 = *v90;
                  if ( *v90 == v126 )
                    goto LABEL_129;
                  ++v97;
                }
                v98 = *(_DWORD *)(a1 + 224);
                if ( v4 )
                  v90 = (_QWORD *)v4;
                ++*(_QWORD *)(a1 + 208);
                v4 = (unsigned int)(v98 + 1);
                if ( 4 * (int)v4 >= (unsigned int)(3 * v3) )
                  goto LABEL_178;
                if ( (int)v3 - *(_DWORD *)(a1 + 228) - (int)v4 <= (unsigned int)v3 >> 3 )
                {
                  sub_1467110(a1 + 208, v3);
                  v99 = *(_DWORD *)(a1 + 232);
                  if ( v99 )
                  {
                    v100 = v99 - 1;
                    v101 = *(_QWORD *)(a1 + 216);
                    v102 = 0;
                    v103 = v100 & (((unsigned int)v126 >> 4) ^ ((unsigned int)v126 >> 9));
                    v104 = 1;
                    v4 = (unsigned int)(*(_DWORD *)(a1 + 224) + 1);
                    v90 = (_QWORD *)(v101 + 8LL * v103);
                    v105 = *v90;
                    if ( *v90 != v126 )
                    {
                      while ( v105 != -8 )
                      {
                        if ( v105 == -16 && !v102 )
                          v102 = v90;
                        v103 = v100 & (v104 + v103);
                        v90 = (_QWORD *)(v101 + 8LL * v103);
                        v105 = *v90;
                        if ( *v90 == v126 )
                          goto LABEL_168;
                        ++v104;
                      }
                      if ( v102 )
                        v90 = v102;
                    }
                    goto LABEL_168;
                  }
                  goto LABEL_244;
                }
LABEL_168:
                *(_DWORD *)(a1 + 224) = v4;
                if ( *v90 != -8 )
                  --*(_DWORD *)(a1 + 228);
                v3 = v126;
                *v90 = v126;
LABEL_129:
                v38 = v153;
                v124 = v123;
LABEL_130:
                if ( v120 == ++v121 )
                {
                  v10 = v38;
                  goto LABEL_5;
                }
                goto LABEL_56;
              }
              goto LABEL_71;
            }
            break;
          }
          v72 = (__int64 *)v58[1];
LABEL_141:
          if ( v4 != 2 )
          {
            if ( v4 != 3 )
            {
              if ( v4 != 1 )
                goto LABEL_69;
              goto LABEL_144;
            }
            if ( v39 == *v72 )
              goto LABEL_107;
            ++v72;
          }
          if ( v39 == *v72 )
          {
LABEL_107:
            v134 = v72;
            goto LABEL_69;
          }
          ++v72;
LABEL_144:
          if ( v39 != *v72 )
            v72 = v134;
          v134 = v72;
          goto LABEL_69;
        }
      }
      else
      {
        ++v136;
      }
      v3 = 2 * v139;
      sub_193C400((__int64)&v136, v3);
      if ( v139 )
      {
        v28 = (int)v137;
        v4 = (v139 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v31 = (__int64)&v137[11 * v4];
        v29 = *(_QWORD *)v31;
        v114 = v138 + 1;
        if ( v8 != *(_QWORD *)v31 )
        {
          v3 = 1;
          v119 = 0;
          while ( v29 != -8 )
          {
            if ( v29 == -16 && !v119 )
              v119 = v31;
            v4 = (v139 - 1) & ((_DWORD)v3 + (_DWORD)v4);
            v31 = (__int64)&v137[11 * (unsigned int)v4];
            v29 = *(_QWORD *)v31;
            if ( v8 == *(_QWORD *)v31 )
              goto LABEL_190;
            v3 = (unsigned int)(v3 + 1);
          }
          if ( v119 )
            v31 = v119;
        }
        goto LABEL_190;
      }
      goto LABEL_242;
    }
LABEL_5:
    v11 = *(_QWORD *)(v10 - 24);
    if ( !*(_BYTE *)(v10 - 8) )
    {
      v12 = *(__int64 **)(v11 + 24);
      *(_BYTE *)(v10 - 8) = 1;
      *(_QWORD *)(v10 - 16) = v12;
      goto LABEL_9;
    }
    while ( 1 )
    {
LABEL_8:
      v12 = *(__int64 **)(v10 - 16);
LABEL_9:
      if ( *(__int64 **)(v11 + 32) == v12 )
      {
        v153 -= 24LL;
        v6 = v152;
        v10 = v153;
        if ( v153 != v152 )
          goto LABEL_5;
        v5 = v152;
        goto LABEL_2;
      }
      *(_QWORD *)(v10 - 16) = v12 + 1;
      v13 = *v12;
      v14 = v147;
      if ( v148 == v147 )
      {
        v15 = &v147[HIDWORD(v149)];
        if ( v147 != v15 )
        {
          v3 = 0;
          do
          {
            while ( 1 )
            {
              v4 = *v14;
              if ( v13 == *v14 )
                goto LABEL_8;
              if ( v4 != -2 )
                break;
              v3 = (unsigned __int64)v14;
              if ( v15 == v14 + 1 )
                goto LABEL_17;
              ++v14;
            }
            ++v14;
          }
          while ( v15 != v14 );
          if ( v3 )
          {
LABEL_17:
            *(_QWORD *)v3 = v13;
            --v150;
            ++v146;
            goto LABEL_18;
          }
        }
        if ( HIDWORD(v149) < (unsigned int)v149 )
          break;
      }
      v3 = v13;
      sub_16CCBA0((__int64)&v146, v13);
      if ( (_BYTE)v4 )
        goto LABEL_18;
    }
    ++HIDWORD(v149);
    *v15 = v13;
    ++v146;
LABEL_18:
    v3 = (unsigned __int64)&v140;
    v140 = v13;
    LOBYTE(v142) = 0;
    sub_13B8390(&v152, (__int64)&v140);
    v5 = v153;
  }
  while ( v153 != v152 );
LABEL_19:
  if ( v5 )
    j_j___libc_free_0(v5, v154 - v5);
  if ( v148 != v147 )
    _libc_free((unsigned __int64)v148);
  v16 = *(_QWORD ***)(a1 + 64);
  for ( i = &v16[*(unsigned int *)(a1 + 72)]; i != v16; ++v16 )
  {
    while ( 1 )
    {
      v22 = *(_DWORD *)(a1 + 232);
      v23 = *v16;
      if ( v22 )
        break;
LABEL_28:
      sub_15F20C0(v23);
      if ( i == ++v16 )
        goto LABEL_29;
    }
    v18 = v22 - 1;
    v19 = *(_QWORD *)(a1 + 216);
    v20 = v18 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
    v21 = *(_QWORD **)(v19 + 8LL * v20);
    if ( v23 != v21 )
    {
      v96 = 1;
      while ( v21 != (_QWORD *)-8LL )
      {
        v20 = v18 & (v96 + v20);
        v21 = *(_QWORD **)(v19 + 8LL * v20);
        if ( v23 == v21 )
          goto LABEL_26;
        ++v96;
      }
      goto LABEL_28;
    }
LABEL_26:
    ;
  }
LABEL_29:
  if ( v139 )
  {
    v24 = v137;
    v25 = &v137[11 * v139];
    do
    {
      if ( *v24 != -16 && *v24 != -8 )
      {
        v26 = v24[1];
        if ( (_QWORD *)v26 != v24 + 3 )
          _libc_free(v26);
      }
      v24 += 11;
    }
    while ( v25 != v24 );
  }
  j___libc_free_0(v137);
  return v124;
}
