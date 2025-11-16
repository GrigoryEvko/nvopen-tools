// Function: sub_35290F0
// Address: 0x35290f0
//
__int64 __fastcall sub_35290F0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  int v4; // ecx
  __int64 v5; // rsi
  __int64 v6; // rdi
  int v7; // ecx
  unsigned int v8; // edx
  __int64 *v9; // rax
  __int64 v10; // r8
  unsigned int v11; // ebx
  void *v12; // rax
  char v13; // al
  __int64 v14; // rdi
  __int64 (*v15)(); // rax
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned int *v21; // rdi
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rcx
  _QWORD *v29; // rbx
  __int64 v30; // rdx
  __int64 v31; // r15
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // rcx
  __int64 v35; // r14
  __int64 v36; // rbx
  _BYTE *v37; // r12
  _BYTE *v38; // r13
  _BYTE *v39; // rbx
  bool v40; // zf
  unsigned int v41; // r13d
  int v42; // esi
  _BYTE *v43; // r15
  __int64 v44; // rax
  unsigned __int64 v45; // rdx
  unsigned int v46; // r12d
  unsigned int v47; // ebx
  __int64 v48; // rdx
  unsigned int v49; // ebx
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rdi
  __int64 (*v53)(); // rax
  unsigned __int64 v54; // rax
  int v55; // r13d
  unsigned __int64 v56; // r14
  unsigned int v57; // ebx
  unsigned int v58; // r12d
  unsigned int v59; // eax
  bool v60; // r15
  __int64 *v61; // r13
  __int64 v62; // r12
  __int64 *v63; // rbx
  __int64 v64; // rsi
  _DWORD *v65; // rdi
  __int64 v66; // rsi
  _DWORD *v67; // rax
  unsigned __int64 v68; // rdi
  int v69; // edx
  int *v70; // rcx
  int v71; // edi
  __int64 v72; // rax
  __int64 v73; // r15
  unsigned int v74; // eax
  unsigned int v75; // r15d
  int v76; // ecx
  int v77; // eax
  __int64 v78; // rcx
  unsigned int v79; // r15d
  unsigned int v80; // edx
  __int64 *v81; // rax
  __int64 v82; // rsi
  int v83; // eax
  unsigned __int64 v84; // rax
  __int64 v85; // rcx
  __int64 v86; // rax
  unsigned int v87; // eax
  int v88; // r10d
  _DWORD *v89; // rcx
  int v90; // edx
  unsigned int v91; // esi
  unsigned __int16 v92; // cx
  __int64 v93; // r9
  unsigned int v94; // edx
  _QWORD *v95; // rax
  char v96; // al
  __int64 v97; // rcx
  __int64 *v98; // rcx
  __int64 v99; // rcx
  __int64 v100; // rdx
  unsigned int v101; // ecx
  __int64 v102; // rsi
  unsigned int v103; // edx
  int v104; // edx
  unsigned __int64 v105; // rax
  unsigned __int64 v106; // rax
  int v107; // ebx
  __int64 v108; // r12
  _DWORD *v109; // rax
  _DWORD *i; // rdx
  int v111; // eax
  int v112; // eax
  _QWORD *v113; // rdx
  unsigned int v114; // ecx
  unsigned __int64 v115; // rax
  __int64 v116; // rdx
  unsigned int v117; // r12d
  __int64 v118; // r9
  __int64 v119; // rcx
  __int64 v120; // r9
  unsigned int v121; // r14d
  __int64 (*v122)(void); // rax
  __int64 v123; // rcx
  _QWORD *v124; // rax
  _QWORD *v125; // rdx
  int v126; // eax
  int v127; // eax
  int v128; // edi
  unsigned int *v129; // rbx
  __int64 v130; // rax
  __int64 v131; // rdx
  __int64 v132; // rdx
  __int64 v133; // rdi
  __int64 v134; // rdx
  __int64 v135; // rcx
  __int64 v136; // r8
  __int64 v137; // r9
  _DWORD *v138; // rsi
  int v139; // eax
  int v140; // esi
  __int64 v141; // r12
  int v142; // eax
  __int64 v143; // rdx
  __int64 v144; // rcx
  __int64 v145; // r8
  __int64 v146; // r9
  int v147; // edi
  __int64 v148; // [rsp+0h] [rbp-4D0h]
  __int64 v149; // [rsp+8h] [rbp-4C8h]
  unsigned int *v150; // [rsp+28h] [rbp-4A8h]
  char v151; // [rsp+46h] [rbp-48Ah]
  char v152; // [rsp+47h] [rbp-489h]
  __int64 v153; // [rsp+48h] [rbp-488h]
  __int64 v154; // [rsp+58h] [rbp-478h]
  _QWORD *v155; // [rsp+70h] [rbp-460h]
  unsigned int *v156; // [rsp+80h] [rbp-450h]
  _QWORD *v157; // [rsp+88h] [rbp-448h]
  _QWORD *v158; // [rsp+90h] [rbp-440h]
  unsigned int v159; // [rsp+A0h] [rbp-430h]
  unsigned __int8 v160; // [rsp+A5h] [rbp-42Bh]
  unsigned __int8 v161; // [rsp+A6h] [rbp-42Ah]
  char v162; // [rsp+A7h] [rbp-429h]
  __int64 *v163; // [rsp+A8h] [rbp-428h]
  __int64 v165; // [rsp+B8h] [rbp-418h]
  __int64 *v166; // [rsp+C0h] [rbp-410h]
  unsigned int v167; // [rsp+C8h] [rbp-408h]
  unsigned int v168; // [rsp+C8h] [rbp-408h]
  int v169; // [rsp+D0h] [rbp-400h]
  unsigned __int64 v170; // [rsp+D0h] [rbp-400h]
  __int64 v171; // [rsp+D0h] [rbp-400h]
  unsigned int *v172; // [rsp+D0h] [rbp-400h]
  __int64 v174[2]; // [rsp+E0h] [rbp-3F0h] BYREF
  __int64 *v175; // [rsp+F0h] [rbp-3E0h]
  __int64 v176; // [rsp+F8h] [rbp-3D8h]
  __int64 v177[2]; // [rsp+100h] [rbp-3D0h] BYREF
  __int64 v178; // [rsp+110h] [rbp-3C0h] BYREF
  _DWORD *v179; // [rsp+118h] [rbp-3B8h]
  __int64 v180; // [rsp+120h] [rbp-3B0h]
  unsigned int v181; // [rsp+128h] [rbp-3A8h]
  unsigned int *v182; // [rsp+130h] [rbp-3A0h] BYREF
  __int64 v183; // [rsp+138h] [rbp-398h]
  _BYTE v184[64]; // [rsp+140h] [rbp-390h] BYREF
  __int64 *v185; // [rsp+180h] [rbp-350h] BYREF
  __int64 v186; // [rsp+188h] [rbp-348h]
  _BYTE v187[128]; // [rsp+190h] [rbp-340h] BYREF
  __int64 *v188; // [rsp+210h] [rbp-2C0h] BYREF
  __int64 v189; // [rsp+218h] [rbp-2B8h]
  __int64 v190; // [rsp+220h] [rbp-2B0h] BYREF
  unsigned int v191; // [rsp+228h] [rbp-2A8h]
  _QWORD *v192; // [rsp+2A0h] [rbp-230h] BYREF
  __int64 v193; // [rsp+2A8h] [rbp-228h]
  _BYTE v194[128]; // [rsp+2B0h] [rbp-220h] BYREF
  _QWORD *v195; // [rsp+330h] [rbp-1A0h] BYREF
  __int64 v196; // [rsp+338h] [rbp-198h]
  _DWORD v197[32]; // [rsp+340h] [rbp-190h] BYREF
  unsigned __int64 v198[2]; // [rsp+3C0h] [rbp-110h] BYREF
  _BYTE v199[192]; // [rsp+3D0h] [rbp-100h] BYREF
  unsigned __int64 v200; // [rsp+490h] [rbp-40h]
  unsigned int v201; // [rsp+498h] [rbp-38h]

  v165 = *(_QWORD *)(a2 + 56);
  v3 = *(_QWORD *)(a1 + 312);
  v4 = *(_DWORD *)(v3 + 24);
  v5 = *(_QWORD *)(v3 + 8);
  if ( v4 )
  {
    v6 = a2;
    v7 = v4 - 1;
    v8 = v7 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v5 + 16LL * v8);
    v10 = *v9;
    if ( v6 == *v9 )
    {
LABEL_3:
      v153 = v9[1];
      goto LABEL_4;
    }
    v139 = 1;
    while ( v10 != -4096 )
    {
      v147 = v139 + 1;
      v8 = v7 & (v139 + v8);
      v9 = (__int64 *)(v5 + 16LL * v8);
      v10 = *v9;
      if ( a2 == *v9 )
        goto LABEL_3;
      v139 = v147;
    }
  }
  v153 = 0;
LABEL_4:
  if ( !*(_QWORD *)(a1 + 328) )
  {
    v141 = *(_QWORD *)(a1 + 320);
    v142 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 208) + 720LL))(*(_QWORD *)(a1 + 208));
    *(_QWORD *)(a1 + 328) = sub_2EE9EC0(v141, v142, v143, v144, v145, v146);
  }
  v200 = 0;
  v198[0] = (unsigned __int64)v199;
  v198[1] = 0x800000000LL;
  v201 = 0;
  v11 = *(_DWORD *)(*(_QWORD *)(a1 + 216) + 44LL);
  if ( v11 )
  {
    v12 = _libc_calloc(v11, 1u);
    if ( !v12 )
      sub_C64F00("Allocation failed", 1u);
    v200 = (unsigned __int64)v12;
    v201 = v11;
  }
  v13 = sub_2EE68A0(a2, *(_QWORD *)(a1 + 344), *(__int64 **)(a1 + 336));
  v14 = *(_QWORD *)(a1 + 208);
  v160 = 0;
  v151 = v13;
  v15 = *(__int64 (**)())(*(_QWORD *)v14 + 608LL);
  if ( v15 != sub_2FDC5A0 )
    v160 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v15)(v14, a2, a1 + 352);
  v162 = 0;
  v154 = 0;
  v161 = 0;
  v155 = (_QWORD *)(a2 + 48);
  if ( v165 != a2 + 48 )
  {
    while ( 1 )
    {
      v16 = v165;
      if ( !v165 )
LABEL_221:
        BUG();
      if ( (*(_BYTE *)v165 & 4) == 0 && (*(_BYTE *)(v165 + 44) & 8) != 0 )
      {
        do
          v16 = *(_QWORD *)(v16 + 8);
        while ( (*(_BYTE *)(v16 + 44) & 8) != 0 );
      }
      v17 = *(_QWORD **)(v16 + 8);
      v182 = (unsigned int *)v184;
      v158 = v17;
      v183 = 0x1000000000LL;
      v152 = (*(__int64 (__fastcall **)(_QWORD, __int64, unsigned int **, _QWORD))(**(_QWORD **)(a1 + 208) + 600LL))(
               *(_QWORD *)(a1 + 208),
               v165,
               &v182,
               v160);
      if ( v152 )
        break;
      v21 = v182;
      if ( v182 != (unsigned int *)v184 )
        goto LABEL_18;
LABEL_19:
      v165 = (__int64)v158;
      if ( v155 == v158 )
      {
        if ( (v161 & (unsigned __int8)v162) != 0 )
        {
          sub_2EE9070(*(_QWORD **)(a1 + 320), a2, v18, (__int64)v158, v19, v20);
          v161 &= v162;
        }
        goto LABEL_22;
      }
    }
    if ( (_BYTE)qword_503D2E8 )
    {
      v21 = v182;
      v172 = &v182[(unsigned int)v183];
      if ( v172 == v182 )
        goto LABEL_182;
      v129 = v182;
      do
      {
        v132 = *v129;
        v192 = v194;
        v195 = v197;
        v193 = 0x1000000000LL;
        v196 = 0x1000000000LL;
        v188 = 0;
        v133 = *(_QWORD *)(a1 + 208);
        v189 = 0;
        v190 = 0;
        v191 = 0;
        (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD **, _QWORD **, __int64 **))(*(_QWORD *)v133 + 672LL))(
          v133,
          v165,
          v132,
          &v192,
          &v195,
          &v188);
        if ( (_DWORD)v193 && (sub_2FF7B70(a1 + 672) || sub_2FF7B90(a1 + 672)) )
        {
          v130 = sub_2EEE3C0(*(_QWORD *)(a1 + 328), a2, v134, v135, v136, v137);
          sub_3528EA0(a1, v165, (__int64 *)&v192, (__int64)&v195, v130, v131);
        }
        sub_C7D6A0(v189, 8LL * v191, 4);
        if ( v195 != (_QWORD *)v197 )
          _libc_free((unsigned __int64)v195);
        if ( v192 != (_QWORD *)v194 )
          _libc_free((unsigned __int64)v192);
        ++v129;
      }
      while ( v172 != v129 );
    }
    v21 = v182;
    v150 = &v182[(unsigned int)v183];
    if ( v150 == v182 )
      goto LABEL_182;
    v156 = v182;
    while ( 1 )
    {
      v23 = *v156;
      v188 = &v190;
      v178 = 0;
      v185 = (__int64 *)v187;
      v186 = 0x1000000000LL;
      v189 = 0x1000000000LL;
      v179 = 0;
      v24 = *(_QWORD *)(a1 + 208);
      v159 = v23;
      v180 = 0;
      v181 = 0;
      (*(void (__fastcall **)(__int64, __int64, __int64, __int64 **, __int64 **, __int64 *))(*(_QWORD *)v24 + 672LL))(
        v24,
        v165,
        v23,
        &v185,
        &v188,
        &v178);
      v28 = (unsigned int)v186;
      if ( !(_DWORD)v186 )
      {
        sub_C7D6A0((__int64)v179, 8LL * v181, 4);
        v68 = (unsigned __int64)v188;
        if ( v188 == &v190 )
          goto LABEL_80;
LABEL_79:
        _libc_free(v68);
        goto LABEL_80;
      }
      if ( v158 != (_QWORD *)v154 && v162 )
      {
        sub_2EEC510(*(_QWORD *)(a1 + 328), v154, (__int64)v158, (__int64)v198);
        v154 = (__int64)v158;
      }
      if ( v160 == 1 && v159 > 3 )
      {
        v112 = (*(__int64 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 208) + 632LL))(
                 *(_QWORD *)(a1 + 208),
                 v159);
        if ( v112 == 1 )
        {
          v113 = *(_QWORD **)(a2 + 56);
          v114 = 0;
          if ( v113 == v155 )
            goto LABEL_217;
          do
          {
            v113 = (_QWORD *)v113[1];
            ++v114;
          }
          while ( v113 != v155 );
          if ( (unsigned int)qword_503D4A8 >= v114 )
          {
LABEL_217:
            LOBYTE(v112) = v162;
          }
          else
          {
            v154 = (__int64)v158;
            v162 = v152;
          }
          sub_3528980(
            a2,
            (unsigned __int64 *)v165,
            (__int64)&v185,
            (__int64)&v188,
            *(_QWORD *)(a1 + 328),
            (__int64)v198,
            *(_QWORD *)(a1 + 208),
            v159,
            v112);
          v115 = *v158 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v115 )
            goto LABEL_221;
          v116 = *(_QWORD *)v115;
          v158 = (_QWORD *)(*v158 & 0xFFFFFFFFFFFFFFF8LL);
          if ( (*(_QWORD *)v115 & 4) == 0 && (*(_BYTE *)(v115 + 44) & 4) != 0 )
          {
            while ( (*(_BYTE *)((v116 & 0xFFFFFFFFFFFFFFF8LL) + 44) & 4) != 0 )
              v116 = *(_QWORD *)(v116 & 0xFFFFFFFFFFFFFFF8LL);
            v158 = (_QWORD *)(v116 & 0xFFFFFFFFFFFFFFF8LL);
          }
          goto LABEL_177;
        }
      }
      if ( v153
        && (*(unsigned __int8 (__fastcall **)(_QWORD, _QWORD))(**(_QWORD **)(a1 + 208) + 624LL))(
             *(_QWORD *)(a1 + 208),
             v159)
        || v151 && (unsigned int)v189 > (unsigned int)v186 )
      {
        sub_3528980(
          a2,
          (unsigned __int64 *)v165,
          (__int64)&v185,
          (__int64)&v188,
          *(_QWORD *)(a1 + 328),
          (__int64)v198,
          *(_QWORD *)(a1 + 208),
          v159,
          v162);
        goto LABEL_177;
      }
      v29 = (_QWORD *)sub_2EEE3C0(*(_QWORD *)(a1 + 328), a2, v25, v28, v26, v27);
      v31 = v30;
      v149 = (__int64)v29;
      v148 = v30;
      nullsub_1629();
      v192 = v29;
      v195 = v197;
      v196 = 0x1000000000LL;
      v193 = v31;
      v34 = (__int64)&v185[(unsigned int)v186];
      v157 = v29;
      v163 = (__int64 *)v34;
      if ( v185 == (__int64 *)v34 )
      {
        v46 = v197[(unsigned int)(v186 - 1)];
        goto LABEL_59;
      }
      v166 = v185;
      do
      {
        v35 = *v166;
        v36 = *(_QWORD *)(*v166 + 32);
        v37 = (_BYTE *)(v36 + 40LL * (*(_DWORD *)(*v166 + 40) & 0xFFFFFF));
        v38 = (_BYTE *)(v36 + 40LL * (unsigned int)sub_2E88FE0(*v166));
        if ( v37 == v38 )
        {
LABEL_114:
          v41 = 0;
          goto LABEL_53;
        }
        while ( 1 )
        {
          v39 = v38;
          if ( (unsigned __int8)sub_2E2FA70(v38) )
            break;
          v38 += 40;
          if ( v37 == v38 )
            goto LABEL_114;
        }
        v40 = v37 == v38;
        v41 = 0;
        if ( !v40 )
        {
          while ( 1 )
          {
            v42 = *((_DWORD *)v39 + 2);
            if ( v42 >= 0 )
              goto LABEL_48;
            v32 = (__int64)v179;
            if ( v181 )
            {
              v33 = v181 - 1;
              v69 = v33 & (37 * v42);
              v70 = &v179[2 * v69];
              v71 = *v70;
              if ( v42 == *v70 )
              {
LABEL_86:
                if ( v70 != &v179[2 * v181] )
                {
                  v72 = (unsigned int)v70[1];
                  v73 = v185[v72];
                  v169 = *((_DWORD *)v195 + v72);
                  v167 = sub_2E8E710(v73, v42, 0, 0, 0);
                  v74 = sub_2E89C70(v35, *((_DWORD *)v39 + 2), 0, 0);
                  v75 = v169 + sub_2FF8170(a1 + 672, v73, v167, v35, v74);
                  goto LABEL_88;
                }
              }
              else
              {
                v76 = 1;
                while ( v71 != -1 )
                {
                  v88 = v76 + 1;
                  v69 = v33 & (v76 + v69);
                  v70 = &v179[2 * v69];
                  v71 = *v70;
                  if ( v42 == *v70 )
                    goto LABEL_86;
                  v76 = v88;
                }
              }
            }
            if ( *v39 )
              goto LABEL_48;
            v170 = sub_2EBEE90(*(_QWORD *)(a1 + 304), v42);
            if ( !v170 )
              goto LABEL_48;
            v77 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 208) + 720LL))(*(_QWORD *)(a1 + 208));
            v33 = v170;
            if ( v77 == 1 && a2 != *(_QWORD *)(v170 + 24) )
              goto LABEL_48;
            v75 = *((_DWORD *)v157 + 100);
            v78 = v157[48];
            if ( v75 )
            {
              v79 = v75 - 1;
              v80 = v79 & (((unsigned int)v170 >> 9) ^ ((unsigned int)v170 >> 4));
              v81 = (__int64 *)(v78 + 16LL * v80);
              v82 = *v81;
              if ( v170 == *v81 )
              {
LABEL_99:
                v75 = *((_DWORD *)v81 + 2);
              }
              else
              {
                v111 = 1;
                while ( v82 != -4096 )
                {
                  v128 = v111 + 1;
                  v80 = v79 & (v111 + v80);
                  v81 = (__int64 *)(v78 + 16LL * v80);
                  v82 = *v81;
                  if ( v170 == *v81 )
                    goto LABEL_99;
                  v111 = v128;
                }
                v75 = 0;
              }
            }
            v83 = *(unsigned __int16 *)(v170 + 68);
            if ( (_WORD)v83 == 20 )
            {
              v89 = *(_DWORD **)(v170 + 32);
              if ( (*v89 & 0xFFF00) != 0 )
                goto LABEL_106;
              LODWORD(v86) = v89[12];
              v90 = v89[2];
              v91 = v86 - 1;
              v92 = (v89[10] >> 8) & 0xFFF;
              if ( v92 )
              {
                if ( v91 <= 0x3FFFFFFE || (unsigned int)(v90 - 1) <= 0x3FFFFFFE )
                  goto LABEL_106;
                v86 = (*(__int64 (__fastcall **)(_QWORD, unsigned __int64, unsigned __int64, _QWORD))(**(_QWORD **)(a1 + 216) + 256LL))(
                        *(_QWORD *)(a1 + 216),
                        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 304) + 56LL) + 16 * (v86 & 0x7FFFFFFF))
                      & 0xFFFFFFFFFFFFFFF8LL,
                        *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 304) + 56LL) + 16LL * (v90 & 0x7FFFFFFF))
                      & 0xFFFFFFFFFFFFFFF8LL,
                        v92);
                v33 = v170;
                LOBYTE(v86) = v86 != 0;
                goto LABEL_105;
              }
              if ( v91 <= 0x3FFFFFFE )
              {
                if ( (unsigned int)(v90 - 1) > 0x3FFFFFFE )
                {
                  v98 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 304) + 56LL) + 16LL * (v90 & 0x7FFFFFFF))
                                  & 0xFFFFFFFFFFFFFFF8LL);
LABEL_128:
                  v99 = *v98;
                  v100 = (unsigned int)v86 >> 3;
                  if ( (unsigned int)v100 >= *(unsigned __int16 *)(v99 + 22) )
                    goto LABEL_106;
                  LOBYTE(v86) = ((int)*(unsigned __int8 *)(*(_QWORD *)(v99 + 8) + v100) >> (v86 & 7)) & 1;
                }
                else
                {
                  LOBYTE(v86) = v90 == (_DWORD)v86;
                }
LABEL_105:
                if ( (_BYTE)v86 )
                  goto LABEL_88;
                goto LABEL_106;
              }
              if ( (int)v86 >= 0 )
                goto LABEL_106;
              v97 = *(_QWORD *)(a1 + 304);
              if ( v90 >= 0 )
              {
                v98 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(v97 + 56) + 16 * (v86 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
                if ( (unsigned int)(v90 - 1) <= 0x3FFFFFFE )
                {
                  LODWORD(v86) = v90;
                  goto LABEL_128;
                }
LABEL_106:
                v171 = v33;
                v168 = sub_2E89C70(v35, *((_DWORD *)v39 + 2), 0, 0);
                v87 = sub_2E8E710(v171, *((_DWORD *)v39 + 2), 0, 0, 0);
                v75 += sub_2FF8170(a1 + 672, v171, v87, v35, v168);
                goto LABEL_88;
              }
              v123 = *(_QWORD *)(v97 + 56);
              v124 = (_QWORD *)(*(_QWORD *)(v123 + 16 * (v86 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
              v125 = (_QWORD *)(*(_QWORD *)(v123 + 16LL * (v90 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL);
              LODWORD(v123) = *(_DWORD *)(v125[1] + 4 * ((unsigned __int64)*(unsigned __int16 *)(*v124 + 24LL) >> 5));
              if ( !_bittest((const int *)&v123, *(unsigned __int16 *)(*v124 + 24LL)) )
              {
                v126 = *(_DWORD *)(v124[1] + 4 * ((unsigned __int64)*(unsigned __int16 *)(*v125 + 24LL) >> 5));
                if ( !_bittest(&v126, *(unsigned __int16 *)(*v125 + 24LL)) )
                  goto LABEL_106;
              }
            }
            else if ( (_WORD)v83 )
            {
              v84 = (unsigned int)(v83 - 9);
              if ( (unsigned __int16)v84 > 0x3Bu || (v85 = 0x800000000000409LL, !_bittest64(&v85, v84)) )
              {
                v86 = (*(_QWORD *)(*(_QWORD *)(v170 + 16) + 24LL) >> 4) & 1LL;
                goto LABEL_105;
              }
            }
LABEL_88:
            if ( v41 < v75 )
              v41 = v75;
LABEL_48:
            v43 = v39 + 40;
            if ( v39 + 40 != v37 )
            {
              while ( 1 )
              {
                v39 = v43;
                if ( (unsigned __int8)sub_2E2FA70(v43) )
                  break;
                v43 += 40;
                if ( v37 == v43 )
                  goto LABEL_53;
              }
              if ( v37 != v43 )
                continue;
            }
            break;
          }
        }
LABEL_53:
        v44 = (unsigned int)v196;
        v34 = HIDWORD(v196);
        v45 = (unsigned int)v196 + 1LL;
        if ( v45 > HIDWORD(v196) )
        {
          sub_C8D5F0((__int64)&v195, v197, v45, 4u, v32, v33);
          v44 = (unsigned int)v196;
        }
        ++v166;
        *((_DWORD *)v195 + v44) = v41;
        LODWORD(v196) = v196 + 1;
      }
      while ( v163 != v166 );
      v46 = *((_DWORD *)v195 + (unsigned int)(v186 - 1));
      if ( v195 != (_QWORD *)v197 )
        _libc_free((unsigned __int64)v195);
      v157 = v192;
LABEL_59:
      v47 = *((_DWORD *)v157 + 100);
      v48 = v157[48];
      if ( v47 )
      {
        v49 = v47 - 1;
        v34 = v49 & (((unsigned int)v165 >> 9) ^ ((unsigned int)v165 >> 4));
        v50 = v48 + 16 * v34;
        v51 = *(_QWORD *)v50;
        if ( v165 == *(_QWORD *)v50 )
        {
LABEL_61:
          v47 = *(_DWORD *)(v50 + 8);
        }
        else
        {
          v127 = 1;
          while ( v51 != -4096 )
          {
            v140 = v127 + 1;
            v34 = v49 & (v127 + (_DWORD)v34);
            v50 = v48 + 16LL * (unsigned int)v34;
            v51 = *(_QWORD *)v50;
            if ( v165 == *(_QWORD *)v50 )
              goto LABEL_61;
            v127 = v140;
          }
          v47 = 0;
        }
      }
      if ( v159 <= 3
        || !(*(unsigned int (__fastcall **)(_QWORD, _QWORD, __int64))(**(_QWORD **)(a1 + 208) + 632LL))(
              *(_QWORD *)(a1 + 208),
              v159,
              v48) )
      {
        v60 = v46 < v47;
        if ( v46 >= v47 )
          goto LABEL_69;
LABEL_119:
        v174[0] = v149;
        v174[1] = v148;
        if ( sub_2FF7B70(a1 + 672) )
        {
          v177[0] = a2;
          v176 = 0x100000001LL;
          v175 = v177;
          v117 = sub_2EE9380(v174, v177, 1, 0, 0, v93, 0, 0);
          v193 = 0x1000000000LL;
          v195 = v197;
          v196 = 0x1000000000LL;
          v192 = v194;
          sub_35286D0(a1, (__int64 *)&v185, (__int64)&v192, 0x1000000000LL, (__int64)&v195, v118);
          sub_35286D0(a1, (__int64 *)&v188, (__int64)&v195, v119, (__int64)&v195, v120);
          v121 = sub_2EE9380(
                   v174,
                   v175,
                   (unsigned int)v176,
                   v192,
                   (unsigned int)v193,
                   (unsigned int)v176,
                   v195,
                   (unsigned int)v196);
          v122 = *(__int64 (**)(void))(**(_QWORD **)(a1 + 208) + 696LL);
          if ( v122 != sub_2FDC5F0 )
            v117 += v122();
          if ( v195 != (_QWORD *)v197 )
            _libc_free((unsigned __int64)v195);
          if ( v192 != (_QWORD *)v194 )
            _libc_free((unsigned __int64)v192);
          if ( v175 != v177 )
            _libc_free((unsigned __int64)v175);
          if ( v117 < v121 )
            goto LABEL_69;
        }
        v94 = 0;
        v95 = *(_QWORD **)(a2 + 56);
        if ( v95 == v155 )
          goto LABEL_215;
        do
        {
          v95 = (_QWORD *)v95[1];
          ++v94;
        }
        while ( v95 != v155 );
        if ( (unsigned int)qword_503D4A8 >= v94 )
        {
LABEL_215:
          v96 = v162;
          v60 = v162;
        }
        else
        {
          v154 = (__int64)v158;
          v96 = 1;
        }
        sub_3528980(
          a2,
          (unsigned __int64 *)v165,
          (__int64)&v185,
          (__int64)&v188,
          *(_QWORD *)(a1 + 328),
          (__int64)v198,
          *(_QWORD *)(a1 + 208),
          v159,
          v96);
        v162 = v60;
LABEL_177:
        sub_C7D6A0((__int64)v179, 8LL * v181, 4);
        if ( v188 != &v190 )
          _libc_free((unsigned __int64)v188);
        if ( v185 != (__int64 *)v187 )
          _libc_free((unsigned __int64)v185);
        v21 = v182;
        v161 = v152;
LABEL_182:
        if ( v21 == (unsigned int *)v184 )
          goto LABEL_19;
LABEL_18:
        _libc_free((unsigned __int64)v21);
        goto LABEL_19;
      }
      v52 = *(_QWORD *)(a1 + 208);
      v53 = *(__int64 (**)())(*(_QWORD *)v52 + 680LL);
      if ( v53 == sub_2FDC5E0 || ((unsigned __int8 (__fastcall *)(__int64, __int64))v53)(v52, v165) )
      {
        v54 = sub_3528EA0(a1, v165, (__int64 *)&v185, (__int64)&v188, (__int64)v192, v193);
        v55 = v54;
        v56 = HIDWORD(v54);
      }
      else
      {
        v55 = sub_2FF8080(a1 + 672, v185[(unsigned int)v186 - 1], 1);
        LODWORD(v56) = sub_2FF8080(a1 + 672, v165, 1);
      }
      v57 = v56 + v47;
      v58 = v55 + v46;
      v59 = v57 + sub_2EE9190(&v192, v165);
      if ( !v162 )
        v57 = v59;
      v60 = v57 >= v58;
      if ( v57 >= v58 )
        goto LABEL_119;
LABEL_69:
      v61 = v185;
      v62 = *(_QWORD *)(a2 + 32);
      v63 = &v185[(unsigned int)v186];
      if ( v63 != v185 )
      {
        do
        {
          v64 = *v61++;
          sub_2E790D0(v62, v64, v48, v34, v32, v33);
        }
        while ( v63 != v61 );
      }
      ++v178;
      v65 = v179;
      if ( !(_DWORD)v180 )
      {
        if ( !HIDWORD(v180) )
        {
          v66 = 2LL * v181;
          goto LABEL_78;
        }
        if ( v181 > 0x40 )
        {
          sub_C7D6A0((__int64)v179, 8LL * v181, 4);
          v65 = 0;
          v66 = 0;
          v179 = 0;
          v180 = 0;
          v181 = 0;
          goto LABEL_78;
        }
LABEL_74:
        v66 = 2LL * v181;
        v67 = &v179[v66];
        if ( v179 != &v179[v66] )
        {
          do
          {
            *v65 = -1;
            v65 += 2;
          }
          while ( v67 != v65 );
          v65 = v179;
          v66 = 2LL * v181;
        }
        v180 = 0;
        goto LABEL_78;
      }
      v101 = 4 * v180;
      if ( (unsigned int)(4 * v180) < 0x40 )
        v101 = 64;
      if ( v181 <= v101 )
        goto LABEL_74;
      v102 = 2LL * v181;
      if ( (_DWORD)v180 == 1 )
      {
        v108 = 1024;
        v107 = 128;
LABEL_138:
        sub_C7D6A0((__int64)v179, v102 * 4, 4);
        v181 = v107;
        v109 = (_DWORD *)sub_C7D670(v108, 4);
        v180 = 0;
        v179 = v109;
        v65 = v109;
        v66 = 2LL * v181;
        for ( i = &v109[v66]; i != v109; v109 += 2 )
        {
          if ( v109 )
            *v109 = -1;
        }
        goto LABEL_78;
      }
      _BitScanReverse(&v103, v180 - 1);
      v104 = 1 << (33 - (v103 ^ 0x1F));
      if ( v104 < 64 )
        v104 = 64;
      if ( v181 != v104 )
      {
        v105 = (4 * v104 / 3u + 1) | ((unsigned __int64)(4 * v104 / 3u + 1) >> 1);
        v106 = ((v105 | (v105 >> 2)) >> 4)
             | v105
             | (v105 >> 2)
             | ((((v105 | (v105 >> 2)) >> 4) | v105 | (v105 >> 2)) >> 8);
        v107 = (v106 | (v106 >> 16)) + 1;
        v108 = 8 * ((v106 | (v106 >> 16)) + 1);
        goto LABEL_138;
      }
      v180 = 0;
      v138 = &v179[v102];
      do
      {
        if ( v65 )
          *v65 = -1;
        v65 += 2;
      }
      while ( v138 != v65 );
      v65 = v179;
      v66 = 2LL * v181;
LABEL_78:
      sub_C7D6A0((__int64)v65, v66 * 4, 4);
      v68 = (unsigned __int64)v188;
      if ( v188 != &v190 )
        goto LABEL_79;
LABEL_80:
      if ( v185 != (__int64 *)v187 )
        _libc_free((unsigned __int64)v185);
      if ( v150 == ++v156 )
      {
        v21 = v182;
        goto LABEL_182;
      }
    }
  }
LABEL_22:
  if ( v200 )
    _libc_free(v200);
  if ( (_BYTE *)v198[0] != v199 )
    _libc_free(v198[0]);
  return v161;
}
