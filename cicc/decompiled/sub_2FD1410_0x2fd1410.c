// Function: sub_2FD1410
// Address: 0x2fd1410
//
__int64 __fastcall sub_2FD1410(char **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v6; // rax
  __int64 v7; // rbx
  unsigned int v8; // ebx
  unsigned int v9; // r13d
  __int64 *v10; // rsi
  __int64 *v11; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // r10
  __int64 v16; // rcx
  __int64 v17; // rax
  unsigned int v20; // r12d
  __int64 v21; // rbx
  __int64 v22; // r15
  __int64 v23; // rdx
  int v24; // eax
  unsigned int v25; // eax
  __int64 v26; // rcx
  int v27; // edx
  unsigned int v28; // r11d
  unsigned int v29; // esi
  int v30; // ecx
  unsigned __int64 v31; // r8
  bool v32; // zf
  char v33; // cl
  __int64 v34; // rdx
  __int64 v37; // rdx
  unsigned __int64 *v38; // rdx
  unsigned __int64 *v39; // rax
  __int64 v40; // rcx
  __int64 v41; // r14
  int *v42; // r15
  int v43; // eax
  unsigned int v44; // edx
  unsigned int v45; // r11d
  unsigned int v46; // esi
  __int64 v47; // rbx
  int v48; // ecx
  unsigned __int64 v49; // r10
  __int64 v50; // rdx
  unsigned __int64 v51; // r10
  unsigned __int64 v52; // r14
  int v55; // eax
  char v56; // r14
  __int64 v57; // r15
  __int64 v58; // rdi
  unsigned __int8 v59; // si
  __int64 v60; // rax
  __int64 v61; // rcx
  unsigned int v62; // edx
  unsigned __int64 *v63; // r14
  __int64 v64; // rax
  float *v65; // rax
  char **v66; // r15
  char *v67; // rbx
  char *v68; // r12
  _DWORD *v69; // rcx
  __int64 v70; // rdx
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // r9
  __int64 v75; // rbx
  __int64 v76; // r12
  int v77; // esi
  __int64 v78; // rax
  __int64 v79; // rcx
  __int64 *v80; // rax
  __int64 *v81; // rsi
  __int64 v82; // rdx
  __int64 v83; // r14
  __int64 v84; // rax
  _DWORD *v85; // r8
  __int64 v86; // rcx
  __int64 j; // rdi
  int v88; // esi
  __int64 v89; // r12
  char **v90; // rax
  __int64 v91; // r15
  char **v92; // r14
  char *v93; // rdi
  __int64 (*v94)(); // rax
  __int64 v95; // rax
  char *v96; // rdi
  __int64 v97; // rbx
  __int64 (__fastcall *v98)(__int64, __int64, __int64, _DWORD *); // r10
  __int64 (*v99)(); // rax
  __int64 *v100; // rbx
  __int64 *v101; // r12
  __int64 v102; // rdi
  __int64 v103; // r13
  int v104; // eax
  __int64 v105; // r11
  __int64 v106; // r12
  int i; // eax
  int v108; // edx
  unsigned int v109; // eax
  char *v110; // rcx
  int v111; // edx
  unsigned int v112; // edi
  unsigned int v113; // esi
  __int64 v114; // r8
  int v115; // ecx
  unsigned __int64 v116; // r10
  char v117; // cl
  __int64 v118; // rdx
  unsigned __int64 v119; // r10
  unsigned __int64 *v122; // rbx
  unsigned __int64 *v123; // r12
  _QWORD *v125; // rax
  __int64 v126; // rsi
  __int64 v127; // rcx
  __int64 v128; // r8
  __int64 v129; // r9
  __int64 v130; // rax
  unsigned __int64 v131; // rdx
  unsigned int v132; // r10d
  __int64 v133; // r13
  __int64 v134; // rdi
  __int64 (__fastcall *v135)(__int64, __int64, __int64, int *); // rax
  __int64 (*v136)(); // rax
  int v137; // eax
  unsigned int v138; // r10d
  int v139; // edx
  __int64 v140; // r9
  __int64 v141; // rax
  int v142; // eax
  char *v143; // rax
  _QWORD *v144; // r13
  _QWORD *v145; // rax
  unsigned __int64 v146; // rcx
  __int64 v147; // rbx
  __int64 v148; // rax
  unsigned __int64 v149; // rdi
  __int64 v150; // r8
  __int64 v151; // r9
  __int64 v152; // rdx
  __int64 v153; // rcx
  __int64 v154; // r8
  __int64 v155; // r9
  unsigned __int64 v156; // rdx
  unsigned __int64 v157; // rdx
  _QWORD *v158; // rax
  char *v160; // rax
  __int64 *v161; // [rsp+8h] [rbp-468h]
  __int64 v163; // [rsp+20h] [rbp-450h]
  __int64 *v164; // [rsp+28h] [rbp-448h]
  unsigned __int8 v165; // [rsp+57h] [rbp-419h]
  __int64 *v166; // [rsp+58h] [rbp-418h]
  unsigned int v167; // [rsp+58h] [rbp-418h]
  unsigned int v168; // [rsp+58h] [rbp-418h]
  int v169; // [rsp+60h] [rbp-410h]
  __int64 v170; // [rsp+68h] [rbp-408h]
  int v171; // [rsp+70h] [rbp-400h]
  int v172; // [rsp+70h] [rbp-400h]
  __int64 v173; // [rsp+70h] [rbp-400h]
  __int64 v174; // [rsp+70h] [rbp-400h]
  __int64 v175; // [rsp+78h] [rbp-3F8h]
  __int64 v176; // [rsp+78h] [rbp-3F8h]
  int v177; // [rsp+80h] [rbp-3F0h] BYREF
  int v178; // [rsp+84h] [rbp-3ECh] BYREF
  int v179; // [rsp+88h] [rbp-3E8h] BYREF
  int v180; // [rsp+8Ch] [rbp-3E4h] BYREF
  void *v181; // [rsp+90h] [rbp-3E0h] BYREF
  __int64 v182; // [rsp+98h] [rbp-3D8h]
  _BYTE v183[48]; // [rsp+A0h] [rbp-3D0h] BYREF
  unsigned int v184; // [rsp+D0h] [rbp-3A0h]
  void *s; // [rsp+E0h] [rbp-390h] BYREF
  __int64 v186; // [rsp+E8h] [rbp-388h]
  _DWORD v187[16]; // [rsp+F0h] [rbp-380h] BYREF
  void *v188; // [rsp+130h] [rbp-340h] BYREF
  __int64 v189; // [rsp+138h] [rbp-338h]
  _DWORD v190[16]; // [rsp+140h] [rbp-330h] BYREF
  __int64 *v191; // [rsp+180h] [rbp-2F0h] BYREF
  __int64 v192; // [rsp+188h] [rbp-2E8h]
  char v193[8]; // [rsp+190h] [rbp-2E0h] BYREF
  __int64 v194; // [rsp+198h] [rbp-2D8h]
  _BYTE *v195; // [rsp+1A0h] [rbp-2D0h]
  __int64 v196; // [rsp+1A8h] [rbp-2C8h]
  _BYTE v197[64]; // [rsp+1B0h] [rbp-2C0h] BYREF
  _BYTE *v198; // [rsp+1F0h] [rbp-280h]
  __int64 v199; // [rsp+1F8h] [rbp-278h]
  _BYTE v200[32]; // [rsp+200h] [rbp-270h] BYREF
  __int16 v201; // [rsp+220h] [rbp-250h]
  __int64 v202; // [rsp+224h] [rbp-24Ch]
  unsigned __int64 *v203; // [rsp+230h] [rbp-240h] BYREF
  __int64 v204; // [rsp+238h] [rbp-238h]
  _OWORD v205[35]; // [rsp+240h] [rbp-230h] BYREF

  v6 = *a1;
  v7 = *((_QWORD *)v6 + 2) - *((_QWORD *)v6 + 1);
  v186 = 0x1000000000LL;
  v8 = -858993459 * (v7 >> 3) - *((_DWORD *)v6 + 8);
  s = v187;
  if ( v8 > 0x10 )
  {
    sub_C8D5F0((__int64)&s, v187, v8, 4u, a5, a6);
    memset(s, 255, 4LL * v8);
    v188 = v190;
    LODWORD(v186) = v8;
    v189 = 0x1000000000LL;
    sub_C8D5F0((__int64)&v188, v190, v8, 4u, v150, v151);
    memset(v188, 0, 4LL * v8);
    v203 = (unsigned __int64 *)v205;
    LODWORD(v189) = v8;
    v204 = 0x1000000000LL;
    sub_2FD0D40((__int64)&v203, v8, v152, v153, v154, v155);
    v39 = &v203[4 * v8];
    v38 = &v203[4 * (unsigned int)v204];
    if ( v39 == v38 )
    {
LABEL_227:
      LODWORD(v204) = v8;
      v9 = (v8 + 63) >> 6;
      v181 = v183;
      v182 = 0x600000000LL;
      if ( v9 > 6 )
      {
        sub_C8D5F0((__int64)&v181, v183, v9, 8u, a5, a6);
        memset(v181, 0, 8LL * v9);
      }
      else if ( v9 && 8LL * v9 )
      {
        memset(v183, 0, 8LL * v9);
      }
      goto LABEL_4;
    }
    do
    {
LABEL_224:
      if ( v38 )
      {
        *((_DWORD *)v38 + 2) = 0;
        *v38 = (unsigned __int64)(v38 + 2);
        *((_DWORD *)v38 + 3) = 4;
      }
      v38 += 4;
    }
    while ( v39 != v38 );
    goto LABEL_227;
  }
  if ( v8 )
  {
    v37 = 4LL * v8;
    if ( v37 )
    {
      if ( (unsigned int)v37 >= 8 )
      {
        *(_QWORD *)((char *)&v187[-2] + (unsigned int)v37) = -1;
        memset(v187, 0xFFu, 8LL * ((unsigned int)(v37 - 1) >> 3));
      }
      else if ( (v37 & 4) != 0 )
      {
        v187[0] = -1;
        *(_DWORD *)((char *)&v187[-1] + (unsigned int)v37) = -1;
      }
      else if ( (_DWORD)v37 )
      {
        LOBYTE(v187[0]) = -1;
      }
    }
    LODWORD(v186) = v8;
    v188 = v190;
    HIDWORD(v189) = 16;
    if ( v37 )
    {
      if ( (unsigned int)v37 >= 8 )
      {
        *(_QWORD *)((char *)&v190[-2] + (unsigned int)v37) = 0;
        memset(v190, 0, 8LL * ((unsigned int)(v37 - 1) >> 3));
      }
      else if ( (v37 & 4) != 0 )
      {
        v190[0] = 0;
        *(_DWORD *)((char *)&v190[-1] + (unsigned int)v37) = 0;
      }
      else if ( (_DWORD)v37 )
      {
        LOBYTE(v190[0]) = 0;
      }
    }
    v38 = (unsigned __int64 *)v205;
    LODWORD(v189) = v8;
    v204 = 0x1000000000LL;
    v203 = (unsigned __int64 *)v205;
    v39 = (unsigned __int64 *)&v205[2 * v8];
    goto LABEL_224;
  }
  v189 = 0x1000000000LL;
  v188 = v190;
  v203 = (unsigned __int64 *)v205;
  LODWORD(v186) = 0;
  v181 = v183;
  HIDWORD(v182) = 6;
  v204 = 0x1000000000LL;
  v9 = 0;
LABEL_4:
  v10 = (__int64 *)a1[6];
  v11 = (__int64 *)a1[5];
  LODWORD(v182) = v9;
  v184 = v8;
  v164 = v10;
  if ( v11 == v10 )
  {
    sub_2FD1370(a1 + 5);
    v165 = 0;
    goto LABEL_137;
  }
  v166 = v11;
  v161 = (__int64 *)(a1 + 228);
  v165 = 0;
  do
  {
    v175 = *v166;
    v169 = *(_DWORD *)(*v166 + 112) - 0x40000000;
    v163 = *(unsigned __int8 *)(*((_QWORD *)*a1 + 1) + 40LL * (unsigned int)(*((_DWORD *)*a1 + 8) + v169) + 20);
    v13 = 72 * v163;
    v14 = (__int64)&a1[208][72 * v163];
    v15 = *(_QWORD *)v14;
    if ( (_BYTE)qword_50263C8 )
      goto LABEL_49;
    v16 = *(unsigned int *)(v14 + 64);
    if ( !(_DWORD)v16 )
      goto LABEL_49;
    a6 = (unsigned int)(v16 - 1) >> 6;
    v17 = 0;
    while ( 1 )
    {
      _RDX = *(_QWORD *)(v15 + 8 * v17);
      if ( a6 == v17 )
        break;
      if ( _RDX )
        goto LABEL_13;
      if ( (_DWORD)a6 + 1 == ++v17 )
        goto LABEL_49;
    }
    v16 = (unsigned int)-(int)v16;
    _RDX &= 0xFFFFFFFFFFFFFFFFLL >> v16;
    if ( !_RDX )
    {
LABEL_49:
      v20 = *(_DWORD *)&a1[205][4 * v163];
      v40 = v20;
      *(_QWORD *)(v15 + 8LL * (v20 >> 6)) |= 1LL << v20;
      v41 = (__int64)&a1[185][v13];
      v170 = 1LL << v20;
      v173 = 8LL * (v20 >> 6);
      v42 = (int *)&a1[205][4 * v163];
      v43 = *(_DWORD *)(v41 + 64);
      v44 = *v42 + 1;
      if ( v43 == v44 || (v45 = v44 >> 6, v46 = (unsigned int)(v43 - 1) >> 6, v44 >> 6 > v46) )
      {
LABEL_159:
        v55 = -1;
      }
      else
      {
        v47 = *(_QWORD *)v41;
        v48 = 64 - (v44 & 0x3F);
        v49 = 0xFFFFFFFFFFFFFFFFLL >> v48;
        if ( v48 == 64 )
          v49 = 0;
        v50 = v45;
        v51 = ~v49;
        v52 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v43;
        while ( 1 )
        {
          _RAX = *(_QWORD *)(v47 + 8 * v50);
          v40 = (unsigned int)v50;
          if ( v45 == (_DWORD)v50 )
            _RAX = v51 & *(_QWORD *)(v47 + 8 * v50);
          if ( v46 == (_DWORD)v50 )
            _RAX &= v52;
          if ( _RAX )
            break;
          if ( v46 < (unsigned int)++v50 )
            goto LABEL_159;
        }
        __asm { tzcnt   rax, rax }
        v40 = (unsigned int)((_DWORD)v50 << 6);
        v55 = v40 + _RAX;
      }
      v21 = (int)v20;
      *v42 = v55;
      v56 = 0;
      v22 = 232LL * (int)v20;
      goto LABEL_62;
    }
LABEL_13:
    __asm { tzcnt   rdx, rdx }
    v20 = ((_DWORD)v17 << 6) + _RDX;
    if ( v20 == -1 )
      goto LABEL_49;
    while ( 1 )
    {
      v21 = (int)v20;
      v22 = 232LL * (int)v20;
      v23 = (__int64)&a1[241][v22];
      if ( *(_QWORD *)(v23 + 8) )
        break;
      if ( !*(_QWORD *)v23 )
        goto LABEL_47;
      a6 = *(unsigned int *)(v175 + 8);
      if ( !(_DWORD)a6 || !(unsigned __int8)sub_2E09D90(*(__int64 ***)v23, v175, *(__int64 **)v175) )
        goto LABEL_47;
LABEL_20:
      v25 = v20 + 1;
      v26 = (__int64)&a1[208][v13];
      v27 = *(_DWORD *)(v26 + 64);
      v15 = *(_QWORD *)v26;
      if ( v27 != v20 + 1 )
      {
        v28 = v25 >> 6;
        v29 = (unsigned int)(v27 - 1) >> 6;
        if ( v25 >> 6 <= v29 )
        {
          v30 = 64 - (v25 & 0x3F);
          v31 = 0xFFFFFFFFFFFFFFFFLL >> v30;
          v32 = v30 == 64;
          v33 = v27;
          v34 = v28;
          if ( v32 )
            v31 = 0;
          a6 = 0xFFFFFFFFFFFFFFFFLL >> -v33;
          a5 = ~v31;
          while ( 1 )
          {
            _RAX = *(_QWORD *)(v15 + 8 * v34);
            if ( v28 == (_DWORD)v34 )
              _RAX = a5 & *(_QWORD *)(v15 + 8 * v34);
            if ( v29 == (_DWORD)v34 )
              _RAX &= a6;
            if ( _RAX )
              break;
            if ( v29 < (unsigned int)++v34 )
              goto LABEL_49;
          }
          __asm { tzcnt   rax, rax }
          v16 = (unsigned int)((_DWORD)v34 << 6);
          v20 = v16 + _RAX;
          if ( (_DWORD)v16 + (_DWORD)_RAX != -1 )
            continue;
        }
      }
      goto LABEL_49;
    }
    v191 = *(__int64 **)(v23 + 8);
    v195 = v197;
    v192 = v175;
    v198 = v200;
    v194 = 0;
    v196 = 0x400000000LL;
    v199 = 0x400000000LL;
    v201 = 0;
    v202 = 0;
    v24 = sub_2E1AC90((__int64)&v191, 1u, (__int64)v197, v16, a5, (__int64)v200);
    a6 = (__int64)v200;
    if ( v198 != v200 )
    {
      v171 = v24;
      _libc_free((unsigned __int64)v198);
      v24 = v171;
    }
    if ( v195 != v197 )
    {
      v172 = v24;
      _libc_free((unsigned __int64)v195);
      v24 = v172;
    }
    if ( v24 )
      goto LABEL_20;
LABEL_47:
    if ( *(_BYTE *)(*((_QWORD *)*a1 + 1) + 40LL * (*((_DWORD *)*a1 + 8) + v20) + 20) != *(_BYTE *)(*((_QWORD *)*a1 + 1)
                                                                                                 + 40LL
                                                                                                 * (unsigned int)(v169 + *((_DWORD *)*a1 + 8))
                                                                                                 + 20) )
    {
      v15 = *(_QWORD *)&a1[208][72 * v163];
      goto LABEL_49;
    }
    v40 = v20;
    v56 = 1;
    v173 = 8LL * (v20 >> 6);
    v170 = 1LL << v20;
LABEL_62:
    v57 = (__int64)&a1[241][v22];
    if ( *(_QWORD *)(v57 + 8) )
    {
      sub_2E1D1A0(*(_DWORD **)(v57 + 8), v175, v175, v40, a5);
    }
    else if ( *(_QWORD *)v57 )
    {
      *(_DWORD *)(v57 + 16) = 0;
      *(_QWORD *)(v57 + 224) = v161;
      memset((void *)(v57 + 24), 0, 0xC0u);
      v125 = (_QWORD *)(v57 + 24);
      *(_QWORD *)(v57 + 216) = 0;
      do
      {
        *v125 = 0;
        v125 += 2;
        *(v125 - 1) = 0;
      }
      while ( (_QWORD *)(v57 + 152) != v125 );
      v126 = *(_QWORD *)v57;
      *(_QWORD *)(v57 + 8) = v57 + 16;
      sub_2E1D1A0((_DWORD *)(v57 + 16), v126, v126, 0, a5);
      sub_2E1D1A0(*(_DWORD **)(v57 + 8), v175, v175, v127, v128);
      *(_QWORD *)v57 = 0;
    }
    else
    {
      *(_QWORD *)v57 = v175;
    }
    v58 = (__int64)*a1;
    v59 = a1[170][v169];
    v60 = *((_QWORD *)*a1 + 1) + 40LL * (*((_DWORD *)*a1 + 8) + v20);
    if ( v56 && *(_BYTE *)(v60 + 16) >= v59 )
    {
      v61 = *(unsigned int *)&a1[175][4 * v169];
      v62 = *(_DWORD *)&a1[175][4 * v169];
      goto LABEL_67;
    }
    *(_BYTE *)(v60 + 16) = v59;
    a5 = *(_QWORD *)(v58 + 8);
    if ( (*(_BYTE *)(a5 + 40LL * (*(_DWORD *)(v58 + 32) + v20) + 20) & 0xFD) == 0 )
      sub_2E76F70(v58, v59);
    v62 = *(_DWORD *)&a1[175][4 * v169];
    v60 = *((_QWORD *)*a1 + 1) + 40LL * (*((_DWORD *)*a1 + 8) + v20);
    if ( !v56 )
      goto LABEL_68;
    v61 = v62;
LABEL_67:
    if ( *(_QWORD *)(v60 + 8) < v61 )
LABEL_68:
      *(_QWORD *)(v60 + 8) = v62;
    *((_DWORD *)s + v169) = v20;
    v63 = &v203[4 * v21];
    v64 = *((unsigned int *)v63 + 2);
    if ( v64 + 1 > (unsigned __int64)*((unsigned int *)v63 + 3) )
    {
      sub_C8D5F0((__int64)&v203[4 * v21], v63 + 2, v64 + 1, 4u, a5, a6);
      v64 = *((unsigned int *)v63 + 2);
    }
    *(_DWORD *)(*v63 + 4 * v64) = v169;
    v65 = (float *)v188;
    ++*((_DWORD *)v63 + 2);
    v65[v21] = v65[v21] + *(float *)(v175 + 116);
    *(_QWORD *)((char *)v181 + v173) |= v170;
    ++v166;
    v165 |= v169 != v20;
  }
  while ( v164 != v166 );
  v66 = a1;
  if ( v165 )
  {
    sub_2FD0390((__int64)a1[2], (int *)s, (unsigned int)v186);
    v143 = a1[2];
    v144 = (_QWORD *)*((_QWORD *)v143 + 15);
    if ( v144 )
    {
      v145 = (_QWORD *)*((_QWORD *)v143 + 15);
      v146 = 0;
      do
      {
        v145 = (_QWORD *)*v145;
        ++v146;
      }
      while ( v145 );
      v67 = v66[5];
      if ( v146 > (v66[7] - v67) >> 3 )
      {
        if ( v146 > 0xFFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
        v147 = 8 * v146;
        v148 = sub_22077B0(8 * v146);
        v68 = (char *)v148;
        do
        {
          v148 += 8;
          *(_QWORD *)(v148 - 8) = v144 + 2;
          v144 = (_QWORD *)*v144;
        }
        while ( v144 );
        v149 = (unsigned __int64)v66[5];
        if ( v149 )
          j_j___libc_free_0(v149);
        v67 = &v68[v147];
        v66[5] = v68;
        v66[6] = v67;
        v66[7] = v67;
        goto LABEL_242;
      }
      v156 = (v66[6] - v67) >> 3;
      if ( v146 > v156 )
      {
        v157 = v156 - 1;
        v158 = v144;
        if ( v66[6] != v67 )
        {
          do
            v158 = (_QWORD *)*v158;
          while ( v157-- != 0 );
          if ( v144 == v158 )
          {
            v67 = v66[6];
          }
          else
          {
            do
            {
              v67 += 8;
              *((_QWORD *)v67 - 1) = v144 + 2;
              v144 = (_QWORD *)*v144;
            }
            while ( v158 != v144 );
            v67 = v66[6];
            if ( !v158 )
              goto LABEL_241;
          }
        }
        do
        {
          v67 += 8;
          *((_QWORD *)v67 - 1) = v158 + 2;
          v158 = (_QWORD *)*v158;
        }
        while ( v158 );
LABEL_241:
        v66[6] = v67;
        v68 = v66[5];
        goto LABEL_242;
      }
      do
      {
        v67 += 8;
        *((_QWORD *)v67 - 1) = v144 + 2;
        v144 = (_QWORD *)*v144;
      }
      while ( v144 );
      v160 = v66[6];
      v68 = v66[5];
    }
    else
    {
      v67 = v66[5];
      v160 = v66[6];
      v68 = v67;
    }
    if ( v67 != v160 )
      v66[6] = v67;
LABEL_242:
    if ( v67 != v68 )
      goto LABEL_74;
    sub_2FD1370(v66 + 5);
LABEL_77:
    v75 = *((unsigned int *)v66 + 18);
    v76 = 0;
    if ( (_DWORD)v75 )
    {
      do
      {
        while ( 1 )
        {
          v77 = *((_DWORD *)s + v76);
          if ( v77 != -1 && v77 != (_DWORD)v76 )
          {
            v78 = sub_2F3F610(*(_QWORD *)(a2 + 352), v77, v71, v72, v73, v74);
            v79 = (__int64)&v66[8][80 * v76];
            v71 = v78;
            v80 = *(__int64 **)v79;
            v72 = *(unsigned int *)(v79 + 8);
            v81 = &v80[v72];
            if ( v81 != v80 )
              break;
          }
          if ( v75 == ++v76 )
            goto LABEL_86;
        }
        v71 |= 4uLL;
        do
        {
          v72 = *v80++;
          *(_QWORD *)v72 = v71;
        }
        while ( v81 != v80 );
        ++v76;
      }
      while ( v75 != v76 );
    }
LABEL_86:
    v174 = *(_QWORD *)(a2 + 328);
    if ( a2 + 320 == v174 )
    {
LABEL_120:
      v104 = *((_DWORD *)v66 + 372);
      if ( v104 )
      {
        v105 = 0;
        v106 = 0;
        v176 = 4LL * (unsigned int)(v104 - 1) + 4;
        do
        {
          for ( i = *(_DWORD *)&v66[205][v106]; i != -1; i = ((_DWORD)v118 << 6) + _RAX )
          {
            v108 = i;
            v109 = i + 1;
            *(_QWORD *)(*((_QWORD *)*v66 + 1) + 40LL * (unsigned int)(*((_DWORD *)*v66 + 8) + v108) + 8) = -1;
            v110 = &v66[185][v105];
            v111 = *((_DWORD *)v110 + 16);
            if ( v111 == v109 )
              break;
            v112 = v109 >> 6;
            v113 = (unsigned int)(v111 - 1) >> 6;
            if ( v109 >> 6 > v113 )
              break;
            v114 = *(_QWORD *)v110;
            v115 = 64 - (v109 & 0x3F);
            v116 = 0xFFFFFFFFFFFFFFFFLL >> v115;
            v32 = v115 == 64;
            v117 = v111;
            v118 = v112;
            if ( v32 )
              v116 = 0;
            v119 = ~v116;
            while ( 1 )
            {
              _RAX = *(_QWORD *)(v114 + 8 * v118);
              if ( v112 == (_DWORD)v118 )
                _RAX = v119 & *(_QWORD *)(v114 + 8 * v118);
              if ( (_DWORD)v118 == v113 )
                _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -v117;
              if ( _RAX )
                break;
              if ( v113 < (unsigned int)++v118 )
                goto LABEL_135;
            }
            __asm { tzcnt   rax, rax }
          }
LABEL_135:
          v106 += 4;
          v105 += 72;
        }
        while ( v176 != v106 );
      }
      v165 = 1;
      goto LABEL_137;
    }
    while ( 1 )
    {
      v82 = *(_QWORD *)(v174 + 56);
      v83 = v174 + 48;
      if ( v174 + 48 != v82 )
      {
        do
        {
          v84 = *(_QWORD *)(v82 + 32);
          v85 = s;
          v86 = 5LL * (*(_DWORD *)(v82 + 40) & 0xFFFFFF);
          for ( j = v84 + 40LL * (*(_DWORD *)(v82 + 40) & 0xFFFFFF); j != v84; v84 += 40 )
          {
            if ( *(_BYTE *)v84 == 5 )
            {
              v86 = *(unsigned int *)(v84 + 24);
              if ( (int)v86 >= 0 )
              {
                v88 = v85[(int)v86];
                if ( v88 != -1 && (_DWORD)v86 != v88 )
                  *(_DWORD *)(v84 + 24) = v88;
              }
            }
          }
          if ( (*(_BYTE *)v82 & 4) == 0 )
          {
            while ( (*(_BYTE *)(v82 + 44) & 8) != 0 )
              v82 = *(_QWORD *)(v82 + 8);
          }
          v82 = *(_QWORD *)(v82 + 8);
        }
        while ( v83 != v82 );
        v89 = *(_QWORD *)(v174 + 56);
        v191 = (__int64 *)v193;
        v192 = 0x400000000LL;
        if ( v83 != v89 )
          break;
      }
LABEL_119:
      v174 = *(_QWORD *)(v174 + 8);
      if ( a2 + 320 == v174 )
        goto LABEL_120;
    }
    v90 = v66;
    v91 = v174 + 48;
    v92 = v90;
    while ( 1 )
    {
      if ( (_DWORD)qword_50262E8 != -1 && (int)qword_50262E8 <= 0 )
      {
LABEL_111:
        v66 = v92;
        v100 = &v191[(unsigned int)v192];
        if ( v191 != v100 )
        {
          v101 = v191;
          do
          {
            v102 = (__int64)v92[4];
            v103 = *v101;
            if ( v102 )
              sub_2FAD510(v102, *v101);
            ++v101;
            sub_2E88E20(v103);
          }
          while ( v100 != v101 );
          v100 = v191;
        }
        if ( v100 != (__int64 *)v193 )
          _libc_free((unsigned __int64)v100);
        goto LABEL_119;
      }
      v93 = v92[1];
      v94 = *(__int64 (**)())(*(_QWORD *)v93 + 152LL);
      if ( v94 != sub_2FCE890
        && ((unsigned __int8 (__fastcall *)(char *, __int64, int *, int *, _DWORD *))v94)(v93, v89, &v177, &v178, v85)
        && v178 == v177
        && v177 != -1 )
      {
        v130 = (unsigned int)v192;
        v86 = HIDWORD(v192);
        v131 = (unsigned int)v192 + 1LL;
        if ( v131 > HIDWORD(v192) )
        {
          sub_C8D5F0((__int64)&v191, v193, v131, 8u, (__int64)v85, v129);
          v130 = (unsigned int)v192;
        }
        v191[v130] = v89;
        LODWORD(v192) = v192 + 1;
        goto LABEL_108;
      }
      if ( !v89 )
        goto LABEL_255;
      v95 = v89;
      if ( (*(_BYTE *)v89 & 4) == 0 && (*(_BYTE *)(v89 + 44) & 8) != 0 )
      {
        do
          v95 = *(_QWORD *)(v95 + 8);
        while ( (*(_BYTE *)(v95 + 44) & 8) != 0 );
      }
      v96 = v92[1];
      v97 = *(_QWORD *)(v95 + 8);
      v179 = 0;
      v180 = 0;
      v98 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _DWORD *))(*(_QWORD *)v96 + 96LL);
      if ( v98 != sub_2FCED00 )
        break;
      v99 = *(__int64 (**)())(*(_QWORD *)v96 + 88LL);
      if ( v99 != sub_2E97330 )
      {
        v132 = ((__int64 (__fastcall *)(char *, __int64, int *, __int64, _DWORD *))v99)(v96, v89, &v177, v86, v85);
        goto LABEL_175;
      }
LABEL_108:
      if ( !v89 )
        goto LABEL_255;
      if ( (*(_BYTE *)v89 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v89 + 44) & 8) != 0 )
          v89 = *(_QWORD *)(v89 + 8);
      }
      v89 = *(_QWORD *)(v89 + 8);
      if ( v91 == v89 )
        goto LABEL_111;
    }
    v132 = ((__int64 (__fastcall *)(char *, __int64, int *, int *, _DWORD *))v98)(v96, v89, &v177, &v179, v85);
LABEL_175:
    if ( !v132 || v91 == v97 )
      goto LABEL_108;
    v133 = v89;
    while ( (unsigned __int16)(*(_WORD *)(v97 + 68) - 14) <= 4u )
    {
      if ( (*(_BYTE *)v97 & 4) == 0 && (*(_BYTE *)(v97 + 44) & 8) != 0 )
      {
        do
          v97 = *(_QWORD *)(v97 + 8);
        while ( (*(_BYTE *)(v97 + 44) & 8) != 0 );
      }
      v97 = *(_QWORD *)(v97 + 8);
      if ( !v133 )
        goto LABEL_255;
      if ( (*(_BYTE *)v133 & 4) != 0 )
      {
        v133 = *(_QWORD *)(v133 + 8);
        if ( v91 == v97 )
          goto LABEL_189;
      }
      else
      {
        while ( (*(_BYTE *)(v133 + 44) & 8) != 0 )
          v133 = *(_QWORD *)(v133 + 8);
        v133 = *(_QWORD *)(v133 + 8);
        if ( v91 == v97 )
          goto LABEL_189;
      }
    }
    if ( v91 != v97 )
    {
      v134 = (__int64)v92[1];
      v135 = *(__int64 (__fastcall **)(__int64, __int64, __int64, int *))(*(_QWORD *)v134 + 128LL);
      if ( v135 != sub_2FCED30 )
      {
        v168 = v132;
        v142 = v135(v134, v97, (__int64)&v178, &v180);
        v138 = v168;
        v139 = v142;
        goto LABEL_194;
      }
      v180 = 0;
      v136 = *(__int64 (**)())(*(_QWORD *)v134 + 120LL);
      if ( v136 != sub_2F4C0B0 )
      {
        v167 = v132;
        v137 = ((__int64 (__fastcall *)(__int64, __int64, int *))v136)(v134, v97, &v178);
        v138 = v167;
        v139 = v137;
LABEL_194:
        if ( v139 )
        {
          LOBYTE(v86) = v139 != v138 || v178 != v177;
          if ( !(_BYTE)v86
            && v177 != -1
            && v179 == v180
            && *(_BYTE *)(*((_QWORD *)*v92 + 1) + 40LL * (unsigned int)(*((_DWORD *)*v92 + 8) + v177) + 18) )
          {
            v32 = (unsigned int)sub_2E89C70(v97, v138, 0, 1) == -1;
            v141 = (unsigned int)v192;
            if ( !v32 )
            {
              if ( (unsigned __int64)(unsigned int)v192 + 1 > HIDWORD(v192) )
              {
                sub_C8D5F0((__int64)&v191, v193, (unsigned int)v192 + 1LL, 8u, (__int64)v85, v140);
                v141 = (unsigned int)v192;
              }
              v191[v141] = v89;
              v141 = (unsigned int)(v192 + 1);
              LODWORD(v192) = v192 + 1;
            }
            v86 = HIDWORD(v192);
            if ( v141 + 1 > (unsigned __int64)HIDWORD(v192) )
            {
              sub_C8D5F0((__int64)&v191, v193, v141 + 1, 8u, (__int64)v85, v140);
              v141 = (unsigned int)v192;
            }
            v191[v141] = v97;
            LODWORD(v192) = v192 + 1;
            if ( !v133 )
LABEL_255:
              BUG();
            if ( (*(_BYTE *)v133 & 4) == 0 )
            {
              while ( (*(_BYTE *)(v133 + 44) & 8) != 0 )
                v133 = *(_QWORD *)(v133 + 8);
            }
            v89 = *(_QWORD *)(v133 + 8);
            goto LABEL_108;
          }
        }
      }
    }
LABEL_189:
    v89 = v133;
    goto LABEL_108;
  }
  v67 = a1[6];
  v68 = a1[5];
  if ( v67 == v68 )
  {
    sub_2FD1370(a1 + 5);
  }
  else
  {
LABEL_74:
    v69 = v188;
    do
    {
      v70 = *(_QWORD *)v68;
      v68 += 8;
      *(_DWORD *)(v70 + 116) = v69[*(_DWORD *)(v70 + 112) - 0x40000000];
    }
    while ( v67 != v68 );
    sub_2FD1370(v66 + 5);
    if ( v165 )
      goto LABEL_77;
  }
LABEL_137:
  if ( v181 != v183 )
    _libc_free((unsigned __int64)v181);
  v122 = v203;
  v123 = &v203[4 * (unsigned int)v204];
  if ( v203 != v123 )
  {
    do
    {
      v123 -= 4;
      if ( (unsigned __int64 *)*v123 != v123 + 2 )
        _libc_free(*v123);
    }
    while ( v122 != v123 );
    v123 = v203;
  }
  if ( v123 != (unsigned __int64 *)v205 )
    _libc_free((unsigned __int64)v123);
  if ( v188 != v190 )
    _libc_free((unsigned __int64)v188);
  if ( s != v187 )
    _libc_free((unsigned __int64)s);
  return v165;
}
