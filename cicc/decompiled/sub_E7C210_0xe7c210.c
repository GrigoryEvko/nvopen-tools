// Function: sub_E7C210
// Address: 0xe7c210
//
__int64 __fastcall sub_E7C210(__int64 *a1, __int64 a2, unsigned __int8 a3)
{
  __int64 v4; // r15
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 result; // rax
  __int64 v8; // rdx
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  _QWORD *v13; // r12
  __int64 v14; // rsi
  _QWORD *v15; // r13
  _QWORD *v16; // rbx
  _QWORD *v17; // r14
  _QWORD *v18; // rdi
  __int64 v19; // rdi
  __int64 v20; // rdx
  __int64 v21; // r8
  __int64 v22; // rcx
  __int64 *v23; // r14
  char v24; // al
  unsigned int v25; // ebx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // rdi
  __int64 v32; // r13
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rcx
  __int64 v39; // r8
  int v40; // r8d
  unsigned int v41; // r8d
  unsigned __int64 v42; // rbx
  char v43; // cl
  __int64 *v44; // r9
  __int64 v45; // r13
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r8
  __int64 v53; // rsi
  __int64 *v54; // r8
  __int64 v55; // rsi
  __int64 v56; // rsi
  unsigned int v57; // r15d
  int v58; // r12d
  __int64 v59; // rbx
  __int64 v60; // rdx
  __int64 v61; // rcx
  __int64 v62; // r8
  char v63; // al
  __int64 v64; // r11
  unsigned int v65; // r13d
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // rsi
  __int64 v72; // rax
  __int64 v73; // rsi
  int v74; // r13d
  __int64 v75; // r9
  __int64 v76; // rax
  int v77; // esi
  unsigned int v78; // eax
  __int64 v79; // r8
  __int64 *v80; // r8
  int v81; // esi
  __int64 *v82; // r13
  unsigned int v83; // eax
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // rbx
  __int64 v87; // rsi
  unsigned __int64 v88; // rdx
  __int64 v89; // r12
  __int64 v90; // r13
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r15
  char v95; // r14
  _QWORD *v96; // rbx
  unsigned __int8 v97; // dl
  __int64 v98; // rsi
  unsigned int v99; // eax
  __int64 v100; // r8
  __int64 v101; // rsi
  __int64 v102; // rdx
  __int64 v103; // rsi
  unsigned __int64 v104; // rax
  __int64 v105; // r13
  __int64 v106; // rdx
  unsigned int v107; // eax
  __int64 v108; // rcx
  __int64 v109; // r8
  __int64 v110; // rax
  __int64 v111; // rax
  __int64 v112; // rax
  __int64 v113; // rcx
  __int64 v114; // r8
  __int64 v115; // rdx
  __int64 *v116; // r12
  _QWORD *v117; // rbx
  _QWORD *v118; // r13
  _QWORD *v119; // rdi
  __int64 v120; // rdi
  bool v121; // zf
  __int64 v122; // rdx
  __int64 v123; // rcx
  __int64 v124; // rdx
  __int64 v125; // rsi
  unsigned __int64 v126; // r8
  __int64 v127; // [rsp+8h] [rbp-148h]
  __int64 v128; // [rsp+10h] [rbp-140h]
  char v129; // [rsp+18h] [rbp-138h]
  unsigned int *v130; // [rsp+18h] [rbp-138h]
  int v131; // [rsp+20h] [rbp-130h]
  __int64 v132; // [rsp+20h] [rbp-130h]
  int v133; // [rsp+28h] [rbp-128h]
  __int64 v134; // [rsp+28h] [rbp-128h]
  __int64 v135; // [rsp+28h] [rbp-128h]
  unsigned int v136; // [rsp+28h] [rbp-128h]
  unsigned int v137; // [rsp+30h] [rbp-120h]
  __int64 v138; // [rsp+30h] [rbp-120h]
  __int64 *v139; // [rsp+38h] [rbp-118h]
  __int64 v140; // [rsp+40h] [rbp-110h]
  char v141; // [rsp+50h] [rbp-100h]
  __int64 v142; // [rsp+58h] [rbp-F8h]
  __int64 v143; // [rsp+58h] [rbp-F8h]
  __int64 v144; // [rsp+58h] [rbp-F8h]
  __int64 v145; // [rsp+60h] [rbp-F0h]
  int v146; // [rsp+60h] [rbp-F0h]
  char v147; // [rsp+68h] [rbp-E8h]
  char v148; // [rsp+69h] [rbp-E7h]
  char v149; // [rsp+6Ah] [rbp-E6h]
  char v150; // [rsp+6Ah] [rbp-E6h]
  char v152; // [rsp+6Ch] [rbp-E4h]
  int v153; // [rsp+6Ch] [rbp-E4h]
  char v154; // [rsp+70h] [rbp-E0h]
  __int64 v155; // [rsp+70h] [rbp-E0h]
  char v156; // [rsp+78h] [rbp-D8h]
  __int64 v157; // [rsp+78h] [rbp-D8h]
  __int64 v158; // [rsp+78h] [rbp-D8h]
  __int64 v159; // [rsp+80h] [rbp-D0h]
  __int64 v160; // [rsp+80h] [rbp-D0h]
  char v161; // [rsp+80h] [rbp-D0h]
  __int64 v162; // [rsp+80h] [rbp-D0h]
  __int64 v163; // [rsp+88h] [rbp-C8h]
  __int64 v164; // [rsp+88h] [rbp-C8h]
  __int64 v165; // [rsp+88h] [rbp-C8h]
  int v166; // [rsp+88h] [rbp-C8h]
  unsigned int v167; // [rsp+88h] [rbp-C8h]
  __int64 v168; // [rsp+90h] [rbp-C0h]
  _QWORD *v169; // [rsp+90h] [rbp-C0h]
  _QWORD *v170; // [rsp+98h] [rbp-B8h]
  __int64 v171; // [rsp+98h] [rbp-B8h]
  __int64 v172; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v173; // [rsp+A8h] [rbp-A8h]
  unsigned __int8 v174; // [rsp+B0h] [rbp-A0h]
  __int64 *v175; // [rsp+B8h] [rbp-98h]
  unsigned __int64 v176; // [rsp+C8h] [rbp-88h]
  __int64 v177; // [rsp+D0h] [rbp-80h]
  __int16 v178; // [rsp+D8h] [rbp-78h]
  unsigned __int64 v179; // [rsp+E8h] [rbp-68h]
  __int64 v180; // [rsp+F0h] [rbp-60h]
  __int16 v181; // [rsp+F8h] [rbp-58h]
  _BYTE *v182; // [rsp+100h] [rbp-50h] BYREF
  __int64 v183; // [rsp+108h] [rbp-48h]
  unsigned __int64 v184; // [rsp+110h] [rbp-40h]
  _BYTE v185[56]; // [rsp+118h] [rbp-38h] BYREF

  v4 = a1[1];
  v172 = 0;
  v5 = *(_QWORD *)(v4 + 168);
  v6 = *(_QWORD *)(v4 + 152);
  v173 = 0;
  v174 = a3;
  v140 = v5;
  v142 = v6;
  v175 = a1;
  result = sub_E98E20();
  v170 = (_QWORD *)result;
  v159 = v8;
  if ( !a3 )
  {
    v9 = *(_QWORD *)(v5 + 112);
    goto LABEL_3;
  }
  v140 = v5;
  v95 = *(_BYTE *)(v5 + 9) ^ 1;
  if ( *(_QWORD *)(v140 + 64) )
  {
    sub_E98E40(a1, a2);
    v96 = v170;
    result = 96 * v159;
    v169 = &v170[12 * v159];
    if ( v169 != v170 )
    {
      v97 = 0;
      while ( 1 )
      {
        result = v96[9];
        if ( !result )
          goto LABEL_89;
        if ( !v97 )
        {
          (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 176))(a1, *(_QWORD *)(v140 + 64), 0);
          v103 = 0xFFFFFFFFLL;
          if ( *(_DWORD *)(v142 + 8) )
          {
            _BitScanReverse64(&v104, *(unsigned int *)(v142 + 8));
            v103 = 63 - ((unsigned int)v104 ^ 0x3F);
          }
          (*(void (__fastcall **)(__int64 *, __int64, _QWORD, __int64, _QWORD))(*a1 + 608))(a1, v103, 0, 1, 0);
          result = v96[9];
        }
        v95 |= *(_DWORD *)(v140 + 16) == result;
        v97 = a3;
        if ( !(_DWORD)result )
          goto LABEL_89;
        v167 = result;
        v105 = v175[1];
        v106 = *(_QWORD *)(v105 + 168);
        if ( (_DWORD)result != *(_DWORD *)(v106 + 16) && v96[3] )
          v167 = result | 0x40000000;
        v146 = *(_DWORD *)(v106 + 16);
        v153 = result;
        v107 = sub_E71E20(v105, *(_DWORD *)(v106 + 12));
        sub_E9A500(v109, *v96, v107, v108);
        v155 = *v96;
        v157 = sub_E808D0(v96[1], 0, v105, 0);
        v110 = sub_E808D0(v155, 0, v105, 0);
        v158 = sub_E81A00(18, v157, v110, v105, 0);
        v111 = sub_E81A90(0, v105, 0, 0);
        v112 = sub_E81A00(18, v158, v111, v105, 0);
        sub_E71DA0(v175, v112, 4, v113, v114);
        (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v175 + 536))(v175, v167, 4);
        v115 = *(unsigned int *)(*(_QWORD *)(v175[1] + 152) + 8LL);
        if ( v153 == v146 )
        {
          (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v175 + 536))(v175, 0, v115);
          v102 = (unsigned int)sub_E71E20(v175[1], *((_DWORD *)v96 + 16));
LABEL_101:
          v96 += 12;
          result = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)v100 + 536LL))(v100, 0, v102);
          v97 = a3;
          if ( v169 == v96 )
            break;
        }
        else
        {
          v98 = v96[2];
          if ( v98 )
            sub_E9A500(v175, v98, v115, 0);
          else
            (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v175 + 536))(v175, 0, v115);
          v99 = sub_E71E20(v175[1], *((_DWORD *)v96 + 16));
          v101 = v96[3];
          v102 = v99;
          if ( !v101 )
            goto LABEL_101;
          result = sub_E9A500(v100, v101, v99, 0);
          v97 = a3;
LABEL_89:
          v96 += 12;
          if ( v169 == v96 )
            break;
        }
      }
    }
  }
  if ( !v95 )
    return result;
  v9 = *(_QWORD *)(v140 + 464);
LABEL_3:
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, v9, 0);
  v128 = sub_E6C430(v4, v9, v10, v11, v12);
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 208))(a1, v128, 0);
  v141 = *(_BYTE *)(v140 + 10);
  v127 = 96 * v159;
  if ( (unsigned __int64)(96 * v159) > 0x7FFFFFFFFFFFFFE0LL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  v139 = 0;
  if ( v127 )
    v139 = (__int64 *)sub_22077B0(v127);
  v168 = sub_E79D70(v170, &v170[12 * v159], (__int64)v139);
  sub_E79920((__int64 *)&v182, v139, 0xAAAAAAAAAAAAAAABLL * ((v168 - (__int64)v139) >> 5));
  if ( v184 )
    sub_E75D60((__int64)v139, v168, (_QWORD *)v184, v183);
  else
    sub_E7C160(v139, v168);
  v13 = (_QWORD *)v184;
  v14 = 96 * v183;
  v15 = (_QWORD *)(v184 + 96 * v183);
  if ( (_QWORD *)v184 != v15 )
  {
    do
    {
      v16 = (_QWORD *)v13[5];
      v17 = (_QWORD *)v13[4];
      if ( v16 != v17 )
      {
        do
        {
          v18 = (_QWORD *)v17[9];
          if ( v18 != v17 + 11 )
            j_j___libc_free_0(v18, v17[11] + 1LL);
          v19 = v17[6];
          if ( v19 )
            j_j___libc_free_0(v19, v17[8] - v19);
          v17 += 13;
        }
        while ( v16 != v17 );
        v17 = (_QWORD *)v13[4];
      }
      if ( v17 )
        j_j___libc_free_0(v17, v13[6] - (_QWORD)v17);
      v13 += 12;
    }
    while ( v15 != v13 );
    v15 = (_QWORD *)v184;
    v14 = 96 * v183;
  }
  result = j_j___libc_free_0(v15, v14);
  v22 = v168;
  if ( v139 != (__int64 *)v168 )
  {
    v147 = 0;
    v23 = v139;
    v148 = 0;
    v131 = -1;
    v129 = 0;
    v149 = 0;
    v133 = -1;
    v137 = 0;
    v143 = 0;
    v171 = 0;
    while ( 1 )
    {
      v23 += 12;
      if ( !v141 )
        break;
      result = *(unsigned int *)(v140 + 16);
      if ( *(v23 - 3) == result || !a3 )
        break;
LABEL_41:
      if ( v23 == (__int64 *)v168 )
      {
        v116 = v139;
        do
        {
          v117 = (_QWORD *)v116[5];
          v118 = (_QWORD *)v116[4];
          if ( v117 != v118 )
          {
            do
            {
              v119 = (_QWORD *)v118[9];
              result = (__int64)(v118 + 11);
              if ( v119 != v118 + 11 )
                result = j_j___libc_free_0(v119, v118[11] + 1LL);
              v120 = v118[6];
              if ( v120 )
                result = j_j___libc_free_0(v120, v118[8] - v120);
              v118 += 13;
            }
            while ( v117 != v118 );
            v118 = (_QWORD *)v116[4];
          }
          if ( v118 )
            result = j_j___libc_free_0(v118, v116[6] - (_QWORD)v118);
          v116 += 12;
        }
        while ( (__int64 *)v168 != v116 );
        goto LABEL_122;
      }
    }
    v57 = *((_DWORD *)v23 - 9);
    v58 = *((_DWORD *)v23 - 8);
    v145 = *(v23 - 10);
    v156 = *((_BYTE *)v23 - 16);
    v161 = *((_BYTE *)v23 - 15);
    v166 = *((_DWORD *)v23 - 3);
    v152 = *((_BYTE *)v23 - 8);
    v154 = *((_BYTE *)v23 - 7);
    if ( !v171 )
      goto LABEL_48;
    if ( a3 )
    {
      v22 = v143;
      if ( v145 != v143 )
        goto LABEL_48;
      v179 = __PAIR64__(v58, v57);
      v176 = __PAIR64__(v133, v137);
      if ( __PAIR64__(v58, v57) != __PAIR64__(v133, v137) )
        goto LABEL_48;
      v22 = 0xFFFFFFFF0000FFFFLL;
      LOBYTE(v180) = v156;
      BYTE1(v180) = v161;
      HIDWORD(v180) = v166;
      LOBYTE(v177) = v149;
      BYTE1(v177) = v129;
      HIDWORD(v177) = v131;
      if ( ((v177 ^ v180) & 0xFFFFFFFF0000FFFFLL) != 0
        || (LOBYTE(v181) = v152, HIBYTE(v181) = v154, LOBYTE(v178) = v148, HIBYTE(v178) = v147, v181 != v178) )
      {
LABEL_48:
        v176 = __PAIR64__(v58, v57);
        LOBYTE(v177) = v156;
        BYTE1(v177) = v161;
        HIDWORD(v177) = v166;
        LOBYTE(v178) = v152;
        HIBYTE(v178) = v154;
        v59 = v175[1];
        v130 = *(unsigned int **)(v59 + 160);
        v138 = *(_QWORD *)(v59 + 168);
        v171 = sub_E6C430(v59, v14, v20, v22, v21);
        (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*v175 + 208))(v175, v171, 0);
        v144 = sub_E6C430(v59, v171, v60, v61, v62);
        if ( v174 )
        {
          v150 = 0;
          v64 = 4;
          v65 = 4;
        }
        else
        {
          v63 = *(_BYTE *)(v59 + 1906);
          v150 = v63;
          if ( v63 )
          {
            if ( v63 != 1 )
LABEL_152:
              BUG();
            v65 = 8;
            (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v175 + 536))(v175, 0xFFFFFFFFLL, 4);
            v64 = 12;
          }
          else
          {
            v64 = 4;
            v65 = 4;
          }
        }
        v132 = v64;
        v134 = sub_E808D0(v144, 0, v59, 0);
        v66 = sub_E808D0(v171, 0, v59, 0);
        v135 = sub_E81A00(18, v134, v66, v59, 0);
        v67 = sub_E81A90(v132, v59, 0, 0);
        v68 = sub_E81A00(18, v135, v67, v59, 0);
        sub_E71DA0(v175, v68, v65, v69, v70);
        v71 = 0;
        if ( !v174 )
        {
          v71 = -1;
          if ( v150 != 1 )
            v71 = 0xFFFFFFFFLL;
        }
        (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*v175 + 536))(v175, v71, v65);
        if ( v174 )
        {
          v73 = 1;
          v74 = 1;
        }
        else
        {
          v72 = (unsigned int)*(unsigned __int16 *)(v59 + 1904) - 2;
          if ( (unsigned int)v72 > 3 )
            goto LABEL_152;
          v73 = dword_3F80260[v72];
          v74 = dword_3F80260[v72];
        }
        (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v175 + 536))(v175, v73, 1);
        if ( v174 )
        {
          v121 = *(v23 - 10) == 0;
          v185[0] = 122;
          v122 = 1;
          v182 = v185;
          v184 = 8;
          v183 = 1;
          if ( !v121 )
          {
            v185[1] = 80;
            v122 = 2;
            v183 = 2;
          }
          if ( *(v23 - 9) )
          {
            v185[v122] = 76;
            v122 = v183 + 1;
            v126 = v183 + 2;
            ++v183;
            if ( v126 > v184 )
            {
              sub_C8D290((__int64)&v182, v185, v126, 1u, v126, v75);
              v122 = v183;
            }
          }
          v182[v122] = 82;
          v123 = v183;
          v121 = *((_BYTE *)v23 - 16) == 0;
          v124 = ++v183;
          if ( !v121 )
          {
            if ( v123 + 2 > v184 )
            {
              sub_C8D290((__int64)&v182, v185, v123 + 2, 1u, v123 + 2, v75);
              v124 = v183;
            }
            v182[v124] = 83;
            v124 = ++v183;
          }
          if ( *((_BYTE *)v23 - 8) )
          {
            if ( v124 + 1 > v184 )
            {
              sub_C8D290((__int64)&v182, v185, v124 + 1, 1u, v124 + 1, v75);
              v124 = v183;
            }
            v182[v124] = 66;
            v124 = ++v183;
          }
          if ( *((_BYTE *)v23 - 7) )
          {
            if ( v124 + 1 > v184 )
            {
              sub_C8D290((__int64)&v182, v185, v124 + 1, 1u, v124 + 1, v75);
              v124 = v183;
            }
            v182[v124] = 71;
            ++v183;
          }
          v125 = (__int64)v182;
          (*(void (__fastcall **)(__int64 *, _BYTE *))(*v175 + 512))(v175, v182);
          if ( v182 != v185 )
            _libc_free(v182, v125);
        }
        (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v175 + 536))(v175, 0, 1);
        if ( v74 == 4 )
        {
          (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v175 + 536))(
            v175,
            *(unsigned int *)(*(_QWORD *)(v59 + 152) + 8LL),
            1);
          (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v175 + 536))(v175, 0, 1);
        }
        sub_E98EB0(v175, *(unsigned int *)(*(_QWORD *)(v59 + 152) + 28LL), 0);
        v76 = *(_QWORD *)(v175[1] + 152);
        v77 = *(_DWORD *)(v76 + 12);
        if ( !*(_BYTE *)(v76 + 17) )
          v77 = -v77;
        sub_E990E0(v175, v77);
        v78 = *((_DWORD *)v23 - 3);
        if ( v78 == 0x7FFFFFFF )
          v78 = (*(__int64 (__fastcall **)(unsigned int *, _QWORD, _QWORD))(*(_QWORD *)v130 + 16LL))(
                  v130,
                  v130[5],
                  v174);
        if ( v74 == 1 )
          (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v175 + 536))(v175, v78, 1);
        else
          sub_E98EB0(v175, v78, 0);
        if ( v174 )
        {
          v80 = v175;
          v81 = 1;
          if ( *(v23 - 10) )
            v81 = sub_E71E20(v175[1], *((_DWORD *)v23 - 9)) + 2;
          sub_E98EB0(v80, v81 - ((unsigned int)(*(v23 - 9) == 0) - 1), 0);
          if ( *(v23 - 10) )
          {
            (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v175 + 536))(v175, *((unsigned int *)v23 - 9), 1);
            v82 = v175;
            v136 = *((_DWORD *)v23 - 9);
            (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, __int64 *))(**(_QWORD **)(v175[1] + 152) + 24LL))(
              *(_QWORD *)(v175[1] + 152),
              *(v23 - 10),
              v136,
              v175);
            v83 = sub_E71E20(v82[1], v136);
            sub_E9A5B0(v82, v85, v83, v84);
          }
          if ( *(v23 - 9) )
            (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v175 + 536))(v175, *((unsigned int *)v23 - 8), 1);
          (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*v175 + 536))(v175, *(unsigned int *)(v138 + 12), 1);
        }
        v86 = *(_QWORD *)(v59 + 152);
        if ( !*((_BYTE *)v23 - 15) )
          sub_E71F00(
            (__int64)&v172,
            *(__int64 **)(v86 + 360),
            0x4EC4EC4EC4EC4EC5LL * ((__int64)(*(_QWORD *)(v86 + 368) - *(_QWORD *)(v86 + 360)) >> 3),
            0,
            v79);
        v87 = 2;
        v173 = v172;
        if ( !v174 )
        {
          v87 = 0xFFFFFFFFLL;
          if ( *(_DWORD *)(v86 + 8) )
          {
            _BitScanReverse64(&v88, *(unsigned int *)(v86 + 8));
            v87 = 63 - ((unsigned int)v88 ^ 0x3F);
          }
        }
        (*(void (__fastcall **)(__int64 *, __int64, _QWORD, __int64, _QWORD))(*v175 + 608))(v175, v87, 0, 1, 0);
        v14 = v144;
        (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*v175 + 208))(v175, v144, 0);
        v133 = v58;
        v137 = v57;
        v147 = v154;
        v148 = v152;
        v131 = v166;
        v129 = v161;
        v149 = v156;
        v143 = v145;
      }
    }
    v89 = v175[1];
    v90 = sub_E6C430(v89, v14, v20, v22, v21);
    v94 = sub_E6C430(v89, v14, v91, v92, v93);
    v162 = *(_QWORD *)(v89 + 168);
    v172 = v173;
    if ( v174 || (v24 = *(_BYTE *)(v89 + 1906)) == 0 )
    {
      v25 = 4;
    }
    else
    {
      if ( v24 != 1 )
        goto LABEL_152;
      v25 = 8;
      (*(void (__fastcall **)(__int64 *, __int64, __int64))(*v175 + 536))(v175, 0xFFFFFFFFLL, 4);
    }
    v163 = sub_E808D0(v94, 0, v89, 0);
    v26 = sub_E808D0(v90, 0, v89, 0);
    v164 = sub_E81A00(18, v163, v26, v89, 0);
    v27 = sub_E81A90(0, v89, 0, 0);
    v28 = sub_E81A00(18, v164, v27, v89, 0);
    sub_E71DA0(v175, v28, v25, v29, v30);
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*v175 + 208))(v175, v90, 0);
    v165 = *(_QWORD *)(v89 + 152);
    if ( v174 )
    {
      v31 = v171;
      v32 = sub_E808D0(v90, 0, v89, 0);
      v33 = v89;
    }
    else
    {
      if ( *(_BYTE *)(v165 + 348) )
      {
        sub_E9A500(v175, v171, v25, *(unsigned __int8 *)(v165 + 259));
        goto LABEL_29;
      }
      v31 = v128;
      v32 = sub_E808D0(v171, 0, v89, 0);
      v33 = v89;
    }
    v34 = sub_E808D0(v31, 0, v33, 0);
    v35 = sub_E81A00(18, v32, v34, v89, 0);
    v36 = sub_E81A90(0, v89, 0, 0);
    v37 = sub_E81A00(18, v35, v36, v89, 0);
    sub_E71DA0(v175, v37, v25, v38, v39);
LABEL_29:
    LOBYTE(v40) = 0;
    if ( v174 )
      v40 = *(_DWORD *)(v162 + 12);
    v42 = (unsigned int)sub_E71E20(v175[1], v40);
    sub_E71E80(v44, *(v23 - 12), v41, v43);
    v160 = *(v23 - 12);
    v45 = sub_E808D0(*(v23 - 11), 0, v89, 0);
    v46 = sub_E808D0(v160, 0, v89, 0);
    v47 = sub_E81A00(18, v45, v46, v89, 0);
    v48 = sub_E81A90(0, v89, 0, 0);
    v49 = sub_E81A00(18, v47, v48, v89, 0);
    sub_E71DA0(v175, v49, (unsigned int)v42, v50, v51);
    if ( v174 )
    {
      v53 = 0;
      v54 = v175;
      if ( *(v23 - 9) )
        v53 = (unsigned int)sub_E71E20(v175[1], *((_DWORD *)v23 - 8));
      sub_E98EB0(v54, v53, 0);
      v55 = *(v23 - 9);
      if ( v55 )
        sub_E71E80(v175, v55, *((unsigned int *)v23 - 8), 1);
    }
    sub_E71F00(
      (__int64)&v172,
      (__int64 *)*(v23 - 8),
      0x4EC4EC4EC4EC4EC5LL * ((*(v23 - 7) - *(v23 - 8)) >> 3),
      *(v23 - 12),
      v52);
    if ( v23 == (__int64 *)v168 )
      v42 = *(unsigned int *)(v165 + 8);
    v56 = 0xFFFFFFFFLL;
    if ( v42 )
    {
      _BitScanReverse64(&v42, v42);
      v56 = 63 - ((unsigned int)v42 ^ 0x3F);
    }
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD, __int64, _QWORD))(*v175 + 608))(v175, v56, 0, 1, 0);
    v14 = v94;
    result = (*(__int64 (__fastcall **)(__int64 *, __int64, _QWORD))(*v175 + 208))(v175, v94, 0);
    goto LABEL_41;
  }
LABEL_122:
  if ( v139 )
    return j_j___libc_free_0(v139, v127);
  return result;
}
