// Function: sub_280FC50
// Address: 0x280fc50
//
__int64 __fastcall sub_280FC50(__int64 a1, __int64 *a2, __int64 **a3)
{
  __int64 v4; // r14
  char v5; // r13
  unsigned int v6; // r14d
  char v8; // r13
  __int64 v9; // r14
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rbx
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // r13
  __int64 v26; // rbx
  _QWORD *v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // r15
  __int64 v33; // rdx
  _QWORD *v34; // rax
  _QWORD *v35; // rdx
  bool v36; // zf
  __int64 v37; // rbx
  _QWORD *v38; // rcx
  __int64 v39; // r13
  __int64 v40; // r14
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rsi
  _BYTE *v45; // rax
  _BYTE *v46; // rax
  _BYTE *v47; // rbx
  __int64 v48; // r15
  __int64 v49; // rbx
  char v50; // al
  _QWORD *v51; // rax
  _QWORD *v52; // rax
  __int64 v53; // r14
  unsigned __int8 v54; // cl
  __int64 v55; // rsi
  _QWORD *v56; // rax
  _QWORD *v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rdi
  __int64 v60; // rcx
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // r14
  __int64 v64; // rax
  __int64 v65; // rdi
  __int64 v66; // rcx
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 i; // r15
  int v70; // eax
  __int64 v71; // r14
  char *v72; // rax
  char *v73; // rdx
  char v74; // al
  __int64 v75; // rax
  __int64 v76; // rax
  __int64 v77; // r8
  __int64 v78; // rax
  __int64 v79; // rax
  unsigned __int64 v80; // rdx
  __int64 v81; // rbx
  __int64 *v82; // rsi
  int v83; // ecx
  unsigned __int8 **v84; // r9
  unsigned __int64 v85; // rax
  int v86; // edx
  __int64 v87; // rbx
  int v88; // eax
  bool v89; // of
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // rsi
  __int64 v93; // rax
  unsigned int v94; // eax
  __int64 v95; // rax
  __int64 v96; // rdi
  __int64 v97; // rcx
  __int64 v98; // rax
  __int64 v99; // rax
  _BYTE *v100; // rdi
  __int64 *v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r8
  __int64 v104; // r9
  _QWORD *v105; // rax
  __int64 *v106; // rax
  _BYTE *v107; // rax
  __int64 v108; // rax
  _BYTE *v109; // rdi
  _BYTE *v110; // rsi
  __int64 v111; // rbx
  __int64 v112; // rsi
  _QWORD *v113; // rax
  _QWORD *v114; // rdx
  _BYTE *v115; // rax
  __int64 *v116; // [rsp+10h] [rbp-130h]
  unsigned __int8 v117; // [rsp+27h] [rbp-119h]
  __int64 v118; // [rsp+28h] [rbp-118h]
  __int64 v119; // [rsp+30h] [rbp-110h]
  _BYTE *v120; // [rsp+40h] [rbp-100h]
  int v121; // [rsp+40h] [rbp-100h]
  __int64 v122; // [rsp+40h] [rbp-100h]
  __int64 *v123; // [rsp+48h] [rbp-F8h]
  unsigned __int8 v124; // [rsp+48h] [rbp-F8h]
  unsigned __int8 v125; // [rsp+50h] [rbp-F0h]
  __int64 v126; // [rsp+50h] [rbp-F0h]
  int v127; // [rsp+58h] [rbp-E8h]
  unsigned __int8 v128; // [rsp+58h] [rbp-E8h]
  __int64 v129; // [rsp+58h] [rbp-E8h]
  _BYTE *v131; // [rsp+60h] [rbp-E0h]
  __int64 v132; // [rsp+68h] [rbp-D8h]
  unsigned __int64 v133; // [rsp+70h] [rbp-D0h] BYREF
  unsigned __int64 v134; // [rsp+78h] [rbp-C8h]
  __int64 v135; // [rsp+80h] [rbp-C0h] BYREF
  int v136; // [rsp+88h] [rbp-B8h]
  unsigned __int8 v137; // [rsp+8Ch] [rbp-B4h]
  _QWORD v138[4]; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v139; // [rsp+B0h] [rbp-90h] BYREF
  char *v140; // [rsp+B8h] [rbp-88h]
  __int64 v141; // [rsp+C0h] [rbp-80h]
  int v142; // [rsp+C8h] [rbp-78h]
  char v143; // [rsp+CCh] [rbp-74h]
  char v144; // [rsp+D0h] [rbp-70h] BYREF

  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE *)(a1 + 208);
  v140 = &v144;
  v139 = 0;
  v141 = 8;
  v142 = 0;
  v143 = 1;
  if ( !(unsigned __int8)sub_D4B3D0(v4) )
    goto LABEL_2;
  if ( !(unsigned __int8)sub_D4B750(v4, (__int64)a2) )
    goto LABEL_2;
  if ( !(unsigned __int8)sub_280F570(
                           v4,
                           (__int64)&v139,
                           a1 + 16,
                           (__int64 *)(a1 + 32),
                           (__int64 *)(a1 + 112),
                           (__int64 *)(a1 + 128),
                           a2,
                           v5) )
    goto LABEL_2;
  v8 = *(_BYTE *)(a1 + 208);
  v9 = *(_QWORD *)a1;
  if ( !(unsigned __int8)sub_D4B3D0(*(_QWORD *)a1) )
    goto LABEL_2;
  if ( !(unsigned __int8)sub_D4B750(v9, (__int64)a2) )
    goto LABEL_2;
  if ( !(unsigned __int8)sub_280F570(
                           v9,
                           (__int64)&v139,
                           a1 + 24,
                           (__int64 *)(a1 + 40),
                           (__int64 *)(a1 + 120),
                           (__int64 *)(a1 + 136),
                           a2,
                           v8) )
    goto LABEL_2;
  if ( !(unsigned __int8)sub_D48480(*(_QWORD *)a1, *(_QWORD *)(a1 + 32), v10, v11) )
    goto LABEL_2;
  v117 = sub_D48480(*(_QWORD *)a1, *(_QWORD *)(a1 + 40), v12, v13);
  if ( !v117 )
    goto LABEL_2;
  v137 = 1;
  v134 = (unsigned __int64)v138;
  v14 = *(_QWORD *)(a1 + 24);
  v135 = 0x100000004LL;
  v138[0] = v14;
  v15 = *(_QWORD *)(a1 + 8);
  v136 = 0;
  v133 = 1;
  v16 = sub_AA5930(**(_QWORD **)(v15 + 32));
  v18 = v17;
  v19 = v16;
  while ( v18 != v19 )
  {
    if ( *(_QWORD *)(a1 + 16) == v19
      || *(_BYTE *)(a1 + 208) && (*(_QWORD *)(a1 + 216) == v19 || *(_QWORD *)(a1 + 224) == v19) )
    {
      if ( !v19 )
        BUG();
      goto LABEL_19;
    }
    v58 = sub_D4B130(*(_QWORD *)(a1 + 8));
    v59 = *(_QWORD *)(v19 - 8);
    v60 = v58;
    if ( (*(_DWORD *)(v19 + 4) & 0x7FFFFFF) != 0 )
    {
      v61 = 0;
      while ( v60 != *(_QWORD *)(v59 + 32LL * *(unsigned int *)(v19 + 72) + 8 * v61) )
      {
        if ( (*(_DWORD *)(v19 + 4) & 0x7FFFFFF) == (_DWORD)++v61 )
          goto LABEL_166;
      }
      v62 = 32 * v61;
    }
    else
    {
LABEL_166:
      v62 = 0x1FFFFFFFE0LL;
    }
    v63 = *(_QWORD *)(v59 + v62);
    v64 = sub_D47930(*(_QWORD *)(a1 + 8));
    v65 = *(_QWORD *)(v19 - 8);
    v66 = v64;
    if ( (*(_DWORD *)(v19 + 4) & 0x7FFFFFF) != 0 )
    {
      v67 = 0;
      while ( v66 != *(_QWORD *)(v65 + 32LL * *(unsigned int *)(v19 + 72) + 8 * v67) )
      {
        if ( (*(_DWORD *)(v19 + 4) & 0x7FFFFFF) == (_DWORD)++v67 )
          goto LABEL_165;
      }
      v68 = 32 * v67;
    }
    else
    {
LABEL_165:
      v68 = 0x1FFFFFFFE0LL;
    }
    if ( *(_BYTE *)v63 != 84 || *(_QWORD *)(v63 + 40) != **(_QWORD **)(*(_QWORD *)a1 + 32LL) )
      goto LABEL_120;
    v129 = *(_QWORD *)(v65 + v68);
    v95 = sub_D47930(*(_QWORD *)a1);
    v96 = *(_QWORD *)(v63 - 8);
    v97 = v95;
    if ( (*(_DWORD *)(v63 + 4) & 0x7FFFFFF) != 0 )
    {
      v98 = 0;
      while ( v97 != *(_QWORD *)(v96 + 32LL * *(unsigned int *)(v63 + 72) + 8 * v98) )
      {
        if ( (*(_DWORD *)(v63 + 4) & 0x7FFFFFF) == (_DWORD)++v98 )
          goto LABEL_196;
      }
      v99 = 32 * v98;
    }
    else
    {
LABEL_196:
      v99 = 0x1FFFFFFFE0LL;
    }
    v100 = *(_BYTE **)(v96 + v99);
    if ( *v100 != 84 || v129 != sub_B48DC0((__int64)v100) )
    {
LABEL_120:
      v6 = v137;
      if ( !v137 )
        goto LABEL_121;
      goto LABEL_2;
    }
    if ( !v137 )
      goto LABEL_224;
    v105 = (_QWORD *)v134;
    v102 = HIDWORD(v135);
    v101 = (__int64 *)(v134 + 8LL * HIDWORD(v135));
    if ( (__int64 *)v134 != v101 )
    {
      while ( v63 != *v105 )
      {
        if ( v101 == ++v105 )
          goto LABEL_237;
      }
      goto LABEL_182;
    }
LABEL_237:
    if ( HIDWORD(v135) < (unsigned int)v135 )
    {
      v102 = (unsigned int)++HIDWORD(v135);
      *v101 = v63;
      ++v133;
    }
    else
    {
LABEL_224:
      sub_C8CC70((__int64)&v133, v63, (__int64)v101, v102, v103, v104);
    }
LABEL_182:
    if ( !*(_BYTE *)(a1 + 172) )
      goto LABEL_236;
    v106 = *(__int64 **)(a1 + 152);
    v102 = *(unsigned int *)(a1 + 164);
    v101 = &v106[v102];
    if ( v106 != v101 )
    {
      while ( *v106 != v19 )
      {
        if ( v101 == ++v106 )
          goto LABEL_186;
      }
      goto LABEL_19;
    }
LABEL_186:
    if ( (unsigned int)v102 < *(_DWORD *)(a1 + 160) )
    {
      *(_DWORD *)(a1 + 164) = v102 + 1;
      *v101 = v19;
      ++*(_QWORD *)(a1 + 144);
    }
    else
    {
LABEL_236:
      sub_C8CC70(a1 + 144, v19, (__int64)v101, v102, v103, v104);
    }
LABEL_19:
    v20 = *(_QWORD *)(v19 + 32);
    if ( !v20 )
      BUG();
    v19 = 0;
    if ( *(_BYTE *)(v20 - 24) == 84 )
      v19 = v20 - 24;
  }
  v21 = sub_AA5930(**(_QWORD **)(*(_QWORD *)a1 + 32LL));
  v25 = v24;
  v26 = v21;
  while ( v25 != v26 )
  {
    if ( !*(_BYTE *)(a1 + 208) || v26 != *(_QWORD *)(a1 + 216) && v26 != *(_QWORD *)(a1 + 224) )
    {
      if ( v137 )
      {
        v27 = (_QWORD *)v134;
        v28 = (_QWORD *)(v134 + 8LL * HIDWORD(v135));
        if ( (_QWORD *)v134 == v28 )
          goto LABEL_120;
        while ( v26 != *v27 )
        {
          if ( v28 == ++v27 )
            goto LABEL_120;
        }
      }
      else if ( !sub_C8CA60((__int64)&v133, v26) )
      {
        goto LABEL_120;
      }
    }
    if ( !v26 )
      BUG();
    v29 = *(_QWORD *)(v26 + 32);
    if ( !v29 )
      BUG();
    v26 = 0;
    if ( *(_BYTE *)(v29 - 24) == 84 )
      v26 = v29 - 24;
  }
  if ( !v137 )
    _libc_free(v134);
  v30 = *(_QWORD *)(a1 + 16);
  if ( *(_QWORD *)(v30 + 8) != *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL) )
  {
LABEL_2:
    v6 = 0;
    goto LABEL_3;
  }
  v116 = *(__int64 **)(*(_QWORD *)a1 + 40LL);
  if ( *(__int64 **)(*(_QWORD *)a1 + 32LL) != v116 )
  {
    v123 = *(__int64 **)(*(_QWORD *)a1 + 32LL);
    v31 = 0;
    v127 = 0;
    while ( 1 )
    {
      v32 = *v123;
      v33 = *(_QWORD *)(a1 + 8);
      if ( *(_BYTE *)(v33 + 84) )
      {
        v34 = *(_QWORD **)(v33 + 64);
        v35 = &v34[*(unsigned int *)(v33 + 76)];
        if ( v34 == v35 )
          goto LABEL_123;
        while ( v32 != *v34 )
        {
          if ( v35 == ++v34 )
            goto LABEL_123;
        }
      }
      else if ( !sub_C8CA60(v33 + 56, *v123) )
      {
LABEL_123:
        v126 = v32 + 48;
        for ( i = *(_QWORD *)(v32 + 56); v126 != i; i = *(_QWORD *)(i + 8) )
        {
          if ( !i )
            BUG();
          v70 = *(unsigned __int8 *)(i - 24);
          v71 = i - 24;
          if ( (_BYTE)v70 != 84 && (unsigned int)(v70 - 30) > 0xA )
          {
            LOBYTE(v94) = sub_991A70((unsigned __int8 *)(i - 24), 0, 0, 0, 0, 1u, 0);
            if ( !(_BYTE)v94 )
            {
              v6 = v94;
              goto LABEL_3;
            }
          }
          if ( v143 )
          {
            v72 = v140;
            v73 = &v140[8 * HIDWORD(v141)];
            if ( v140 != v73 )
            {
              while ( v71 != *(_QWORD *)v72 )
              {
                v72 += 8;
                if ( v73 == v72 )
                  goto LABEL_135;
              }
              continue;
            }
          }
          else if ( sub_C8CA60((__int64)&v139, i - 24) )
          {
            continue;
          }
LABEL_135:
          v74 = *(_BYTE *)(i - 24);
          if ( v74 == 31 )
          {
            v75 = *(_DWORD *)(i - 20) & 0x7FFFFFF;
            if ( (_DWORD)v75 != 1 || *(_QWORD *)(i - 56) != **(_QWORD **)(*(_QWORD *)(a1 + 8) + 32LL) )
            {
              v76 = 32 * v75;
              if ( (*(_BYTE *)(i - 17) & 0x40) == 0 )
                goto LABEL_157;
              goto LABEL_138;
            }
          }
          else if ( v74 != 46
                 || ((v90 = *(_QWORD *)(i - 88),
                      v91 = *(_QWORD *)(i - 56),
                      v92 = *(_QWORD *)(a1 + 32),
                      v93 = *(_QWORD *)(a1 + 24),
                      v93 != v90)
                  || v91 != v92)
                 && (v91 != v93 || v92 != v90) )
          {
            v76 = 32LL * (*(_DWORD *)(i - 20) & 0x7FFFFFF);
            if ( (*(_BYTE *)(i - 17) & 0x40) == 0 )
            {
LABEL_157:
              v77 = v71 - v76;
              v78 = i - 24;
              goto LABEL_139;
            }
LABEL_138:
            v77 = *(_QWORD *)(i - 32);
            v78 = v77 + v76;
LABEL_139:
            v79 = v78 - v77;
            v134 = 0x400000000LL;
            v80 = v79 >> 5;
            v133 = (unsigned __int64)&v135;
            v81 = v79 >> 5;
            if ( (unsigned __int64)v79 > 0x80 )
            {
              v118 = v79;
              v119 = v77;
              v122 = v79 >> 5;
              sub_C8D5F0((__int64)&v133, &v135, v80, 8u, v77, v23);
              v84 = (unsigned __int8 **)v133;
              v83 = v134;
              LODWORD(v80) = v122;
              v77 = v119;
              v79 = v118;
              v82 = (__int64 *)(v133 + 8LL * (unsigned int)v134);
            }
            else
            {
              v82 = &v135;
              v83 = 0;
              v84 = (unsigned __int8 **)&v135;
            }
            if ( v79 > 0 )
            {
              v85 = 0;
              do
              {
                v82[v85 / 8] = *(_QWORD *)(v77 + 4 * v85);
                v85 += 8LL;
                --v81;
              }
              while ( v81 );
              v84 = (unsigned __int8 **)v133;
              v83 = v134;
            }
            LODWORD(v134) = v80 + v83;
            v87 = sub_DFCEF0(a3, (unsigned __int8 *)(i - 24), v84, (unsigned int)(v80 + v83), 3);
            if ( (__int64 *)v133 != &v135 )
            {
              v121 = v86;
              _libc_free(v133);
              v86 = v121;
            }
            v88 = 1;
            if ( v86 != 1 )
              v88 = v127;
            v89 = __OFADD__(v87, v31);
            v31 += v87;
            v127 = v88;
            if ( v89 )
            {
              v31 = 0x8000000000000000LL;
              if ( v87 > 0 )
                v31 = 0x7FFFFFFFFFFFFFFFLL;
            }
          }
        }
      }
      if ( v116 == ++v123 )
      {
        if ( v127 || (unsigned int)qword_4FFF2E8 < v31 )
          goto LABEL_2;
        v30 = *(_QWORD *)(a1 + 16);
        break;
      }
    }
  }
  v36 = *(_BYTE *)(a1 + 208) == 0;
  v133 = 0;
  v135 = 4;
  v134 = (unsigned __int64)v138;
  v37 = *(_QWORD *)(a1 + 32);
  v136 = 0;
  v137 = 1;
  v120 = (_BYTE *)v37;
  if ( !v36 && (unsigned __int8)(*(_BYTE *)v37 - 68) <= 1u )
  {
    if ( (*(_BYTE *)(v37 + 7) & 0x40) != 0 )
      v38 = *(_QWORD **)(v37 - 8);
    else
      v38 = (_QWORD *)(v37 - 32LL * (*(_DWORD *)(v37 + 4) & 0x7FFFFFF));
    v120 = (_BYTE *)*v38;
  }
  v39 = *(_QWORD *)(v30 + 16);
  if ( !v39 )
  {
    v53 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL);
    if ( v53 )
      goto LABEL_98;
    goto LABEL_108;
  }
  do
  {
    v40 = *(_QWORD *)(v39 + 24);
    if ( v40 == *(_QWORD *)(a1 + 112) )
      goto LABEL_96;
    if ( *(_BYTE *)v40 == 67 )
    {
      v108 = *(_QWORD *)(v40 + 16);
      if ( !v108 || *(_QWORD *)(v108 + 8) )
        goto LABEL_120;
      v40 = *(_QWORD *)(v108 + 24);
    }
    if ( *(_QWORD *)(*(_QWORD *)(a1 + 128) - 96LL) == v40 )
      goto LABEL_96;
    v41 = *(_QWORD *)(a1 + 16);
    if ( *(_BYTE *)v40 == 42 )
    {
      v42 = *(_QWORD *)(v40 - 64);
      v107 = *(_BYTE **)(v40 - 32);
      if ( v41 != v42 || (v132 = *(_QWORD *)(v40 - 32), !v107) )
      {
        if ( !v42 || (_BYTE *)v41 != v107 )
        {
          v131 = 0;
          v132 = 0;
          v128 = 0;
          goto LABEL_192;
        }
        v132 = *(_QWORD *)(v40 - 64);
      }
      v128 = 0;
      v131 = 0;
      if ( *(_BYTE *)v132 == 46 )
      {
        v109 = *(_BYTE **)(a1 + 24);
        v110 = *(_BYTE **)(v132 - 32);
        v131 = *(_BYTE **)(v132 - 64);
        if ( v109 == v131 && v110 )
        {
          v131 = *(_BYTE **)(v132 - 32);
          v128 = v117;
        }
        else if ( v109 != v110 || (v128 = v117, !v131) )
        {
          v131 = 0;
          v128 = 0;
        }
      }
LABEL_192:
      if ( *(_BYTE *)v42 == 67 && v41 == *(_QWORD *)(v42 - 32) && v107 )
      {
        v132 = *(_QWORD *)(v40 - 32);
      }
      else
      {
        if ( *v107 != 67 || v41 != *((_QWORD *)v107 - 4) )
        {
LABEL_194:
          if ( !v131 )
            goto LABEL_120;
          v125 = 0;
          v124 = 0;
          goto LABEL_77;
        }
        v132 = *(_QWORD *)(v40 - 64);
      }
      if ( *(_BYTE *)v132 == 46 )
      {
        v41 = *(_QWORD *)(v132 - 64);
        v42 = *(_QWORD *)(a1 + 24);
        v115 = *(_BYTE **)(v132 - 32);
        if ( *(_BYTE *)v41 == 67 && v42 == *(_QWORD *)(v41 - 32) && v115 )
          goto LABEL_233;
        if ( *v115 == 67 && v42 == *((_QWORD *)v115 - 4) )
        {
          v115 = *(_BYTE **)(v132 - 64);
LABEL_233:
          v131 = v115;
          v124 = 0;
          v125 = v117;
          goto LABEL_77;
        }
      }
      goto LABEL_194;
    }
    if ( *(_BYTE *)v40 != 63 || (*(_DWORD *)(v40 + 4) & 0x7FFFFFF) != 2 )
      goto LABEL_120;
    v42 = v40 - 64;
    if ( (*(_BYTE *)(v40 + 7) & 0x40) != 0 )
      v42 = *(_QWORD *)(v40 - 8);
    v43 = *(_QWORD *)v42;
    if ( **(_BYTE **)v42 != 63 || (*(_DWORD *)(v43 + 4) & 0x7FFFFFF) != 2 )
      goto LABEL_120;
    v44 = v43 - 64;
    if ( (*(_BYTE *)(v43 + 7) & 0x40) != 0 )
      v44 = *(_QWORD *)(v43 - 8);
    v45 = *(_BYTE **)(v44 + 32);
    v132 = (__int64)v45;
    if ( !v45 || v41 != *(_QWORD *)(v42 + 32) || *v45 != 46 )
      goto LABEL_120;
    v41 = *(_QWORD *)(a1 + 24);
    v46 = (_BYTE *)*((_QWORD *)v45 - 8);
    v47 = *(_BYTE **)(*(_QWORD *)(v44 + 32) - 32LL);
    v131 = v47;
    if ( (_BYTE *)v41 != v46 || !v47 )
    {
      if ( !v46 || (_BYTE *)v41 != v47 )
        goto LABEL_120;
      v131 = v46;
    }
    v125 = 0;
    v128 = 0;
    v124 = v117;
LABEL_77:
    v48 = *(_QWORD *)(v132 + 16);
    if ( v48 )
    {
      v49 = 0;
      do
      {
        v50 = sub_F50EE0(*(unsigned __int8 **)(v48 + 24), 0);
        v48 = *(_QWORD *)(v48 + 8);
        v49 += v50 == 0;
      }
      while ( v48 );
      if ( v49 > 1 )
        goto LABEL_120;
    }
    if ( !*(_BYTE *)(a1 + 208) )
      goto LABEL_199;
    if ( v124 || v128 )
    {
      if ( (unsigned __int8)(*v131 - 68) <= 1u )
        v131 = (_BYTE *)*((_QWORD *)v131 - 4);
LABEL_199:
      if ( !(v124 | v125) && !v128 )
        goto LABEL_120;
      goto LABEL_85;
    }
    if ( !v125 )
      goto LABEL_120;
LABEL_85:
    if ( v120 != v131 )
      goto LABEL_120;
    if ( !v137 )
      goto LABEL_245;
    v51 = (_QWORD *)v134;
    v41 = HIDWORD(v135);
    v42 = v134 + 8LL * HIDWORD(v135);
    if ( v134 != v42 )
    {
      while ( v132 != *v51 )
      {
        if ( (_QWORD *)v42 == ++v51 )
          goto LABEL_249;
      }
      goto LABEL_91;
    }
LABEL_249:
    if ( HIDWORD(v135) < (unsigned int)v135 )
    {
      v41 = (unsigned int)++HIDWORD(v135);
      *(_QWORD *)v42 = v132;
      ++v133;
    }
    else
    {
LABEL_245:
      sub_C8CC70((__int64)&v133, v132, v41, v42, v22, v23);
    }
LABEL_91:
    if ( !*(_BYTE *)(a1 + 76) )
      goto LABEL_256;
    v52 = *(_QWORD **)(a1 + 56);
    v41 = *(unsigned int *)(a1 + 68);
    v42 = (__int64)&v52[v41];
    if ( v52 == (_QWORD *)v42 )
    {
LABEL_254:
      if ( (unsigned int)v41 < *(_DWORD *)(a1 + 64) )
      {
        *(_DWORD *)(a1 + 68) = v41 + 1;
        *(_QWORD *)v42 = v40;
        ++*(_QWORD *)(a1 + 48);
        goto LABEL_96;
      }
LABEL_256:
      sub_C8CC70(a1 + 48, v40, v41, v42, v22, v23);
      goto LABEL_96;
    }
    while ( v40 != *v52 )
    {
      if ( (_QWORD *)v42 == ++v52 )
        goto LABEL_254;
    }
LABEL_96:
    v39 = *(_QWORD *)(v39 + 8);
  }
  while ( v39 );
  v53 = *(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL);
  if ( v53 )
  {
LABEL_98:
    v54 = v137;
    do
    {
      v55 = *(_QWORD *)(v53 + 24);
      if ( v55 != *(_QWORD *)(a1 + 120) )
      {
        if ( *(_BYTE *)v55 == 67 )
        {
          v111 = *(_QWORD *)(v55 + 16);
          if ( v111 )
          {
            while ( 1 )
            {
              v112 = *(_QWORD *)(v111 + 24);
              if ( v137 )
              {
                v113 = (_QWORD *)v134;
                v114 = (_QWORD *)(v134 + 8LL * HIDWORD(v135));
                if ( (_QWORD *)v134 == v114 )
                  goto LABEL_2;
                while ( v112 != *v113 )
                {
                  if ( v114 == ++v113 )
                    goto LABEL_2;
                }
              }
              else if ( !sub_C8CA60((__int64)&v133, v112) )
              {
                goto LABEL_120;
              }
              v111 = *(_QWORD *)(v111 + 8);
              if ( !v111 )
                goto LABEL_214;
            }
          }
        }
        else if ( v54 )
        {
          v56 = (_QWORD *)v134;
          v57 = (_QWORD *)(v134 + 8LL * HIDWORD(v135));
          if ( (_QWORD *)v134 == v57 )
            goto LABEL_2;
          while ( v55 != *v56 )
          {
            if ( v57 == ++v56 )
              goto LABEL_2;
          }
        }
        else
        {
          if ( !sub_C8CA60((__int64)&v133, v55) )
            goto LABEL_120;
LABEL_214:
          v54 = v137;
        }
      }
      v53 = *(_QWORD *)(v53 + 8);
    }
    while ( v53 );
  }
  if ( v137 )
  {
LABEL_108:
    v6 = v117;
    goto LABEL_3;
  }
  v6 = v117;
LABEL_121:
  _libc_free(v134);
LABEL_3:
  if ( !v143 )
    _libc_free((unsigned __int64)v140);
  return v6;
}
