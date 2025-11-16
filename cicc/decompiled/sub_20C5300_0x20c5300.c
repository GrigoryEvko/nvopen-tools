// Function: sub_20C5300
// Address: 0x20c5300
//
__int64 __fastcall sub_20C5300(__int64 a1, int a2, _QWORD *a3, _QWORD *a4)
{
  __int64 *v4; // r14
  unsigned int *v5; // rdi
  __int64 v6; // rdi
  unsigned int v7; // r12d
  __int64 v8; // rbx
  unsigned int v9; // r14d
  unsigned int *v10; // rdi
  _QWORD *v11; // rax
  int *v12; // r9
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned __int64 v15; // rdi
  __int64 v16; // r8
  __int64 v17; // rax
  __int64 v18; // r9
  unsigned int v19; // edx
  __int16 *v20; // rsi
  __int16 v21; // ax
  __int16 *v22; // rsi
  unsigned __int16 v23; // dx
  __int16 *v24; // rax
  __int16 v25; // cx
  __int16 *v27; // rsi
  __int16 v28; // ax
  __int16 *v29; // rsi
  unsigned __int16 v30; // dx
  __int16 v31; // cx
  __int16 *v32; // rax
  __int64 *v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rbx
  __int64 v36; // r15
  _QWORD *v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rdx
  int v40; // eax
  unsigned int *v41; // r12
  _QWORD *v42; // rbx
  __int32 v43; // esi
  __int64 v44; // rdi
  __int64 (*v45)(); // rax
  __int64 v46; // rdx
  unsigned int v47; // r13d
  unsigned int *v48; // rdx
  unsigned int *v49; // rax
  __int64 v50; // rax
  __int64 v51; // r15
  __int64 v52; // rbx
  unsigned int v53; // r13d
  _QWORD *v54; // rax
  int *v55; // rsi
  __int64 v56; // rdx
  __int64 v57; // rsi
  __int64 v58; // r8
  _QWORD *v59; // rdi
  unsigned int v60; // edx
  __int16 v61; // ax
  _WORD *v62; // rdx
  _WORD *v63; // r9
  unsigned __int16 v64; // cx
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rdx
  __int64 v68; // rax
  __int16 v69; // cx
  __int16 v70; // ax
  __int64 v71; // rax
  __int64 v72; // rdx
  __int64 v73; // rbx
  __int64 v74; // r15
  __int64 v75; // r12
  unsigned int v76; // eax
  __int64 v77; // rax
  __int64 v78; // rdx
  __int64 v79; // rbx
  __int64 i; // r15
  __int64 v81; // rax
  __int64 v82; // rsi
  _QWORD *v83; // rsi
  _QWORD *v84; // rax
  _QWORD *v85; // rdx
  _BOOL8 v86; // rdi
  __int64 v87; // rdi
  __int64 v88; // rsi
  _QWORD *v89; // r12
  _QWORD *v90; // rbx
  _QWORD *v91; // rax
  bool v92; // al
  _QWORD *v93; // rdx
  __int64 v94; // rdx
  _QWORD *v95; // rdi
  __int64 v96; // rax
  __int64 v97; // [rsp+8h] [rbp-168h]
  _QWORD *v98; // [rsp+10h] [rbp-160h]
  unsigned __int64 v99; // [rsp+18h] [rbp-158h]
  __int64 v100; // [rsp+20h] [rbp-150h]
  unsigned int *v102; // [rsp+38h] [rbp-138h]
  __int64 v104; // [rsp+48h] [rbp-128h]
  __int64 v105; // [rsp+50h] [rbp-120h]
  unsigned int v106; // [rsp+58h] [rbp-118h]
  unsigned int v107; // [rsp+5Ch] [rbp-114h]
  __int64 v108; // [rsp+60h] [rbp-110h]
  unsigned int *v109; // [rsp+68h] [rbp-108h]
  unsigned __int64 v110; // [rsp+68h] [rbp-108h]
  int v111; // [rsp+78h] [rbp-F8h]
  int *v112; // [rsp+80h] [rbp-F0h]
  __int32 v113; // [rsp+80h] [rbp-F0h]
  __int64 *v114; // [rsp+88h] [rbp-E8h]
  unsigned __int8 v115; // [rsp+88h] [rbp-E8h]
  __int64 v116; // [rsp+98h] [rbp-D8h]
  __int32 v117; // [rsp+98h] [rbp-D8h]
  unsigned int v118; // [rsp+ACh] [rbp-C4h] BYREF
  unsigned int *v119; // [rsp+B0h] [rbp-C0h] BYREF
  unsigned int *v120; // [rsp+B8h] [rbp-B8h]
  __int64 v121; // [rsp+C0h] [rbp-B0h]
  __int64 v122; // [rsp+D0h] [rbp-A0h] BYREF
  int v123; // [rsp+D8h] [rbp-98h] BYREF
  _QWORD *v124; // [rsp+E0h] [rbp-90h]
  int *v125; // [rsp+E8h] [rbp-88h]
  int *v126; // [rsp+F0h] [rbp-80h]
  __int64 v127; // [rsp+F8h] [rbp-78h]
  __m128i v128; // [rsp+100h] [rbp-70h] BYREF
  int v129; // [rsp+110h] [rbp-60h]
  unsigned __int16 v130; // [rsp+118h] [rbp-58h]
  _WORD *v131; // [rsp+120h] [rbp-50h]
  int v132; // [rsp+128h] [rbp-48h]
  unsigned __int16 v133; // [rsp+130h] [rbp-40h]
  __int64 v134; // [rsp+138h] [rbp-38h]

  v4 = (__int64 *)a1;
  v109 = *(unsigned int **)(a1 + 72);
  v108 = (__int64)(v109 + 14);
  v119 = 0;
  v120 = 0;
  v121 = 0;
  sub_20C30B0(v109, a2, (__int64)&v119, (__int64)(v109 + 14));
  v5 = v120;
  if ( v119 == v120 )
  {
    v115 = 0;
    goto LABEL_31;
  }
  v123 = 0;
  v6 = v120 - v119;
  v124 = 0;
  v7 = v6;
  v125 = &v123;
  v126 = &v123;
  v127 = 0;
  if ( !(_DWORD)v6 )
  {
    v16 = v4[4];
    goto LABEL_43;
  }
  v114 = v4;
  v8 = 4;
  v116 = 4LL * (unsigned int)v6;
  v7 = *v119;
  v9 = *v119;
  v118 = *v119;
  while ( !sub_20C2FE0(v108, &v118) )
  {
    v10 = v119;
    if ( v116 == v8 )
      goto LABEL_20;
LABEL_5:
    v9 = v10[(unsigned __int64)v8 / 4];
    v118 = v9;
    if ( v7 )
    {
      v27 = (__int16 *)(*(_QWORD *)(v114[4] + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v114[4] + 8) + 24LL * v7 + 8));
      v28 = *v27;
      v29 = v27 + 1;
      v30 = v28 + v7;
      if ( !v28 )
        v29 = 0;
LABEL_39:
      v32 = v29;
      while ( v32 )
      {
        if ( v9 == v30 )
          goto LABEL_6;
        v31 = *v32;
        v29 = 0;
        ++v32;
        v30 += v31;
        if ( !v31 )
          goto LABEL_39;
      }
    }
    else
    {
LABEL_6:
      v7 = v9;
    }
    v8 += 4;
  }
  v11 = v124;
  v12 = &v123;
  if ( !v124 )
    goto LABEL_16;
  do
  {
    while ( 1 )
    {
      v13 = v11[2];
      v14 = v11[3];
      if ( *((_DWORD *)v11 + 8) >= v9 )
        break;
      v11 = (_QWORD *)v11[3];
      if ( !v14 )
        goto LABEL_14;
    }
    v12 = (int *)v11;
    v11 = (_QWORD *)v11[2];
  }
  while ( v13 );
LABEL_14:
  if ( v12 == &v123 || v12[8] > v9 )
  {
LABEL_16:
    v128.m128i_i64[0] = (__int64)&v118;
    v12 = (int *)sub_20C5230(&v122, (__int64)v12, (unsigned int **)&v128);
  }
  v112 = v12;
  sub_20C4220((__int64)&v128, v114, v118);
  v15 = v128.m128i_i64[0];
  if ( &v128 != (__m128i *)(v112 + 10) )
  {
    _libc_free(*((_QWORD *)v112 + 5));
    v15 = 0;
    *(__m128i *)(v112 + 10) = _mm_loadu_si128(&v128);
    v112[14] = v129;
  }
  _libc_free(v15);
  v10 = v119;
  if ( v116 != v8 )
    goto LABEL_5;
LABEL_20:
  v4 = v114;
  v16 = v114[4];
  v17 = v120 - v10;
  if ( (_DWORD)v17 )
  {
    v18 = (__int64)&v10[(unsigned int)(v17 - 1) + 1];
    do
    {
      v19 = *v10;
      if ( v7 != *v10 )
      {
        v20 = (__int16 *)(*(_QWORD *)(v16 + 56) + 2LL * *(unsigned int *)(*(_QWORD *)(v16 + 8) + 24LL * v19 + 8));
        v21 = *v20;
        v22 = v20 + 1;
        v23 = v21 + v19;
        if ( !v21 )
          v22 = 0;
LABEL_25:
        v24 = v22;
        if ( !v22 )
          goto LABEL_29;
        while ( v7 != v23 )
        {
          v25 = *v24;
          v22 = 0;
          ++v24;
          if ( !v25 )
            goto LABEL_25;
          v23 += v25;
          if ( !v24 )
            goto LABEL_29;
        }
      }
      ++v10;
    }
    while ( (unsigned int *)v18 != v10 );
  }
LABEL_43:
  v33 = sub_1F4ABE0(v16, v7, 1);
  v34 = v4[5];
  v99 = (unsigned __int64)v33;
  v35 = *(_QWORD *)v34 + 24LL * *(unsigned __int16 *)(*v33 + 24);
  if ( *(_DWORD *)(v34 + 8) == *(_DWORD *)v35 )
  {
    v113 = *(_DWORD *)(v35 + 4);
    if ( !v113 )
      goto LABEL_29;
  }
  else
  {
    sub_1ED7890(v34, (__int64 **)v33);
    v113 = *(_DWORD *)(v35 + 4);
    if ( !v113 )
      goto LABEL_29;
  }
  v97 = *(_QWORD *)(v35 + 16);
  v36 = (__int64)(a3 + 1);
  v128.m128i_i64[0] = v99;
  v128.m128i_i32[2] = v113;
  sub_20C4C90((__int64)a3, &v128);
  v37 = (_QWORD *)a3[2];
  v98 = a3 + 1;
  if ( !v37 )
  {
    v36 = (__int64)(a3 + 1);
LABEL_117:
    v83 = (_QWORD *)v36;
    v36 = sub_22077B0(48);
    *(_DWORD *)(v36 + 40) = 0;
    *(_QWORD *)(v36 + 32) = v99;
    v84 = sub_20C4D40(a3, v83, (unsigned __int64 *)(v36 + 32));
    if ( v85 )
    {
      v86 = v98 == v85 || v84 || v99 < v85[4];
      sub_220F040(v86, v36, v85, v98);
      ++a3[5];
    }
    else
    {
      v87 = v36;
      v36 = (__int64)v84;
      j_j___libc_free_0(v87, 48);
    }
    goto LABEL_52;
  }
  do
  {
    while ( 1 )
    {
      v38 = v37[2];
      v39 = v37[3];
      if ( v37[4] >= v99 )
        break;
      v37 = (_QWORD *)v37[3];
      if ( !v39 )
        goto LABEL_50;
    }
    v36 = (__int64)v37;
    v37 = (_QWORD *)v37[2];
  }
  while ( v38 );
LABEL_50:
  if ( (_QWORD *)v36 == v98 || *(_QWORD *)(v36 + 32) > v99 )
    goto LABEL_117;
LABEL_52:
  v40 = 0;
  v106 = v7;
  v41 = v109;
  if ( *(_DWORD *)(v36 + 40) != v113 )
    v40 = *(_DWORD *)(v36 + 40);
  v117 = *(_DWORD *)(v36 + 40);
  v111 = v40;
  while ( 2 )
  {
    v42 = (_QWORD *)v4[2];
    v43 = v117;
    if ( !v117 )
      v43 = v113;
    v44 = *(_QWORD *)(*v42 + 16LL);
    v45 = *(__int64 (**)())(*(_QWORD *)v44 + 112LL);
    if ( v45 == sub_1D00B10 )
      BUG();
    v46 = (unsigned int)(v43 - 1);
    v117 = v43 - 1;
    v47 = *(unsigned __int16 *)(v97 + 2 * v46);
    v115 = *(_BYTE *)(*(_QWORD *)(((__int64 (__fastcall *)(__int64, __int64, __int64, __int64))v45)(v44, v97, v46, v38)
                                + 232)
                    + 8LL * (unsigned __int16)v47
                    + 4);
    if ( !v115 )
      goto LABEL_55;
    v38 = v47;
    v107 = (unsigned __int16)v47;
    v105 = 1LL << v47;
    v104 = 8LL * ((unsigned __int16)v47 >> 6);
    if ( (*(_QWORD *)(v42[38] + v104) & (1LL << v47)) != 0 || (unsigned __int16)v47 == v106 )
      goto LABEL_55;
    sub_1ECD820((__int64)a4, a4[2]);
    v48 = v119;
    a4[2] = 0;
    a4[3] = a4 + 1;
    a4[4] = a4 + 1;
    v49 = v120;
    a4[5] = 0;
    v50 = v49 - v48;
    if ( !(_DWORD)v50 )
    {
LABEL_127:
      v88 = a3[2];
      if ( v88 )
      {
        v89 = a3 + 1;
        v90 = (_QWORD *)a3[2];
        while ( 1 )
        {
          if ( v90[4] < v99 )
          {
            v90 = (_QWORD *)v90[3];
          }
          else
          {
            v91 = (_QWORD *)v90[2];
            if ( v90[4] <= v99 )
            {
              v93 = (_QWORD *)v90[3];
              while ( v93 )
              {
                if ( v93[4] <= v99 )
                {
                  v93 = (_QWORD *)v93[3];
                }
                else
                {
                  v89 = v93;
                  v93 = (_QWORD *)v93[2];
                }
              }
              while ( v91 )
              {
                while ( 1 )
                {
                  v94 = v91[3];
                  if ( v91[4] >= v99 )
                    break;
                  v91 = (_QWORD *)v91[3];
                  if ( !v94 )
                    goto LABEL_149;
                }
                v90 = v91;
                v91 = (_QWORD *)v91[2];
              }
LABEL_149:
              if ( (_QWORD *)a3[3] != v90 || v89 != v98 )
              {
                for ( ; v89 != v90; --a3[5] )
                {
                  v95 = v90;
                  v90 = (_QWORD *)sub_220EF30(v90);
                  v96 = sub_220F330(v95, v98);
                  j_j___libc_free_0(v96, 48);
                }
                goto LABEL_138;
              }
LABEL_137:
              sub_20C4400((__int64)a3, v88);
              a3[2] = 0;
              a3[5] = 0;
              a3[3] = v98;
              a3[4] = v98;
              goto LABEL_138;
            }
            v89 = v90;
            v90 = (_QWORD *)v90[2];
          }
          if ( !v90 )
          {
            v92 = v98 == v89;
            goto LABEL_135;
          }
        }
      }
      v92 = v115;
      v89 = a3 + 1;
LABEL_135:
      if ( (_QWORD *)a3[3] == v89 && v92 )
        goto LABEL_137;
LABEL_138:
      v128.m128i_i64[0] = v99;
      v128.m128i_i32[2] = v117;
      sub_20C4C90((__int64)a3, &v128);
      goto LABEL_30;
    }
    v110 = 0;
    v100 = 4LL * (unsigned int)(v50 - 1);
    while ( 2 )
    {
      v118 = v48[v110 / 4];
      if ( v118 != v106 )
      {
        v51 = 0;
        v52 = 1;
        v53 = sub_38D7050(v4[4] + 8, v106);
        if ( v53 )
        {
          v53 = sub_38D6F10(v4[4] + 8, v107, v53);
          v52 = 1LL << v53;
          v51 = 8LL * (v53 >> 6);
        }
        v54 = v124;
        if ( v124 )
          goto LABEL_68;
LABEL_114:
        v55 = &v123;
        goto LABEL_74;
      }
      v54 = v124;
      v51 = v104;
      v52 = v105;
      v53 = v107;
      if ( !v124 )
        goto LABEL_114;
LABEL_68:
      v55 = &v123;
      do
      {
        while ( 1 )
        {
          v38 = v54[2];
          v56 = v54[3];
          if ( *((_DWORD *)v54 + 8) >= v118 )
            break;
          v54 = (_QWORD *)v54[3];
          if ( !v56 )
            goto LABEL_72;
        }
        v55 = (int *)v54;
        v54 = (_QWORD *)v54[2];
      }
      while ( v38 );
LABEL_72:
      if ( v55 == &v123 || v118 < v55[8] )
      {
LABEL_74:
        v128.m128i_i64[0] = (__int64)&v118;
        v55 = (int *)sub_20C5230(&v122, (__int64)v55, (unsigned int **)&v128);
      }
      if ( (*(_QWORD *)(*((_QWORD *)v55 + 5) + v51) & v52) == 0 )
        goto LABEL_55;
      v57 = v4[9];
      v58 = *(_QWORD *)(v57 + 104);
      if ( *(_DWORD *)(v58 + 4LL * v53) != -1 && *(_DWORD *)(*(_QWORD *)(v57 + 128) + 4LL * v53) == -1 )
        goto LABEL_55;
      v38 = *((_QWORD *)v41 + 13);
      if ( *(_DWORD *)(v38 + 4LL * v118) > *(_DWORD *)(*((_QWORD *)v41 + 16) + 4LL * v53) )
        goto LABEL_55;
      v59 = (_QWORD *)v4[4];
      v128.m128i_i32[0] = v53;
      if ( !v59 )
      {
        v128.m128i_i64[1] = 0;
        LOBYTE(v129) = 0;
        v130 = 0;
        v131 = 0;
        v132 = 0;
        v133 = 0;
        v134 = 0;
        BUG();
      }
      LOBYTE(v129) = 0;
      v128.m128i_i64[1] = (__int64)(v59 + 1);
      v130 = 0;
      v131 = 0;
      v133 = 0;
      v132 = 0;
      v134 = 0;
      v60 = *(_DWORD *)(v59[1] + 24LL * v53 + 16);
      v61 = v53 * (v60 & 0xF);
      v62 = (_WORD *)(v59[7] + 2LL * (v60 >> 4));
      v63 = v62 + 1;
      v130 = *v62 + v61;
      v131 = v62 + 1;
      while ( v63 )
      {
        v132 = *(_DWORD *)(v59[6] + 4LL * v130);
        v64 = v132;
        if ( (_WORD)v132 )
        {
          while ( 2 )
          {
            v65 = *(unsigned int *)(v59[1] + 24LL * v64 + 8);
            v66 = v59[7];
            v133 = v64;
            v67 = v66 + 2 * v65;
            v134 = v67;
            while ( v67 )
            {
              v38 = v133;
              v68 = v133;
              if ( v53 != v133 )
              {
                if ( *(_DWORD *)(v58 + 4LL * v133) == -1 || *(_DWORD *)(*(_QWORD *)(v57 + 128) + 4LL * v133) != -1 )
                {
                  do
                  {
                    v38 = *((_QWORD *)v41 + 13);
                    if ( *(_DWORD *)(v38 + 4LL * v118) > *(_DWORD *)(*((_QWORD *)v41 + 16) + 4 * v68) )
                      break;
                    sub_1E1D5E0((__int64)&v128);
                    if ( !v131 )
                      goto LABEL_91;
                    v82 = v4[9];
                    v68 = v133;
                  }
                  while ( *(_DWORD *)(*(_QWORD *)(v82 + 104) + 4LL * v133) == -1
                       || *(_DWORD *)(*(_QWORD *)(v82 + 128) + 4LL * v133) != -1 );
                }
                goto LABEL_55;
              }
              v67 += 2;
              v134 = v67;
              v69 = *(_WORD *)(v67 - 2);
              v133 += v69;
              if ( !v69 )
              {
                v134 = 0;
                break;
              }
            }
            v64 = HIWORD(v132);
            v132 = HIWORD(v132);
            if ( v64 )
              continue;
            break;
          }
        }
        v131 = ++v63;
        v70 = *(v63 - 1);
        v130 += v70;
        if ( !v70 )
        {
          v131 = 0;
          break;
        }
      }
LABEL_91:
      v71 = sub_20C3470(v108, &v118);
      v73 = v72;
      v74 = v71;
      if ( v72 == v71 )
      {
LABEL_97:
        v77 = sub_20C3470(v108, &v118);
        v79 = v78;
        for ( i = v77; v79 != i; i = sub_220EEE0(i) )
        {
          v81 = *(_QWORD *)(i + 40);
          if ( (*(_BYTE *)(v81 + 3) & 0x10) != 0
            && (*(_BYTE *)(v81 + 4) & 4) != 0
            && (unsigned int)sub_1E165A0(*(_QWORD *)(v81 + 16), v53, 0, v4[4]) != -1 )
          {
            goto LABEL_55;
          }
        }
        v128.m128i_i64[0] = __PAIR64__(v53, v118);
        sub_1E948B0(a4, (unsigned int *)&v128);
        if ( v100 == v110 )
          goto LABEL_127;
        v110 += 4LL;
        v48 = v119;
        continue;
      }
      break;
    }
    v102 = v41;
    while ( 1 )
    {
      v75 = *(_QWORD *)(*(_QWORD *)(v74 + 40) + 16LL);
      v76 = sub_1E16810(v75, v53, 0, 1, v4[4]);
      if ( v76 != -1 && (*(_BYTE *)(*(_QWORD *)(v75 + 32) + 40LL * v76 + 4) & 4) != 0 )
        break;
      v74 = sub_220EEE0(v74);
      if ( v74 == v73 )
      {
        v41 = v102;
        goto LABEL_97;
      }
    }
    v41 = v102;
LABEL_55:
    if ( v117 != v111 )
      continue;
    break;
  }
LABEL_29:
  v115 = 0;
LABEL_30:
  sub_20C1D80(v124);
  v5 = v119;
LABEL_31:
  if ( v5 )
    j_j___libc_free_0(v5, v121 - (_QWORD)v5);
  return v115;
}
