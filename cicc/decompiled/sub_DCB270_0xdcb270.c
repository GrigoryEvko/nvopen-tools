// Function: sub_DCB270
// Address: 0xdcb270
//
__int64 __fastcall sub_DCB270(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // r13
  __int64 v5; // rbx
  __int64 *v6; // rsi
  _QWORD *v7; // rax
  __int16 v9; // cx
  __int64 v10; // rdi
  int v11; // eax
  bool v12; // al
  __int64 v13; // rax
  int v14; // eax
  bool v15; // al
  void *v16; // r15
  __int64 v17; // rdx
  __int64 v18; // r14
  __int16 v19; // ax
  __int64 v20; // rsi
  __int64 v21; // rdi
  unsigned int v22; // edx
  unsigned __int64 v23; // rax
  unsigned int v24; // ecx
  unsigned __int64 v25; // rdx
  int v26; // r8d
  __int64 v27; // rdi
  __int64 v28; // rax
  _QWORD *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int16 v33; // ax
  __int64 *v34; // rax
  __int64 v35; // r8
  __int64 *v36; // r13
  __int64 *v37; // r14
  __int64 v38; // rsi
  _QWORD *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 *v44; // rsi
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r8
  __int64 v54; // rax
  int v55; // eax
  __int64 v56; // r8
  __int64 *v57; // r13
  __int64 *v58; // r14
  __int64 v59; // rsi
  _QWORD *v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  char **v65; // rsi
  _QWORD *v66; // rax
  int v67; // eax
  _QWORD *v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rcx
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rdx
  __int64 v74; // rcx
  __int64 v75; // r8
  __int64 v76; // r9
  _QWORD *v77; // rax
  __int64 v78; // rax
  __int64 v79; // r15
  __int64 *v80; // r13
  __int64 v81; // rax
  __int64 v82; // r14
  __int64 v83; // rcx
  __int64 v84; // r8
  __int64 v85; // r9
  _QWORD *v86; // rax
  __int64 *v87; // r13
  __int64 *v88; // r14
  __int64 v89; // rsi
  __int64 v90; // rax
  __int64 v91; // rdx
  __int64 v92; // rcx
  __int64 v93; // r8
  __int64 v94; // r9
  _QWORD *v95; // rax
  char *v96; // rdi
  __int64 v97; // rax
  __int64 v98; // r13
  __int64 v99; // rax
  __int64 v100; // r14
  __int64 *v101; // rax
  __int64 v102; // rcx
  __int64 v103; // r8
  __int64 v104; // r9
  __int64 v105; // rdx
  _QWORD *v106; // rax
  __int64 v107; // rdx
  __int64 v108; // rcx
  __int64 v109; // r8
  __int64 v110; // r9
  __int64 v111; // r11
  __int64 v112; // r9
  __int64 *v113; // rax
  __int64 v114; // [rsp+0h] [rbp-190h]
  __int64 v115; // [rsp+0h] [rbp-190h]
  __int64 v116; // [rsp+8h] [rbp-188h]
  _QWORD *v117; // [rsp+8h] [rbp-188h]
  __int64 v118; // [rsp+10h] [rbp-180h]
  _QWORD *v119; // [rsp+10h] [rbp-180h]
  __int64 v120; // [rsp+10h] [rbp-180h]
  _QWORD *v121; // [rsp+10h] [rbp-180h]
  __int64 v122; // [rsp+18h] [rbp-178h]
  _QWORD *v123; // [rsp+18h] [rbp-178h]
  __int64 v124; // [rsp+18h] [rbp-178h]
  __int64 *v125; // [rsp+18h] [rbp-178h]
  __int64 v126; // [rsp+20h] [rbp-170h]
  __int64 v127; // [rsp+20h] [rbp-170h]
  __int64 v128; // [rsp+20h] [rbp-170h]
  __int64 *v129; // [rsp+20h] [rbp-170h]
  _QWORD *v130; // [rsp+20h] [rbp-170h]
  __int64 *v131; // [rsp+20h] [rbp-170h]
  __int16 v132; // [rsp+28h] [rbp-168h]
  __int64 v133; // [rsp+28h] [rbp-168h]
  __int64 *v134; // [rsp+28h] [rbp-168h]
  int v135; // [rsp+28h] [rbp-168h]
  int v136; // [rsp+30h] [rbp-160h]
  __int64 v137; // [rsp+30h] [rbp-160h]
  __int64 *v138; // [rsp+30h] [rbp-160h]
  __int64 *v139; // [rsp+30h] [rbp-160h]
  __int64 *v140; // [rsp+30h] [rbp-160h]
  __int64 v141; // [rsp+30h] [rbp-160h]
  __int64 v142; // [rsp+30h] [rbp-160h]
  __int16 v143; // [rsp+38h] [rbp-158h]
  int v144; // [rsp+38h] [rbp-158h]
  char *v145; // [rsp+38h] [rbp-158h]
  unsigned int v146; // [rsp+38h] [rbp-158h]
  __int64 v147; // [rsp+38h] [rbp-158h]
  int v148; // [rsp+38h] [rbp-158h]
  _QWORD *v149; // [rsp+38h] [rbp-158h]
  __int64 v150; // [rsp+38h] [rbp-158h]
  __int64 v151; // [rsp+38h] [rbp-158h]
  __int64 *v153; // [rsp+58h] [rbp-138h] BYREF
  __int64 v154[2]; // [rsp+60h] [rbp-130h] BYREF
  char *v155; // [rsp+70h] [rbp-120h] BYREF
  __int64 v156; // [rsp+78h] [rbp-118h]
  _BYTE v157[32]; // [rsp+80h] [rbp-110h] BYREF
  char *v158; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v159; // [rsp+A8h] [rbp-E8h]
  _BYTE v160[32]; // [rsp+B0h] [rbp-E0h] BYREF
  int *v161; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v162; // [rsp+D8h] [rbp-B8h]
  int v163; // [rsp+E0h] [rbp-B0h] BYREF
  __int64 v164; // [rsp+E4h] [rbp-ACh]
  __int64 v165; // [rsp+ECh] [rbp-A4h]

  v3 = (__int64 *)(a1 + 1032);
  v5 = a2;
  v161 = &v163;
  v164 = a2;
  v6 = (__int64 *)&v161;
  v165 = a3;
  v162 = 0x2000000005LL;
  v163 = 7;
  v153 = 0;
  v7 = sub_C65B40(a1 + 1032, (__int64)&v161, (__int64 *)&v153, (__int64)off_49DEA80);
  if ( v7 )
  {
    v5 = (__int64)v7;
    goto LABEL_3;
  }
  v9 = *(_WORD *)(v5 + 24);
  if ( v9 )
  {
    if ( *(_WORD *)(a3 + 24) )
      goto LABEL_15;
  }
  else
  {
    v10 = *(_QWORD *)(v5 + 32);
    if ( *(_DWORD *)(v10 + 32) <= 0x40u )
    {
      v12 = *(_QWORD *)(v10 + 24) == 0;
    }
    else
    {
      v136 = *(_DWORD *)(v10 + 32);
      v143 = *(_WORD *)(v5 + 24);
      v11 = sub_C444A0(v10 + 24);
      v9 = v143;
      v12 = v136 == v11;
    }
    if ( v12 )
      goto LABEL_3;
    if ( *(_WORD *)(a3 + 24) )
      goto LABEL_23;
  }
  v13 = *(_QWORD *)(a3 + 32);
  if ( *(_DWORD *)(v13 + 32) <= 0x40u )
  {
    if ( *(_QWORD *)(v13 + 24) == 1 )
      goto LABEL_3;
    v15 = *(_QWORD *)(v13 + 24) == 0;
  }
  else
  {
    v144 = *(_DWORD *)(v13 + 32);
    v132 = v9;
    v137 = v13 + 24;
    if ( (unsigned int)sub_C444A0(v13 + 24) == v144 - 1 )
      goto LABEL_3;
    v14 = sub_C444A0(v137);
    v9 = v132;
    v15 = v144 == v14;
  }
  if ( v15 )
  {
LABEL_15:
    if ( v9 == 5 )
    {
      v145 = (char *)v5;
      if ( *(_QWORD *)(v5 + 40) != 2
        || (v46 = **(_QWORD **)(v5 + 32), v141 = *(_QWORD *)(v5 + 32), *(_WORD *)(v46 + 24))
        || (v47 = *(_QWORD *)(v46 + 32),
            v134 = (__int64 *)(v47 + 24),
            v48 = (unsigned int)(*(_DWORD *)(v47 + 32) - 1),
            !sub_986C60((__int64 *)(v47 + 24), v48))
        || (unsigned __int8)sub_986B30(v134, v48, v49, v50, (unsigned int)v134) )
      {
LABEL_17:
        v6 = (__int64 *)&v161;
        v153 = 0;
        v5 = (__int64)sub_C65B40((__int64)v3, (__int64)&v161, (__int64 *)&v153, (__int64)off_49DEA80);
        if ( !v5 )
        {
          v16 = sub_C65D30((__int64)&v161, (unsigned __int64 *)(a1 + 1064));
          v18 = v17;
          v5 = sub_A777F0(0x30u, (__int64 *)(a1 + 1064));
          if ( v5 )
          {
            v159 = a3;
            v158 = v145;
            v19 = sub_D95470((__int64 *)&v158, 2);
            *(_QWORD *)(v5 + 8) = v16;
            *(_WORD *)(v5 + 26) = v19;
            *(_WORD *)(v5 + 28) = 0;
            *(_QWORD *)v5 = 0;
            *(_QWORD *)(v5 + 16) = v18;
            *(_WORD *)(v5 + 24) = 7;
            *(_QWORD *)(v5 + 32) = v145;
            *(_QWORD *)(v5 + 40) = a3;
          }
          sub_C657C0(v3, (__int64 *)v5, v153, (__int64)off_49DEA80);
          v6 = (__int64 *)v5;
          v158 = v145;
          v159 = a3;
          sub_DAEE00(a1, v5, (__int64 *)&v158, 2);
        }
        goto LABEL_3;
      }
      v51 = *(_QWORD *)(v141 + 8);
      if ( *(_WORD *)(v51 + 24) == 10 && *(_QWORD *)(v51 + 40) == 2 && !*(_WORD *)(**(_QWORD **)(v51 + 32) + 24LL) )
      {
        v127 = *(_QWORD *)(v141 + 8);
        sub_9865C0((__int64)&v155, (__int64)v134);
        sub_AADAA0((__int64)&v158, (__int64)&v155, v52, (__int64)&v158, v53);
        v54 = **(_QWORD **)(v127 + 32);
        v128 = *(_QWORD *)(v127 + 32);
        if ( !sub_AAD8B0(*(_QWORD *)(v54 + 32) + 24LL, &v158) )
        {
          sub_969240((__int64 *)&v158);
          sub_969240((__int64 *)&v155);
          goto LABEL_17;
        }
        v151 = *(_QWORD *)(v128 + 8);
        sub_969240((__int64 *)&v158);
        sub_969240((__int64 *)&v155);
        if ( a3 == v151 )
        {
          v6 = (__int64 *)sub_D95540(v5);
          v5 = (__int64)sub_DA2C50(a1, (__int64)v6, 0, 0);
          goto LABEL_3;
        }
      }
    }
LABEL_23:
    v145 = (char *)v5;
    goto LABEL_17;
  }
  v20 = sub_D95540(v5);
  v21 = *(_QWORD *)(a3 + 32);
  v22 = *(_DWORD *)(v21 + 32);
  if ( v22 > 0x40 )
  {
    v22 = sub_C444A0(v21 + 24);
  }
  else
  {
    v23 = *(_QWORD *)(v21 + 24);
    v24 = v22 - 64;
    if ( v23 )
    {
      _BitScanReverse64(&v25, v23);
      v22 = v24 + (v25 ^ 0x3F);
    }
  }
  v26 = sub_D97050(a1, v20) - v22;
  v27 = *(_QWORD *)(a3 + 32);
  if ( *(_DWORD *)(v27 + 32) > 0x40u )
  {
    v135 = v26 - 1;
    v148 = v26;
    v55 = sub_C44630(v27 + 24);
    v26 = v148;
    if ( v55 == 1 )
      v26 = v135;
  }
  else
  {
    v28 = *(_QWORD *)(v27 + 24);
    if ( v28 && (v28 & (v28 - 1)) == 0 )
      --v26;
  }
  v146 = v26 + sub_D97050(a1, v20);
  v29 = (_QWORD *)sub_B2BE50(*(_QWORD *)a1);
  v147 = sub_BCCE00(v29, v146);
  v33 = *(_WORD *)(v5 + 24);
  if ( v33 != 8 )
    goto LABEL_40;
  v126 = sub_D33D80((_QWORD *)v5, a1, v30, v31, v32);
  if ( *(_WORD *)(v126 + 24) )
    goto LABEL_82;
  v133 = *(_QWORD *)(v126 + 32) + 24LL;
  v122 = *(_QWORD *)(a3 + 32) + 24LL;
  sub_C4B490((__int64)&v158, v133, v122);
  if ( !sub_9867B0((__int64)&v158)
    || (v121 = sub_DC2B70(a1, v5, v147, 0),
        v115 = *(_QWORD *)(v5 + 48),
        v117 = sub_DC2B70(a1, v126, v147, 0),
        v86 = sub_DC2B70(a1, **(_QWORD **)(v5 + 32), v147, 0),
        v121 != sub_DC1960(a1, (__int64)v86, (__int64)v117, v115, 0)) )
  {
    sub_969240((__int64 *)&v158);
    v34 = *(__int64 **)(v5 + 32);
    if ( !*(_WORD *)(*v34 + 24) )
    {
      v118 = *v34;
      sub_C4B490((__int64)&v158, v122, v133);
      if ( !sub_9867B0((__int64)&v158)
        || (v114 = v118,
            v123 = sub_DC2B70(a1, v5, v147, 0),
            v116 = *(_QWORD *)(v5 + 48),
            v119 = sub_DC2B70(a1, v126, v147, 0),
            v66 = sub_DC2B70(a1, **(_QWORD **)(v5 + 32), v147, 0),
            v123 != sub_DC1960(a1, (__int64)v66, (__int64)v119, v116, 0)) )
      {
        sub_969240((__int64 *)&v158);
        v33 = *(_WORD *)(v5 + 24);
LABEL_40:
        if ( v33 == 6 )
        {
          v155 = v157;
          v156 = 0x400000000LL;
          v35 = *(_QWORD *)(v5 + 32);
          if ( v35 + 8LL * *(_QWORD *)(v5 + 40) != v35 )
          {
            v138 = v3;
            v36 = (__int64 *)(v35 + 8LL * *(_QWORD *)(v5 + 40));
            v37 = *(__int64 **)(v5 + 32);
            do
            {
              v38 = *v37++;
              v39 = sub_DC2B70(a1, v38, v147, 0);
              sub_D9B3A0((__int64)&v155, (__int64)v39, v40, v41, v42, v43);
            }
            while ( v36 != v37 );
            v3 = v138;
          }
          v44 = (__int64 *)&v155;
          v139 = sub_DC2B70(a1, v5, v147, 0);
          if ( v139 == sub_DC8BD0((__int64 *)a1, (__int64)&v155, 0, 0) )
          {
            v78 = *(_QWORD *)(v5 + 40);
            if ( (_DWORD)v78 )
            {
              v125 = v3;
              v142 = 8LL * (unsigned int)v78;
              v79 = 0;
              while ( 1 )
              {
                v80 = *(__int64 **)(*(_QWORD *)(v5 + 32) + v79);
                v44 = v80;
                v81 = sub_DCB270(a1, v80, a3);
                v82 = v81;
                if ( *(_WORD *)(v81 + 24) != 7 )
                {
                  v44 = (__int64 *)v81;
                  if ( v80 == sub_DCA690((__int64 *)a1, v81, a3, 0, 0) )
                    break;
                }
                v79 += 8;
                if ( v142 == v79 )
                {
                  v3 = v125;
                  goto LABEL_46;
                }
              }
              sub_D9C9D0(&v158, *(const void **)(v5 + 32), *(_QWORD *)(v5 + 40), v83, v84, v85);
              sub_D91D30((__int64)&v155, &v158, v107, v108, v109, v110);
              v111 = v79;
              v112 = v82;
              if ( v158 != v160 )
              {
                _libc_free(v158, &v158);
                v111 = v79;
                v112 = v82;
              }
              v6 = (__int64 *)&v155;
              *(_QWORD *)&v155[v111] = v112;
              v113 = sub_DC8BD0((__int64 *)a1, (__int64)&v155, 0, 0);
              v96 = v155;
              v5 = (__int64)v113;
              if ( v155 != v157 )
                goto LABEL_94;
              goto LABEL_3;
            }
          }
LABEL_46:
          if ( v155 != v157 )
            _libc_free(v155, v44);
          v33 = *(_WORD *)(v5 + 24);
        }
        if ( v33 != 7 )
        {
          if ( v33 == 5 )
          {
            v158 = v160;
            v159 = 0x400000000LL;
            v56 = *(_QWORD *)(v5 + 32);
            if ( v56 + 8LL * *(_QWORD *)(v5 + 40) != v56 )
            {
              v129 = v3;
              v57 = (__int64 *)(v56 + 8LL * *(_QWORD *)(v5 + 40));
              v58 = *(__int64 **)(v5 + 32);
              do
              {
                v59 = *v58++;
                v60 = sub_DC2B70(a1, v59, v147, 0);
                sub_D9B3A0((__int64)&v158, (__int64)v60, v61, v62, v63, v64);
              }
              while ( v57 != v58 );
              v3 = v129;
            }
            v65 = &v158;
            v149 = sub_DC2B70(a1, v5, v147, 0);
            if ( v149 == sub_DC7EB0((__int64 *)a1, (__int64)&v158, 0, 0) )
            {
              LODWORD(v159) = 0;
              v97 = *(_QWORD *)(v5 + 40);
              v150 = 8LL * (unsigned int)v97;
              if ( (_DWORD)v97 )
              {
                v131 = v3;
                v98 = 0;
                do
                {
                  v65 = *(char ***)(*(_QWORD *)(v5 + 32) + v98);
                  v99 = sub_DCB270(a1, v65, a3);
                  v100 = v99;
                  if ( *(_WORD *)(v99 + 24) == 7 )
                    break;
                  v65 = (char **)v99;
                  v101 = sub_DCA690((__int64 *)a1, v99, a3, 0, 0);
                  v105 = *(_QWORD *)(v5 + 32);
                  if ( *(__int64 **)(v105 + v98) != v101 )
                    break;
                  v65 = (char **)v100;
                  v98 += 8;
                  sub_D9B3A0((__int64)&v158, v100, v105, v102, v103, v104);
                }
                while ( v98 != v150 );
                v3 = v131;
              }
              if ( *(_QWORD *)(v5 + 40) == (unsigned int)v159 )
              {
                v6 = (__int64 *)&v158;
                v106 = sub_DC7EB0((__int64 *)a1, (__int64)&v158, 0, 0);
                v96 = v158;
                v5 = (__int64)v106;
                if ( v158 == v160 )
                  goto LABEL_3;
LABEL_94:
                _libc_free(v96, v6);
                goto LABEL_3;
              }
            }
            if ( v158 != v160 )
              _libc_free(v158, v65);
            v9 = *(_WORD *)(v5 + 24);
            if ( v9 )
              goto LABEL_15;
          }
          else if ( v33 )
          {
            goto LABEL_23;
          }
          sub_C4A1D0((__int64)&v158, *(_QWORD *)(v5 + 32) + 24LL, *(_QWORD *)(a3 + 32) + 24LL);
          v6 = (__int64 *)&v158;
          v5 = (__int64)sub_DA26C0((__int64 *)a1, (__int64)&v158);
          sub_969240((__int64 *)&v158);
          goto LABEL_3;
        }
        v45 = *(_QWORD *)(v5 + 40);
        if ( *(_WORD *)(v45 + 24) )
          goto LABEL_23;
        LOBYTE(v155) = 0;
        v140 = (__int64 *)&v158;
        sub_C49BE0((__int64)&v158, *(_QWORD *)(v45 + 32) + 24LL, *(_QWORD *)(a3 + 32) + 24LL, (bool *)&v155);
        if ( (_BYTE)v155 )
        {
          v6 = *(__int64 **)(*(_QWORD *)(a3 + 32) + 8LL);
          v5 = (__int64)sub_DA2C50(a1, (__int64)v6, 0, 0);
        }
        else
        {
          v77 = sub_DA26C0((__int64 *)a1, (__int64)&v158);
          v6 = *(__int64 **)(v5 + 32);
          v5 = sub_DCB270(a1, v6, v77);
        }
LABEL_53:
        sub_969240(v140);
        goto LABEL_3;
      }
      sub_969240((__int64 *)&v158);
      v124 = *(_QWORD *)(v114 + 32) + 24LL;
      sub_C4B490((__int64)v154, v124, v133);
      if ( !sub_D94970((__int64)v154, 0) )
      {
        v120 = *(_QWORD *)(v5 + 48);
        sub_9865C0((__int64)&v155, v124);
        sub_C46B40((__int64)&v155, v154);
        v67 = v156;
        LODWORD(v156) = 0;
        LODWORD(v159) = v67;
        v158 = v155;
        v68 = sub_DA26C0((__int64 *)a1, (__int64)&v158);
        v130 = sub_DC1960(a1, (__int64)v68, v126, v120, 1u);
        sub_969240((__int64 *)&v158);
        sub_969240((__int64 *)&v155);
        if ( (_QWORD *)v5 != v130 )
        {
          v140 = v154;
          LODWORD(v162) = 0;
          sub_9C8C60((__int64)&v161, 7);
          sub_D953B0((__int64)&v161, (__int64)v130, v69, v70, v71, v72);
          sub_D953B0((__int64)&v161, a3, v73, v74, v75, v76);
          v6 = (__int64 *)&v161;
          v153 = 0;
          v5 = (__int64)sub_C65B40((__int64)v3, (__int64)&v161, (__int64 *)&v153, (__int64)off_49DEA80);
          if ( v5 )
            goto LABEL_53;
          v5 = (__int64)v130;
        }
      }
      sub_969240(v154);
    }
LABEL_82:
    v33 = *(_WORD *)(v5 + 24);
    goto LABEL_40;
  }
  sub_969240((__int64 *)&v158);
  v87 = *(__int64 **)(v5 + 32);
  v158 = v160;
  v159 = 0x400000000LL;
  v88 = &v87[*(_QWORD *)(v5 + 40)];
  while ( v88 != v87 )
  {
    v89 = *v87++;
    v90 = sub_DCB270(a1, v89, a3);
    sub_D9B3A0((__int64)&v158, v90, v91, v92, v93, v94);
  }
  v6 = (__int64 *)&v158;
  v95 = sub_DBFF60(a1, (unsigned int *)&v158, *(_QWORD *)(v5 + 48), 1u);
  v96 = v158;
  v5 = (__int64)v95;
  if ( v158 != v160 )
    goto LABEL_94;
LABEL_3:
  if ( v161 != &v163 )
    _libc_free(v161, v6);
  return v5;
}
