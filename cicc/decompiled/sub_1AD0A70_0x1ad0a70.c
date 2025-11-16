// Function: sub_1AD0A70
// Address: 0x1ad0a70
//
char __fastcall sub_1AD0A70(size_t **a1, size_t **a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // r11
  size_t **v5; // r14
  size_t **v6; // r11
  size_t *v7; // rcx
  size_t v8; // r15
  int v9; // r10d
  _DWORD *v10; // r9
  int v11; // r8d
  bool v12; // zf
  bool v13; // sf
  bool v14; // of
  size_t *v15; // rbx
  size_t v16; // r12
  int v17; // r13d
  bool v18; // zf
  bool v19; // sf
  bool v20; // of
  size_t *v21; // r8
  size_t *v22; // r15
  size_t *v23; // r12
  size_t v24; // r10
  size_t **v25; // r13
  size_t *v26; // r14
  size_t *v27; // rcx
  size_t **v28; // r9
  int v29; // r15d
  size_t v30; // rax
  int v31; // edx
  size_t **v32; // rbx
  const void *v33; // rdi
  size_t v34; // rax
  bool v35; // zf
  bool v36; // sf
  bool v37; // of
  int v38; // eax
  size_t v39; // r11
  size_t v40; // r8
  const void *v41; // rsi
  int v42; // eax
  size_t v43; // r8
  size_t v44; // rbx
  const void *v45; // rsi
  const void *v46; // rdi
  int v47; // eax
  int v48; // eax
  size_t v49; // rbx
  const void *v50; // rdi
  size_t v51; // r12
  const void *v52; // rsi
  int v53; // eax
  bool v54; // zf
  bool v55; // sf
  bool v56; // of
  __int64 v57; // r13
  __int64 v58; // r12
  __int64 i; // rbx
  char *v60; // r12
  size_t **v61; // rbx
  size_t *v62; // rcx
  __int64 v63; // r13
  int v64; // eax
  size_t v65; // r9
  const void *v66; // rsi
  size_t v67; // r8
  const void *v68; // rdi
  bool v69; // cc
  int v70; // eax
  size_t v71; // r10
  size_t v72; // r15
  const void *v73; // rsi
  const void *v74; // rdi
  bool v75; // zf
  bool v76; // sf
  bool v77; // of
  int v78; // eax
  size_t v79; // r12
  const void *v80; // rsi
  size_t v81; // r13
  const void *v82; // rdi
  int v83; // eax
  size_t v84; // r12
  size_t v85; // r13
  const void *v86; // rsi
  const void *v87; // rdi
  int v88; // eax
  int v89; // eax
  int v90; // eax
  int v91; // eax
  size_t **v93; // [rsp+8h] [rbp-88h]
  size_t **v94; // [rsp+10h] [rbp-80h]
  __int64 v95; // [rsp+18h] [rbp-78h]
  size_t *v96; // [rsp+20h] [rbp-70h]
  size_t *v97; // [rsp+20h] [rbp-70h]
  size_t **v98; // [rsp+28h] [rbp-68h]
  size_t **v99; // [rsp+28h] [rbp-68h]
  size_t **v100; // [rsp+28h] [rbp-68h]
  size_t **v101; // [rsp+28h] [rbp-68h]
  size_t **v102; // [rsp+28h] [rbp-68h]
  size_t **v103; // [rsp+28h] [rbp-68h]
  size_t **v104; // [rsp+28h] [rbp-68h]
  size_t **v105; // [rsp+28h] [rbp-68h]
  size_t v106; // [rsp+30h] [rbp-60h]
  size_t v107; // [rsp+30h] [rbp-60h]
  size_t *v108; // [rsp+30h] [rbp-60h]
  size_t *v109; // [rsp+30h] [rbp-60h]
  size_t *v110; // [rsp+30h] [rbp-60h]
  size_t *v111; // [rsp+30h] [rbp-60h]
  size_t *v112; // [rsp+30h] [rbp-60h]
  size_t *v113; // [rsp+30h] [rbp-60h]
  size_t *v114; // [rsp+30h] [rbp-60h]
  size_t *v115; // [rsp+30h] [rbp-60h]
  size_t v116; // [rsp+38h] [rbp-58h]
  size_t v117; // [rsp+38h] [rbp-58h]
  size_t **v118; // [rsp+38h] [rbp-58h]
  size_t **v119; // [rsp+38h] [rbp-58h]
  int v120; // [rsp+38h] [rbp-58h]
  size_t v121; // [rsp+38h] [rbp-58h]
  size_t v122; // [rsp+38h] [rbp-58h]
  int v123; // [rsp+38h] [rbp-58h]
  int v124; // [rsp+38h] [rbp-58h]
  size_t v125; // [rsp+38h] [rbp-58h]
  size_t v126; // [rsp+40h] [rbp-50h]
  size_t v127; // [rsp+40h] [rbp-50h]
  size_t v128; // [rsp+40h] [rbp-50h]
  size_t v129; // [rsp+40h] [rbp-50h]
  _DWORD *v130; // [rsp+40h] [rbp-50h]
  int v131; // [rsp+40h] [rbp-50h]
  int v132; // [rsp+40h] [rbp-50h]
  _DWORD *v133; // [rsp+40h] [rbp-50h]
  _DWORD *v134; // [rsp+40h] [rbp-50h]
  int v135; // [rsp+40h] [rbp-50h]
  void *s1; // [rsp+48h] [rbp-48h]
  void *s1a; // [rsp+48h] [rbp-48h]
  int s1b; // [rsp+48h] [rbp-48h]
  void *s1c; // [rsp+48h] [rbp-48h]
  _DWORD *s1d; // [rsp+48h] [rbp-48h]
  int s1e; // [rsp+48h] [rbp-48h]
  void *s1f; // [rsp+48h] [rbp-48h]
  void *s1g; // [rsp+48h] [rbp-48h]
  size_t **s1h; // [rsp+48h] [rbp-48h]
  size_t **s1i; // [rsp+48h] [rbp-48h]
  size_t **s1j; // [rsp+48h] [rbp-48h]
  size_t **s1k; // [rsp+48h] [rbp-48h]
  size_t **v148; // [rsp+50h] [rbp-40h]
  size_t **v149; // [rsp+50h] [rbp-40h]
  size_t *v150; // [rsp+50h] [rbp-40h]
  size_t *v151; // [rsp+50h] [rbp-40h]
  size_t *v152; // [rsp+58h] [rbp-38h]
  size_t **v153; // [rsp+58h] [rbp-38h]
  size_t *v154; // [rsp+58h] [rbp-38h]
  size_t *v155; // [rsp+58h] [rbp-38h]

  v3 = (char *)a2 - (char *)a1;
  v95 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return v3;
  v4 = (__int64)a1;
  if ( !a3 )
  {
    v153 = a2;
    goto LABEL_54;
  }
  v5 = a1;
  v6 = a2;
  v93 = a1 + 2;
  while ( 2 )
  {
    v7 = v5[1];
    --v95;
    v8 = v7[1];
    v9 = *(_DWORD *)(v8 + 80);
    v148 = &v5[(__int64)(v6 - v5 + ((unsigned __int64)((char *)v6 - (char *)v5) >> 63)) >> 1];
    v10 = (_DWORD *)(*v148)[1];
    v152 = *v148;
    v11 = v10[20];
    v14 = __OFSUB__(v9, v11);
    v12 = v9 == v11;
    v13 = v9 - v11 < 0;
    if ( v9 != v11
      || (v48 = v10[21],
          v14 = __OFSUB__(*(_DWORD *)(v8 + 84), v48),
          v12 = *(_DWORD *)(v8 + 84) == v48,
          v13 = *(_DWORD *)(v8 + 84) - v48 < 0,
          *(_DWORD *)(v8 + 84) != v48) )
    {
      v15 = *(v6 - 1);
      v16 = v15[1];
      v17 = *(_DWORD *)(v16 + 80);
      if ( v13 ^ v14 | v12 )
      {
LABEL_50:
        v56 = __OFSUB__(v9, v17);
        v54 = v9 == v17;
        v55 = v9 - v17 < 0;
        if ( v9 == v17
          && (v70 = *(_DWORD *)(v16 + 84),
              v56 = __OFSUB__(*(_DWORD *)(v8 + 84), v70),
              v54 = *(_DWORD *)(v8 + 84) == v70,
              v55 = *(_DWORD *)(v8 + 84) - v70 < 0,
              *(_DWORD *)(v8 + 84) == v70) )
        {
          v71 = *v15;
          v72 = *v7;
          v73 = v15 + 2;
          v74 = v7 + 2;
          if ( *v15 >= *v7 )
          {
            if ( v72 )
            {
              v102 = v6;
              v112 = v7;
              v122 = *v15;
              v132 = v11;
              s1d = v10;
              v89 = memcmp(v74, v73, *v7);
              v10 = s1d;
              v11 = v132;
              v71 = v122;
              v7 = v112;
              v6 = v102;
              if ( v89 )
              {
LABEL_104:
                if ( v89 >= 0 )
                  goto LABEL_70;
                v22 = *v5;
                goto LABEL_52;
              }
            }
            if ( v71 == v72 )
              goto LABEL_70;
          }
          else
          {
            if ( !v71 )
            {
LABEL_70:
              v22 = *v5;
              goto LABEL_71;
            }
            v104 = v6;
            v114 = v7;
            v124 = v11;
            v134 = v10;
            s1f = (void *)*v15;
            v89 = memcmp(v74, v73, *v15);
            v71 = (size_t)s1f;
            v10 = v134;
            v11 = v124;
            v7 = v114;
            v6 = v104;
            if ( v89 )
              goto LABEL_104;
          }
          v69 = v71 <= v72;
          v22 = *v5;
          if ( v69 )
            goto LABEL_71;
        }
        else
        {
          v22 = *v5;
          if ( v55 ^ v56 | v54 )
          {
LABEL_71:
            v77 = __OFSUB__(v11, v17);
            v75 = v11 == v17;
            v76 = v11 - v17 < 0;
            if ( v11 == v17
              && (v78 = *(_DWORD *)(v16 + 84),
                  v77 = __OFSUB__(v10[21], v78),
                  v75 = v10[21] == v78,
                  v76 = v10[21] - v78 < 0,
                  v10[21] == v78) )
            {
              v79 = *v15;
              v80 = v15 + 2;
              v81 = *v152;
              v82 = v152 + 2;
              if ( *v15 < *v152 )
              {
                if ( !v79 )
                  goto LABEL_79;
                s1k = v6;
                v91 = memcmp(v82, v80, *v15);
                v6 = s1k;
                if ( v91 )
                  goto LABEL_113;
              }
              else
              {
                if ( v81 )
                {
                  s1i = v6;
                  v91 = memcmp(v82, v80, *v152);
                  v6 = s1i;
                  if ( v91 )
                  {
LABEL_113:
                    if ( v91 < 0 )
                      goto LABEL_73;
LABEL_79:
                    *v5 = v152;
                    *v148 = v22;
                    v22 = v5[1];
                    v7 = *v5;
                    v23 = *(v6 - 1);
                    goto LABEL_9;
                  }
                }
                if ( v79 == v81 )
                  goto LABEL_79;
              }
              if ( v79 <= v81 )
                goto LABEL_79;
            }
            else if ( v76 ^ v77 | v75 )
            {
              goto LABEL_79;
            }
LABEL_73:
            *v5 = v15;
            v23 = v22;
            *(v6 - 1) = v22;
            v7 = *v5;
            v22 = v5[1];
            goto LABEL_9;
          }
        }
LABEL_52:
        *v5 = v7;
        v5[1] = v22;
        v23 = *(v6 - 1);
        goto LABEL_9;
      }
      goto LABEL_6;
    }
    v49 = *v7;
    v50 = v7 + 2;
    v51 = *v152;
    v52 = v152 + 2;
    if ( *v152 < *v7 )
    {
      if ( !v51 )
        goto LABEL_49;
      v103 = v6;
      v113 = v5[1];
      v123 = v10[20];
      v133 = (_DWORD *)(*v148)[1];
      s1e = *(_DWORD *)(v8 + 80);
      v53 = memcmp(v50, v52, *v152);
      v9 = s1e;
      v10 = v133;
      v11 = v123;
      v7 = v113;
      v6 = v103;
      if ( v53 )
        goto LABEL_48;
LABEL_87:
      v69 = v51 <= v49;
      v15 = *(v6 - 1);
      v16 = v15[1];
      v17 = *(_DWORD *)(v16 + 80);
      if ( v69 )
        goto LABEL_50;
      goto LABEL_6;
    }
    if ( !v49 )
      goto LABEL_86;
    v100 = v6;
    v110 = v5[1];
    v120 = v10[20];
    v130 = (_DWORD *)(*v148)[1];
    s1b = *(_DWORD *)(v8 + 80);
    v53 = memcmp(v50, v52, *v7);
    v9 = s1b;
    v10 = v130;
    v11 = v120;
    v7 = v110;
    v6 = v100;
    if ( !v53 )
    {
LABEL_86:
      if ( v51 == v49 )
        goto LABEL_49;
      goto LABEL_87;
    }
LABEL_48:
    if ( v53 >= 0 )
    {
LABEL_49:
      v15 = *(v6 - 1);
      v16 = v15[1];
      v17 = *(_DWORD *)(v16 + 80);
      goto LABEL_50;
    }
    v15 = *(v6 - 1);
    v16 = v15[1];
    v17 = *(_DWORD *)(v16 + 80);
LABEL_6:
    v20 = __OFSUB__(v11, v17);
    v18 = v11 == v17;
    v19 = v11 - v17 < 0;
    if ( v11 != v17
      || (v64 = *(_DWORD *)(v16 + 84),
          v20 = __OFSUB__(v10[21], v64),
          v18 = v10[21] == v64,
          v19 = v10[21] - v64 < 0,
          v10[21] != v64) )
    {
      v21 = *v5;
      if ( v19 ^ v20 | v18 )
      {
LABEL_64:
        v69 = v9 <= v17;
        if ( v9 == v17 && (v83 = *(_DWORD *)(v16 + 84), v69 = *(_DWORD *)(v8 + 84) <= v83, *(_DWORD *)(v8 + 84) == v83) )
        {
          v84 = *v15;
          v85 = *v7;
          v86 = v15 + 2;
          v87 = v7 + 2;
          if ( *v15 < *v7 )
          {
            if ( !v84 )
              goto LABEL_85;
            s1j = v6;
            v151 = v7;
            v155 = v21;
            v90 = memcmp(v87, v86, *v15);
            v21 = v155;
            v7 = v151;
            v6 = s1j;
            if ( v90 )
              goto LABEL_110;
          }
          else
          {
            if ( v85 )
            {
              s1h = v6;
              v150 = v7;
              v154 = v21;
              v90 = memcmp(v87, v86, *v7);
              v21 = v154;
              v7 = v150;
              v6 = s1h;
              if ( v90 )
              {
LABEL_110:
                if ( v90 < 0 )
                  goto LABEL_66;
LABEL_85:
                *v5 = v7;
                v22 = v21;
                v5[1] = v21;
                v23 = *(v6 - 1);
                goto LABEL_9;
              }
            }
            if ( v84 == v85 )
              goto LABEL_85;
          }
          if ( v84 <= v85 )
            goto LABEL_85;
        }
        else if ( v69 )
        {
          goto LABEL_85;
        }
LABEL_66:
        *v5 = v15;
        v23 = v21;
        *(v6 - 1) = v21;
        v22 = v5[1];
        v7 = *v5;
        goto LABEL_9;
      }
      goto LABEL_8;
    }
    v65 = *v15;
    v66 = v15 + 2;
    v67 = *v152;
    v68 = v152 + 2;
    if ( *v15 < *v152 )
    {
      if ( !v65 )
      {
LABEL_63:
        v21 = *v5;
        goto LABEL_64;
      }
      v105 = v6;
      v115 = v7;
      v125 = *v152;
      v135 = v9;
      s1g = (void *)*v15;
      v88 = memcmp(v68, v66, *v15);
      v65 = (size_t)s1g;
      v9 = v135;
      v67 = v125;
      v7 = v115;
      v6 = v105;
      if ( v88 )
        goto LABEL_107;
LABEL_92:
      v69 = v65 <= v67;
      v21 = *v5;
      if ( v69 )
        goto LABEL_64;
      goto LABEL_8;
    }
    if ( !v67 )
      goto LABEL_91;
    v101 = v6;
    v111 = v7;
    v121 = *v15;
    v131 = v9;
    s1c = (void *)*v152;
    v88 = memcmp(v68, v66, *v152);
    v67 = (size_t)s1c;
    v9 = v131;
    v65 = v121;
    v7 = v111;
    v6 = v101;
    if ( !v88 )
    {
LABEL_91:
      if ( v65 == v67 )
        goto LABEL_63;
      goto LABEL_92;
    }
LABEL_107:
    if ( v88 >= 0 )
      goto LABEL_63;
    v21 = *v5;
LABEL_8:
    *v5 = v152;
    *v148 = v21;
    v22 = v5[1];
    v7 = *v5;
    v23 = *(v6 - 1);
LABEL_9:
    v24 = v7[1];
    v149 = v5;
    v25 = v6;
    v26 = v7;
    v94 = v6;
    v27 = v22;
    v28 = v93;
    v29 = *(_DWORD *)(v24 + 80);
    while ( 1 )
    {
      v153 = v28 - 1;
      v30 = v27[1];
      if ( v29 != *(_DWORD *)(v30 + 80) )
      {
        if ( v29 >= *(_DWORD *)(v30 + 80) )
          goto LABEL_15;
        goto LABEL_11;
      }
      v31 = *(_DWORD *)(v24 + 84);
      if ( *(_DWORD *)(v30 + 84) != v31 )
      {
        if ( *(_DWORD *)(v30 + 84) <= v31 )
          goto LABEL_15;
        goto LABEL_11;
      }
      v43 = *v26;
      v44 = *v27;
      v45 = v26 + 2;
      v46 = v27 + 2;
      if ( *v26 >= *v27 )
      {
        if ( v44 )
        {
          v108 = v27;
          v118 = v28;
          v128 = v24;
          s1 = (void *)*v26;
          v47 = memcmp(v46, v45, *v27);
          v43 = (size_t)s1;
          v24 = v128;
          v28 = v118;
          v27 = v108;
          if ( v47 )
            break;
        }
        if ( v43 == v44 )
          goto LABEL_15;
        goto LABEL_35;
      }
      if ( !v43 )
        goto LABEL_15;
      v109 = v27;
      v119 = v28;
      v129 = v24;
      s1a = (void *)*v26;
      v47 = memcmp(v46, v45, *v26);
      v43 = (size_t)s1a;
      v24 = v129;
      v28 = v119;
      v27 = v109;
      if ( v47 )
        break;
LABEL_35:
      if ( v43 <= v44 )
        goto LABEL_15;
LABEL_11:
      v27 = *v28++;
    }
    if ( v47 < 0 )
      goto LABEL_11;
LABEL_15:
    v32 = v25 - 1;
    v33 = v26 + 2;
    while ( 2 )
    {
      v34 = v23[1];
      v25 = v32;
      v37 = __OFSUB__(v29, *(_DWORD *)(v34 + 80));
      v35 = v29 == *(_DWORD *)(v34 + 80);
      v36 = v29 - *(_DWORD *)(v34 + 80) < 0;
      if ( v29 != *(_DWORD *)(v34 + 80)
        || (v38 = *(_DWORD *)(v34 + 84),
            v37 = __OFSUB__(*(_DWORD *)(v24 + 84), v38),
            v35 = *(_DWORD *)(v24 + 84) == v38,
            v36 = *(_DWORD *)(v24 + 84) - v38 < 0,
            *(_DWORD *)(v24 + 84) != v38) )
      {
        --v32;
        if ( v36 ^ v37 | v35 )
          break;
        goto LABEL_17;
      }
      v39 = *v23;
      v40 = *v26;
      v41 = v23 + 2;
      if ( *v23 < *v26 )
      {
        if ( !v39 )
          break;
        v97 = v27;
        v99 = v28;
        v107 = v24;
        v117 = *v26;
        v127 = *v23;
        v42 = memcmp(v33, v41, *v23);
        v39 = v127;
        v40 = v117;
        v24 = v107;
        v28 = v99;
        v27 = v97;
        if ( v42 )
          goto LABEL_28;
LABEL_24:
        if ( v39 <= v40 )
          break;
        goto LABEL_25;
      }
      if ( !v40 )
        goto LABEL_23;
      v96 = v27;
      v98 = v28;
      v106 = v24;
      v116 = *v23;
      v126 = *v26;
      v42 = memcmp(v33, v41, *v26);
      v40 = v126;
      v39 = v116;
      v24 = v106;
      v28 = v98;
      v27 = v96;
      if ( !v42 )
      {
LABEL_23:
        if ( v39 == v40 )
          break;
        goto LABEL_24;
      }
LABEL_28:
      if ( v42 < 0 )
      {
LABEL_25:
        --v32;
LABEL_17:
        v23 = *v32;
        continue;
      }
      break;
    }
    if ( v25 > v153 )
    {
      *(v28 - 1) = v23;
      *v25 = v27;
      v23 = *(v25 - 1);
      v26 = *v149;
      v24 = (*v149)[1];
      v29 = *(_DWORD *)(v24 + 80);
      goto LABEL_11;
    }
    v5 = v149;
    sub_1AD0A70(v153, v94, v95);
    v3 = (char *)v153 - (char *)v149;
    if ( (char *)v153 - (char *)v149 > 128 )
    {
      if ( v95 )
      {
        v6 = v153;
        continue;
      }
      v4 = (__int64)v149;
LABEL_54:
      v57 = v4;
      v58 = v3 >> 3;
      for ( i = ((v3 >> 3) - 2) >> 1; ; --i )
      {
        sub_1ACFF60(v57, i, v58, *(size_t **)(v57 + 8 * i));
        if ( !i )
          break;
      }
      v60 = (char *)v57;
      v61 = v153 - 1;
      do
      {
        v62 = *v61;
        v63 = (char *)v61-- - v60;
        v61[1] = *(size_t **)v60;
        LOBYTE(v3) = sub_1ACFF60((__int64)v60, 0, v63 >> 3, v62);
      }
      while ( v63 > 8 );
    }
    return v3;
  }
}
