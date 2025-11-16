// Function: sub_26D3D20
// Address: 0x26d3d20
//
void __fastcall sub_26D3D20(__int64 a1, char **a2)
{
  unsigned __int64 *v2; // rbx
  char *v3; // r15
  char *v4; // r13
  __int64 v5; // rax
  char *v6; // rax
  char *v7; // r13
  __int64 v9; // r12
  __int64 v10; // r13
  unsigned __int64 v11; // rdi
  _QWORD *v12; // rax
  _QWORD *v13; // rcx
  unsigned __int64 v14; // rsi
  __int64 v15; // r8
  __int64 *v16; // r13
  unsigned __int64 *v17; // r14
  _QWORD *v18; // rax
  _QWORD ********v19; // r12
  _QWORD *v20; // rdx
  _QWORD *********v21; // rcx
  _QWORD ********v22; // rsi
  _QWORD *******v23; // r9
  _QWORD ******v24; // r8
  _QWORD *****v25; // r10
  _QWORD ****v26; // rdi
  _QWORD ***v27; // rax
  _QWORD *v28; // r10
  _QWORD *v29; // rax
  _QWORD ********v30; // rdx
  _QWORD *v31; // rcx
  _QWORD *********v32; // rsi
  _QWORD ********v33; // r8
  _QWORD *******v34; // r10
  _QWORD ******v35; // r9
  _QWORD *****v36; // r11
  _QWORD ****v37; // rdi
  _QWORD ***v38; // rax
  _QWORD *v39; // r11
  unsigned int v40; // eax
  _QWORD *v41; // r8
  __int64 v42; // r12
  _QWORD *v43; // rax
  _QWORD *v44; // rsi
  __int64 *v45; // r12
  __int64 v46; // rax
  _QWORD *v47; // r13
  __int64 v48; // rdx
  __int64 v49; // r8
  unsigned int v50; // eax
  __int64 *v51; // rcx
  __int64 v52; // rdi
  __int64 v53; // rax
  unsigned int v54; // esi
  __int64 v55; // r12
  __int64 v56; // r14
  __int64 *v57; // r10
  int v58; // eax
  __int64 *v59; // rbx
  unsigned __int64 v60; // rcx
  unsigned __int64 v61; // rdx
  __int64 *v62; // rax
  bool v63; // r8
  __int64 v64; // rax
  _QWORD *v65; // rax
  unsigned __int64 v66; // rsi
  char v67; // al
  unsigned __int64 v68; // rdx
  _QWORD *v69; // r9
  unsigned __int64 v70; // r8
  _QWORD *v71; // r10
  _QWORD **v72; // rax
  _QWORD *v73; // rdx
  __int64 *i; // r13
  unsigned __int64 *v75; // r12
  __int64 v76; // rax
  __int64 v77; // r15
  _QWORD *v78; // rax
  _QWORD *v79; // rax
  _BYTE *v80; // rsi
  __int64 v81; // r12
  __int64 v82; // r13
  __int64 v83; // r12
  unsigned __int64 v84; // r14
  __int64 v85; // rax
  int v86; // edx
  __int64 v87; // rdi
  int v88; // ecx
  unsigned int v89; // edx
  __int64 *v90; // rsi
  __int64 v91; // r8
  _QWORD *v92; // rax
  _QWORD *v93; // rcx
  unsigned __int64 v94; // rsi
  __int64 v95; // r8
  __int64 *v96; // rax
  __int64 *v97; // rdx
  __int64 *j; // rax
  __int64 v99; // rcx
  __int64 v100; // rsi
  unsigned __int64 v101; // rdi
  unsigned __int64 *v102; // rbx
  unsigned __int64 v103; // r12
  unsigned __int64 v104; // rdi
  _QWORD *v105; // rbx
  unsigned __int64 v106; // rdi
  _QWORD *v107; // rax
  int v108; // esi
  int v109; // r9d
  __int64 v110; // rdx
  int v111; // r11d
  int v112; // eax
  size_t v113; // r12
  void *v114; // rax
  _QWORD *v115; // rax
  _QWORD *v116; // rsi
  unsigned __int64 v117; // rdi
  _QWORD *v118; // rcx
  unsigned __int64 v119; // rdx
  _QWORD **v120; // rax
  _QWORD *v121; // rdx
  unsigned __int64 v122; // [rsp+0h] [rbp-140h]
  _QWORD *v123; // [rsp+8h] [rbp-138h]
  _QWORD *v124; // [rsp+8h] [rbp-138h]
  __int64 v125; // [rsp+20h] [rbp-120h]
  char *v126; // [rsp+28h] [rbp-118h]
  char *v127; // [rsp+30h] [rbp-110h]
  _QWORD *v128; // [rsp+30h] [rbp-110h]
  unsigned __int64 v129; // [rsp+30h] [rbp-110h]
  _QWORD *v130; // [rsp+30h] [rbp-110h]
  char v131; // [rsp+38h] [rbp-108h]
  __int64 v132; // [rsp+40h] [rbp-100h] BYREF
  _QWORD v133[2]; // [rsp+48h] [rbp-F8h] BYREF
  __int64 v134; // [rsp+58h] [rbp-E8h] BYREF
  __int64 *v135; // [rsp+60h] [rbp-E0h]
  __int64 *v136; // [rsp+68h] [rbp-D8h]
  __int64 *v137; // [rsp+70h] [rbp-D0h]
  __int64 v138; // [rsp+78h] [rbp-C8h]
  void *s; // [rsp+80h] [rbp-C0h]
  unsigned __int64 v140; // [rsp+88h] [rbp-B8h]
  _QWORD *v141; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v142; // [rsp+98h] [rbp-A8h]
  int v143; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v144; // [rsp+A8h] [rbp-98h]
  _QWORD v145[2]; // [rsp+B0h] [rbp-90h] BYREF
  __int64 v146; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v147; // [rsp+C8h] [rbp-78h]
  unsigned __int64 v148; // [rsp+D0h] [rbp-70h]
  __int64 v149; // [rsp+D8h] [rbp-68h]
  __int64 v150; // [rsp+E0h] [rbp-60h]
  unsigned __int64 *v151; // [rsp+E8h] [rbp-58h]
  _QWORD *v152; // [rsp+F0h] [rbp-50h]
  __int64 v153; // [rsp+F8h] [rbp-48h]
  __int64 v154; // [rsp+100h] [rbp-40h]
  __int64 *v155; // [rsp+108h] [rbp-38h]

  *(_QWORD *)a1 = a1 + 48;
  *(_QWORD *)(a1 + 8) = 1;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_DWORD *)(a1 + 32) = 1065353216;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  v125 = a1 + 56;
  if ( (unsigned __int64)(a2[1] - *a2) <= 8 )
  {
    sub_26BA800(v125, a2);
    return;
  }
  v2 = (unsigned __int64 *)a1;
  sub_26C37A0(a1);
  v3 = *a2;
  v4 = a2[1];
  if ( *a2 == v4 )
  {
    LODWORD(v134) = 0;
    v135 = 0;
    v136 = &v134;
    v137 = &v134;
    v138 = 0;
LABEL_171:
    v140 = 1;
    s = v145;
    v141 = 0;
    v142 = 0;
    v143 = 1065353216;
    v144 = 0;
    v145[0] = 0;
    goto LABEL_57;
  }
  do
  {
    v5 = *(_QWORD *)v3;
    v3 += 8;
    v146 = v5;
    sub_26C6DA0((unsigned __int64 *)a1, (unsigned __int64 *)&v146);
  }
  while ( v4 != v3 );
  v6 = a2[1];
  v7 = *a2;
  LODWORD(v134) = 0;
  v135 = 0;
  v126 = v6;
  v136 = &v134;
  v137 = &v134;
  v138 = 0;
  if ( v7 == v6 )
    goto LABEL_171;
  v127 = v7;
  do
  {
    v9 = *(_QWORD *)v127 + 24LL;
    v10 = *(_QWORD *)(*(_QWORD *)v127 + 40LL);
    if ( v9 == v10 )
      goto LABEL_17;
    do
    {
      while ( 1 )
      {
        v11 = *(_QWORD *)(a1 + 8);
        v12 = *(_QWORD **)(*(_QWORD *)a1 + 8 * (*(_QWORD *)(v10 + 40) % v11));
        if ( v12 )
        {
          v13 = (_QWORD *)*v12;
          if ( *v12 )
            break;
        }
LABEL_16:
        v10 = sub_220EF30(v10);
        if ( v9 == v10 )
          goto LABEL_17;
      }
      v14 = v13[1];
      v15 = 0;
      while ( *(_QWORD *)(v10 + 40) == v14 )
      {
        v13 = (_QWORD *)*v13;
        ++v15;
        if ( !v13 )
          goto LABEL_15;
LABEL_12:
        v14 = v13[1];
        if ( *(_QWORD *)(v10 + 40) % v11 != v14 % v11 )
          goto LABEL_15;
      }
      if ( v15 )
        goto LABEL_65;
      v13 = (_QWORD *)*v13;
      if ( v13 )
        goto LABEL_12;
LABEL_15:
      if ( !v15 )
        goto LABEL_16;
LABEL_65:
      v59 = v135;
      if ( v135 )
      {
        v60 = *(_QWORD *)(v10 + 48);
        while ( 1 )
        {
          v61 = *(_QWORD *)(v59[4] + 16);
          v62 = (__int64 *)v59[3];
          if ( v60 > v61 )
            v62 = (__int64 *)v59[2];
          if ( !v62 )
            break;
          v59 = v62;
        }
        v63 = 1;
        if ( v59 != &v134 )
          v63 = v60 > v61;
      }
      else
      {
        v63 = 1;
        v59 = &v134;
      }
      v131 = v63;
      v64 = sub_22077B0(0x28u);
      *(_QWORD *)(v64 + 32) = v10 + 32;
      sub_220F040(v131, v64, v59, &v134);
      ++v138;
      v10 = sub_220EF30(v10);
    }
    while ( v9 != v10 );
LABEL_17:
    v127 += 8;
  }
  while ( v126 != v127 );
  v2 = (unsigned __int64 *)a1;
  v16 = v136;
  s = v145;
  v140 = 1;
  v141 = 0;
  v142 = 0;
  v143 = 1065353216;
  v144 = 0;
  v145[0] = 0;
  if ( v136 != &v134 )
  {
    while ( 1 )
    {
      v17 = (unsigned __int64 *)v16[4];
      v18 = (_QWORD *)sub_26C6DA0((unsigned __int64 *)a1, v17);
      v19 = (_QWORD ********)*v18;
      v20 = v18;
      if ( v18 != (_QWORD *)*v18 )
      {
        v21 = (_QWORD *********)*v19;
        if ( v19 != *v19 )
        {
          v22 = *v21;
          if ( v21 != *v21 )
          {
            v23 = *v22;
            if ( v22 != *v22 )
            {
              v24 = *v23;
              if ( v23 != *v23 )
              {
                v25 = *v24;
                if ( v24 != *v24 )
                {
                  v26 = *v25;
                  if ( v25 != *v25 )
                  {
                    v27 = sub_26BB420(v26);
                    *v28 = v27;
                    v26 = (_QWORD ****)v27;
                  }
                  *v24 = (_QWORD *****)v26;
                  v24 = (_QWORD ******)v26;
                }
                *v23 = v24;
              }
              *v22 = (_QWORD *******)v24;
              v22 = (_QWORD ********)v24;
            }
            *v21 = v22;
          }
          *v19 = v22;
          v19 = v22;
        }
        *v20 = v19;
      }
      v29 = (_QWORD *)sub_26C6DA0((unsigned __int64 *)a1, v17 + 1);
      v30 = (_QWORD ********)*v29;
      v31 = v29;
      if ( v29 != (_QWORD *)*v29 )
      {
        v32 = (_QWORD *********)*v30;
        if ( v30 != *v30 )
        {
          v33 = *v32;
          if ( v32 != *v32 )
          {
            v34 = *v33;
            if ( v33 != *v33 )
            {
              v35 = *v34;
              if ( v34 != *v34 )
              {
                v36 = *v35;
                if ( v35 != *v35 )
                {
                  v37 = *v36;
                  if ( v36 != *v36 )
                  {
                    v38 = sub_26BB420(v37);
                    *v39 = v38;
                    v37 = (_QWORD ****)v38;
                  }
                  *v35 = (_QWORD *****)v37;
                  v35 = (_QWORD ******)v37;
                }
                *v34 = v35;
              }
              *v33 = (_QWORD *******)v35;
              v33 = (_QWORD ********)v35;
            }
            *v32 = v33;
          }
          *v30 = v33;
          v30 = v33;
        }
        *v31 = v30;
      }
      if ( v19 == v30 )
        goto LABEL_56;
      v40 = *((_DWORD *)v30 + 2);
      if ( *((_DWORD *)v19 + 2) >= v40 )
      {
        *v30 = v19;
        if ( v40 == *((_DWORD *)v19 + 2) )
          *((_DWORD *)v19 + 2) = v40 + 1;
      }
      else
      {
        *v19 = v30;
      }
      v41 = (_QWORD *)*((_QWORD *)s + (unsigned __int64)v17 % v140);
      v42 = (unsigned __int64)v17 % v140;
      if ( !v41 )
        goto LABEL_78;
      v43 = (_QWORD *)*v41;
      if ( v17 != *(unsigned __int64 **)(*v41 + 8LL) )
        break;
LABEL_55:
      if ( !*v41 )
        goto LABEL_78;
LABEL_56:
      v16 = (__int64 *)sub_220EF30((__int64)v16);
      if ( v16 == &v134 )
        goto LABEL_57;
    }
    while ( 1 )
    {
      v44 = (_QWORD *)*v43;
      if ( !*v43 )
        break;
      v41 = v43;
      if ( (unsigned __int64)v17 % v140 != v44[1] % v140 )
        break;
      v43 = (_QWORD *)*v43;
      if ( v17 == (unsigned __int64 *)v44[1] )
        goto LABEL_55;
    }
LABEL_78:
    v65 = (_QWORD *)sub_22077B0(0x10u);
    if ( v65 )
      *v65 = 0;
    v65[1] = v17;
    v66 = v140;
    v128 = v65;
    v67 = sub_222DA10((__int64)&v143, v140, v142, 1);
    v69 = v128;
    v70 = v68;
    if ( !v67 )
    {
      v71 = s;
      v72 = (_QWORD **)((char *)s + v42 * 8);
      v73 = *(_QWORD **)((char *)s + v42 * 8);
      if ( v73 )
      {
LABEL_82:
        *v69 = *v73;
        **v72 = v69;
LABEL_83:
        ++v142;
        goto LABEL_56;
      }
LABEL_165:
      v121 = v141;
      v141 = v69;
      *v69 = v121;
      if ( v121 )
      {
        v71[v121[1] % v140] = v69;
        v72 = (_QWORD **)((char *)s + v42 * 8);
      }
      *v72 = &v141;
      goto LABEL_83;
    }
    if ( v68 == 1 )
    {
      v145[0] = 0;
      v71 = v145;
    }
    else
    {
      if ( v68 > 0xFFFFFFFFFFFFFFFLL )
        sub_4261EA(&v143, v66, v68);
      v113 = 8 * v68;
      v123 = v128;
      v129 = v68;
      v114 = (void *)sub_22077B0(8 * v68);
      v115 = memset(v114, 0, v113);
      v69 = v123;
      v70 = v129;
      v71 = v115;
    }
    v116 = v141;
    v141 = 0;
    if ( !v116 )
    {
LABEL_162:
      if ( s != v145 )
      {
        v122 = v70;
        v124 = v71;
        v130 = v69;
        j_j___libc_free_0((unsigned __int64)s);
        v70 = v122;
        v71 = v124;
        v69 = v130;
      }
      v140 = v70;
      s = v71;
      v42 = (unsigned __int64)v17 % v70;
      v72 = (_QWORD **)&v71[v42];
      v73 = (_QWORD *)v71[v42];
      if ( v73 )
        goto LABEL_82;
      goto LABEL_165;
    }
    v117 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v118 = v116;
        v116 = (_QWORD *)*v116;
        v119 = v118[1] % v70;
        v120 = (_QWORD **)&v71[v119];
        if ( !*v120 )
          break;
        *v118 = **v120;
        **v120 = v118;
LABEL_158:
        if ( !v116 )
          goto LABEL_162;
      }
      *v118 = v141;
      v141 = v118;
      *v120 = &v141;
      if ( !*v118 )
      {
        v117 = v119;
        goto LABEL_158;
      }
      v71[v117] = v118;
      v117 = v119;
      if ( !v116 )
        goto LABEL_162;
    }
  }
LABEL_57:
  v146 = 0;
  v148 = 0;
  v149 = 0;
  v150 = 0;
  v151 = 0;
  v152 = 0;
  v153 = 0;
  v154 = 0;
  v155 = 0;
  v147 = 8;
  v146 = sub_22077B0(0x40u);
  v45 = (__int64 *)(v146 + ((4 * v147 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v46 = sub_22077B0(0x200u);
  v47 = v141;
  v151 = (unsigned __int64 *)v45;
  *v45 = v46;
  v149 = v46;
  v150 = v46 + 512;
  v155 = v45;
  v153 = v46;
  v154 = v46 + 512;
  v148 = v46;
  v152 = (_QWORD *)v46;
  if ( v47 )
  {
    while ( 1 )
    {
      v132 = v47[1];
      v53 = sub_26C6DA0(v2, (unsigned __int64 *)(v132 + 8));
      v54 = *(_DWORD *)(v53 + 40);
      v55 = v53;
      v56 = v53 + 16;
      if ( !v54 )
        break;
      v48 = v132;
      v49 = *(_QWORD *)(v53 + 24);
      v50 = (v54 - 1) & (((unsigned int)v132 >> 9) ^ ((unsigned int)v132 >> 4));
      v51 = (__int64 *)(v49 + 8LL * v50);
      v52 = *v51;
      if ( *v51 != v132 )
      {
        v111 = 1;
        v57 = 0;
        while ( v52 != -4096 )
        {
          if ( v52 == -8192 && !v57 )
            v57 = v51;
          v50 = (v54 - 1) & (v111 + v50);
          v51 = (__int64 *)(v49 + 8LL * v50);
          v52 = *v51;
          if ( v132 == *v51 )
            goto LABEL_60;
          ++v111;
        }
        if ( !v57 )
          v57 = v51;
        v133[0] = v57;
        v112 = *(_DWORD *)(v55 + 32);
        ++*(_QWORD *)(v55 + 16);
        v58 = v112 + 1;
        if ( 4 * v58 < 3 * v54 )
        {
          if ( v54 - *(_DWORD *)(v55 + 36) - v58 > v54 >> 3 )
            goto LABEL_149;
          goto LABEL_64;
        }
LABEL_63:
        v54 *= 2;
LABEL_64:
        sub_26D3B50(v56, v54);
        sub_26C9620(v56, &v132, v133);
        v48 = v132;
        v57 = (__int64 *)v133[0];
        v58 = *(_DWORD *)(v55 + 32) + 1;
LABEL_149:
        *(_DWORD *)(v55 + 32) = v58;
        if ( *v57 != -4096 )
          --*(_DWORD *)(v55 + 36);
        *v57 = v48;
      }
LABEL_60:
      v47 = (_QWORD *)*v47;
      if ( !v47 )
        goto LABEL_84;
    }
    v133[0] = 0;
    ++*(_QWORD *)(v53 + 16);
    goto LABEL_63;
  }
LABEL_84:
  for ( i = v136; i != &v134; i = (__int64 *)sub_220EF30((__int64)i) )
  {
    while ( 1 )
    {
      v75 = (unsigned __int64 *)i[4];
      v76 = sub_26C6DA0(v2, v75);
      v77 = v76;
      if ( !*(_BYTE *)(v76 + 12) && !*(_DWORD *)(v76 + 32) )
        break;
      i = (__int64 *)sub_220EF30((__int64)i);
      if ( i == &v134 )
        goto LABEL_94;
    }
    v78 = v152;
    if ( v152 == (_QWORD *)(v154 - 8) )
    {
      sub_26C96D0((unsigned __int64 *)&v146, v75);
    }
    else
    {
      if ( v152 )
      {
        *v152 = *v75;
        v78 = v152;
      }
      v152 = v78 + 1;
    }
    *(_BYTE *)(v77 + 12) = 1;
  }
LABEL_94:
  v79 = (_QWORD *)v148;
  if ( v152 != (_QWORD *)v148 )
  {
LABEL_95:
    v133[0] = *v79;
    if ( v79 == (_QWORD *)(v150 - 8) )
    {
      j_j___libc_free_0(v149);
      v80 = (_BYTE *)v2[8];
      v110 = *++v151 + 512;
      v149 = *v151;
      v150 = v110;
      v148 = v149;
      if ( v80 != (_BYTE *)v2[9] )
        goto LABEL_97;
    }
    else
    {
      v80 = (_BYTE *)v2[8];
      v148 = (unsigned __int64)(v79 + 1);
      if ( v80 != (_BYTE *)v2[9] )
      {
LABEL_97:
        v81 = v133[0];
        if ( v80 )
        {
          *(_QWORD *)v80 = v133[0];
          v80 = (_BYTE *)v2[8];
        }
        v2[8] = (unsigned __int64)(v80 + 8);
LABEL_100:
        v82 = *(_QWORD *)(v81 + 40);
        v83 = v81 + 24;
        if ( v83 == v82 )
          goto LABEL_114;
        while ( 1 )
        {
          v84 = v82 + 32;
          v85 = sub_26C6DA0(v2, (unsigned __int64 *)(v82 + 40));
          v86 = *(_DWORD *)(v85 + 40);
          v87 = *(_QWORD *)(v85 + 24);
          if ( v86 )
          {
            v88 = v86 - 1;
            v89 = (v86 - 1) & (((unsigned int)v84 >> 9) ^ ((unsigned int)v84 >> 4));
            v90 = (__int64 *)(v87 + 8LL * v89);
            v91 = *v90;
            if ( v84 == *v90 )
            {
LABEL_103:
              *v90 = -8192;
              --*(_DWORD *)(v85 + 32);
              ++*(_DWORD *)(v85 + 36);
            }
            else
            {
              v108 = 1;
              while ( v91 != -4096 )
              {
                v109 = v108 + 1;
                v89 = v88 & (v108 + v89);
                v90 = (__int64 *)(v87 + 8LL * v89);
                v91 = *v90;
                if ( v84 == *v90 )
                  goto LABEL_103;
                v108 = v109;
              }
            }
          }
          v92 = (_QWORD *)*((_QWORD *)s + v84 % v140);
          if ( !v92 )
            goto LABEL_113;
          v93 = (_QWORD *)*v92;
          if ( !*v92 )
            goto LABEL_113;
          v94 = v93[1];
          v95 = 0;
          while ( v84 == v94 )
          {
            v93 = (_QWORD *)*v93;
            ++v95;
            if ( !v93 )
              goto LABEL_112;
LABEL_109:
            v94 = v93[1];
            if ( v84 % v140 != v94 % v140 )
              goto LABEL_112;
          }
          if ( v95 )
            goto LABEL_128;
          v93 = (_QWORD *)*v93;
          if ( v93 )
            goto LABEL_109;
LABEL_112:
          if ( !v95 )
            goto LABEL_113;
LABEL_128:
          if ( !*(_DWORD *)(sub_26C6DA0(v2, (unsigned __int64 *)(v82 + 40)) + 32) )
          {
            v107 = v152;
            if ( v152 == (_QWORD *)(v154 - 8) )
            {
              sub_26C96D0((unsigned __int64 *)&v146, (_QWORD *)(v82 + 40));
              v82 = sub_220EF30(v82);
              if ( v83 == v82 )
                goto LABEL_114;
              continue;
            }
            if ( v152 )
            {
              *v152 = *(_QWORD *)(v82 + 40);
              v107 = v152;
            }
            v152 = v107 + 1;
            v82 = sub_220EF30(v82);
            if ( v83 == v82 )
            {
LABEL_114:
              v79 = (_QWORD *)v148;
              if ( v152 == (_QWORD *)v148 )
                goto LABEL_115;
              goto LABEL_95;
            }
          }
          else
          {
LABEL_113:
            v82 = sub_220EF30(v82);
            if ( v83 == v82 )
              goto LABEL_114;
          }
        }
      }
    }
    sub_26C7040(v125, v80, v133);
    v81 = v133[0];
    goto LABEL_100;
  }
LABEL_115:
  v96 = (__int64 *)v2[8];
  v97 = (__int64 *)v2[7];
  if ( v97 != v96 )
  {
    for ( j = v96 - 1; j > v97; j[1] = v99 )
    {
      v99 = *v97;
      v100 = *j;
      ++v97;
      --j;
      *(v97 - 1) = v100;
    }
  }
  v101 = v146;
  if ( v146 )
  {
    v102 = v151;
    v103 = (unsigned __int64)(v155 + 1);
    if ( v155 + 1 > (__int64 *)v151 )
    {
      do
      {
        v104 = *v102++;
        j_j___libc_free_0(v104);
      }
      while ( v103 > (unsigned __int64)v102 );
      v101 = v146;
    }
    j_j___libc_free_0(v101);
  }
  v105 = v141;
  while ( v105 )
  {
    v106 = (unsigned __int64)v105;
    v105 = (_QWORD *)*v105;
    j_j___libc_free_0(v106);
  }
  memset(s, 0, 8 * v140);
  v142 = 0;
  v141 = 0;
  if ( s != v145 )
    j_j___libc_free_0((unsigned __int64)s);
  sub_26BB660((unsigned __int64)v135);
}
