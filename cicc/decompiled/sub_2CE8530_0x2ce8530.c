// Function: sub_2CE8530
// Address: 0x2ce8530
//
__int64 __fastcall sub_2CE8530(__int64 a1, __int64 a2, unsigned __int8 *a3, _QWORD *a4, _QWORD *a5, int *a6)
{
  _QWORD *v8; // r13
  _QWORD *v11; // rdx
  __int64 v12; // r9
  _QWORD *v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rsi
  _QWORD *v18; // rax
  _QWORD *v19; // rsi
  __int64 v20; // rcx
  __int64 v21; // rdx
  unsigned int v22; // r10d
  __int64 v23; // rsi
  char v24; // al
  unsigned __int64 *v25; // rdx
  __int64 v26; // r9
  _QWORD *v27; // r8
  unsigned __int64 v28; // r11
  char v29; // al
  char v30; // al
  char v31; // al
  int v32; // r12d
  __int64 v33; // rcx
  __int64 v34; // rax
  int v35; // eax
  char v37; // al
  unsigned __int64 v38; // r11
  _QWORD *v39; // r12
  _QWORD *v40; // rdx
  _QWORD *v41; // r14
  _QWORD *v42; // rdi
  _QWORD *v43; // rax
  int v44; // eax
  unsigned __int64 v45; // rsi
  char v46; // al
  unsigned __int64 *v47; // r10
  __int64 v48; // r12
  unsigned __int8 v49; // al
  unsigned int v50; // r11d
  __int64 v51; // rcx
  __int64 v52; // rsi
  _QWORD *v53; // rax
  _QWORD *v54; // rdx
  _QWORD *v55; // r8
  _QWORD *v56; // r9
  bool v57; // r11
  __int64 v58; // rax
  unsigned int v59; // eax
  unsigned __int64 v60; // r15
  _DWORD *v61; // rax
  __int64 v62; // rax
  _QWORD *v63; // rax
  _QWORD *v64; // rdx
  char v65; // di
  __int64 i; // rax
  int v67; // r12d
  _DWORD *v68; // rax
  unsigned __int64 v69; // rax
  _QWORD *v70; // rdi
  char v71; // al
  int v72; // r11d
  unsigned __int64 v73; // rdx
  _QWORD *v74; // rax
  _QWORD *v75; // rdx
  char v76; // al
  _QWORD *v77; // r8
  unsigned __int8 *v78; // r12
  int v79; // eax
  char v80; // al
  int v81; // r10d
  unsigned __int64 v82; // r12
  int *v83; // rax
  int v84; // edx
  unsigned __int64 v85; // rsi
  int v86; // r10d
  __int64 v87; // r12
  int v88; // edx
  unsigned int v89; // eax
  __int64 *v90; // r12
  int v91; // r10d
  _QWORD *v92; // r8
  __int64 v93; // r12
  int *v94; // r13
  char v95; // bl
  int *v96; // rax
  unsigned __int8 *v97; // rdx
  int v98; // eax
  int v99; // r10d
  _DWORD *v100; // rax
  __int64 *v101; // r12
  int v102; // r15d
  _QWORD *v103; // r14
  char v104; // r11
  int v105; // eax
  _QWORD *v106; // rdx
  _QWORD *v107; // rcx
  _QWORD *v108; // rdx
  unsigned __int64 v109; // rcx
  _QWORD *j; // rax
  _QWORD *v111; // rdi
  unsigned __int64 *v112; // rax
  _QWORD *v113; // rax
  const __m128i *v114; // rax
  __int64 v115; // rax
  __int64 v116; // r8
  __int64 v117; // r12
  __int64 v118; // r11
  unsigned __int64 v119; // rcx
  int *v120; // r12
  _QWORD *v121; // rbx
  __int64 v122; // rdx
  __int64 v123; // rax
  int v124; // eax
  __int64 v125; // rax
  int v126; // eax
  _QWORD *v127; // [rsp+0h] [rbp-E0h]
  _QWORD *v128; // [rsp+0h] [rbp-E0h]
  unsigned __int64 v129; // [rsp+0h] [rbp-E0h]
  __int64 v130; // [rsp+0h] [rbp-E0h]
  __int64 v131; // [rsp+8h] [rbp-D8h]
  __int64 v132; // [rsp+8h] [rbp-D8h]
  _QWORD *v133; // [rsp+10h] [rbp-D0h]
  int v134; // [rsp+18h] [rbp-C8h]
  char v135; // [rsp+18h] [rbp-C8h]
  _QWORD *v136; // [rsp+18h] [rbp-C8h]
  char v137; // [rsp+20h] [rbp-C0h]
  _QWORD *v138; // [rsp+20h] [rbp-C0h]
  __int64 v139; // [rsp+20h] [rbp-C0h]
  _QWORD *v140; // [rsp+20h] [rbp-C0h]
  __int64 v142; // [rsp+28h] [rbp-B8h]
  _QWORD *v143; // [rsp+28h] [rbp-B8h]
  _QWORD *v144; // [rsp+28h] [rbp-B8h]
  _QWORD *v145; // [rsp+28h] [rbp-B8h]
  __int64 v146; // [rsp+28h] [rbp-B8h]
  _QWORD *v147; // [rsp+28h] [rbp-B8h]
  _QWORD *v148; // [rsp+28h] [rbp-B8h]
  char v149; // [rsp+28h] [rbp-B8h]
  _QWORD *v150; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v151; // [rsp+30h] [rbp-B0h]
  unsigned int v152; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v153; // [rsp+30h] [rbp-B0h]
  _QWORD *v154; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v155; // [rsp+30h] [rbp-B0h]
  unsigned int v156; // [rsp+30h] [rbp-B0h]
  _QWORD *v157; // [rsp+30h] [rbp-B0h]
  unsigned int v158; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v159[2]; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v160; // [rsp+48h] [rbp-98h] BYREF
  __int64 v161; // [rsp+50h] [rbp-90h] BYREF
  int v162; // [rsp+58h] [rbp-88h] BYREF
  unsigned __int64 v163; // [rsp+60h] [rbp-80h]
  int *v164; // [rsp+68h] [rbp-78h]
  int *v165; // [rsp+70h] [rbp-70h]
  __int64 v166; // [rsp+78h] [rbp-68h]
  unsigned __int64 *v167; // [rsp+80h] [rbp-60h] BYREF
  __int64 v168; // [rsp+88h] [rbp-58h] BYREF
  unsigned __int64 v169; // [rsp+90h] [rbp-50h] BYREF
  __int64 *v170; // [rsp+98h] [rbp-48h]
  __int64 *v171; // [rsp+A0h] [rbp-40h]
  __int64 v172; // [rsp+A8h] [rbp-38h]

  v8 = a4;
  v159[0] = (unsigned __int64)a3;
  v11 = (_QWORD *)a4[2];
  if ( v11 )
  {
    v12 = (__int64)(a4 + 1);
    v13 = (_QWORD *)a4[2];
    v14 = a4 + 1;
    do
    {
      while ( 1 )
      {
        v15 = v13[2];
        v16 = v13[3];
        if ( v13[4] >= (unsigned __int64)a3 )
          break;
        v13 = (_QWORD *)v13[3];
        if ( !v16 )
          goto LABEL_6;
      }
      v14 = v13;
      v13 = (_QWORD *)v13[2];
    }
    while ( v15 );
LABEL_6:
    if ( (_QWORD *)v12 != v14 )
    {
      v17 = v12;
      if ( v14[4] <= (unsigned __int64)a3 )
      {
        while ( 1 )
        {
          v33 = v11[2];
          v34 = v11[3];
          if ( v11[4] < (unsigned __int64)a3 )
          {
            v11 = (_QWORD *)v11[3];
            if ( !v34 )
              goto LABEL_26;
          }
          else
          {
            v17 = (__int64)v11;
            v11 = (_QWORD *)v11[2];
            if ( !v33 )
            {
LABEL_26:
              if ( v12 == v17 || *(_QWORD *)(v17 + 32) > (unsigned __int64)a3 )
              {
                v167 = v159;
                v17 = sub_2CE6580(v8, v17, &v167);
              }
              v35 = *(_DWORD *)(v17 + 40);
              *a6 = v35;
              return v35 != 0;
            }
          }
        }
      }
    }
  }
  v18 = (_QWORD *)a5[2];
  if ( v18 )
  {
    v19 = a5 + 1;
    do
    {
      while ( 1 )
      {
        v20 = v18[2];
        v21 = v18[3];
        if ( v18[4] >= (unsigned __int64)a3 )
          break;
        v18 = (_QWORD *)v18[3];
        if ( !v21 )
          goto LABEL_13;
      }
      v19 = v18;
      v18 = (_QWORD *)v18[2];
    }
    while ( v20 );
LABEL_13:
    if ( a5 + 1 != v19 )
    {
      v22 = 2;
      if ( v19[4] <= (unsigned __int64)a3 )
        return v22;
    }
  }
  v23 = a2;
  v150 = a5 + 1;
  v24 = sub_2CE8360(a1, a2, a3);
  v26 = (__int64)v150;
  v27 = a5;
  v137 = v24;
  if ( !v24 )
  {
LABEL_176:
    v100 = (_DWORD *)sub_2CE6630(v8, v159);
    v22 = 0;
    *v100 = 0;
    return v22;
  }
  v28 = v159[0];
  v29 = *(_BYTE *)v159[0];
  if ( *(_BYTE *)v159[0] > 0x1Cu )
  {
    if ( v29 == 77 )
    {
      *a6 = 1;
      v61 = (_DWORD *)sub_2CE6630(v8, v159);
      v22 = 1;
      *v61 = 1;
      return v22;
    }
    if ( v29 == 93 )
    {
      v47 = &v169;
      v169 = v159[0];
      v167 = &v169;
      v168 = 0x400000001LL;
      v48 = *(_QWORD *)(v159[0] - 32);
      v49 = *(_BYTE *)v48;
      if ( *(_BYTE *)v48 != 93 )
      {
        v50 = 1;
        v51 = 1;
        goto LABEL_55;
      }
      v25 = &v169;
      for ( i = 1; ; i = (unsigned int)v168 )
      {
        v25[i] = v48;
        v50 = v168 + 1;
        LODWORD(v168) = v168 + 1;
        v48 = *(_QWORD *)(v48 - 32);
        v49 = *(_BYTE *)v48;
        if ( *(_BYTE *)v48 != 93 )
          break;
        v73 = v50 + 1LL;
        if ( v73 > HIDWORD(v168) )
        {
          v136 = v27;
          sub_C8D5F0((__int64)&v167, &v169, v73, 8u, (__int64)v27, v26);
          v27 = v136;
        }
        v25 = v167;
      }
      v51 = v50;
      v26 = (__int64)v150;
      v47 = v167;
      if ( v50 )
      {
LABEL_55:
        while ( 1 )
        {
          v52 = v47[v51 - 1];
          if ( v49 <= 0x1Cu )
            break;
          if ( v49 == 61 )
          {
            if ( *(_QWORD *)(v48 + 8) == *(_QWORD *)(*(_QWORD *)(v52 - 32) + 8LL) )
            {
              LODWORD(v168) = v50 - 1;
              if ( v50 != 1 )
              {
LABEL_93:
                sub_2CE66B0(v8, v159[0], 0, v27);
                v22 = 0;
LABEL_68:
                if ( v167 != &v169 )
                {
                  v152 = v22;
                  _libc_free((unsigned __int64)v167);
                  return v152;
                }
                return v22;
              }
LABEL_60:
              v143 = v27;
              v139 = v26;
              v53 = sub_23FDE00((__int64)v27, v159);
              v55 = v143;
              if ( v54 )
              {
                v56 = (_QWORD *)v139;
                v57 = v53 || (_QWORD *)v139 == v54 || v159[0] < v54[4];
                v133 = v143;
                v135 = v57;
                v140 = v54;
                v144 = v56;
                v58 = sub_22077B0(0x28u);
                *(_QWORD *)(v58 + 32) = v159[0];
                sub_220F040(v135, v58, v140, v144);
                v55 = v133;
                ++v133[5];
              }
              v145 = v55;
              v59 = sub_2CE8530(a1, a2, v48, v8, v55, a6);
              v22 = v59;
              if ( v59 )
              {
                if ( v59 == 1 )
                {
                  sub_2CE66B0(v8, v159[0], *a6, v145);
                  v22 = 1;
                }
              }
              else
              {
                sub_2CE66B0(v8, v159[0], 0, v145);
                v22 = 0;
              }
              goto LABEL_68;
            }
LABEL_92:
            if ( (_DWORD)v168 )
              goto LABEL_93;
            goto LABEL_60;
          }
          if ( v49 != 94 || *(_QWORD *)(*(_QWORD *)(v52 - 32) + 8LL) != *(_QWORD *)(v48 + 8) )
            goto LABEL_58;
          v147 = v27;
          v71 = sub_2CDD660(v48, v52, (__int64)v25, v51, (unsigned int)v27);
          v27 = v147;
          if ( v71 )
          {
            LODWORD(v168) = v72 - 1;
            v48 = *(_QWORD *)(v48 - 32);
          }
          else
          {
            v48 = *(_QWORD *)(v48 - 64);
          }
          v51 = (unsigned int)v168;
          v50 = v168;
          if ( !(_DWORD)v168 )
            goto LABEL_59;
          v49 = *(_BYTE *)v48;
        }
        if ( v49 == 22 )
        {
          if ( *(_QWORD *)(*(_QWORD *)(v52 - 32) + 8LL) == *(_QWORD *)(v48 + 8) )
          {
            LODWORD(v168) = 0;
            goto LABEL_60;
          }
          goto LABEL_92;
        }
LABEL_58:
        if ( (_DWORD)v168 )
          goto LABEL_93;
      }
LABEL_59:
      v138 = v27;
      v142 = v26;
      sub_BD84D0(v159[0], v48);
      v27 = v138;
      v26 = v142;
      goto LABEL_60;
    }
    if ( v29 == 60 )
    {
      if ( !*(_BYTE *)(a1 + 1) )
        goto LABEL_176;
      v67 = *(_DWORD *)(*(_QWORD *)(v159[0] + 8) + 8LL) >> 8;
      if ( !v67 )
        v67 = 5;
    }
    else
    {
      if ( v29 != 79 )
      {
        if ( v29 != 85 )
        {
          if ( v29 == 78 )
          {
            v87 = *(_QWORD *)(v159[0] - 32);
          }
          else
          {
            if ( v29 != 63 )
              goto LABEL_46;
            v87 = *(_QWORD *)(v159[0] - 32LL * (*(_DWORD *)(v159[0] + 4) & 0x7FFFFFF));
          }
          v88 = *(_DWORD *)(*(_QWORD *)(v87 + 8) + 8LL) >> 8;
          *a6 = v88;
          if ( !v88 )
          {
LABEL_152:
            sub_2CE14D0((__int64)a5, v159);
            v89 = sub_2CE8530(a1, a2, v87, v8, a5, a6);
            v22 = v89;
            if ( v89 )
            {
              if ( v89 == 1 )
              {
                sub_2CE66B0(v8, v159[0], *a6, a5);
                return 1;
              }
            }
            else
            {
              sub_2CE66B0(v8, v159[0], 0, a5);
              return 0;
            }
            return v22;
          }
          goto LABEL_155;
        }
        v69 = *(_QWORD *)(v159[0] - 32);
        if ( v69 )
        {
          if ( !*(_BYTE *)v69
            && *(_QWORD *)(v69 + 24) == *(_QWORD *)(v159[0] + 80)
            && (*(_BYTE *)(v69 + 33) & 0x20) != 0 )
          {
            v99 = *(_DWORD *)(*(_QWORD *)(v159[0] + 8) + 8LL) >> 8;
            if ( v99 )
            {
              *a6 = v99;
              *(_DWORD *)sub_2CE6630(v8, v159) = v99;
              return 1;
            }
            if ( *(_DWORD *)(v69 + 36) != 8170 )
              goto LABEL_176;
            v88 = *a6;
            if ( !*a6 )
            {
              v87 = *(_QWORD *)(v159[0] - 32LL * (*(_DWORD *)(v159[0] + 4) & 0x7FFFFFF));
              goto LABEL_152;
            }
LABEL_155:
            sub_2CE66B0(v8, v28, v88, a5);
            return 1;
          }
          v70 = *(_QWORD **)(a1 + 16);
          if ( v70 )
          {
            if ( !*(_BYTE *)v69 && *(_QWORD *)(v69 + 24) == *(_QWORD *)(v159[0] + 80) )
            {
              v167 = *(unsigned __int64 **)(v159[0] - 32);
              v106 = (_QWORD *)v70[2];
              if ( v106 )
              {
                v107 = v70 + 1;
                do
                {
                  if ( v106[4] < v69 )
                  {
                    v106 = (_QWORD *)v106[3];
                  }
                  else
                  {
                    v107 = v106;
                    v106 = (_QWORD *)v106[2];
                  }
                }
                while ( v106 );
                if ( v107 != v70 + 1 && v107[4] <= v69 )
                {
                  v157 = a5;
                  v83 = (int *)sub_2CE67C0(v70, (unsigned __int64 *)&v167);
                  goto LABEL_147;
                }
              }
            }
          }
        }
        sub_2CE66B0(v8, v159[0], 0, a5);
        return 0;
      }
      v67 = *(_DWORD *)(*(_QWORD *)(v159[0] + 8) + 8LL) >> 8;
      if ( !v67 )
        v67 = *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v159[0] - 32) + 8LL) + 8LL) >> 8;
    }
    *a6 = v67;
    v68 = (_DWORD *)sub_2CE6630(v8, v159);
    v22 = 1;
    *v68 = v67;
    return v22;
  }
  if ( v29 == 22 )
  {
    if ( unk_50142AD )
    {
      v30 = sub_CE9220(a2);
      v28 = v159[0];
      if ( v30 )
      {
        v31 = sub_B2D680(v159[0]);
        v28 = v159[0];
        if ( !v31 )
        {
          *a6 = 1;
          v32 = 1;
LABEL_22:
          *(_DWORD *)sub_2CE6630(v8, v159) = v32;
          return *a6 != 0;
        }
      }
    }
    v151 = v28;
    v37 = sub_B2D680(v28);
    v38 = v151;
    if ( v37 )
    {
      v46 = sub_CE9220(a2);
      v38 = v151;
      if ( !v46 )
      {
        *a6 = 5;
        v32 = 5;
        goto LABEL_22;
      }
    }
    v39 = *(_QWORD **)(a1 + 8);
    if ( !v39 )
      goto LABEL_70;
    v40 = (_QWORD *)v39[2];
    v41 = v39 + 1;
    if ( !v40 )
      goto LABEL_70;
    v42 = v39 + 1;
    v43 = (_QWORD *)v39[2];
    do
    {
      if ( v43[4] < v38 )
      {
        v43 = (_QWORD *)v43[3];
      }
      else
      {
        v42 = v43;
        v43 = (_QWORD *)v43[2];
      }
    }
    while ( v43 );
    if ( v42 == v41 || v42[4] > v38 )
    {
LABEL_70:
      *a6 = 0;
      v32 = 0;
      goto LABEL_22;
    }
    v60 = (unsigned __int64)(v39 + 1);
    do
    {
      if ( v40[4] < v38 )
      {
        v40 = (_QWORD *)v40[3];
      }
      else
      {
        v60 = (unsigned __int64)v40;
        v40 = (_QWORD *)v40[2];
      }
    }
    while ( v40 );
    if ( v41 == (_QWORD *)v60 || *(_QWORD *)(v60 + 32) > v38 )
    {
      v153 = v38;
      v146 = v60;
      v62 = sub_22077B0(0x30u);
      *(_DWORD *)(v62 + 40) = 0;
      v60 = v62;
      *(_QWORD *)(v62 + 32) = v153;
      v63 = sub_2CBBA50(v39, v146, (unsigned __int64 *)(v62 + 32));
      if ( v64 )
      {
        v65 = v41 == v64 || v63 || v64[4] > v153;
        sub_220F040(v65, v60, v64, v39 + 1);
        ++v39[5];
      }
      else
      {
        v154 = v63;
        j_j___libc_free_0(v60);
        v60 = (unsigned __int64)v154;
      }
    }
    v32 = *(_DWORD *)(v60 + 40);
LABEL_80:
    *a6 = v32;
    goto LABEL_22;
  }
  if ( v29 == 3 )
  {
    v32 = *(_DWORD *)(*(_QWORD *)(v159[0] + 8) + 8LL) >> 8;
    goto LABEL_80;
  }
LABEL_46:
  if ( (v29 & 0xFD) == 0x54 )
  {
    v86 = *(_DWORD *)(*(_QWORD *)(v159[0] + 8) + 8LL) >> 8;
    *a6 = v86;
    if ( v86 )
    {
      sub_2CE66B0(v8, v28, v86, a5);
      return 1;
    }
    v164 = &v162;
    v165 = &v162;
    v162 = 0;
    v163 = 0;
    v166 = 0;
    LODWORD(v168) = 0;
    v169 = 0;
    v170 = &v168;
    v171 = &v168;
    v172 = 0;
    sub_2CE1D30(v28, &v161, &v167, (__int64)v8);
    v90 = v170;
    v91 = 0;
    v92 = a5;
    if ( v170 != &v168 )
    {
      do
      {
        v160 = v90[4];
        sub_2CE14D0((__int64)a5, (unsigned __int64 *)&v160);
        v90 = (__int64 *)sub_220EF30((__int64)v90);
      }
      while ( v90 != &v168 );
      v91 = 0;
      v92 = a5;
    }
    v93 = (__int64)v164;
    if ( v164 == &v162 )
      goto LABEL_179;
    v148 = v8;
    v94 = a6;
    v95 = 0;
    do
    {
      v97 = *(unsigned __int8 **)(v93 + 32);
      v98 = *v97;
      if ( (_BYTE)v98 != 20 && (unsigned int)(v98 - 12) > 1 )
      {
        LODWORD(v160) = *(_DWORD *)(*((_QWORD *)v97 + 1) + 8LL) >> 8;
        if ( (_DWORD)v160 )
          goto LABEL_166;
        v128 = v92;
        v105 = sub_2CE8530(a1, a2, v97, v148, v92, &v160);
        v92 = v128;
        if ( !v105 )
        {
LABEL_178:
          a6 = v94;
          v91 = 0;
          v8 = v148;
          goto LABEL_179;
        }
        if ( v105 != 2 )
        {
LABEL_166:
          if ( v95 )
          {
            if ( *v94 != (_DWORD)v160 )
              goto LABEL_178;
          }
          else
          {
            *v94 = v160;
          }
          v95 = v137;
        }
      }
      v127 = v92;
      v96 = (int *)sub_220EF30(v93);
      v92 = v127;
      v93 = (__int64)v96;
    }
    while ( v96 != &v162 );
    v104 = v95;
    v91 = 0;
    a6 = v94;
    v8 = v148;
    if ( !v104 )
LABEL_179:
      *a6 = 0;
    else
      v91 = 1;
    v101 = v170;
    v102 = v91;
    v103 = v92;
    if ( v170 != &v168 )
    {
      do
      {
        sub_2CE66B0(v8, v101[4], *a6, v103);
        v101 = (__int64 *)sub_220EF30((__int64)v101);
      }
      while ( v101 != &v168 );
      v91 = v102;
    }
    v158 = v91;
    sub_2CDF380(v169);
    sub_2CDF380(v163);
    return v158;
  }
  if ( v29 != 61 )
  {
    if ( *(_BYTE *)v159[0] == 5 )
    {
      v44 = sub_2CDD760(v159[0], a2);
      v45 = v159[0];
      *a6 = v44;
      sub_2CE66B0(v8, v45, v44, a5);
      return *a6 != 0;
    }
    goto LABEL_176;
  }
  *a6 = 0;
  v161 = v28;
  v155 = v28;
  if ( (unsigned int)sub_2CDFE70(v28) == 4 )
  {
    *a6 = 1;
    sub_2CE66B0(v8, v155, 1, a5);
    return 1;
  }
  v74 = *(_QWORD **)(a1 + 512);
  if ( v74 )
  {
    v75 = (_QWORD *)(a1 + 504);
    do
    {
      v23 = v74[3];
      if ( v74[4] < v155 )
      {
        v74 = (_QWORD *)v74[3];
      }
      else
      {
        v75 = v74;
        v74 = (_QWORD *)v74[2];
      }
    }
    while ( v74 );
    if ( v75 != (_QWORD *)(a1 + 504) && v75[4] <= v155 )
    {
      v157 = a5;
      v83 = (int *)sub_2CE50D0((_QWORD *)(a1 + 496), (unsigned __int64 *)&v161);
LABEL_147:
      v84 = *v83;
      v85 = v159[0];
      *a6 = *v83;
      sub_2CE66B0(v8, v85, v84, v157);
      return 1;
    }
  }
  v76 = sub_CE9220(a2);
  v77 = a5;
  if ( v76 && unk_50142AD )
  {
    v78 = sub_BD3990(*(unsigned __int8 **)(v161 - 32), v23);
    v79 = sub_2CDFE70(v161);
    v77 = a5;
    if ( v79 == 101 )
    {
      *a6 = 1;
      if ( *v78 != 22 )
        goto LABEL_143;
LABEL_141:
      v80 = sub_B2D680((__int64)v78);
      v77 = a5;
      if ( v80 )
      {
        *a6 = 1;
LABEL_143:
        v81 = 1;
LABEL_144:
        v82 = v159[0];
        goto LABEL_145;
      }
      goto LABEL_197;
    }
    if ( *v78 == 22 )
      goto LABEL_141;
  }
LABEL_197:
  v81 = *a6;
  if ( *a6 )
    goto LABEL_143;
  v108 = (_QWORD *)(a1 + 312);
  v109 = *(_QWORD *)(v161 - 32);
  for ( j = *(_QWORD **)(a1 + 320); j; j = v111 )
  {
    v111 = (_QWORD *)j[3];
    if ( j[4] >= v109 )
    {
      v111 = (_QWORD *)j[2];
      v108 = j;
    }
  }
  if ( (_QWORD *)(a1 + 312) == v108 )
    goto LABEL_144;
  if ( v109 < v108[4] )
    goto LABEL_144;
  v167 = *(unsigned __int64 **)(v161 - 32);
  v131 = (__int64)v77;
  v112 = (unsigned __int64 *)sub_2CE2990((_QWORD *)(a1 + 304), (unsigned __int64 *)&v167);
  v113 = sub_2CE0E10(a1 + 256, v112);
  v81 = 0;
  v77 = (_QWORD *)v131;
  if ( (_QWORD *)(a1 + 264) == v113 )
    goto LABEL_144;
  v167 = *(unsigned __int64 **)(v161 - 32);
  v114 = (const __m128i *)sub_2CE2990((_QWORD *)(a1 + 304), (unsigned __int64 *)&v167);
  v115 = sub_2CE4DC0((_QWORD *)(a1 + 256), v114);
  v116 = v131;
  v117 = v115;
  if ( *(_BYTE *)(*(_QWORD *)(v161 + 8) + 8LL) == 14 )
  {
    sub_2CE14D0(v131, v159);
    v116 = v131;
  }
  v118 = *(_QWORD *)(v117 + 24);
  v149 = 0;
  v132 = v117 + 8;
  v119 = v159[0];
  v120 = a6;
  v121 = (_QWORD *)v116;
  while ( v132 != v118 )
  {
    v122 = *(_QWORD *)(v118 + 32);
    v123 = *(_QWORD *)(v122 + 8);
    if ( *(_BYTE *)(v123 + 8) != 14 || v123 != *(_QWORD *)(v119 + 8) )
    {
LABEL_225:
      v81 = 0;
      v77 = v121;
      a6 = v120;
      v82 = v119;
LABEL_226:
      *a6 = 0;
      goto LABEL_145;
    }
    v124 = *(_DWORD *)(v123 + 8) >> 8;
    if ( v124 )
    {
      LODWORD(v167) = v124;
LABEL_214:
      if ( v149 )
      {
        if ( v134 != (_DWORD)v167 )
          goto LABEL_225;
      }
      else
      {
        v134 = (int)v167;
        v149 = v137;
      }
      goto LABEL_216;
    }
    v130 = v118;
    v126 = sub_2CE8530(a1, a2, v122, v8, v121, &v167);
    v118 = v130;
    if ( !v126 )
    {
      v77 = v121;
      v81 = 0;
      a6 = v120;
      v82 = v159[0];
      goto LABEL_226;
    }
    v119 = v159[0];
    if ( v126 != 2 )
      goto LABEL_214;
LABEL_216:
    v129 = v119;
    v125 = sub_220EF30(v118);
    v119 = v129;
    v118 = v125;
  }
  v77 = v121;
  v81 = 0;
  a6 = v120;
  v82 = v119;
  if ( !v149 )
    goto LABEL_226;
  v81 = 1;
  *a6 = v134;
LABEL_145:
  v156 = v81;
  sub_2CE66B0(v8, v82, *a6, v77);
  return v156;
}
