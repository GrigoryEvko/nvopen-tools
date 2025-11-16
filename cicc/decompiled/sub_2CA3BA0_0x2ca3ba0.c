// Function: sub_2CA3BA0
// Address: 0x2ca3ba0
//
unsigned __int64 __fastcall sub_2CA3BA0(__int64 a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 result; // rax
  __int64 v6; // rdx
  unsigned __int64 v8; // r10
  unsigned int v9; // esi
  __int64 v10; // rdi
  unsigned int v11; // ebx
  unsigned int v12; // edx
  unsigned __int64 *v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 *v16; // rcx
  __int64 v17; // rsi
  unsigned int v18; // edx
  __int64 **v19; // rdi
  __int64 *v20; // r9
  __int64 v21; // rcx
  __int64 v22; // r14
  __int64 v23; // rdi
  _QWORD *v24; // r13
  unsigned int v25; // edx
  _QWORD *v26; // rdi
  _QWORD *v27; // r8
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // r13
  int v32; // r15d
  __int64 *v33; // r14
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // rbx
  __int64 v37; // r12
  __int64 v38; // rax
  _BYTE *v39; // rsi
  __int64 v40; // rdi
  int v41; // eax
  int v42; // ecx
  unsigned int v43; // eax
  _QWORD *v44; // rsi
  _QWORD *v45; // r8
  _BYTE *v46; // rsi
  _BYTE *v47; // rax
  unsigned __int64 v48; // r12
  _BYTE *v49; // rsi
  unsigned __int64 v50; // rdi
  int v51; // r10d
  _QWORD *v52; // rbx
  unsigned __int64 v53; // rdx
  _QWORD *v54; // r14
  __int64 v55; // rcx
  unsigned __int64 *v56; // r13
  unsigned __int64 *v57; // r9
  unsigned __int64 *v58; // rbx
  unsigned __int64 v59; // rbx
  char *v60; // rax
  char *v61; // rcx
  _BYTE *v62; // rbx
  int v63; // edi
  int v64; // r10d
  int v65; // eax
  unsigned int v66; // eax
  __int64 **v67; // rsi
  __int64 *v68; // r8
  int v69; // edi
  unsigned __int64 v70; // rax
  unsigned __int64 *v71; // r14
  unsigned __int64 v72; // r12
  int v73; // esi
  int v74; // r9d
  int v75; // r10d
  int v76; // eax
  __int64 v77; // r14
  unsigned __int64 v78; // r13
  unsigned __int64 v79; // rax
  bool v80; // cf
  unsigned __int64 v81; // r13
  __int64 v82; // rdi
  __int64 v83; // rax
  bool v84; // zf
  _QWORD *v85; // r14
  _QWORD *v86; // r13
  _BYTE *v87; // rax
  unsigned __int64 v88; // r8
  __int64 v89; // rax
  char *v90; // rdi
  size_t v91; // r14
  char *v92; // rax
  unsigned __int64 *v93; // r13
  unsigned __int64 *v94; // r12
  unsigned __int64 *v95; // rbx
  unsigned __int64 *v96; // r14
  unsigned __int64 v97; // rdx
  unsigned __int64 v98; // rsi
  unsigned __int64 v99; // rdi
  int v100; // esi
  int v101; // r9d
  int v102; // r10d
  int v103; // r10d
  __int64 v104; // r8
  unsigned int v105; // edi
  __int64 v106; // rsi
  int v107; // ecx
  int v108; // r10d
  int v109; // r10d
  __int64 v110; // r8
  int v111; // ecx
  unsigned int v112; // edi
  __int64 v113; // rsi
  int v114; // r11d
  unsigned __int64 *v115; // r10
  int v116; // edx
  int v117; // r11d
  int v118; // r11d
  __int64 v119; // r9
  unsigned int v120; // ecx
  unsigned __int64 v121; // r8
  int v122; // edi
  unsigned __int64 *v123; // rsi
  int v124; // r10d
  int v125; // r10d
  __int64 v126; // r8
  unsigned __int64 *v127; // rcx
  unsigned int v128; // ebx
  int v129; // esi
  unsigned __int64 v130; // rdi
  __int64 v132; // [rsp+18h] [rbp-108h]
  __int64 v134; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v136; // [rsp+40h] [rbp-E0h]
  __int64 v137; // [rsp+48h] [rbp-D8h]
  unsigned int v138; // [rsp+50h] [rbp-D0h]
  unsigned int v139; // [rsp+54h] [rbp-CCh]
  __int64 v140; // [rsp+58h] [rbp-C8h]
  _QWORD *v142; // [rsp+68h] [rbp-B8h]
  unsigned __int64 *v143; // [rsp+78h] [rbp-A8h]
  unsigned __int64 *v144; // [rsp+80h] [rbp-A0h]
  unsigned __int64 v145; // [rsp+80h] [rbp-A0h]
  unsigned __int64 *v146; // [rsp+80h] [rbp-A0h]
  unsigned __int64 *v147; // [rsp+80h] [rbp-A0h]
  __int64 v148; // [rsp+88h] [rbp-98h]
  unsigned int v149; // [rsp+90h] [rbp-90h]
  unsigned __int64 *v150; // [rsp+90h] [rbp-90h]
  __int64 *v151; // [rsp+98h] [rbp-88h]
  __int64 *v152; // [rsp+A0h] [rbp-80h] BYREF
  _QWORD *v153; // [rsp+A8h] [rbp-78h] BYREF
  __int64 v154; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v155; // [rsp+B8h] [rbp-68h] BYREF
  __int64 v156; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v157; // [rsp+C8h] [rbp-58h] BYREF
  void *src; // [rsp+D0h] [rbp-50h] BYREF
  _BYTE *v159; // [rsp+D8h] [rbp-48h]
  _BYTE *v160; // [rsp+E0h] [rbp-40h]

  result = *a2;
  v6 = (__int64)(a2[1] - *a2) >> 3;
  if ( (_DWORD)v6 )
  {
    v134 = 0;
    v132 = 8LL * (unsigned int)(v6 - 1);
    while ( 1 )
    {
      v8 = *(_QWORD *)(result + v134);
      v9 = *(_DWORD *)(a3 + 24);
      v136 = v8;
      if ( !v9 )
        break;
      v10 = *(_QWORD *)(a3 + 8);
      v11 = ((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4);
      v12 = (v9 - 1) & v11;
      v13 = (unsigned __int64 *)(v10 + 16LL * v12);
      v14 = *v13;
      if ( v8 != *v13 )
      {
        v114 = 1;
        v115 = 0;
        while ( v14 != -4096 )
        {
          if ( !v115 && v14 == -8192 )
            v115 = v13;
          v12 = (v9 - 1) & (v114 + v12);
          v13 = (unsigned __int64 *)(v10 + 16LL * v12);
          v14 = *v13;
          if ( v136 == *v13 )
            goto LABEL_5;
          ++v114;
        }
        if ( v115 )
          v13 = v115;
        ++*(_QWORD *)a3;
        v116 = *(_DWORD *)(a3 + 16) + 1;
        if ( 4 * v116 < 3 * v9 )
        {
          if ( v9 - *(_DWORD *)(a3 + 20) - v116 <= v9 >> 3 )
          {
            sub_2C93320(a3, v9);
            v124 = *(_DWORD *)(a3 + 24);
            if ( !v124 )
            {
LABEL_189:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v125 = v124 - 1;
            v126 = *(_QWORD *)(a3 + 8);
            v127 = 0;
            v128 = v125 & v11;
            v116 = *(_DWORD *)(a3 + 16) + 1;
            v129 = 1;
            v13 = (unsigned __int64 *)(v126 + 16LL * v128);
            v130 = *v13;
            if ( v136 != *v13 )
            {
              while ( v130 != -4096 )
              {
                if ( v130 == -8192 && !v127 )
                  v127 = v13;
                v128 = v125 & (v129 + v128);
                v13 = (unsigned __int64 *)(v126 + 16LL * v128);
                v130 = *v13;
                if ( v136 == *v13 )
                  goto LABEL_149;
                ++v129;
              }
              if ( v127 )
                v13 = v127;
            }
          }
          goto LABEL_149;
        }
LABEL_153:
        sub_2C93320(a3, 2 * v9);
        v117 = *(_DWORD *)(a3 + 24);
        if ( !v117 )
          goto LABEL_189;
        v118 = v117 - 1;
        v119 = *(_QWORD *)(a3 + 8);
        v116 = *(_DWORD *)(a3 + 16) + 1;
        v120 = v118 & (((unsigned int)v136 >> 9) ^ ((unsigned int)v136 >> 4));
        v13 = (unsigned __int64 *)(v119 + 16LL * v120);
        v121 = *v13;
        if ( v136 != *v13 )
        {
          v122 = 1;
          v123 = 0;
          while ( v121 != -4096 )
          {
            if ( !v123 && v121 == -8192 )
              v123 = v13;
            v120 = v118 & (v122 + v120);
            v13 = (unsigned __int64 *)(v119 + 16LL * v120);
            v121 = *v13;
            if ( v136 == *v13 )
              goto LABEL_149;
            ++v122;
          }
          if ( v123 )
            v13 = v123;
        }
LABEL_149:
        *(_DWORD *)(a3 + 16) = v116;
        if ( *v13 != -4096 )
          --*(_DWORD *)(a3 + 20);
        v13[1] = 0;
        v151 = 0;
        *v13 = v136;
        goto LABEL_6;
      }
LABEL_5:
      v151 = (__int64 *)v13[1];
LABEL_6:
      result = (unsigned __int64)v151;
      v15 = *v151;
      if ( *v151 == v151[1] )
        goto LABEL_54;
      v139 = 1;
      v137 = 0;
      v138 = ((unsigned int)v136 >> 9) ^ ((unsigned int)v136 >> 4);
      do
      {
        v16 = *(__int64 **)(v15 + 8 * v137);
        v17 = *(_QWORD *)(a4 + 8);
        result = *(unsigned int *)(a4 + 24);
        v152 = v16;
        if ( !(_DWORD)result )
          goto LABEL_70;
        v18 = (result - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v19 = (__int64 **)(v17 + 8LL * v18);
        v20 = *v19;
        if ( v16 != *v19 )
        {
          v69 = 1;
          while ( v20 != (__int64 *)-4096LL )
          {
            v75 = v69 + 1;
            v18 = (result - 1) & (v69 + v18);
            v19 = (__int64 **)(v17 + 8LL * v18);
            v20 = *v19;
            if ( v16 == *v19 )
              goto LABEL_10;
            v69 = v75;
          }
LABEL_70:
          v140 = v139;
LABEL_71:
          v137 = v140;
          result = (unsigned __int64)v151;
          v15 = *v151;
          v21 = v151[1];
          goto LABEL_53;
        }
LABEL_10:
        v140 = v139;
        v137 = v139;
        if ( v19 == (__int64 **)(v17 + 8LL * (unsigned int)result) )
          goto LABEL_71;
        v159 = 0;
        v160 = 0;
        v21 = v151[1];
        src = 0;
        v15 = *v151;
        if ( v139 == (v21 - *v151) >> 3 )
          goto LABEL_53;
        v22 = a1;
        v149 = v139;
        v23 = v139;
        while ( 1 )
        {
          v24 = *(_QWORD **)(v15 + 8 * v23);
          v153 = v24;
          if ( !(_DWORD)result )
            goto LABEL_28;
          v25 = (result - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
          v26 = (_QWORD *)(v17 + 8LL * v25);
          v27 = (_QWORD *)*v26;
          if ( v24 == (_QWORD *)*v26 )
          {
LABEL_15:
            if ( v26 == (_QWORD *)(v17 + 8 * result) )
              goto LABEL_28;
            v28 = *v152;
            v29 = v152[1] - *v152;
            if ( v29 != v24[1] - *v24 )
              goto LABEL_28;
            if ( v29 )
            {
              v142 = v24;
              v30 = 0;
              v31 = v22;
              v32 = 0;
              v33 = v152;
              do
              {
                v36 = *(_QWORD *)(v28 + 8 * v30);
                sub_2C9EEF0(v31, *(unsigned int **)v36, *(_QWORD ***)(v36 + 8), &v154, &v155, *(_QWORD *)(v31 + 200), 0);
                v37 = *(_QWORD *)(*v142 + 8 * v30);
                sub_2C9EEF0(v31, *(unsigned int **)v37, *(_QWORD ***)(v37 + 8), &v156, &v157, *(_QWORD *)(v31 + 200), 0);
                if ( v154 != v156
                  || v155 != v157
                  || (v38 = *(_QWORD *)(***(_QWORD ***)v36 + 8LL), *(_WORD *)(v38 + 24) != 5)
                  || (v34 = *(_QWORD *)(***(_QWORD ***)v37 + 8LL), *(_WORD *)(v34 + 24) != 5)
                  || (v35 = *(_QWORD *)(v34 + 40), *(_QWORD *)(v38 + 40) != v35)
                  || v35 != 2
                  || **(_QWORD **)(v38 + 32) != **(_QWORD **)(v34 + 32) )
                {
                  v22 = v31;
                  v21 = v151[1];
                  goto LABEL_28;
                }
                v28 = *v33;
                v30 = (unsigned int)++v32;
              }
              while ( v32 != (v33[1] - *v33) >> 3 );
              v22 = v31;
            }
            v39 = v159;
            if ( v159 == src )
            {
              if ( v159 == v160 )
              {
                sub_2C90710((__int64)&src, v159, &v152);
              }
              else
              {
                if ( v159 )
                {
                  *(_QWORD *)v159 = v152;
                  v39 = v159;
                }
                v159 = v39 + 8;
              }
              v40 = *(_QWORD *)(a4 + 8);
              v65 = *(_DWORD *)(a4 + 24);
              if ( v65 )
              {
                v42 = v65 - 1;
                v66 = (v65 - 1) & (((unsigned int)v152 >> 9) ^ ((unsigned int)v152 >> 4));
                v67 = (__int64 **)(v40 + 8LL * v66);
                v68 = *v67;
                if ( *v67 == v152 )
                {
LABEL_66:
                  *v67 = (__int64 *)-8192LL;
                  --*(_DWORD *)(a4 + 16);
                  ++*(_DWORD *)(a4 + 20);
                  goto LABEL_32;
                }
                v100 = 1;
                while ( v68 != (__int64 *)-4096LL )
                {
                  v101 = v100 + 1;
                  v66 = v42 & (v100 + v66);
                  v67 = (__int64 **)(v40 + 8LL * v66);
                  v68 = *v67;
                  if ( v152 == *v67 )
                    goto LABEL_66;
                  v100 = v101;
                }
LABEL_34:
                v43 = v42 & (((unsigned int)v153 >> 9) ^ ((unsigned int)v153 >> 4));
                v44 = (_QWORD *)(v40 + 8LL * v43);
                v45 = (_QWORD *)*v44;
                if ( v153 == (_QWORD *)*v44 )
                {
LABEL_35:
                  *v44 = -8192;
                  --*(_DWORD *)(a4 + 16);
                  ++*(_DWORD *)(a4 + 20);
                }
                else
                {
                  v73 = 1;
                  while ( v45 != (_QWORD *)-4096LL )
                  {
                    v74 = v73 + 1;
                    v43 = v42 & (v73 + v43);
                    v44 = (_QWORD *)(v40 + 8LL * v43);
                    v45 = (_QWORD *)*v44;
                    if ( v153 == (_QWORD *)*v44 )
                      goto LABEL_35;
                    v73 = v74;
                  }
                }
              }
            }
            else
            {
LABEL_32:
              v40 = *(_QWORD *)(a4 + 8);
              v41 = *(_DWORD *)(a4 + 24);
              if ( v41 )
              {
                v42 = v41 - 1;
                goto LABEL_34;
              }
            }
            v46 = v159;
            if ( v159 == v160 )
            {
              sub_2C90710((__int64)&src, v159, &v153);
              v21 = v151[1];
            }
            else
            {
              if ( v159 )
              {
                *(_QWORD *)v159 = v153;
                v46 = v159;
              }
              v159 = v46 + 8;
              v21 = v151[1];
            }
            goto LABEL_28;
          }
          v63 = 1;
          while ( v27 != (_QWORD *)-4096LL )
          {
            v64 = v63 + 1;
            v25 = (result - 1) & (v63 + v25);
            v26 = (_QWORD *)(v17 + 8LL * v25);
            v27 = (_QWORD *)*v26;
            if ( v24 == (_QWORD *)*v26 )
              goto LABEL_15;
            v63 = v64;
          }
LABEL_28:
          v23 = ++v149;
          v15 = *v151;
          if ( v149 == (v21 - *v151) >> 3 )
            break;
          v17 = *(_QWORD *)(a4 + 8);
          result = *(unsigned int *)(a4 + 24);
        }
        v47 = v159;
        v48 = (unsigned __int64)src;
        a1 = v22;
        if ( v159 == src )
          goto LABEL_50;
        v49 = (_BYTE *)*(unsigned int *)(a5 + 24);
        if ( !(_DWORD)v49 )
        {
          ++*(_QWORD *)a5;
          goto LABEL_125;
        }
        v50 = *(_QWORD *)(a5 + 8);
        v51 = 1;
        v52 = 0;
        v53 = ((_DWORD)v49 - 1) & v138;
        v54 = (_QWORD *)(v50 + 32 * v53);
        v55 = *v54;
        if ( v136 != *v54 )
        {
          while ( v55 != -4096 )
          {
            if ( !v52 && v55 == -8192 )
              v52 = v54;
            v53 = ((_DWORD)v49 - 1) & (unsigned int)(v51 + v53);
            v54 = (_QWORD *)(v50 + 32LL * (unsigned int)v53);
            v55 = *v54;
            if ( v136 == *v54 )
              goto LABEL_43;
            ++v51;
          }
          if ( !v52 )
            v52 = v54;
          ++*(_QWORD *)a5;
          v76 = *(_DWORD *)(a5 + 16) + 1;
          if ( 4 * v76 >= (unsigned int)(3 * (_DWORD)v49) )
          {
LABEL_125:
            sub_2C943F0(a5, 2 * (_DWORD)v49);
            v102 = *(_DWORD *)(a5 + 24);
            if ( v102 )
            {
              v103 = v102 - 1;
              v104 = *(_QWORD *)(a5 + 8);
              v105 = v103 & v138;
              v76 = *(_DWORD *)(a5 + 16) + 1;
              v52 = (_QWORD *)(v104 + 32LL * (v103 & v138));
              v106 = *v52;
              if ( v136 == *v52 )
                goto LABEL_94;
              v107 = 1;
              v53 = 0;
              while ( v106 != -4096 )
              {
                if ( !v53 && v106 == -8192 )
                  v53 = (unsigned __int64)v52;
                v105 = v103 & (v107 + v105);
                v52 = (_QWORD *)(v104 + 32LL * v105);
                v106 = *v52;
                if ( v136 == *v52 )
                  goto LABEL_94;
                ++v107;
              }
LABEL_138:
              if ( v53 )
                v52 = (_QWORD *)v53;
              goto LABEL_94;
            }
          }
          else
          {
            v53 = (unsigned int)((_DWORD)v49 - *(_DWORD *)(a5 + 20) - v76);
            if ( (unsigned int)v53 > (unsigned int)v49 >> 3 )
            {
LABEL_94:
              v50 = a5;
              *(_DWORD *)(a5 + 16) = v76;
              if ( *v52 != -4096 )
                --*(_DWORD *)(a5 + 20);
              v58 = v52 + 1;
              v57 = 0;
              *v58 = 0;
              v58[1] = 0;
              *(v58 - 1) = v136;
              v58[2] = 0;
LABEL_97:
              v72 = *v58;
              v77 = (__int64)v57 - *v58;
              v78 = 0xAAAAAAAAAAAAAAABLL * (v77 >> 3);
              if ( v78 == 0x555555555555555LL )
                sub_4262D8((__int64)"vector::_M_realloc_insert");
              v79 = 1;
              if ( v78 )
                v79 = 0xAAAAAAAAAAAAAAABLL * ((__int64)((__int64)v57 - *v58) >> 3);
              v80 = __CFADD__(v79, v78);
              v81 = v79 - 0x5555555555555555LL * ((__int64)((__int64)v57 - *v58) >> 3);
              v148 = v81;
              if ( !v80 )
              {
                v150 = 0;
                if ( v81 )
                {
                  v82 = 0x555555555555555LL;
                  if ( v81 <= 0x555555555555555LL )
                    v82 = v79 - 0x5555555555555555LL * ((__int64)((__int64)v57 - *v58) >> 3);
                  v148 = v82;
                  v50 = 24 * v82;
LABEL_105:
                  v144 = v57;
                  v83 = sub_22077B0(v50);
                  v57 = v144;
                  v150 = (unsigned __int64 *)v83;
                }
                v84 = (unsigned __int64 *)((char *)v150 + v77) == 0;
                v85 = (unsigned __int64 *)((char *)v150 + v77);
                v86 = v85;
                if ( !v84 )
                {
                  v87 = v159;
                  v49 = src;
                  *v85 = 0;
                  v85[1] = 0;
                  v85[2] = 0;
                  v88 = v87 - v49;
                  if ( v87 == v49 )
                  {
                    v91 = 0;
                    v90 = 0;
                  }
                  else
                  {
                    v143 = v57;
                    if ( v88 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_177:
                      sub_4261EA(v50, v49, v53);
                    v145 = v87 - v49;
                    v89 = sub_22077B0(v88);
                    v49 = src;
                    v88 = v145;
                    v90 = (char *)v89;
                    v87 = v159;
                    v57 = v143;
                    v91 = v159 - (_BYTE *)src;
                  }
                  *v86 = v90;
                  v86[1] = v90;
                  v86[2] = &v90[v88];
                  if ( v49 != v87 )
                  {
                    v146 = v57;
                    v92 = (char *)memmove(v90, v49, v91);
                    v57 = v146;
                    v90 = v92;
                  }
                  v86[1] = &v90[v91];
                }
                v71 = v150;
                if ( v57 != (unsigned __int64 *)v72 )
                {
                  v93 = (unsigned __int64 *)v72;
                  v147 = v58;
                  v94 = v150;
                  v95 = v57;
                  v96 = v93;
                  do
                  {
                    v98 = v93[2];
                    v99 = *v93;
                    if ( v94 )
                    {
                      *v94 = v99;
                      v97 = v93[1];
                      v94[2] = v98;
                      v94[1] = v97;
                      v93[2] = 0;
                      *v93 = 0;
                    }
                    else if ( v99 )
                    {
                      j_j___libc_free_0(v99);
                    }
                    v93 += 3;
                    v94 += 3;
                  }
                  while ( v95 != v93 );
                  v70 = (unsigned __int64)v96;
                  v58 = v147;
                  v71 = v94;
                  v72 = v70;
                }
                if ( v72 )
                  j_j___libc_free_0(v72);
                v58[1] = (unsigned __int64)(v71 + 3);
                *v58 = (unsigned __int64)v150;
                v48 = (unsigned __int64)src;
                v58[2] = (unsigned __int64)&v150[3 * v148];
                goto LABEL_50;
              }
              v50 = 0x7FFFFFFFFFFFFFF8LL;
              v148 = 0x555555555555555LL;
              goto LABEL_105;
            }
            sub_2C943F0(a5, (int)v49);
            v108 = *(_DWORD *)(a5 + 24);
            if ( v108 )
            {
              v109 = v108 - 1;
              v110 = *(_QWORD *)(a5 + 8);
              v53 = 0;
              v111 = 1;
              v112 = v109 & v138;
              v76 = *(_DWORD *)(a5 + 16) + 1;
              v52 = (_QWORD *)(v110 + 32LL * (v109 & v138));
              v113 = *v52;
              if ( v136 == *v52 )
                goto LABEL_94;
              while ( v113 != -4096 )
              {
                if ( !v53 && v113 == -8192 )
                  v53 = (unsigned __int64)v52;
                v112 = v109 & (v111 + v112);
                v52 = (_QWORD *)(v110 + 32LL * v112);
                v113 = *v52;
                if ( v136 == *v52 )
                  goto LABEL_94;
                ++v111;
              }
              goto LABEL_138;
            }
          }
          ++*(_DWORD *)(a5 + 16);
          BUG();
        }
LABEL_43:
        v56 = (unsigned __int64 *)v54[2];
        v57 = (unsigned __int64 *)v54[3];
        v58 = v54 + 1;
        if ( v56 == v57 )
          goto LABEL_97;
        if ( !v56 )
          goto LABEL_49;
        *v56 = 0;
        v59 = (unsigned __int64)&v47[-v48];
        v56[1] = 0;
        v56[2] = 0;
        if ( (unsigned __int64)&v47[-v48] > 0x7FFFFFFFFFFFFFF8LL )
          goto LABEL_177;
        v60 = (char *)sub_22077B0(v59);
        *v56 = (unsigned __int64)v60;
        v61 = v60;
        v48 = (unsigned __int64)src;
        v56[1] = (unsigned __int64)v60;
        v56[2] = (unsigned __int64)&v60[v59];
        v62 = &v159[-v48];
        if ( v159 != (_BYTE *)v48 )
          v61 = (char *)memmove(v60, (const void *)v48, (size_t)&v159[-v48]);
        v56[1] = (unsigned __int64)&v61[(_QWORD)v62];
        v56 = (unsigned __int64 *)v54[2];
LABEL_49:
        v54[2] = v56 + 3;
LABEL_50:
        if ( v48 )
          j_j___libc_free_0(v48);
        result = (unsigned __int64)v151;
        v15 = *v151;
        v21 = v151[1];
LABEL_53:
        ++v139;
      }
      while ( (v21 - v15) >> 3 != v140 );
LABEL_54:
      if ( v134 == v132 )
        return result;
      v134 += 8;
      result = *a2;
    }
    ++*(_QWORD *)a3;
    goto LABEL_153;
  }
  return result;
}
