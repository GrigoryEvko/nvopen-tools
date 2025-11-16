// Function: sub_1C66A70
// Address: 0x1c66a70
//
unsigned __int64 __fastcall sub_1C66A70(_QWORD *a1, unsigned __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 result; // rax
  __int64 v6; // rdx
  __int64 v8; // rdx
  unsigned int v9; // esi
  __int64 v10; // r8
  unsigned int v11; // ecx
  _QWORD *v12; // rax
  __int64 v13; // rdi
  _QWORD *v14; // r14
  __int64 v15; // rcx
  __int64 *v16; // rdx
  __int64 v17; // rsi
  unsigned int v18; // ecx
  __int64 **v19; // rdi
  __int64 *v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rdi
  _QWORD *v23; // r13
  unsigned int v24; // ecx
  _QWORD *v25; // rdi
  _QWORD *v26; // r8
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // r12
  _QWORD *v30; // r13
  int v31; // r15d
  __int64 *v32; // r14
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rbx
  __int64 v36; // r12
  __int64 v37; // rax
  _BYTE *v38; // rsi
  int v39; // eax
  __int64 v40; // rdi
  int v41; // ecx
  unsigned int v42; // eax
  _QWORD *v43; // rsi
  _QWORD *v44; // r8
  _BYTE *v45; // rsi
  _BYTE *v46; // rax
  _BYTE *v47; // r12
  _BYTE *v48; // rsi
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // rdx
  __int64 *v52; // rbx
  __int64 v53; // rdi
  __int64 *v54; // r15
  __int64 v55; // r13
  char *v56; // rax
  char *v57; // rcx
  __int64 v58; // r13
  __int64 v59; // rsi
  int v60; // eax
  unsigned int v61; // eax
  __int64 **v62; // rsi
  __int64 *v63; // r8
  int v64; // edi
  int v65; // edi
  int v66; // r10d
  int v67; // esi
  int v68; // r9d
  int v69; // esi
  int v70; // r9d
  int v71; // r13d
  __int64 *v72; // r11
  int v73; // eax
  __int64 v74; // r12
  __int64 v75; // rax
  _QWORD *v76; // r12
  _BYTE *v77; // rax
  unsigned __int64 v78; // r8
  __int64 v79; // rax
  char *v80; // rdi
  size_t v81; // r13
  __int64 *v82; // r13
  _QWORD *i; // r12
  __int64 v84; // rax
  __int64 v85; // rsi
  __int64 v86; // rdi
  __int64 v87; // rsi
  int v88; // r11d
  int v89; // r11d
  __int64 v90; // r10
  unsigned int v91; // edx
  __int64 v92; // r8
  int v93; // edi
  __int64 *v94; // rsi
  int v95; // r11d
  int v96; // r11d
  __int64 v97; // r10
  int v98; // edi
  unsigned int v99; // edx
  __int64 v100; // r8
  int v101; // ebx
  _QWORD *v102; // r11
  int v103; // ecx
  __int64 v104; // rbx
  int v105; // r10d
  unsigned __int64 v106; // rcx
  unsigned __int64 v107; // rax
  __int64 v109; // [rsp+20h] [rbp-110h]
  __int64 v111; // [rsp+30h] [rbp-100h]
  __int64 v113; // [rsp+48h] [rbp-E8h]
  unsigned int v114; // [rsp+54h] [rbp-DCh]
  __int64 v115; // [rsp+58h] [rbp-D8h]
  _QWORD *v117; // [rsp+68h] [rbp-C8h]
  unsigned __int64 v118; // [rsp+78h] [rbp-B8h]
  __int64 v119; // [rsp+80h] [rbp-B0h]
  unsigned __int64 v120; // [rsp+88h] [rbp-A8h]
  unsigned int v121; // [rsp+90h] [rbp-A0h]
  __int64 *v122; // [rsp+90h] [rbp-A0h]
  __int64 *v123; // [rsp+98h] [rbp-98h]
  __int64 v124; // [rsp+A8h] [rbp-88h] BYREF
  __int64 *v125; // [rsp+B0h] [rbp-80h] BYREF
  _QWORD *v126; // [rsp+B8h] [rbp-78h] BYREF
  __int64 v127; // [rsp+C0h] [rbp-70h] BYREF
  unsigned __int64 v128; // [rsp+C8h] [rbp-68h] BYREF
  __int64 v129; // [rsp+D0h] [rbp-60h] BYREF
  unsigned __int64 v130; // [rsp+D8h] [rbp-58h] BYREF
  void *src; // [rsp+E0h] [rbp-50h] BYREF
  _BYTE *v132; // [rsp+E8h] [rbp-48h]
  _BYTE *v133; // [rsp+F0h] [rbp-40h]

  result = *a2;
  v6 = (__int64)(a2[1] - *a2) >> 3;
  if ( !(_DWORD)v6 )
    return result;
  v111 = 0;
  v109 = 8LL * (unsigned int)(v6 - 1);
  while ( 2 )
  {
    v8 = *(_QWORD *)(result + v111);
    v9 = *(_DWORD *)(a3 + 24);
    v124 = v8;
    if ( !v9 )
    {
      ++*(_QWORD *)a3;
LABEL_142:
      v104 = a3;
      v9 *= 2;
      goto LABEL_143;
    }
    v10 = *(_QWORD *)(a3 + 8);
    v11 = (v9 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
    v12 = (_QWORD *)(v10 + 16LL * v11);
    v13 = *v12;
    if ( v8 == *v12 )
    {
LABEL_5:
      v123 = (__int64 *)v12[1];
      goto LABEL_6;
    }
    v101 = 1;
    v102 = 0;
    while ( v13 != -8 )
    {
      if ( !v102 && v13 == -16 )
        v102 = v12;
      v11 = (v9 - 1) & (v101 + v11);
      v12 = (_QWORD *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( v8 == *v12 )
        goto LABEL_5;
      ++v101;
    }
    if ( v102 )
      v12 = v102;
    ++*(_QWORD *)a3;
    v103 = *(_DWORD *)(a3 + 16) + 1;
    if ( 4 * v103 >= 3 * v9 )
      goto LABEL_142;
    v104 = a3;
    if ( v9 - *(_DWORD *)(a3 + 20) - v103 <= v9 >> 3 )
    {
LABEL_143:
      sub_1C532A0(v104, v9);
      sub_1C507A0(v104, &v124, &src);
      v12 = src;
      v8 = v124;
      v103 = *(_DWORD *)(v104 + 16) + 1;
    }
    *(_DWORD *)(a3 + 16) = v103;
    if ( *v12 != -8 )
      --*(_DWORD *)(a3 + 20);
    *v12 = v8;
    v12[1] = 0;
    v123 = 0;
LABEL_6:
    result = (unsigned __int64)v123;
    v14 = a1;
    v114 = 1;
    v113 = 0;
    v15 = *v123;
    if ( *v123 == v123[1] )
      goto LABEL_54;
    do
    {
      v16 = *(__int64 **)(v15 + 8 * v113);
      result = *(unsigned int *)(a4 + 24);
      v125 = v16;
      if ( !(_DWORD)result )
        goto LABEL_66;
      v17 = *(_QWORD *)(a4 + 8);
      v18 = (result - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v19 = (__int64 **)(v17 + 8LL * v18);
      v20 = *v19;
      if ( v16 != *v19 )
      {
        v64 = 1;
        while ( v20 != (__int64 *)-8LL )
        {
          v105 = v64 + 1;
          v18 = (result - 1) & (v64 + v18);
          v19 = (__int64 **)(v17 + 8LL * v18);
          v20 = *v19;
          if ( v16 == *v19 )
            goto LABEL_9;
          v64 = v105;
        }
LABEL_66:
        v115 = v114;
LABEL_67:
        v113 = v115;
        result = (unsigned __int64)v123;
        v15 = *v123;
        v21 = v123[1];
        goto LABEL_52;
      }
LABEL_9:
      v115 = v114;
      v113 = v114;
      if ( v19 == (__int64 **)(v17 + 8LL * (unsigned int)result) )
        goto LABEL_67;
      src = 0;
      v132 = 0;
      v133 = 0;
      v21 = v123[1];
      v15 = *v123;
      if ( v114 == (v21 - *v123) >> 3 )
        goto LABEL_52;
      v121 = v114;
      v22 = v114;
      while ( 1 )
      {
        v23 = *(_QWORD **)(v15 + 8 * v22);
        v126 = v23;
        if ( !(_DWORD)result )
          goto LABEL_27;
        v24 = (result - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v25 = (_QWORD *)(v17 + 8LL * v24);
        v26 = (_QWORD *)*v25;
        if ( v23 == (_QWORD *)*v25 )
        {
LABEL_14:
          if ( v25 == (_QWORD *)(v17 + 8 * result) )
            goto LABEL_27;
          v27 = *v125;
          v28 = v125[1] - *v125;
          if ( v28 != v23[1] - *v23 )
            goto LABEL_27;
          if ( v28 )
          {
            v117 = v23;
            v29 = 0;
            v30 = v14;
            v31 = 0;
            v32 = v125;
            do
            {
              v35 = *(_QWORD *)(v27 + 8 * v29);
              sub_1C620D0(v30, *(unsigned int **)v35, *(__int64 ***)(v35 + 8), &v127, &v128, v30[25], 0);
              v36 = *(_QWORD *)(*v117 + 8 * v29);
              sub_1C620D0(v30, *(unsigned int **)v36, *(__int64 ***)(v36 + 8), &v129, &v130, v30[25], 0);
              if ( v127 != v129
                || v128 != v130
                || (v37 = *(_QWORD *)(***(_QWORD ***)v35 + 8LL), *(_WORD *)(v37 + 24) != 4)
                || (v33 = *(_QWORD *)(***(_QWORD ***)v36 + 8LL), *(_WORD *)(v33 + 24) != 4)
                || (v34 = *(_QWORD *)(v33 + 40), *(_QWORD *)(v37 + 40) != v34)
                || v34 != 2
                || **(_QWORD **)(v37 + 32) != **(_QWORD **)(v33 + 32) )
              {
                v14 = v30;
                v21 = v123[1];
                goto LABEL_27;
              }
              v27 = *v32;
              v29 = (unsigned int)++v31;
            }
            while ( v31 != (v32[1] - *v32) >> 3 );
            v14 = v30;
          }
          v38 = v132;
          if ( v132 == src )
          {
            if ( v132 == v133 )
            {
              sub_1C50BF0((__int64)&src, v132, &v125);
            }
            else
            {
              if ( v132 )
              {
                *(_QWORD *)v132 = v125;
                v38 = v132;
              }
              v132 = v38 + 8;
            }
            v60 = *(_DWORD *)(a4 + 24);
            if ( v60 )
            {
              v41 = v60 - 1;
              v40 = *(_QWORD *)(a4 + 8);
              v61 = (v60 - 1) & (((unsigned int)v125 >> 9) ^ ((unsigned int)v125 >> 4));
              v62 = (__int64 **)(v40 + 8LL * v61);
              v63 = *v62;
              if ( *v62 == v125 )
              {
LABEL_62:
                *v62 = (__int64 *)-16LL;
                --*(_DWORD *)(a4 + 16);
                ++*(_DWORD *)(a4 + 20);
                goto LABEL_31;
              }
              v69 = 1;
              while ( v63 != (__int64 *)-8LL )
              {
                v70 = v69 + 1;
                v61 = v41 & (v69 + v61);
                v62 = (__int64 **)(v40 + 8LL * v61);
                v63 = *v62;
                if ( v125 == *v62 )
                  goto LABEL_62;
                v69 = v70;
              }
LABEL_33:
              v42 = v41 & (((unsigned int)v126 >> 9) ^ ((unsigned int)v126 >> 4));
              v43 = (_QWORD *)(v40 + 8LL * v42);
              v44 = (_QWORD *)*v43;
              if ( v126 == (_QWORD *)*v43 )
              {
LABEL_34:
                *v43 = -16;
                --*(_DWORD *)(a4 + 16);
                ++*(_DWORD *)(a4 + 20);
              }
              else
              {
                v67 = 1;
                while ( v44 != (_QWORD *)-8LL )
                {
                  v68 = v67 + 1;
                  v42 = v41 & (v67 + v42);
                  v43 = (_QWORD *)(v40 + 8LL * v42);
                  v44 = (_QWORD *)*v43;
                  if ( v126 == (_QWORD *)*v43 )
                    goto LABEL_34;
                  v67 = v68;
                }
              }
            }
          }
          else
          {
LABEL_31:
            v39 = *(_DWORD *)(a4 + 24);
            if ( v39 )
            {
              v40 = *(_QWORD *)(a4 + 8);
              v41 = v39 - 1;
              goto LABEL_33;
            }
          }
          v45 = v132;
          if ( v132 == v133 )
          {
            sub_1C50BF0((__int64)&src, v132, &v126);
            v21 = v123[1];
          }
          else
          {
            if ( v132 )
            {
              *(_QWORD *)v132 = v126;
              v45 = v132;
            }
            v132 = v45 + 8;
            v21 = v123[1];
          }
          goto LABEL_27;
        }
        v65 = 1;
        while ( v26 != (_QWORD *)-8LL )
        {
          v66 = v65 + 1;
          v24 = (result - 1) & (v65 + v24);
          v25 = (_QWORD *)(v17 + 8LL * v24);
          v26 = (_QWORD *)*v25;
          if ( v23 == (_QWORD *)*v25 )
            goto LABEL_14;
          v65 = v66;
        }
LABEL_27:
        v22 = ++v121;
        v15 = *v123;
        if ( v121 == (v21 - *v123) >> 3 )
          break;
        v17 = *(_QWORD *)(a4 + 8);
        result = *(unsigned int *)(a4 + 24);
      }
      v46 = v132;
      v47 = src;
      if ( v132 == src )
        goto LABEL_75;
      v48 = (_BYTE *)*(unsigned int *)(a5 + 24);
      if ( !(_DWORD)v48 )
      {
        ++*(_QWORD *)a5;
        goto LABEL_113;
      }
      v49 = v124;
      v50 = *(_QWORD *)(a5 + 8);
      v51 = ((_DWORD)v48 - 1) & (((unsigned int)v124 >> 9) ^ ((unsigned int)v124 >> 4));
      v52 = (__int64 *)(v50 + 32 * v51);
      v53 = *v52;
      if ( v124 != *v52 )
      {
        v71 = 1;
        v72 = 0;
        while ( v53 != -8 )
        {
          if ( !v72 && v53 == -16 )
            v72 = v52;
          v51 = ((_DWORD)v48 - 1) & (unsigned int)(v51 + v71);
          v52 = (__int64 *)(v50 + 32 * v51);
          v53 = *v52;
          if ( v124 == *v52 )
            goto LABEL_42;
          ++v71;
        }
        if ( v72 )
          v52 = v72;
        ++*(_QWORD *)a5;
        v73 = *(_DWORD *)(a5 + 16) + 1;
        if ( 4 * v73 < (unsigned int)(3 * (_DWORD)v48) )
        {
          if ( (int)v48 - *(_DWORD *)(a5 + 20) - v73 > (unsigned int)v48 >> 3 )
          {
LABEL_91:
            *(_DWORD *)(a5 + 16) = v73;
            if ( *v52 != -8 )
              --*(_DWORD *)(a5 + 20);
            *v52 = v49;
            v74 = 0;
            v54 = 0;
            v52[1] = 0;
            v52[2] = 0;
            v52[3] = 0;
            v120 = 1;
            v122 = 0;
            goto LABEL_94;
          }
          sub_1C53C20(a5, (int)v48);
          v95 = *(_DWORD *)(a5 + 24);
          if ( v95 )
          {
            v49 = v124;
            v96 = v95 - 1;
            v97 = *(_QWORD *)(a5 + 8);
            v94 = 0;
            v98 = 1;
            v99 = v96 & (((unsigned int)v124 >> 9) ^ ((unsigned int)v124 >> 4));
            v73 = *(_DWORD *)(a5 + 16) + 1;
            v52 = (__int64 *)(v97 + 32LL * v99);
            v100 = *v52;
            if ( *v52 == v124 )
              goto LABEL_91;
            while ( v100 != -8 )
            {
              if ( v100 == -16 && !v94 )
                v94 = v52;
              v99 = v96 & (v98 + v99);
              v52 = (__int64 *)(v97 + 32LL * v99);
              v100 = *v52;
              if ( v124 == *v52 )
                goto LABEL_91;
              ++v98;
            }
            goto LABEL_117;
          }
          goto LABEL_168;
        }
LABEL_113:
        sub_1C53C20(a5, 2 * (_DWORD)v48);
        v88 = *(_DWORD *)(a5 + 24);
        if ( v88 )
        {
          v49 = v124;
          v89 = v88 - 1;
          v90 = *(_QWORD *)(a5 + 8);
          v91 = v89 & (((unsigned int)v124 >> 9) ^ ((unsigned int)v124 >> 4));
          v73 = *(_DWORD *)(a5 + 16) + 1;
          v52 = (__int64 *)(v90 + 32LL * v91);
          v92 = *v52;
          if ( *v52 == v124 )
            goto LABEL_91;
          v93 = 1;
          v94 = 0;
          while ( v92 != -8 )
          {
            if ( !v94 && v92 == -16 )
              v94 = v52;
            v91 = v89 & (v93 + v91);
            v52 = (__int64 *)(v90 + 32LL * v91);
            v92 = *v52;
            if ( v124 == *v52 )
              goto LABEL_91;
            ++v93;
          }
LABEL_117:
          if ( v94 )
            v52 = v94;
          goto LABEL_91;
        }
LABEL_168:
        ++*(_DWORD *)(a5 + 16);
        BUG();
      }
LABEL_42:
      v54 = (__int64 *)v52[2];
      if ( v54 == (__int64 *)v52[3] )
      {
        v74 = (__int64)v54 - v52[1];
        v122 = (__int64 *)v52[1];
        v106 = 0xAAAAAAAAAAAAAAABLL * (v74 >> 3);
        if ( v106 == 0x555555555555555LL )
          sub_4262D8((__int64)"vector::_M_realloc_insert");
        v107 = 1;
        if ( v106 )
          v107 = 0xAAAAAAAAAAAAAAABLL * (v74 >> 3);
        v120 = v106 + v107;
        if ( __CFADD__(v106, v107) )
        {
          v53 = 0x7FFFFFFFFFFFFFF8LL;
          v120 = 0x555555555555555LL;
LABEL_97:
          v119 = sub_22077B0(v53);
          goto LABEL_98;
        }
        if ( !(v106 + v107) )
        {
          v119 = 0;
LABEL_98:
          v76 = (_QWORD *)(v119 + v74);
          if ( v76 )
          {
            v77 = v132;
            v48 = src;
            *v76 = 0;
            v76[1] = 0;
            v76[2] = 0;
            v78 = v77 - v48;
            if ( v77 == v48 )
            {
              v81 = 0;
              v80 = 0;
            }
            else
            {
              if ( v78 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_162:
                sub_4261EA(v53, v48, v51);
              v118 = v77 - v48;
              v79 = sub_22077B0(v78);
              v48 = src;
              v78 = v118;
              v80 = (char *)v79;
              v77 = v132;
              v81 = v132 - (_BYTE *)src;
            }
            *v76 = v80;
            v76[1] = v80;
            v76[2] = &v80[v78];
            if ( v48 != v77 )
              v80 = (char *)memmove(v80, v48, v81);
            v76[1] = &v80[v81];
          }
          v82 = v122;
          for ( i = (_QWORD *)v119; v54 != v82; i += 3 )
          {
            v85 = v82[2];
            v86 = *v82;
            if ( i )
            {
              *i = v86;
              v84 = v82[1];
              i[2] = v85;
              i[1] = v84;
              v82[2] = 0;
              *v82 = 0;
            }
            else
            {
              v87 = v85 - v86;
              if ( v86 )
                j_j___libc_free_0(v86, v87);
            }
            v82 += 3;
          }
          if ( v122 )
            j_j___libc_free_0(v122, v52[3] - (_QWORD)v122);
          v52[2] = (__int64)(i + 3);
          v52[1] = v119;
          v52[3] = v119 + 24 * v120;
          v47 = src;
LABEL_75:
          v59 = v133 - v47;
          goto LABEL_49;
        }
LABEL_94:
        v75 = 0x555555555555555LL;
        if ( v120 <= 0x555555555555555LL )
          v75 = v120;
        v120 = v75;
        v53 = 24 * v75;
        goto LABEL_97;
      }
      if ( !v54 )
        goto LABEL_48;
      *v54 = 0;
      v55 = v46 - v47;
      v54[1] = 0;
      v54[2] = 0;
      if ( (unsigned __int64)(v46 - v47) > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_162;
      v56 = (char *)sub_22077B0(v55);
      *v54 = (__int64)v56;
      v57 = v56;
      v54[1] = (__int64)v56;
      v54[2] = (__int64)&v56[v55];
      v47 = src;
      v58 = v132 - (_BYTE *)src;
      if ( v132 != src )
        v57 = (char *)memmove(v56, src, v132 - (_BYTE *)src);
      v54[1] = (__int64)&v57[v58];
      v54 = (__int64 *)v52[2];
LABEL_48:
      v52[2] = (__int64)(v54 + 3);
      v59 = v133 - v47;
LABEL_49:
      if ( v47 )
        j_j___libc_free_0(v47, v59);
      result = (unsigned __int64)v123;
      v15 = *v123;
      v21 = v123[1];
LABEL_52:
      ++v114;
    }
    while ( (v21 - v15) >> 3 != v115 );
    a1 = v14;
LABEL_54:
    if ( v111 != v109 )
    {
      v111 += 8;
      result = *a2;
      continue;
    }
    return result;
  }
}
