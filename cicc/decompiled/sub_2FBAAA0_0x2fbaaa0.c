// Function: sub_2FBAAA0
// Address: 0x2fbaaa0
//
void __fastcall sub_2FBAAA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  __int64 v10; // rax
  unsigned __int64 v11; // r13
  _QWORD *i; // rax
  unsigned int v13; // ebx
  _OWORD *v14; // rdx
  _QWORD *v15; // rax
  __int64 j; // rdx
  __int64 *v17; // rbx
  __int64 v18; // r14
  __int64 v19; // r13
  __int64 v20; // rdi
  __int64 *v21; // rcx
  __int64 v22; // rdi
  unsigned int *v23; // r13
  unsigned int **v24; // rax
  __int64 v25; // r14
  __int64 v26; // r14
  unsigned __int64 m; // rbx
  __int64 *v28; // rdx
  __int64 v29; // rcx
  unsigned __int64 v30; // r8
  __int64 *v31; // rax
  __int64 *v32; // r15
  unsigned int *v33; // r13
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rsi
  __int64 *v41; // r15
  int v42; // edi
  int v43; // edi
  unsigned int k; // eax
  __int64 v45; // rsi
  __int64 v46; // rax
  __int64 v47; // rdi
  unsigned int v48; // r8d
  __int64 v49; // rcx
  __int64 v50; // rax
  __int64 v51; // r11
  unsigned int v52; // r10d
  __int64 v53; // rcx
  __int64 v54; // rdx
  __int64 v55; // rax
  bool v56; // cf
  __int64 v57; // rax
  int v58; // esi
  int v59; // edi
  int v60; // r15d
  _DWORD *v61; // r10
  __int64 *v62; // rbx
  __int64 v63; // r15
  __int64 v64; // r14
  __int64 v65; // rdi
  int *v66; // r14
  _QWORD *v67; // rax
  unsigned int v68; // eax
  int v69; // edi
  __int64 v70; // rax
  unsigned __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rdx
  _QWORD *v74; // r9
  unsigned int v75; // edi
  __int64 v76; // r8
  __int64 *v77; // rcx
  int v78; // eax
  __int64 v79; // rax
  __int64 v80; // r9
  __int64 v81; // rsi
  int v82; // esi
  int v83; // edi
  int v84; // r15d
  __int64 v85; // r10
  int v86; // eax
  __int64 v87; // rsi
  bool v88; // al
  __int64 v89; // rax
  __int64 v90; // [rsp+10h] [rbp-1A0h]
  __int64 v91; // [rsp+18h] [rbp-198h]
  __int64 v92; // [rsp+20h] [rbp-190h]
  __int64 *v93; // [rsp+28h] [rbp-188h]
  __int64 v94; // [rsp+28h] [rbp-188h]
  _QWORD *v95; // [rsp+28h] [rbp-188h]
  __int64 *v96; // [rsp+28h] [rbp-188h]
  __int64 v97; // [rsp+30h] [rbp-180h] BYREF
  __int64 v98; // [rsp+38h] [rbp-178h]
  __int64 v99; // [rsp+40h] [rbp-170h]
  __int64 v100; // [rsp+48h] [rbp-168h]
  _QWORD *v101; // [rsp+50h] [rbp-160h] BYREF
  __int64 v102; // [rsp+58h] [rbp-158h]
  _QWORD v103[8]; // [rsp+60h] [rbp-150h] BYREF
  _BYTE *v104; // [rsp+A0h] [rbp-110h] BYREF
  __int64 v105; // [rsp+A8h] [rbp-108h]
  _BYTE v106[64]; // [rsp+B0h] [rbp-100h] BYREF
  _OWORD *v107; // [rsp+F0h] [rbp-C0h] BYREF
  __int64 v108; // [rsp+F8h] [rbp-B8h]
  _OWORD v109[11]; // [rsp+100h] [rbp-B0h] BYREF

  v90 = sub_2DF8570(
          *(_QWORD *)(a1 + 8),
          *(_DWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 72) + 16LL) + 4LL * *(unsigned int *)(*(_QWORD *)(a1 + 72) + 64LL)),
          *(unsigned int *)(*(_QWORD *)(a1 + 72) + 64LL),
          *(_QWORD *)(*(_QWORD *)(a1 + 72) + 16LL),
          a5,
          a6);
  v10 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
  v108 = 0x800000000LL;
  v11 = *(unsigned int *)(v10 + 72);
  v91 = v10;
  i = v109;
  v107 = v109;
  v13 = v11;
  if ( v11 )
  {
    v7 = (__int64)v109;
    if ( v11 > 8 )
    {
      sub_C8D5F0((__int64)&v107, v109, v11, 0x10u, v8, v9);
      v7 = (__int64)v107;
      v14 = &v107[v11];
      for ( i = &v107[(unsigned int)v108]; v14 != (_OWORD *)i; i += 2 )
      {
LABEL_4:
        if ( i )
        {
          *i = 0;
          i[1] = 0;
        }
      }
    }
    else
    {
      v14 = &v109[v11];
      if ( v14 != v109 )
        goto LABEL_4;
    }
    LODWORD(v108) = v11;
    v13 = *(_DWORD *)(v91 + 72);
  }
  v15 = v103;
  j = 0x800000000LL;
  v101 = v103;
  v102 = 0x800000000LL;
  if ( !v13 )
    goto LABEL_15;
  if ( v13 > 8uLL )
  {
    sub_C8D5F0((__int64)&v101, v103, v13, 8u, v8, v9);
    v15 = &v101[(unsigned int)v102];
    for ( j = (__int64)&v101[v13]; (_QWORD *)j != v15; ++v15 )
    {
LABEL_11:
      if ( v15 )
        *v15 = 0;
    }
  }
  else
  {
    j = (__int64)&v103[v13];
    if ( v103 != (_QWORD *)j )
      goto LABEL_11;
  }
  LODWORD(v102) = v13;
LABEL_15:
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v17 = *(__int64 **)(v90 + 64);
  v100 = 0;
  v93 = &v17[*(unsigned int *)(v90 + 72)];
  if ( v17 == v93 )
    goto LABEL_25;
  while ( 1 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        while ( 1 )
        {
LABEL_16:
          v18 = *v17;
          v19 = *(_QWORD *)(*v17 + 8);
          if ( (v19 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            goto LABEL_24;
          v20 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
          v21 = (__int64 *)sub_2E09D00((__int64 *)v20, *(_QWORD *)(*v17 + 8));
          if ( v21 != (__int64 *)(*(_QWORD *)v20 + 24LL * *(unsigned int *)(v20 + 8))
            && (*(_DWORD *)((*v21 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v21 >> 1) & 3) <= (*(_DWORD *)((v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v19 >> 1) & 3) )
          {
            break;
          }
          v22 = *(_QWORD *)(a1 + 72);
          v23 = 0;
          if ( *(_BYTE *)(v22 + 172) )
            goto LABEL_20;
LABEL_41:
          if ( !sub_C8CA60(v22 + 144, (__int64)v23) )
            goto LABEL_42;
LABEL_24:
          if ( v93 == ++v17 )
            goto LABEL_25;
        }
        v22 = *(_QWORD *)(a1 + 72);
        v23 = (unsigned int *)v21[2];
        if ( !*(_BYTE *)(v22 + 172) )
          goto LABEL_41;
LABEL_20:
        v24 = *(unsigned int ***)(v22 + 152);
        v7 = (__int64)&v24[*(unsigned int *)(v22 + 164)];
        if ( v24 != (unsigned int **)v7 )
        {
          while ( *v24 != v23 )
          {
            if ( (unsigned int **)v7 == ++v24 )
              goto LABEL_42;
          }
          goto LABEL_24;
        }
LABEL_42:
        v8 = *(_QWORD *)(v18 + 8);
        v7 = *(_QWORD *)(a1 + 8);
        v38 = *(_QWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 16);
        if ( v38 )
        {
          v9 = *(_QWORD *)(v38 + 24);
        }
        else
        {
          v79 = *(_QWORD *)(v7 + 32);
          v7 = *(unsigned int *)(v79 + 304);
          v80 = *(_QWORD *)(v79 + 296);
          if ( *(_DWORD *)(v79 + 304) )
          {
            do
            {
              while ( 1 )
              {
                v81 = v7 >> 1;
                j = v80 + 16 * (v7 >> 1);
                if ( (*(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v8 >> 1) & 3) < (*(_DWORD *)((*(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)j >> 1) & 3) )
                  break;
                v80 = j + 16;
                v7 = v7 - v81 - 1;
                if ( v7 <= 0 )
                  goto LABEL_124;
              }
              v7 >>= 1;
            }
            while ( v81 > 0 );
          }
LABEL_124:
          v9 = *(_QWORD *)(v80 - 8);
        }
        v39 = *((_QWORD *)v23 + 1);
        v40 = *v23;
        v41 = (__int64 *)&v107[v40];
        if ( v39 != *(_QWORD *)(v18 + 8) )
          break;
LABEL_101:
        *v41 = v9;
        ++v17;
        v41[1] = v39;
        if ( v93 == v17 )
          goto LABEL_25;
      }
      v42 = *(_DWORD *)(a1 + 416);
      if ( !v42 )
        break;
      v43 = v42 - 1;
      j = 1;
      for ( k = v43 & (((0xBF58476D1CE4E5B9LL * (unsigned int)(37 * v40)) >> 31) ^ (756364221 * v40)); ; k = v43 & v78 )
      {
        v7 = *(_QWORD *)(a1 + 400) + 16LL * k;
        if ( !*(_DWORD *)v7 )
          break;
        if ( *(_DWORD *)v7 == -1 && *(_DWORD *)(v7 + 4) == -1 )
          goto LABEL_50;
LABEL_115:
        v78 = j + k;
        j = (unsigned int)(j + 1);
      }
      if ( (_DWORD)v40 != *(_DWORD *)(v7 + 4) )
        goto LABEL_115;
      if ( (*(_QWORD *)(v7 + 8) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        break;
      if ( v93 == ++v17 )
        goto LABEL_25;
    }
LABEL_50:
    v45 = *v41;
    if ( !*v41 )
    {
      v39 = *(_QWORD *)(v18 + 8);
      goto LABEL_101;
    }
    if ( v45 == v9 )
      break;
    v46 = *(_QWORD *)(*(_QWORD *)(v45 + 32) + 328LL);
    if ( v45 != v46 )
    {
      if ( v9 == v46 )
        goto LABEL_146;
      v47 = *(_QWORD *)(a1 + 32);
      v48 = *(_DWORD *)(v47 + 32);
      v49 = (unsigned int)(*(_DWORD *)(v45 + 24) + 1);
      v50 = 0;
      if ( (unsigned int)v49 < v48 )
        v50 = *(_QWORD *)(*(_QWORD *)(v47 + 24) + 8 * v49);
      if ( v9 )
      {
        v51 = (unsigned int)(*(_DWORD *)(v9 + 24) + 1);
        v52 = *(_DWORD *)(v9 + 24) + 1;
      }
      else
      {
        v51 = 0;
        v52 = 0;
      }
      v53 = 0;
      if ( v48 > v52 )
        v53 = *(_QWORD *)(*(_QWORD *)(v47 + 24) + 8 * v51);
      for ( ; v50 != v53; v50 = *(_QWORD *)(v50 + 8) )
      {
        if ( *(_DWORD *)(v50 + 16) < *(_DWORD *)(v53 + 16) )
        {
          v54 = v50;
          v50 = v53;
          v53 = v54;
        }
      }
      v46 = *(_QWORD *)v53;
    }
    if ( v9 == v46 )
    {
LABEL_146:
      v89 = *(_QWORD *)(v18 + 8);
      *v41 = v9;
      v41[1] = v89;
      goto LABEL_68;
    }
    if ( v45 != v46 )
    {
      *v41 = v46;
      v41[1] = 0;
    }
LABEL_68:
    v55 = sub_2E39EA0(*(__int64 **)(a1 + 56), v9);
    v7 = (__int64)&v101[*v23];
    v56 = __CFADD__(*(_QWORD *)v7, v55);
    v57 = *(_QWORD *)v7 + v55;
    if ( v56 )
    {
      *(_QWORD *)v7 = -1;
      goto LABEL_24;
    }
    *(_QWORD *)v7 = v57;
    if ( v93 == ++v17 )
      goto LABEL_25;
  }
  v87 = v41[1];
  if ( (v87 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    v92 = *(_QWORD *)(v18 + 8);
    v88 = sub_2DF8300((__int64 *)(v18 + 8), v87);
    v8 = v92;
    if ( !v88 )
      goto LABEL_24;
  }
  v41[1] = v8;
  if ( v93 != ++v17 )
    goto LABEL_16;
LABEL_25:
  v25 = *(unsigned int *)(v91 + 72);
  if ( (_DWORD)v25 )
  {
    v26 = 8 * v25;
    for ( m = 0; v26 != m; m += 8LL )
    {
      while ( 1 )
      {
        v32 = (__int64 *)&v107[m / 8];
        if ( !*v32 || (v32[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          goto LABEL_32;
        v33 = *(unsigned int **)(*(_QWORD *)(v91 + 64) + m);
        v34 = *((_QWORD *)v33 + 1) & 0xFFFFFFFFFFFFFFF8LL;
        v35 = *(_QWORD *)(v34 + 16);
        if ( v35 )
        {
          v36 = *(_QWORD *)(v35 + 24);
        }
        else
        {
          v72 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
          v73 = *(unsigned int *)(v72 + 304);
          v74 = *(_QWORD **)(v72 + 296);
          if ( *(_DWORD *)(v72 + 304) )
          {
            v75 = *(_DWORD *)(v34 + 24) | (*((__int64 *)v33 + 1) >> 1) & 3;
            do
            {
              while ( 1 )
              {
                v76 = v73 >> 1;
                v77 = &v74[2 * (v73 >> 1)];
                if ( v75 < (*(_DWORD *)((*v77 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v77 >> 1) & 3) )
                  break;
                v74 = v77 + 2;
                v73 = v73 - v76 - 1;
                if ( v73 <= 0 )
                  goto LABEL_113;
              }
              v73 >>= 1;
            }
            while ( v76 > 0 );
          }
LABEL_113:
          v36 = *(v74 - 1);
        }
        v37 = sub_2FB2570((_QWORD *)a1, *v32, v36);
        *v32 = v37;
        if ( *(_DWORD *)(a1 + 84) == 2 )
          break;
LABEL_27:
        v28 = (__int64 *)(*(_QWORD *)(*(_QWORD *)a1 + 56LL) + 16LL * *(unsigned int *)(v37 + 24));
        v29 = *v28;
        if ( (*v28 & 0xFFFFFFFFFFFFFFF8LL) == 0 || (v28[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          v29 = sub_2FB0650((_QWORD *)(*(_QWORD *)a1 + 48LL), *(_QWORD *)(*(_QWORD *)a1 + 40LL), v37, v29, v30);
        if ( (*(_DWORD *)((v29 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v29 >> 1) & 3) > (*(_DWORD *)((*((_QWORD *)v33 + 1) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                              | (unsigned int)(*((__int64 *)v33 + 1) >> 1)
                                                                                              & 3) )
        {
          v94 = v29;
          v31 = (__int64 *)sub_2FB0DC0(
                             (_QWORD *)(*(_QWORD *)a1 + 48LL),
                             *(_QWORD *)(*(_QWORD *)a1 + 40LL),
                             *v32,
                             v29,
                             v30);
          v32[1] = *(_QWORD *)(sub_2FB9FE0((__int64 *)a1, 0, (int *)v33, v94, *v32, v31) + 8);
          goto LABEL_32;
        }
        v58 = v100;
        if ( !(_DWORD)v100 )
        {
          ++v97;
          v104 = 0;
LABEL_149:
          v58 = 2 * v100;
LABEL_150:
          sub_A08C50((__int64)&v97, v58);
          sub_22B31A0((__int64)&v97, (int *)v33, &v104);
          v61 = v104;
          j = (unsigned int)(v99 + 1);
          goto LABEL_79;
        }
        v9 = (unsigned int)(v100 - 1);
        v8 = v98;
        j = (unsigned int)v9 & (37 * *v33);
        v7 = v98 + 4 * j;
        v59 = *(_DWORD *)v7;
        if ( *(_DWORD *)v7 == *v33 )
          goto LABEL_32;
        v60 = 1;
        v61 = 0;
        while ( v59 != -1 )
        {
          if ( v59 != -2 || v61 )
            v7 = (__int64)v61;
          j = (unsigned int)v9 & (v60 + (_DWORD)j);
          v59 = *(_DWORD *)(v98 + 4LL * (unsigned int)j);
          if ( *v33 == v59 )
            goto LABEL_32;
          ++v60;
          v61 = (_DWORD *)v7;
          v7 = v98 + 4LL * (unsigned int)j;
        }
        if ( !v61 )
          v61 = (_DWORD *)v7;
        ++v97;
        j = (unsigned int)(v99 + 1);
        v104 = v61;
        if ( 4 * (int)j >= (unsigned int)(3 * v100) )
          goto LABEL_149;
        v7 = (unsigned int)v100 >> 3;
        if ( (int)v100 - HIDWORD(v99) - (int)j <= (unsigned int)v7 )
          goto LABEL_150;
LABEL_79:
        LODWORD(v99) = j;
        if ( *v61 != -1 )
          --HIDWORD(v99);
        m += 8LL;
        *v61 = *v33;
        if ( v26 == m )
          goto LABEL_82;
      }
      v95 = &v101[*v33];
      if ( *v95 >= (unsigned __int64)sub_2E39EA0(*(__int64 **)(a1 + 56), v37) )
      {
        v37 = *v32;
        goto LABEL_27;
      }
      v82 = v100;
      if ( !(_DWORD)v100 )
      {
        ++v97;
        v104 = 0;
        goto LABEL_152;
      }
      v9 = (unsigned int)(v100 - 1);
      v8 = v98;
      j = (unsigned int)v9 & (37 * *v33);
      v7 = v98 + 4 * j;
      v83 = *(_DWORD *)v7;
      if ( *v33 != *(_DWORD *)v7 )
      {
        v84 = 1;
        v85 = 0;
        while ( v83 != -1 )
        {
          if ( v83 == -2 && !v85 )
            v85 = v7;
          j = (unsigned int)v9 & ((_DWORD)j + v84);
          v7 = v98 + 4 * j;
          v83 = *(_DWORD *)v7;
          if ( *v33 == *(_DWORD *)v7 )
            goto LABEL_32;
          ++v84;
        }
        if ( v85 )
          v7 = v85;
        ++v97;
        v86 = v99 + 1;
        v104 = (_BYTE *)v7;
        if ( 4 * ((int)v99 + 1) < (unsigned int)(3 * v100) )
        {
          j = (unsigned int)(v100 - HIDWORD(v99) - v86);
          if ( (unsigned int)j > (unsigned int)v100 >> 3 )
          {
LABEL_135:
            LODWORD(v99) = v86;
            if ( *(_DWORD *)v7 != -1 )
              --HIDWORD(v99);
            *(_DWORD *)v7 = *v33;
            continue;
          }
LABEL_153:
          sub_A08C50((__int64)&v97, v82);
          sub_22B31A0((__int64)&v97, (int *)v33, &v104);
          v7 = (__int64)v104;
          v86 = v99 + 1;
          goto LABEL_135;
        }
LABEL_152:
        v82 = 2 * v100;
        goto LABEL_153;
      }
LABEL_32:
      ;
    }
  }
LABEL_82:
  v104 = v106;
  v105 = 0x800000000LL;
  v62 = *(__int64 **)(v90 + 64);
  if ( &v62[*(unsigned int *)(v90 + 72)] != v62 )
  {
    v96 = &v62[*(unsigned int *)(v90 + 72)];
    while ( 1 )
    {
      v63 = *v62;
      v64 = *(_QWORD *)(*v62 + 8);
      if ( (v64 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
        goto LABEL_91;
      v65 = *(_QWORD *)(*(_QWORD *)(a1 + 72) + 8LL);
      v7 = sub_2E09D00((__int64 *)v65, *(_QWORD *)(*v62 + 8));
      if ( v7 == *(_QWORD *)v65 + 24LL * *(unsigned int *)(v65 + 8)
        || (*(_DWORD *)((*(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)v7 >> 1) & 3) > (*(_DWORD *)((v64 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v64 >> 1) & 3) )
      {
        BUG();
      }
      v66 = *(int **)(v7 + 16);
      j = (unsigned int)*v66;
      v67 = &v107[j];
      if ( !*v67 || *(_QWORD *)(v63 + 8) == v67[1] )
        goto LABEL_91;
      if ( !(_DWORD)v100 )
        goto LABEL_105;
      v7 = (unsigned int)(v100 - 1);
      v68 = v7 & (37 * j);
      v69 = *(_DWORD *)(v98 + 4LL * v68);
      if ( (_DWORD)j != v69 )
        break;
LABEL_91:
      if ( v96 == ++v62 )
        goto LABEL_92;
    }
    v8 = 1;
    while ( v69 != -1 )
    {
      v9 = (unsigned int)(v8 + 1);
      v68 = v7 & (v8 + v68);
      v69 = *(_DWORD *)(v98 + 4LL * v68);
      if ( (_DWORD)j == v69 )
        goto LABEL_91;
      v8 = (unsigned int)v9;
    }
LABEL_105:
    v70 = (unsigned int)v105;
    v71 = (unsigned int)v105 + 1LL;
    if ( v71 > HIDWORD(v105) )
    {
      sub_C8D5F0((__int64)&v104, v106, v71, 8u, v8, v9);
      v70 = (unsigned int)v105;
    }
    *(_QWORD *)&v104[8 * v70] = v63;
    LODWORD(v105) = v105 + 1;
    sub_2FB7E60(a1, 0, v66);
    goto LABEL_91;
  }
LABEL_92:
  if ( *(_DWORD *)(a1 + 84) == 2 && (_DWORD)v99 )
    sub_2FB8F50(a1, (__int64)&v97, (__int64)&v104, v7, v8, v9);
  sub_2FB8410(a1, (__int64 *)&v104, j, v7, v8, v9);
  if ( v104 != v106 )
    _libc_free((unsigned __int64)v104);
  sub_C7D6A0(v98, 4LL * (unsigned int)v100, 4);
  if ( v101 != v103 )
    _libc_free((unsigned __int64)v101);
  if ( v107 != v109 )
    _libc_free((unsigned __int64)v107);
}
