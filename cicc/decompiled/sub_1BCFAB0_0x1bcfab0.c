// Function: sub_1BCFAB0
// Address: 0x1bcfab0
//
__int64 __fastcall sub_1BCFAB0(__int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned __int64 v3; // r15
  _QWORD *v4; // rbx
  __int64 v5; // r12
  __int64 *v6; // r11
  unsigned int v7; // r12d
  __int64 **v8; // r13
  __int64 v9; // r12
  char *v10; // rdi
  __int64 v12; // rax
  __int64 v13; // r13
  _QWORD *v14; // rsi
  unsigned int *v15; // rdx
  _QWORD *v16; // rax
  __int64 v17; // rbx
  __int64 v18; // r14
  __int64 v19; // rax
  char *v20; // rax
  _QWORD *v21; // r8
  unsigned int v22; // esi
  __int64 v23; // rdi
  int v24; // r10d
  __int64 v25; // r9
  unsigned int v26; // edx
  __int64 v27; // r13
  __int64 v28; // rax
  int v29; // eax
  char v30; // r15
  char *v31; // rdx
  unsigned int *v32; // rax
  __int64 v33; // r14
  __int64 v34; // rdx
  int v35; // eax
  __int64 v36; // rsi
  int v37; // r8d
  __int64 v38; // rax
  unsigned int *v39; // r9
  unsigned int *v40; // r12
  __int64 i; // rax
  unsigned int v42; // r15d
  __int64 v43; // rdx
  unsigned __int64 v44; // rsi
  unsigned int *v45; // rcx
  unsigned int *v46; // rax
  __int64 v47; // rdi
  int v48; // r10d
  __int64 v49; // r15
  __int64 v50; // rcx
  int v51; // r11d
  int *v52; // r14
  unsigned int v53; // edx
  int *v54; // rax
  _BOOL4 v55; // r13d
  __int64 v56; // rax
  int v57; // eax
  char *v58; // rdx
  __int64 v59; // r12
  unsigned int *v60; // r15
  unsigned int v61; // r13d
  int *j; // r14
  unsigned int v63; // edx
  int *v64; // rax
  _BOOL4 v65; // r12d
  __int64 v66; // rax
  bool v67; // zf
  unsigned int v68; // eax
  int *v69; // r13
  unsigned int v70; // edx
  int *v71; // rax
  _BOOL4 v72; // r14d
  __int64 v73; // rax
  char *v74; // r13
  __int64 v75; // rbx
  __int64 v76; // rax
  _QWORD *v77; // [rsp+0h] [rbp-130h]
  _QWORD *v78; // [rsp+0h] [rbp-130h]
  _QWORD *v79; // [rsp+0h] [rbp-130h]
  _QWORD *v80; // [rsp+0h] [rbp-130h]
  unsigned __int64 v81; // [rsp+8h] [rbp-128h]
  unsigned int *v82; // [rsp+8h] [rbp-128h]
  __int64 *v83; // [rsp+10h] [rbp-120h]
  __int64 *v84; // [rsp+10h] [rbp-120h]
  unsigned int v85; // [rsp+10h] [rbp-120h]
  __int64 v86; // [rsp+18h] [rbp-118h]
  __int64 ***v87; // [rsp+18h] [rbp-118h]
  __int64 v88; // [rsp+20h] [rbp-110h]
  unsigned int *v89; // [rsp+30h] [rbp-100h]
  char *v91; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v92; // [rsp+50h] [rbp-E0h] BYREF
  __int64 v93; // [rsp+58h] [rbp-D8h]
  char *v94; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v95; // [rsp+78h] [rbp-B8h]
  _WORD v96[8]; // [rsp+80h] [rbp-B0h] BYREF
  char *v97; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v98; // [rsp+98h] [rbp-98h]
  __int64 v99; // [rsp+A0h] [rbp-90h] BYREF
  unsigned int v100; // [rsp+A8h] [rbp-88h]
  unsigned int *v101; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v102; // [rsp+B8h] [rbp-78h]
  _BYTE v103[24]; // [rsp+C0h] [rbp-70h] BYREF
  int v104; // [rsp+D8h] [rbp-58h] BYREF
  int *v105; // [rsp+E0h] [rbp-50h]
  int *v106; // [rsp+E8h] [rbp-48h]
  int *v107; // [rsp+F0h] [rbp-40h]
  __int64 v108; // [rsp+F8h] [rbp-38h]

  v3 = a3;
  v4 = a2;
  v88 = (__int64)a2;
  v86 = a3;
  sub_1BBCA40(&v92, a2, a3, 0);
  v5 = v92;
  if ( v93 )
  {
    if ( *(_BYTE *)(v93 + 16) != 24 )
    {
      v12 = sub_1BBCD20(a1, v92);
      v13 = v12;
      if ( v12 )
      {
        if ( *(_DWORD *)(v12 + 8) == v3 )
        {
          v15 = (unsigned int *)(8 * v3);
          if ( !(8 * v3) || !memcmp(a2, *(const void **)v12, (size_t)v15) )
          {
LABEL_53:
            v9 = sub_1BD0660(a1, v13, v15);
            if ( *(_DWORD *)(v13 + 8) != v3 )
              return v9;
            v38 = *(unsigned int *)(v13 + 104);
            if ( !(_DWORD)v38 )
              return v9;
            if ( *(_BYTE *)(v9 + 16) == 85 )
              return *(_QWORD *)(v9 - 72);
            v104 = 0;
            v97 = (char *)&v99;
            v101 = (unsigned int *)v103;
            v98 = 0x400000000LL;
            v102 = 0x400000000LL;
            v105 = 0;
            v106 = &v104;
            v107 = &v104;
            v108 = 0;
            v39 = *(unsigned int **)(v13 + 96);
            v87 = (__int64 ***)v9;
            v40 = v39;
            v89 = &v39[v38];
            for ( i = 0; ; i = v108 )
            {
              v42 = *v40;
              if ( i )
              {
                v52 = v105;
                if ( v105 )
                {
                  while ( 1 )
                  {
                    v53 = v52[8];
                    v54 = (int *)*((_QWORD *)v52 + 3);
                    if ( v42 < v53 )
                      v54 = (int *)*((_QWORD *)v52 + 2);
                    if ( !v54 )
                      break;
                    v52 = v54;
                  }
                  if ( v42 < v53 )
                  {
                    if ( v52 != v106 )
                      goto LABEL_134;
                  }
                  else if ( v42 <= v53 )
                  {
                    goto LABEL_63;
                  }
                }
                else
                {
                  v52 = &v104;
                  if ( v106 == &v104 )
                  {
                    v55 = 1;
LABEL_98:
                    v56 = sub_22077B0(40);
                    *(_DWORD *)(v56 + 32) = v42;
                    sub_220F040(v55, v56, v52, &v104);
                    ++v108;
                    goto LABEL_99;
                  }
LABEL_134:
                  if ( v42 <= *(_DWORD *)(sub_220EF80(v52) + 32) || !v52 )
                    goto LABEL_63;
                }
                v55 = 1;
                if ( v52 != &v104 )
                  v55 = v42 < v52[8];
                goto LABEL_98;
              }
              v43 = (unsigned int)v102;
              v44 = (unsigned __int64)v101;
              v45 = &v101[(unsigned int)v102];
              if ( v101 != v45 )
              {
                v46 = v101;
                while ( v42 != *v46 )
                {
                  if ( v45 == ++v46 )
                    goto LABEL_104;
                }
                if ( v45 != v46 )
                  goto LABEL_63;
              }
LABEL_104:
              if ( (unsigned int)v102 <= 3uLL )
              {
                if ( (unsigned int)v102 >= HIDWORD(v102) )
                {
                  sub_16CD150((__int64)&v101, v103, 0, 4, v37, (int)v39);
                  v45 = &v101[(unsigned int)v102];
                }
                *v45 = v42;
                LODWORD(v102) = v102 + 1;
                goto LABEL_99;
              }
              v85 = *v40;
              v82 = v40;
              v59 = (__int64)v105;
              while ( 1 )
              {
                v60 = (unsigned int *)(v44 + 4 * v43 - 4);
                if ( v59 )
                {
                  v61 = *v60;
                  for ( j = (int *)v59; ; j = v64 )
                  {
                    v63 = j[8];
                    v64 = (int *)*((_QWORD *)j + 3);
                    if ( v61 < v63 )
                      v64 = (int *)*((_QWORD *)j + 2);
                    if ( !v64 )
                      break;
                  }
                  if ( v61 >= v63 )
                  {
                    if ( v61 <= v63 )
                      goto LABEL_117;
LABEL_114:
                    v65 = 1;
                    if ( j != &v104 )
                      v65 = v61 < j[8];
LABEL_116:
                    v66 = sub_22077B0(40);
                    *(_DWORD *)(v66 + 32) = *v60;
                    sub_220F040(v65, v66, j, &v104);
                    ++v108;
                    v59 = (__int64)v105;
                    goto LABEL_117;
                  }
                  if ( v106 == j )
                    goto LABEL_114;
                }
                else
                {
                  j = &v104;
                  if ( v106 == &v104 )
                  {
                    v65 = 1;
                    goto LABEL_116;
                  }
                  v61 = *v60;
                }
                if ( v61 > *(_DWORD *)(sub_220EF80(j) + 32) )
                  goto LABEL_114;
LABEL_117:
                v67 = (_DWORD)v102 == 1;
                v68 = v102 - 1;
                LODWORD(v102) = v102 - 1;
                if ( v67 )
                  break;
                v44 = (unsigned __int64)v101;
                v43 = v68;
              }
              v69 = (int *)v59;
              v42 = v85;
              v40 = v82;
              if ( v69 )
              {
                while ( 1 )
                {
                  v70 = v69[8];
                  v71 = (int *)*((_QWORD *)v69 + 3);
                  if ( v85 < v70 )
                    v71 = (int *)*((_QWORD *)v69 + 2);
                  if ( !v71 )
                    break;
                  v69 = v71;
                }
                if ( v85 < v70 )
                {
                  if ( v69 != v106 )
                    goto LABEL_147;
                }
                else if ( v85 <= v70 )
                {
                  goto LABEL_99;
                }
LABEL_130:
                v72 = 1;
                if ( v69 != &v104 )
                  v72 = v85 < v69[8];
LABEL_132:
                v73 = sub_22077B0(40);
                *(_DWORD *)(v73 + 32) = v85;
                sub_220F040(v72, v73, v69, &v104);
                ++v108;
                goto LABEL_99;
              }
              v69 = &v104;
              if ( v106 == &v104 )
              {
                v72 = 1;
                goto LABEL_132;
              }
LABEL_147:
              if ( v85 > *(_DWORD *)(sub_220EF80(v69) + 32) && v69 )
                goto LABEL_130;
LABEL_99:
              v57 = v98;
              if ( (unsigned int)v98 >= HIDWORD(v98) )
              {
                sub_16CD150((__int64)&v97, &v99, 0, 4, v37, (int)v39);
                v57 = v98;
              }
              v58 = &v97[4 * v57];
              if ( v58 )
              {
                *(_DWORD *)v58 = v42;
                v57 = v98;
              }
              LODWORD(v98) = v57 + 1;
LABEL_63:
              if ( v89 == ++v40 )
              {
                v74 = v97;
                v96[0] = 257;
                v75 = (unsigned int)v98;
                v76 = sub_1599EF0(*v87);
                v9 = sub_156A7D0((__int64 *)(a1 + 1400), (__int64)v87, v76, (__int64)v74, v75, (__int64)&v94);
                sub_1BBB240((__int64)v105);
                if ( v101 != (unsigned int *)v103 )
                  _libc_free((unsigned __int64)v101);
                v10 = v97;
                if ( v97 != (char *)&v99 )
                  goto LABEL_10;
                return v9;
              }
            }
          }
        }
        else if ( *(_DWORD *)(v12 + 104) == v3 )
        {
          v14 = &a2[v3];
          v15 = *(unsigned int **)(v12 + 96);
          if ( v14 != v4 )
          {
            v16 = v4;
            while ( *v16 == *(_QWORD *)(*(_QWORD *)v13 + 8LL * *v15) )
            {
              ++v16;
              ++v15;
              if ( v14 == v16 )
                goto LABEL_53;
            }
            goto LABEL_3;
          }
          goto LABEL_53;
        }
      }
    }
  }
LABEL_3:
  v6 = *(__int64 **)v5;
  if ( *(_BYTE *)(v5 + 16) == 55 )
    v6 = **(__int64 ***)(v5 - 48);
  v7 = v3;
  v94 = (char *)v96;
  v95 = 0x400000000LL;
  v101 = (unsigned int *)v103;
  v102 = 0x400000000LL;
  if ( v3 <= 2 )
    goto LABEL_6;
  v97 = 0;
  v21 = &v4[v3];
  v98 = 0;
  v99 = 0;
  v100 = 0;
  if ( v21 == v4 )
  {
    v23 = 0;
    goto LABEL_50;
  }
  v83 = v6;
  v22 = 0;
  v23 = 0;
  v7 = 0;
  v81 = v3;
  while ( 1 )
  {
    v33 = *v4;
    if ( !v22 )
    {
      ++v97;
LABEL_39:
      v77 = v21;
      sub_177C7D0((__int64)&v97, 2 * v22);
      if ( !v100 )
        goto LABEL_166;
      v21 = v77;
      LODWORD(v34) = (v100 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
      v35 = v99 + 1;
      v27 = v98 + 16LL * (unsigned int)v34;
      v36 = *(_QWORD *)v27;
      if ( v33 != *(_QWORD *)v27 )
      {
        v51 = 1;
        v25 = 0;
        while ( v36 != -8 )
        {
          if ( !v25 && v36 == -16 )
            v25 = v27;
          v34 = (v100 - 1) & ((_DWORD)v34 + v51);
          v27 = v98 + 16 * v34;
          v36 = *(_QWORD *)v27;
          if ( v33 == *(_QWORD *)v27 )
            goto LABEL_41;
          ++v51;
        }
        if ( v25 )
          v27 = v25;
      }
      goto LABEL_41;
    }
    v24 = 1;
    v25 = 0;
    v26 = (v22 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
    v27 = v23 + 16LL * v26;
    v28 = *(_QWORD *)v27;
    if ( v33 == *(_QWORD *)v27 )
    {
LABEL_25:
      v29 = v95;
      v30 = 0;
      if ( (unsigned int)v95 < HIDWORD(v95) )
        goto LABEL_26;
LABEL_44:
      v78 = v21;
      sub_16CD150((__int64)&v94, v96, 0, 4, (int)v21, v25);
      v29 = v95;
      v21 = v78;
      goto LABEL_26;
    }
    while ( v28 != -8 )
    {
      if ( !v25 && v28 == -16 )
        v25 = v27;
      v26 = (v22 - 1) & (v24 + v26);
      v27 = v23 + 16LL * v26;
      v28 = *(_QWORD *)v27;
      if ( v33 == *(_QWORD *)v27 )
        goto LABEL_25;
      ++v24;
    }
    if ( v25 )
      v27 = v25;
    ++v97;
    v35 = v99 + 1;
    if ( 4 * ((int)v99 + 1) >= 3 * v22 )
      goto LABEL_39;
    if ( v22 - (v35 + HIDWORD(v99)) <= v22 >> 3 )
    {
      v79 = v21;
      sub_177C7D0((__int64)&v97, v22);
      if ( !v100 )
      {
LABEL_166:
        LODWORD(v99) = v99 + 1;
        BUG();
      }
      v47 = 0;
      v48 = 1;
      LODWORD(v49) = (v100 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
      v21 = v79;
      v35 = v99 + 1;
      v27 = v98 + 16LL * (unsigned int)v49;
      v50 = *(_QWORD *)v27;
      if ( v33 != *(_QWORD *)v27 )
      {
        while ( v50 != -8 )
        {
          if ( v50 == -16 && !v47 )
            v47 = v27;
          LODWORD(v25) = v48 + 1;
          v49 = (v100 - 1) & ((_DWORD)v49 + v48);
          v27 = v98 + 16 * v49;
          v50 = *(_QWORD *)v27;
          if ( v33 == *(_QWORD *)v27 )
            goto LABEL_41;
          ++v48;
        }
        if ( v47 )
          v27 = v47;
      }
    }
LABEL_41:
    LODWORD(v99) = v35;
    if ( *(_QWORD *)v27 != -8 )
      --HIDWORD(v99);
    *(_QWORD *)v27 = v33;
    v30 = 1;
    *(_DWORD *)(v27 + 8) = v7;
    v29 = v95;
    if ( (unsigned int)v95 >= HIDWORD(v95) )
      goto LABEL_44;
LABEL_26:
    v31 = &v94[4 * v29];
    if ( v31 )
    {
      *(_DWORD *)v31 = *(_DWORD *)(v27 + 8);
      v29 = v95;
    }
    v7 = v102;
    LODWORD(v95) = v29 + 1;
    if ( v30 || *(_BYTE *)(v33 + 16) <= 0x10u )
    {
      if ( HIDWORD(v102) <= (unsigned int)v102 )
      {
        v80 = v21;
        sub_16CD150((__int64)&v101, v103, 0, 8, (int)v21, v25);
        v7 = v102;
        v21 = v80;
      }
      v32 = &v101[2 * v7];
      if ( v32 )
      {
        *(_QWORD *)v32 = v33;
        v7 = v102;
      }
      LODWORD(v102) = ++v7;
    }
    ++v4;
    v23 = v98;
    if ( v21 == v4 )
      break;
    v22 = v100;
  }
  LODWORD(v3) = v81;
  v6 = v83;
  if ( v7 != v81 && v7 > 1uLL && (v7 & (v7 - 1)) == 0 )
  {
    v86 = v7;
    v88 = (__int64)v101;
    goto LABEL_51;
  }
LABEL_50:
  LODWORD(v95) = 0;
  v7 = v3;
LABEL_51:
  v84 = v6;
  j___libc_free_0(v23);
  v6 = v84;
LABEL_6:
  v8 = (__int64 **)sub_16463B0(v6, v7);
  v9 = sub_1BCF750(a1, v88, v86, (__int64)v8);
  if ( (_DWORD)v95 )
  {
    v17 = (unsigned int)v95;
    v18 = (__int64)v94;
    v97 = "shuffle";
    LOWORD(v99) = 259;
    v19 = sub_1599EF0(v8);
    v20 = (char *)sub_156A7D0((__int64 *)(a1 + 1400), v9, v19, v18, v17, (__int64)&v97);
    v9 = (__int64)v20;
    if ( (unsigned __int8)v20[16] > 0x17u )
    {
      v91 = v20;
      sub_1BCF290(a1 + 1080, &v91);
      v97 = (char *)*((_QWORD *)v91 + 5);
      sub_1BCF4F0(a1 + 1136, &v97);
    }
  }
  if ( v101 != (unsigned int *)v103 )
    _libc_free((unsigned __int64)v101);
  v10 = v94;
  if ( v94 != (char *)v96 )
LABEL_10:
    _libc_free((unsigned __int64)v10);
  return v9;
}
