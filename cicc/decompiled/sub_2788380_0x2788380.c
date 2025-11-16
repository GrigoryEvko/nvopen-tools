// Function: sub_2788380
// Address: 0x2788380
//
void __fastcall sub_2788380(__int64 a1)
{
  __int64 v2; // r15
  char *v3; // r13
  signed __int64 v4; // r15
  char *v5; // r14
  void **v6; // r12
  void *v7; // rdi
  const void *v8; // rsi
  unsigned __int64 *v9; // rdi
  unsigned __int8 *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned int v13; // ecx
  unsigned __int8 **v14; // rdx
  unsigned __int8 *v15; // r8
  __int64 v16; // rax
  unsigned __int8 *v17; // rdx
  unsigned __int8 *v18; // r13
  _BYTE *v19; // rcx
  _QWORD *v20; // rdx
  _QWORD *i; // r10
  unsigned __int64 v22; // rsi
  _QWORD *v23; // rax
  _QWORD *v24; // r9
  unsigned __int64 v25; // rsi
  _QWORD *v26; // rax
  _QWORD *v27; // rsi
  unsigned __int64 v28; // rdx
  unsigned __int64 v29; // rcx
  __int64 **v30; // r10
  __int64 **v31; // rcx
  __int64 ***v32; // rsi
  __int64 **v33; // r10
  __int64 v34; // rax
  __int64 v35; // rcx
  unsigned int v36; // edi
  __int64 v37; // rdx
  unsigned __int8 *v38; // rsi
  __int64 v39; // rsi
  __int64 v40; // rdx
  bool v41; // al
  unsigned __int8 *v42; // rdx
  const void *v43; // rax
  __int64 v44; // rdx
  unsigned int v45; // eax
  bool v46; // r10
  _QWORD *v47; // rax
  _QWORD *v48; // rdx
  __int64 v49; // rax
  bool v50; // r11
  _QWORD *v51; // rax
  _QWORD *v52; // rdx
  __int64 v53; // rax
  unsigned __int64 *v54; // rax
  int v55; // edx
  __int64 v56; // rdx
  int v57; // edx
  int v58; // r9d
  __int64 ***v59; // rax
  __int64 ***v60; // rax
  int v61; // r8d
  __int64 **v62; // r8
  __int64 **v63; // r8
  __int64 v64; // rax
  __int64 *v65; // rax
  __int64 *v66; // r11
  __int64 *v67; // rdi
  __int64 *v68; // rax
  __int64 v69; // rax
  __int64 *v70; // rax
  __int64 *v71; // r11
  __int64 *v72; // rdi
  __int64 *v73; // rax
  _QWORD *v74; // [rsp+8h] [rbp-158h]
  _QWORD *v75; // [rsp+8h] [rbp-158h]
  __int64 **v76; // [rsp+10h] [rbp-150h]
  _QWORD *v77; // [rsp+10h] [rbp-150h]
  __int64 **v78; // [rsp+18h] [rbp-148h]
  __int64 **v79; // [rsp+18h] [rbp-148h]
  _BYTE *v80; // [rsp+20h] [rbp-140h]
  _QWORD *v81; // [rsp+28h] [rbp-138h]
  __int64 **v82; // [rsp+28h] [rbp-138h]
  _BYTE *v83; // [rsp+30h] [rbp-130h]
  unsigned __int64 v84; // [rsp+30h] [rbp-130h]
  __int64 **v85; // [rsp+38h] [rbp-128h]
  _QWORD *v86; // [rsp+48h] [rbp-118h]
  _QWORD *v87; // [rsp+58h] [rbp-108h]
  __int64 *v88; // [rsp+58h] [rbp-108h]
  __int64 *v89; // [rsp+58h] [rbp-108h]
  _BYTE *v90; // [rsp+60h] [rbp-100h]
  _BYTE *v91; // [rsp+60h] [rbp-100h]
  _QWORD *v92; // [rsp+60h] [rbp-100h]
  __int64 ***v93; // [rsp+60h] [rbp-100h]
  __int64 ***v94; // [rsp+60h] [rbp-100h]
  char v95; // [rsp+68h] [rbp-F8h]
  _QWORD *v96; // [rsp+68h] [rbp-F8h]
  char v97; // [rsp+68h] [rbp-F8h]
  _BYTE *v98; // [rsp+68h] [rbp-F8h]
  __int64 *v99; // [rsp+68h] [rbp-F8h]
  __int64 *v100; // [rsp+68h] [rbp-F8h]
  __int64 v101; // [rsp+70h] [rbp-F0h]
  bool v102; // [rsp+70h] [rbp-F0h]
  bool v103; // [rsp+70h] [rbp-F0h]
  _QWORD *v104; // [rsp+70h] [rbp-F0h]
  _QWORD *v105; // [rsp+70h] [rbp-F0h]
  _QWORD *v106; // [rsp+70h] [rbp-F0h]
  _QWORD *v107; // [rsp+70h] [rbp-F0h]
  _QWORD *v108; // [rsp+70h] [rbp-F0h]
  _QWORD *v109; // [rsp+70h] [rbp-F0h]
  __int64 *v110; // [rsp+70h] [rbp-F0h]
  __int64 *v111; // [rsp+70h] [rbp-F0h]
  unsigned __int8 *v112; // [rsp+78h] [rbp-E8h]
  unsigned __int64 v113; // [rsp+80h] [rbp-E0h] BYREF
  unsigned int v114; // [rsp+88h] [rbp-D8h]
  unsigned __int64 v115; // [rsp+90h] [rbp-D0h]
  unsigned int v116; // [rsp+98h] [rbp-C8h]
  unsigned __int64 v117; // [rsp+A0h] [rbp-C0h] BYREF
  unsigned int v118; // [rsp+A8h] [rbp-B8h]
  unsigned __int64 v119; // [rsp+B0h] [rbp-B0h]
  unsigned int v120; // [rsp+B8h] [rbp-A8h]
  const void *v121; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v122; // [rsp+C8h] [rbp-98h]
  unsigned __int64 v123; // [rsp+D0h] [rbp-90h] BYREF
  unsigned int v124; // [rsp+D8h] [rbp-88h]
  __int64 v125[2]; // [rsp+E0h] [rbp-80h] BYREF
  unsigned __int64 *v126; // [rsp+F0h] [rbp-70h]
  __int64 v127; // [rsp+F8h] [rbp-68h]
  __int64 v128; // [rsp+100h] [rbp-60h]
  void **v129; // [rsp+108h] [rbp-58h]
  unsigned __int64 *v130; // [rsp+110h] [rbp-50h]
  void *dest; // [rsp+118h] [rbp-48h]
  __int64 v132; // [rsp+120h] [rbp-40h]
  void **v133; // [rsp+128h] [rbp-38h]

  v2 = *(unsigned int *)(a1 + 88);
  v3 = *(char **)(a1 + 80);
  v125[0] = 0;
  v4 = 8 * v2;
  v125[1] = 0;
  v126 = 0;
  v5 = &v3[v4];
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  dest = 0;
  v132 = 0;
  v133 = 0;
  sub_2785050(v125, v4 >> 3);
  v6 = v129;
  if ( v129 < v133 )
  {
    do
    {
      v7 = *v6;
      v8 = v3;
      ++v6;
      v3 += 512;
      memmove(v7, v8, 0x200u);
    }
    while ( v133 > v6 );
    v4 = v5 - v3;
  }
  if ( v5 != v3 )
    memmove(dest, v3, v4);
  v86 = (_QWORD *)(a1 + 168);
  while ( 1 )
  {
    v9 = v130;
    if ( v126 == v130 )
      break;
LABEL_8:
    if ( v9 == dest )
    {
      v10 = (unsigned __int8 *)*((_QWORD *)*(v133 - 1) + 63);
      j_j___libc_free_0((unsigned __int64)v9);
      v56 = (__int64)*--v133 + 512;
      dest = *v133;
      v132 = v56;
      v130 = (unsigned __int64 *)((char *)dest + 504);
    }
    else
    {
      v10 = (unsigned __int8 *)*(v9 - 1);
      v130 = v9 - 1;
    }
    v11 = *(unsigned int *)(a1 + 24);
    v12 = *(_QWORD *)(a1 + 8);
    if ( !(_DWORD)v11 )
      goto LABEL_13;
    v13 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
    v14 = (unsigned __int8 **)(v12 + 16LL * v13);
    v15 = *v14;
    if ( *v14 != v10 )
    {
      v57 = 1;
      while ( v15 != (unsigned __int8 *)-4096LL )
      {
        v58 = v57 + 1;
        v13 = (v11 - 1) & (v57 + v13);
        v14 = (unsigned __int8 **)(v12 + 16LL * v13);
        v15 = *v14;
        if ( *v14 == v10 )
          goto LABEL_12;
        v57 = v58;
      }
LABEL_13:
      v112 = v10;
      switch ( *v10 )
      {
        case ')':
        case '+':
        case '-':
        case '/':
        case 'F':
        case 'G':
        case 'S':
          sub_2784F00((__int64)&v121);
          goto LABEL_15;
        case 'H':
        case 'I':
          if ( (v10[7] & 0x40) != 0 )
            v42 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
          else
            v42 = &v10[-32 * (*((_DWORD *)v10 + 1) & 0x7FFFFFF)];
          v43 = (const void *)sub_BCAE30(*(_QWORD *)(*(_QWORD *)v42 + 8LL));
          v122 = v44;
          v121 = v43;
          v45 = sub_CA1930(&v121);
          sub_AADB10((__int64)&v113, v45, 1);
          sub_AB49F0((__int64)&v117, (__int64)&v113, *v10 - 29, qword_4FFB348 + 1);
          sub_2784F30((__int64)&v121, a1, (__int64 *)&v117);
          sub_2787A60(a1, (__int64)v10, (__int64)&v121);
          if ( v124 > 0x40 && v123 )
            j_j___libc_free_0_0(v123);
          if ( (unsigned int)v122 > 0x40 && v121 )
            j_j___libc_free_0_0((unsigned __int64)v121);
          if ( v120 > 0x40 && v119 )
            j_j___libc_free_0_0(v119);
          if ( v118 > 0x40 && v117 )
            j_j___libc_free_0_0(v117);
          if ( v116 > 0x40 && v115 )
            j_j___libc_free_0_0(v115);
          if ( v114 <= 0x40 || !v113 )
            continue;
          j_j___libc_free_0_0(v113);
          v9 = v130;
          if ( v126 != v130 )
            goto LABEL_8;
          goto LABEL_100;
        default:
          sub_2784ED0((__int64)&v121);
LABEL_15:
          sub_2787A60(a1, (__int64)v10, (__int64)&v121);
          if ( v124 > 0x40 && v123 )
            j_j___libc_free_0_0(v123);
          if ( (unsigned int)v122 > 0x40 && v121 )
            j_j___libc_free_0_0((unsigned __int64)v121);
          v16 = 32LL * (*((_DWORD *)v10 + 1) & 0x7FFFFFF);
          v17 = &v10[-v16];
          if ( (v10[7] & 0x40) != 0 )
          {
            v17 = (unsigned __int8 *)*((_QWORD *)v10 - 1);
            v112 = &v17[v16];
          }
          if ( v17 == v112 )
            continue;
          v18 = v17;
          v19 = *(_BYTE **)v17;
          if ( **(_BYTE **)v17 > 0x1Cu )
            goto LABEL_25;
          break;
      }
LABEL_72:
      v117 = 0;
      if ( *v19 != 18 )
      {
        sub_2784ED0((__int64)&v121);
        sub_2787A60(a1, (__int64)v10, (__int64)&v121);
        if ( v124 > 0x40 && v123 )
          j_j___libc_free_0_0(v123);
        if ( (unsigned int)v122 > 0x40 )
        {
          if ( v121 )
            j_j___libc_free_0_0((unsigned __int64)v121);
        }
      }
LABEL_70:
      v18 += 32;
      if ( v112 == v18 )
        continue;
      v19 = *(_BYTE **)v18;
      if ( **(_BYTE **)v18 <= 0x1Cu )
        goto LABEL_72;
LABEL_25:
      v20 = *(_QWORD **)(a1 + 176);
      v117 = (unsigned __int64)v19;
      v121 = &v121;
      v122 = 1;
      v123 = (unsigned __int64)v10;
      if ( v20 )
      {
        for ( i = v20; ; i = v23 )
        {
          v22 = i[6];
          v23 = (_QWORD *)i[3];
          if ( (unsigned __int64)v10 < v22 )
            v23 = (_QWORD *)i[2];
          if ( !v23 )
            break;
        }
        if ( (unsigned __int64)v10 >= v22 )
        {
          if ( v22 >= (unsigned __int64)v10 )
          {
            v121 = &v121;
            v24 = i;
            v122 = 1;
            v123 = (unsigned __int64)v19;
            goto LABEL_35;
          }
LABEL_109:
          v50 = 1;
          if ( v86 != i )
            v50 = (unsigned __int64)v10 < i[6];
LABEL_111:
          v97 = v50;
          v107 = i;
          v51 = (_QWORD *)sub_22077B0(0x38u);
          v51[4] = v51 + 4;
          v52 = v107;
          v51[5] = 1;
          v51[6] = v10;
          v108 = v51;
          sub_220F040(v97, (__int64)v51, v52, v86);
          ++*(_QWORD *)(a1 + 200);
          v19 = (_BYTE *)v117;
          v20 = *(_QWORD **)(a1 + 176);
          v24 = v108;
LABEL_112:
          v121 = &v121;
          v122 = 1;
          v123 = (unsigned __int64)v19;
          if ( !v20 )
          {
            v20 = (_QWORD *)(a1 + 168);
            if ( v86 == *(_QWORD **)(a1 + 184) )
            {
              v46 = 1;
            }
            else
            {
LABEL_114:
              v92 = v24;
              v98 = v19;
              v109 = v20;
              v53 = sub_220EF80((__int64)v20);
              v19 = v98;
              v20 = v109;
              v24 = v92;
              if ( (unsigned __int64)v98 <= *(_QWORD *)(v53 + 48) )
              {
                v20 = (_QWORD *)v53;
LABEL_40:
                v27 = v20;
                if ( v86 != v20 )
                {
LABEL_41:
                  v28 = (unsigned __int64)(v27 + 4);
                  if ( (v27[5] & 1) != 0 || (v28 = v27[4], (*(_BYTE *)(v28 + 8) & 1) != 0) )
                  {
                    if ( v24 != v86 )
                      goto LABEL_47;
                    v31 = 0;
                    goto LABEL_54;
                  }
                  v29 = *(_QWORD *)v28;
                  if ( (*(_BYTE *)(*(_QWORD *)v28 + 8LL) & 1) != 0 )
                  {
                    v28 = *(_QWORD *)v28;
                  }
                  else
                  {
                    v30 = *(__int64 ***)v29;
                    if ( (*(_BYTE *)(*(_QWORD *)v29 + 8LL) & 1) == 0 )
                    {
                      v60 = (__int64 ***)*v30;
                      v94 = (__int64 ***)*v30;
                      if ( ((*v30)[1] & 1) != 0 )
                      {
                        v30 = (__int64 **)*v30;
                      }
                      else
                      {
                        v62 = *v60;
                        if ( ((_BYTE)(*v60)[1] & 1) == 0 )
                        {
                          v89 = *v62;
                          if ( ((*v62)[1] & 1) != 0 )
                          {
                            v62 = (__int64 **)*v62;
                          }
                          else
                          {
                            v69 = **v62;
                            v111 = (__int64 *)v69;
                            if ( (*(_BYTE *)(v69 + 8) & 1) == 0 )
                            {
                              v70 = *(__int64 **)v69;
                              v100 = v70;
                              if ( (v70[1] & 1) == 0 )
                              {
                                v71 = (__int64 *)*v70;
                                if ( (*(_BYTE *)(*v70 + 8) & 1) == 0 )
                                {
                                  v72 = (__int64 *)*v71;
                                  v75 = (_QWORD *)*v70;
                                  if ( (*(_BYTE *)(*v71 + 8) & 1) == 0 )
                                  {
                                    v77 = v24;
                                    v79 = v62;
                                    v80 = (_BYTE *)v27[4];
                                    v82 = *(__int64 ***)v29;
                                    v84 = *(_QWORD *)v28;
                                    v73 = sub_2785310(v72);
                                    v24 = v77;
                                    v62 = v79;
                                    v28 = (unsigned __int64)v80;
                                    v72 = (__int64 *)v73;
                                    *v75 = v73;
                                    v30 = v82;
                                    v29 = v84;
                                  }
                                  v71 = v72;
                                  *v100 = (__int64)v72;
                                }
                                v100 = v71;
                                *v111 = (__int64)v71;
                              }
                              v111 = v100;
                              *v89 = (__int64)v100;
                            }
                            *v62 = v111;
                            v62 = (__int64 **)v111;
                          }
                          *v94 = v62;
                        }
                        *v30 = (__int64 *)v62;
                        v30 = v62;
                      }
                      *(_QWORD *)v29 = v30;
                    }
                    *(_QWORD *)v28 = v30;
                    v28 = (unsigned __int64)v30;
                  }
                  v27[4] = v28;
                  v31 = 0;
                  if ( v24 != v86 )
                  {
LABEL_47:
                    v31 = (__int64 **)(v24 + 4);
                    if ( (v24[5] & 1) == 0 )
                    {
                      v31 = (__int64 **)v24[4];
                      if ( ((_BYTE)v31[1] & 1) == 0 )
                      {
                        v32 = (__int64 ***)*v31;
                        if ( ((*v31)[1] & 1) != 0 )
                        {
                          v31 = (__int64 **)*v31;
                        }
                        else
                        {
                          v33 = *v32;
                          if ( ((_BYTE)(*v32)[1] & 1) == 0 )
                          {
                            v59 = (__int64 ***)*v33;
                            v93 = (__int64 ***)*v33;
                            if ( ((*v33)[1] & 1) != 0 )
                            {
                              v33 = (__int64 **)*v33;
                            }
                            else
                            {
                              v63 = *v59;
                              if ( ((_BYTE)(*v59)[1] & 1) == 0 )
                              {
                                v88 = *v63;
                                if ( ((*v63)[1] & 1) != 0 )
                                {
                                  v63 = (__int64 **)*v63;
                                }
                                else
                                {
                                  v64 = **v63;
                                  v110 = (__int64 *)v64;
                                  if ( (*(_BYTE *)(v64 + 8) & 1) == 0 )
                                  {
                                    v65 = *(__int64 **)v64;
                                    v99 = v65;
                                    if ( (v65[1] & 1) == 0 )
                                    {
                                      v66 = (__int64 *)*v65;
                                      if ( (*(_BYTE *)(*v65 + 8) & 1) == 0 )
                                      {
                                        v67 = (__int64 *)*v66;
                                        v74 = (_QWORD *)*v65;
                                        if ( (*(_BYTE *)(*v66 + 8) & 1) == 0 )
                                        {
                                          v76 = v63;
                                          v78 = *v32;
                                          v81 = v24;
                                          v83 = (_BYTE *)v28;
                                          v85 = (__int64 **)v24[4];
                                          v68 = sub_2785310(v67);
                                          v63 = v76;
                                          v33 = v78;
                                          v67 = (__int64 *)v68;
                                          *v74 = v68;
                                          v24 = v81;
                                          v28 = (unsigned __int64)v83;
                                          v31 = v85;
                                        }
                                        v66 = v67;
                                        *v99 = (__int64)v67;
                                      }
                                      v99 = v66;
                                      *v110 = (__int64)v66;
                                    }
                                    v110 = v99;
                                    *v88 = (__int64)v99;
                                  }
                                  *v63 = v110;
                                  v63 = (__int64 **)v110;
                                }
                                *v93 = v63;
                              }
                              *v33 = (__int64 *)v63;
                              v33 = v63;
                            }
                            *v32 = v33;
                          }
                          *v31 = (__int64 *)v33;
                          v31 = v33;
                        }
                        v24[4] = v31;
                      }
                    }
                  }
                  if ( (__int64 **)v28 != v31 )
                  {
LABEL_54:
                    (*v31)[1] = v28 | (*v31)[1] & 1;
                    *v31 = *(__int64 **)v28;
                    *(_QWORD *)(v28 + 8) &= ~1uLL;
                    *(_QWORD *)v28 = v31;
                  }
LABEL_55:
                  sub_2784ED0((__int64)&v121);
                  v34 = *(unsigned int *)(a1 + 24);
                  v35 = *(_QWORD *)(a1 + 8);
                  if ( (_DWORD)v34 )
                  {
                    v36 = (v34 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
                    v37 = v35 + 16LL * v36;
                    v38 = *(unsigned __int8 **)v37;
                    if ( *(unsigned __int8 **)v37 == v10 )
                    {
LABEL_57:
                      v39 = *(_QWORD *)(a1 + 32);
                      if ( v37 != v35 + 16 * v34 )
                      {
                        v40 = v39 + 40LL * *(unsigned int *)(v37 + 8);
                        goto LABEL_59;
                      }
LABEL_127:
                      v40 = v39 + 40LL * *(unsigned int *)(a1 + 40);
LABEL_59:
                      if ( *(_DWORD *)(v40 + 16) <= 0x40u )
                      {
                        if ( *(const void **)(v40 + 8) == v121 )
                          goto LABEL_61;
                        v41 = 0;
                      }
                      else
                      {
                        v101 = v40;
                        v41 = sub_C43C50(v40 + 8, &v121);
                        v40 = v101;
                        if ( v41 )
                        {
LABEL_61:
                          if ( *(_DWORD *)(v40 + 32) <= 0x40u )
                            v41 = *(_QWORD *)(v40 + 24) == v123;
                          else
                            v41 = sub_C43C50(v40 + 24, (const void **)&v123);
                        }
                      }
                      if ( v124 > 0x40 && v123 )
                      {
                        v102 = v41;
                        j_j___libc_free_0_0(v123);
                        v41 = v102;
                      }
                      if ( (unsigned int)v122 > 0x40 && v121 )
                      {
                        v103 = v41;
                        j_j___libc_free_0_0((unsigned __int64)v121);
                        v41 = v103;
                      }
                      if ( !v41 )
                      {
                        v54 = v130;
                        if ( v130 == (unsigned __int64 *)(v132 - 8) )
                        {
                          sub_2785520((unsigned __int64 *)v125, &v117);
                        }
                        else
                        {
                          if ( v130 )
                          {
                            *v130 = v117;
                            v54 = v130;
                          }
                          v130 = v54 + 1;
                        }
                      }
                      goto LABEL_70;
                    }
                    v55 = 1;
                    while ( v38 != (unsigned __int8 *)-4096LL )
                    {
                      v61 = v55 + 1;
                      v36 = (v34 - 1) & (v55 + v36);
                      v37 = v35 + 16LL * v36;
                      v38 = *(unsigned __int8 **)v37;
                      if ( *(unsigned __int8 **)v37 == v10 )
                        goto LABEL_57;
                      v55 = v61;
                    }
                  }
                  v39 = *(_QWORD *)(a1 + 32);
                  goto LABEL_127;
                }
LABEL_105:
                v28 = 0;
                if ( v24 != v86 )
                  goto LABEL_47;
                goto LABEL_55;
              }
LABEL_102:
              v46 = 1;
              if ( v86 != v20 )
                v46 = (unsigned __int64)v19 < v20[6];
            }
            v87 = v24;
            v90 = v19;
            v104 = v20;
            v95 = v46;
            v47 = (_QWORD *)sub_22077B0(0x38u);
            v48 = v104;
            v47[6] = v90;
            v47[4] = v47 + 4;
            v47[5] = 1;
            v105 = v47;
            sub_220F040(v95, (__int64)v47, v48, v86);
            v27 = v105;
            ++*(_QWORD *)(a1 + 200);
            v24 = v87;
            if ( v86 != v105 )
              goto LABEL_41;
            goto LABEL_105;
          }
          while ( 1 )
          {
LABEL_35:
            v25 = v20[6];
            v26 = (_QWORD *)v20[3];
            if ( (unsigned __int64)v19 < v25 )
              v26 = (_QWORD *)v20[2];
            if ( !v26 )
              break;
            v20 = v26;
          }
          if ( (unsigned __int64)v19 < v25 )
          {
            if ( v20 != *(_QWORD **)(a1 + 184) )
              goto LABEL_114;
          }
          else if ( v25 >= (unsigned __int64)v19 )
          {
            goto LABEL_40;
          }
          goto LABEL_102;
        }
        if ( *(_QWORD **)(a1 + 184) == i )
          goto LABEL_109;
      }
      else
      {
        i = (_QWORD *)(a1 + 168);
        if ( v86 == *(_QWORD **)(a1 + 184) )
        {
          v50 = 1;
          goto LABEL_111;
        }
      }
      v91 = v19;
      v96 = v20;
      v106 = i;
      v49 = sub_220EF80((__int64)i);
      i = v106;
      v20 = v96;
      v19 = v91;
      v24 = (_QWORD *)v49;
      if ( *(_QWORD *)(v49 + 48) < (unsigned __int64)v10 )
        goto LABEL_109;
      goto LABEL_112;
    }
LABEL_12:
    if ( v14 == (unsigned __int8 **)(v12 + 16 * v11) )
      goto LABEL_13;
  }
LABEL_100:
  sub_2784FD0((unsigned __int64 *)v125);
}
