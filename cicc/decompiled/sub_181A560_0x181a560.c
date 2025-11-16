// Function: sub_181A560
// Address: 0x181a560
//
__int64 __fastcall sub_181A560(__int64 *a1, _BYTE *a2, unsigned __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rbx
  __int64 v5; // rax
  _BYTE *v6; // rax
  unsigned __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 *v10; // r15
  _QWORD *v11; // r12
  int v12; // edx
  unsigned int v13; // eax
  __int64 v14; // rsi
  int v15; // edi
  unsigned int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // r12
  __int64 v19; // r14
  __int64 v20; // r13
  __int64 v21; // r15
  unsigned __int64 v22; // r14
  unsigned __int64 v23; // rbx
  __int64 v24; // r13
  _QWORD *v25; // r14
  __int64 v26; // r15
  unsigned __int64 v27; // r12
  unsigned __int64 v28; // rbx
  char v29; // al
  unsigned int v30; // esi
  int v31; // eax
  int v32; // eax
  __int64 v33; // rax
  __int64 v34; // rbx
  __int64 *v35; // rax
  __int64 *v36; // rax
  __int64 *v37; // rax
  __int64 *v38; // rax
  __int64 *v39; // rax
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r14
  int *v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 *v48; // rdx
  __int64 *v49; // rax
  __int64 *v50; // rdx
  __int64 v51; // r14
  __int64 v52; // rax
  __int64 v53; // rdi
  _QWORD *v54; // r14
  __int64 v55; // r12
  _BOOL4 v56; // r15d
  __int64 v57; // rax
  _QWORD *v58; // rax
  __int64 *v59; // rdx
  int v61; // edi
  _QWORD *v62; // rax
  _QWORD *v63; // rsi
  __int64 v64; // rcx
  __int64 v65; // rdx
  char v66; // al
  __int64 v67; // r12
  __int64 v68; // rdi
  _BYTE *v69; // r13
  __int64 v70; // rax
  __int64 v71; // rsi
  __int64 v72; // rax
  __int64 v73; // r14
  __int64 v74; // rbx
  __int64 *v75; // rax
  _BYTE *v76; // rax
  __int64 *v77; // rax
  _BYTE *v78; // rax
  __int64 *v79; // rax
  __int64 v80; // r9
  __int64 v81; // rax
  __int64 v82; // rcx
  __int64 v83; // r8
  __int64 v84; // r9
  __int64 v85; // r14
  __int64 v86; // rcx
  __int64 v87; // r8
  __int64 v88; // r9
  unsigned int v89; // esi
  int v90; // eax
  int v91; // eax
  __int64 v92; // rax
  _QWORD *v93; // rax
  __int64 *v94; // rdx
  __int64 *v95; // r12
  _BOOL4 v96; // r13d
  __int64 v97; // rax
  _QWORD *v98; // rax
  __int64 *v99; // rdx
  _BOOL4 v100; // r14d
  __int64 v101; // rax
  _QWORD *v102; // rax
  _QWORD *v103; // rsi
  unsigned __int64 v104; // [rsp+0h] [rbp-170h]
  __int64 *v105; // [rsp+8h] [rbp-168h]
  __int64 v106; // [rsp+10h] [rbp-160h]
  __int64 v107; // [rsp+10h] [rbp-160h]
  __int64 *v108; // [rsp+18h] [rbp-158h]
  __int64 v109; // [rsp+18h] [rbp-158h]
  unsigned __int64 v110; // [rsp+20h] [rbp-150h]
  __int64 v111; // [rsp+20h] [rbp-150h]
  _QWORD *v112; // [rsp+28h] [rbp-148h]
  unsigned __int64 v113; // [rsp+28h] [rbp-148h]
  _QWORD *v114; // [rsp+30h] [rbp-140h]
  _QWORD *v115; // [rsp+38h] [rbp-138h]
  __int64 *v116; // [rsp+38h] [rbp-138h]
  __int64 *v118; // [rsp+40h] [rbp-130h]
  __int64 v119; // [rsp+40h] [rbp-130h]
  __int64 *v120; // [rsp+40h] [rbp-130h]
  _QWORD *v121; // [rsp+48h] [rbp-128h]
  __int64 *v122; // [rsp+48h] [rbp-128h]
  unsigned __int64 v123; // [rsp+50h] [rbp-120h] BYREF
  _BYTE *v124; // [rsp+58h] [rbp-118h] BYREF
  unsigned __int64 v125; // [rsp+60h] [rbp-110h] BYREF
  unsigned __int64 v126; // [rsp+68h] [rbp-108h]
  _QWORD v127[2]; // [rsp+70h] [rbp-100h] BYREF
  _BYTE *v128; // [rsp+80h] [rbp-F0h] BYREF
  unsigned __int64 v129; // [rsp+88h] [rbp-E8h]
  __int64 **v130; // [rsp+90h] [rbp-E0h]
  __int64 v131[10]; // [rsp+A0h] [rbp-D0h] BYREF
  __int64 *v132; // [rsp+F0h] [rbp-80h] BYREF
  __int64 v133; // [rsp+F8h] [rbp-78h] BYREF
  __int64 v134; // [rsp+100h] [rbp-70h]
  __int64 *v135; // [rsp+108h] [rbp-68h]
  __int64 *v136; // [rsp+110h] [rbp-60h]
  __int64 v137; // [rsp+118h] [rbp-58h]

  v4 = a3;
  v5 = *a1;
  v124 = a2;
  v123 = a3;
  v6 = *(_BYTE **)(v5 + 200);
  if ( v6 == a2 )
    return v4;
  v7 = (unsigned __int64)a2;
  if ( a2 == (_BYTE *)a3 || v6 == (_BYTE *)a3 )
    return v7;
  v8 = *((unsigned int *)a1 + 84);
  v9 = a1[40];
  v10 = a1;
  v11 = (_QWORD *)(v9 + 56 * v8);
  if ( !(_DWORD)v8 )
  {
LABEL_110:
    v115 = v11;
    goto LABEL_24;
  }
  v12 = *((_DWORD *)a1 + 84) - 1;
  v13 = v12 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v14 = *(_QWORD *)(v9 + 56LL * v13);
  v115 = (_QWORD *)(v9 + 56LL * v13);
  v15 = 1;
  if ( v7 != v14 )
  {
    while ( v14 != -8 )
    {
      v13 = v12 & (v15 + v13);
      v14 = *(_QWORD *)(v9 + 56LL * v13);
      if ( v7 == v14 )
      {
        v115 = (_QWORD *)(v9 + 56LL * v13);
        goto LABEL_6;
      }
      ++v15;
    }
    v115 = v11;
    v16 = v12 & (((unsigned int)v4 >> 4) ^ ((unsigned int)v4 >> 9));
    v17 = *(_QWORD *)(v9 + 56LL * v16);
    if ( v17 != v4 )
      goto LABEL_60;
    v11 = (_QWORD *)(v9 + 56LL * (v12 & (((unsigned int)v4 >> 4) ^ ((unsigned int)v4 >> 9))));
LABEL_118:
    if ( v115 != v11 )
    {
      v102 = (_QWORD *)v11[3];
      if ( v102 )
      {
        v103 = v11 + 2;
        do
        {
          if ( v102[4] < v7 )
          {
            v102 = (_QWORD *)v102[3];
          }
          else
          {
            v103 = v102;
            v102 = (_QWORD *)v102[2];
          }
        }
        while ( v102 );
        if ( v11 + 2 != v103 && v103[4] <= v7 )
          return v4;
      }
    }
LABEL_24:
    v125 = v7;
    v126 = v4;
    if ( v7 > v4 )
    {
      v125 = v4;
      v126 = v7;
    }
    v29 = sub_1814640((__int64)(v10 + 35), (__int64 *)&v125, &v132);
    v122 = v132;
    if ( v29 )
    {
      v71 = v132[2];
      if ( v71 && sub_15CC8F0((__int64)(v10 + 2), v71, *(_QWORD *)(a4 + 40)) )
        return v122[3];
LABEL_32:
      sub_17CE510((__int64)v131, a4, 0, 0, 0);
      if ( *((_BYTE *)v10 + 272) )
      {
        LOWORD(v134) = 257;
        v128 = v124;
        v129 = v123;
        v34 = sub_1285290(
                v131,
                *(_QWORD *)(**(_QWORD **)(*v10 + 336) + 24LL),
                *(_QWORD *)(*v10 + 336),
                (int)&v128,
                2,
                (__int64)&v132,
                0);
        v132 = *(__int64 **)(v34 + 56);
        v35 = (__int64 *)sub_16498A0(v34);
        v36 = (__int64 *)sub_1563AB0((__int64 *)&v132, v35, 0, 58);
        *(_QWORD *)(v34 + 56) = v36;
        v132 = v36;
        v37 = (__int64 *)sub_16498A0(v34);
        v38 = (__int64 *)sub_1563AB0((__int64 *)&v132, v37, 1, 58);
        *(_QWORD *)(v34 + 56) = v38;
        v132 = v38;
        v39 = (__int64 *)sub_16498A0(v34);
        *(_QWORD *)(v34 + 56) = sub_1563AB0((__int64 *)&v132, v39, 2, 58);
        v40 = *(_QWORD *)(a4 + 40);
        v122[3] = v34;
        v122[2] = v40;
      }
      else
      {
        v111 = *(_QWORD *)(a4 + 40);
        LOWORD(v134) = 257;
        v72 = sub_12AA0C0(v131, 0x21u, v124, v123, (__int64)&v132);
        v73 = sub_1AA92B0(v72, a4, 0, *(_QWORD *)(*v10 + 384), v10 + 2, 0);
        sub_17CE510((__int64)&v132, v73, 0, 0, 0);
        LOWORD(v130) = 257;
        v127[0] = v124;
        v127[1] = v123;
        v74 = sub_1285290(
                (__int64 *)&v132,
                *(_QWORD *)(**(_QWORD **)(*v10 + 328) + 24LL),
                *(_QWORD *)(*v10 + 328),
                (int)v127,
                2,
                (__int64)&v128,
                0);
        v128 = *(_BYTE **)(v74 + 56);
        v75 = (__int64 *)sub_16498A0(v74);
        v76 = (_BYTE *)sub_1563AB0((__int64 *)&v128, v75, 0, 58);
        *(_QWORD *)(v74 + 56) = v76;
        v128 = v76;
        v77 = (__int64 *)sub_16498A0(v74);
        v78 = (_BYTE *)sub_1563AB0((__int64 *)&v128, v77, 1, 58);
        *(_QWORD *)(v74 + 56) = v78;
        v128 = v78;
        v79 = (__int64 *)sub_16498A0(v74);
        *(_QWORD *)(v74 + 56) = sub_1563AB0((__int64 *)&v128, v79, 2, 58);
        v80 = *(_QWORD *)(*(_QWORD *)(v73 - 24) + 48LL);
        v119 = *(_QWORD *)(v73 - 24);
        LOWORD(v130) = 257;
        if ( v80 )
          v80 -= 24;
        v107 = v80;
        v109 = *(_QWORD *)(*v10 + 176);
        v81 = sub_1648B60(64);
        v85 = v81;
        if ( v81 )
        {
          sub_15F1EA0(v81, v109, 53, 0, 0, v107);
          *(_DWORD *)(v85 + 56) = 2;
          sub_164B780(v85, (__int64 *)&v128);
          sub_1648880(v85, *(_DWORD *)(v85 + 56), 1);
        }
        sub_1704F80(v85, v74, *(_QWORD *)(v74 + 40), v82, v83, v84);
        sub_1704F80(v85, (__int64)v124, v111, v86, v87, v88);
        v122[2] = v119;
        v122[3] = v85;
        sub_17CD270((__int64 *)&v132);
      }
      v41 = *((unsigned int *)v10 + 84);
      LODWORD(v133) = 0;
      v134 = 0;
      v135 = &v133;
      v42 = 7 * v41;
      v43 = v10[40];
      v136 = &v133;
      v137 = 0;
      v44 = v43 + 8 * v42;
      if ( (_QWORD *)v44 == v115 )
      {
        v98 = sub_1819210((__int64)&v132, (unsigned __int64 *)&v124);
        if ( v99 )
        {
          v100 = v98 || v99 == &v133 || (unsigned __int64)v124 < v99[4];
          v120 = v99;
          v101 = sub_22077B0(40);
          *(_QWORD *)(v101 + 32) = v124;
          sub_220F040(v100, v101, v120, &v133);
          ++v137;
          goto LABEL_43;
        }
      }
      else if ( &v132 != v115 + 1 )
      {
        v128 = 0;
        v130 = &v132;
        v129 = 0;
        v45 = (int *)v115[3];
        if ( v45 )
        {
          v46 = sub_1814A30(v45, (__int64)&v133, &v128);
          v47 = v46;
          do
          {
            v48 = (__int64 *)v46;
            v46 = *(_QWORD *)(v46 + 16);
          }
          while ( v46 );
          v135 = v48;
          v49 = (__int64 *)v47;
          do
          {
            v50 = v49;
            v49 = (__int64 *)v49[3];
          }
          while ( v49 );
          v136 = v50;
          v51 = (__int64)v128;
          v52 = v115[6];
          v134 = v47;
          v137 = v52;
          if ( v128 )
          {
            do
            {
              sub_1814E60(*(_QWORD *)(v51 + 24));
              v53 = v51;
              v51 = *(_QWORD *)(v51 + 16);
              j_j___libc_free_0(v53, 40);
            }
            while ( v51 );
LABEL_43:
            v44 = v10[40] + 56LL * *((unsigned int *)v10 + 84);
            goto LABEL_44;
          }
          v44 = v10[40] + 56LL * *((unsigned int *)v10 + 84);
        }
      }
LABEL_44:
      if ( (_QWORD *)v44 == v11 )
      {
        v93 = sub_1819210((__int64)&v132, &v123);
        v95 = v94;
        if ( v94 )
        {
          v96 = v93 || v94 == &v133 || v123 < v94[4];
          v97 = sub_22077B0(40);
          *(_QWORD *)(v97 + 32) = v123;
          sub_220F040(v96, v97, v95, &v133);
          ++v137;
        }
      }
      else
      {
        v54 = v11 + 2;
        v55 = v11[4];
        if ( (_QWORD *)v55 != v54 )
        {
          v116 = v10;
          do
          {
            v58 = sub_1819AD0(&v132, &v133, (unsigned __int64 *)(v55 + 32));
            if ( v59 )
            {
              v56 = v58 || v59 == &v133 || *(_QWORD *)(v55 + 32) < (unsigned __int64)v59[4];
              v118 = v59;
              v57 = sub_22077B0(40);
              *(_QWORD *)(v57 + 32) = *(_QWORD *)(v55 + 32);
              sub_220F040(v56, v57, v118, &v133);
              ++v137;
            }
            v55 = sub_220EF30(v55);
          }
          while ( v54 != (_QWORD *)v55 );
          v10 = v116;
        }
      }
      v66 = sub_1819BD0((__int64)(v10 + 39), v122 + 3, &v128);
      v67 = (__int64)v128;
      if ( v66 )
      {
        v68 = *((_QWORD *)v128 + 3);
        v69 = v128 + 16;
LABEL_75:
        sub_1814E60(v68);
        *(_QWORD *)(v67 + 32) = v69;
        *(_QWORD *)(v67 + 24) = 0;
        *(_QWORD *)(v67 + 40) = v69;
        *(_QWORD *)(v67 + 48) = 0;
        if ( v134 )
        {
          *(_DWORD *)(v67 + 16) = v133;
          v70 = v134;
          *(_QWORD *)(v67 + 24) = v134;
          *(_QWORD *)(v67 + 32) = v135;
          *(_QWORD *)(v67 + 40) = v136;
          *(_QWORD *)(v70 + 8) = v69;
          *(_QWORD *)(v67 + 48) = v137;
          v134 = 0;
          v135 = &v133;
          v136 = &v133;
          v137 = 0;
        }
        v7 = v122[3];
        sub_1814E60(0);
        if ( v131[0] )
          sub_161E7C0((__int64)v131, v131[0]);
        return v7;
      }
      v89 = *((_DWORD *)v10 + 84);
      v90 = *((_DWORD *)v10 + 82);
      ++v10[39];
      v91 = v90 + 1;
      if ( 4 * v91 >= 3 * v89 )
      {
        v89 *= 2;
      }
      else if ( v89 - *((_DWORD *)v10 + 83) - v91 > v89 >> 3 )
      {
LABEL_89:
        *((_DWORD *)v10 + 82) = v91;
        if ( *(_QWORD *)v67 != -8 )
          --*((_DWORD *)v10 + 83);
        v69 = (_BYTE *)(v67 + 16);
        v68 = 0;
        v92 = v122[3];
        *(_QWORD *)(v67 + 32) = v67 + 16;
        *(_DWORD *)(v67 + 16) = 0;
        *(_QWORD *)v67 = v92;
        *(_QWORD *)(v67 + 24) = 0;
        *(_QWORD *)(v67 + 40) = v67 + 16;
        *(_QWORD *)(v67 + 48) = 0;
        goto LABEL_75;
      }
      sub_181A2A0((__int64)(v10 + 39), v89);
      sub_1819BD0((__int64)(v10 + 39), v122 + 3, &v128);
      v67 = (__int64)v128;
      v91 = *((_DWORD *)v10 + 82) + 1;
      goto LABEL_89;
    }
    v30 = *((_DWORD *)v10 + 76);
    v31 = *((_DWORD *)v10 + 74);
    ++v10[35];
    v32 = v31 + 1;
    if ( 4 * v32 >= 3 * v30 )
    {
      v30 *= 2;
    }
    else if ( v30 - *((_DWORD *)v10 + 75) - v32 > v30 >> 3 )
    {
      goto LABEL_29;
    }
    sub_18153E0((__int64)(v10 + 35), v30);
    sub_1814640((__int64)(v10 + 35), (__int64 *)&v125, &v132);
    v122 = v132;
    v32 = *((_DWORD *)v10 + 74) + 1;
LABEL_29:
    *((_DWORD *)v10 + 74) = v32;
    if ( *v122 != -8 || v122[1] != -8 )
      --*((_DWORD *)v10 + 75);
    *v122 = v125;
    v33 = v126;
    v122[2] = 0;
    v122[1] = v33;
    v122[3] = 0;
    goto LABEL_32;
  }
LABEL_6:
  v16 = v12 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v17 = *(_QWORD *)(v9 + 56LL * v16);
  v114 = (_QWORD *)(v9 + 56LL * v16);
  if ( v17 != v4 )
  {
LABEL_60:
    v61 = 1;
    while ( v17 != -8 )
    {
      v16 = v12 & (v61 + v16);
      v17 = *(_QWORD *)(v9 + 56LL * v16);
      if ( v17 == v4 )
      {
        v114 = (_QWORD *)(v9 + 56LL * v16);
        goto LABEL_7;
      }
      ++v61;
    }
    if ( v115 == v11 )
      goto LABEL_110;
    v121 = v115 + 2;
    goto LABEL_64;
  }
LABEL_7:
  if ( v115 == v11 )
  {
    v11 = v114;
    goto LABEL_118;
  }
  v121 = v115 + 2;
  if ( v11 != v114 )
  {
    v18 = v114[4];
    v19 = v115[4];
    v112 = v114 + 2;
    if ( (_QWORD *)v18 == v114 + 2 )
      return v7;
    if ( (_QWORD *)v19 == v115 + 2 )
    {
      if ( (_QWORD *)v18 == v112 )
        return v7;
    }
    else
    {
      v110 = v7;
      v20 = v115[4];
      v105 = v10;
      v21 = v114[4];
      v106 = v20;
      v104 = v4;
      do
      {
        v22 = *(_QWORD *)(v21 + 32);
        v23 = *(_QWORD *)(v20 + 32);
        if ( v22 < v23 )
        {
          v24 = v18;
          v25 = v114 + 2;
          v26 = v106;
          v108 = v105;
          v113 = v104;
          goto LABEL_22;
        }
        v20 = sub_220EF30(v20);
        if ( v22 <= v23 )
          v21 = sub_220EF30(v21);
        if ( v112 == (_QWORD *)v21 )
          return v110;
      }
      while ( v121 != (_QWORD *)v20 );
      v7 = v110;
      if ( v112 == (_QWORD *)v21 )
        return v7;
      v24 = v18;
      v108 = v105;
      v26 = v106;
      v25 = v114 + 2;
      v113 = v104;
      do
      {
LABEL_22:
        v27 = *(_QWORD *)(v26 + 32);
        v28 = *(_QWORD *)(v24 + 32);
        if ( v27 < v28 )
        {
          v7 = v110;
          v10 = v108;
          v4 = v113;
          v11 = v114;
          goto LABEL_24;
        }
        v24 = sub_220EF30(v24);
        if ( v27 <= v28 )
          v26 = sub_220EF30(v26);
        if ( (_QWORD *)v26 == v121 )
          return v113;
      }
      while ( v25 != (_QWORD *)v24 );
      v19 = v26;
      v7 = v110;
      v10 = v108;
      v4 = v113;
    }
    if ( (_QWORD *)v19 != v121 )
    {
      v11 = v114;
      goto LABEL_24;
    }
    return v4;
  }
LABEL_64:
  v62 = (_QWORD *)v115[3];
  if ( !v62 )
    goto LABEL_24;
  v63 = v121;
  do
  {
    while ( 1 )
    {
      v64 = v62[2];
      v65 = v62[3];
      if ( v62[4] >= v4 )
        break;
      v62 = (_QWORD *)v62[3];
      if ( !v65 )
        goto LABEL_69;
    }
    v63 = v62;
    v62 = (_QWORD *)v62[2];
  }
  while ( v64 );
LABEL_69:
  if ( v121 == v63 || v63[4] > v4 )
    goto LABEL_24;
  return v7;
}
