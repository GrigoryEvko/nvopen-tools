// Function: sub_1CC43E0
// Address: 0x1cc43e0
//
__int64 __fastcall sub_1CC43E0(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // rax
  unsigned __int8 *v8; // r12
  size_t v9; // r15
  unsigned int v10; // r13d
  _QWORD *v11; // r9
  __int64 v12; // rax
  __int64 *v13; // r9
  __int64 v14; // rcx
  __int64 v15; // rdi
  unsigned int v16; // r15d
  __int64 v17; // rbx
  unsigned __int64 v18; // r12
  unsigned __int64 v19; // r13
  unsigned __int8 *v20; // rax
  size_t v21; // rdx
  int v22; // eax
  unsigned __int64 v23; // rax
  _QWORD *i; // rdi
  __int64 v25; // rbx
  __int64 v26; // r14
  __int64 v27; // rbx
  __int64 j; // r13
  char *v29; // r8
  __int64 v30; // rbx
  int *v31; // rax
  unsigned __int64 v32; // rsi
  int *v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rdx
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rax
  __int64 v38; // r13
  __int64 v39; // rbx
  unsigned int v40; // r12d
  unsigned __int64 v41; // r14
  __int64 v42; // r13
  __int64 v43; // r8
  __int64 v44; // rdi
  unsigned __int64 v45; // rdx
  __int64 v46; // r14
  __int64 v47; // rax
  void *v48; // rax
  unsigned __int64 v49; // r12
  _QWORD *v50; // rax
  __int64 v51; // rdx
  __int64 v52; // r15
  _BOOL4 v53; // r8d
  __int64 v54; // rax
  _QWORD *v55; // rax
  _QWORD *v56; // rax
  _QWORD *v57; // rdx
  __int64 v58; // rdi
  _BYTE *v59; // rax
  __int64 v60; // r8
  __int64 v61; // rax
  const char *v62; // rax
  unsigned __int64 v63; // rdx
  __int64 v64; // r8
  const char *v65; // r10
  size_t v66; // r9
  unsigned __int64 v67; // rax
  char *v68; // rdx
  __int64 v69; // rdx
  __int64 v70; // r13
  __int64 v71; // rbx
  __int64 v72; // r12
  char v73; // al
  unsigned int v74; // r13d
  unsigned __int64 v75; // r8
  __int64 v76; // r12
  __int64 v77; // rbx
  unsigned __int64 v78; // rdi
  __int64 v79; // rdi
  __int64 *v80; // rbx
  __int64 *v81; // r12
  __int64 v82; // rdi
  __int64 *v84; // r15
  __int64 v85; // rdx
  size_t v86; // rdx
  char *v87; // rsi
  __int64 v88; // rdi
  _BYTE *v89; // rax
  _QWORD *v90; // rax
  __int64 v91; // rdx
  __int64 v92; // r13
  _BOOL4 v93; // r12d
  __int64 v94; // rax
  __int64 v95; // rax
  _QWORD *v96; // rdi
  size_t n; // [rsp+0h] [rbp-150h]
  unsigned int v98; // [rsp+8h] [rbp-148h]
  unsigned int v99; // [rsp+Ch] [rbp-144h]
  const char *src; // [rsp+10h] [rbp-140h]
  __int64 v102; // [rsp+20h] [rbp-130h]
  __int64 v103; // [rsp+28h] [rbp-128h]
  __int64 v104; // [rsp+30h] [rbp-120h]
  __int64 v105; // [rsp+30h] [rbp-120h]
  __int64 v106; // [rsp+30h] [rbp-120h]
  __int64 *v107; // [rsp+38h] [rbp-118h]
  __int64 v108; // [rsp+38h] [rbp-118h]
  __int64 v109; // [rsp+38h] [rbp-118h]
  __int64 v110; // [rsp+40h] [rbp-110h]
  __int64 *v111; // [rsp+40h] [rbp-110h]
  char v112; // [rsp+48h] [rbp-108h]
  __int64 *v113; // [rsp+48h] [rbp-108h]
  _BOOL4 v114; // [rsp+48h] [rbp-108h]
  unsigned __int64 v115; // [rsp+58h] [rbp-F8h] BYREF
  unsigned __int64 v116; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v117; // [rsp+68h] [rbp-E8h]
  __int64 v118; // [rsp+70h] [rbp-E0h]
  char *v119; // [rsp+80h] [rbp-D0h] BYREF
  char *v120; // [rsp+88h] [rbp-C8h]
  _QWORD v121[2]; // [rsp+90h] [rbp-C0h] BYREF
  char v122[8]; // [rsp+A0h] [rbp-B0h] BYREF
  int v123; // [rsp+A8h] [rbp-A8h] BYREF
  int *v124; // [rsp+B0h] [rbp-A0h]
  int *v125; // [rsp+B8h] [rbp-98h]
  int *v126; // [rsp+C0h] [rbp-90h]
  __int64 v127; // [rsp+C8h] [rbp-88h]
  __int64 v128; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v129; // [rsp+D8h] [rbp-78h]
  _QWORD *v130; // [rsp+E0h] [rbp-70h]
  __int64 v131; // [rsp+E8h] [rbp-68h]
  __int64 v132; // [rsp+F0h] [rbp-60h]
  unsigned __int64 v133; // [rsp+F8h] [rbp-58h]
  _QWORD *v134; // [rsp+100h] [rbp-50h]
  _QWORD *v135; // [rsp+108h] [rbp-48h]
  __int64 v136; // [rsp+110h] [rbp-40h]
  __int64 *v137; // [rsp+118h] [rbp-38h]

  v128 = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v129 = 8;
  v128 = sub_22077B0(64);
  v1 = (__int64 *)(v128 + ((4 * v129 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v2 = sub_22077B0(512);
  v3 = qword_4FBEFC0;
  v133 = (unsigned __int64)v1;
  *v1 = v2;
  v131 = v2;
  v135 = (_QWORD *)v2;
  v130 = (_QWORD *)v2;
  v134 = (_QWORD *)v2;
  v118 = 0x1000000000LL;
  v132 = v2 + 512;
  v103 = qword_4FBEFC8;
  v4 = (qword_4FBEFC8 - v3) >> 5;
  v137 = v1;
  v136 = v132;
  v123 = 0;
  v124 = 0;
  v125 = &v123;
  v126 = &v123;
  v127 = 0;
  v116 = 0;
  v117 = 0;
  v102 = v3;
  if ( (_DWORD)v4 )
  {
    v5 = 0;
    v104 = 32LL * (unsigned int)(v4 - 1);
    v6 = v3;
    while ( 1 )
    {
      v7 = v5 + v6;
      v8 = *(unsigned __int8 **)v7;
      v9 = *(_QWORD *)(v7 + 8);
      v10 = sub_16D19C0((__int64)&v116, *(unsigned __int8 **)v7, v9);
      v11 = (_QWORD *)(v116 + 8LL * v10);
      if ( *v11 )
      {
        if ( *v11 != -8 )
        {
          if ( v104 == v5 )
            goto LABEL_12;
          goto LABEL_4;
        }
        LODWORD(v118) = v118 - 1;
      }
      v107 = (__int64 *)(v116 + 8LL * v10);
      v12 = malloc(v9 + 17);
      v13 = v107;
      v14 = v12;
      if ( v12 )
        break;
      if ( v9 != -17 || (v47 = malloc(1u), v13 = v107, v14 = 0, !v47) )
      {
        v109 = v14;
        v111 = v13;
        sub_16BD1C0("Allocation failed", 1u);
        v13 = v111;
        v15 = 16;
        v14 = v109;
LABEL_10:
        if ( v9 + 1 <= 1 )
          goto LABEL_11;
        goto LABEL_67;
      }
      v15 = v47 + 16;
      v14 = v47;
LABEL_67:
      v110 = v14;
      v113 = v13;
      v48 = memcpy((void *)v15, v8, v9);
      v14 = v110;
      v13 = v113;
      v15 = (__int64)v48;
LABEL_11:
      *(_BYTE *)(v15 + v9) = 0;
      *(_QWORD *)v14 = v9;
      *(_BYTE *)(v14 + 8) = 0;
      *v13 = v14;
      ++HIDWORD(v117);
      sub_16D1CD0((__int64)&v116, v10);
      if ( v104 == v5 )
        goto LABEL_12;
LABEL_4:
      v6 = qword_4FBEFC0;
      v5 += 32;
    }
    v15 = v12 + 16;
    goto LABEL_10;
  }
LABEL_12:
  if ( (unsigned __int64)(qword_4FBEEE8 - qword_4FBEEE0) <= 4
    || (v99 = *(_DWORD *)qword_4FBEEE0, v98 = *(_DWORD *)(qword_4FBEEE0 + 4), *(_DWORD *)qword_4FBEEE0 > v98) )
  {
    if ( v103 == v102 )
    {
      v74 = 0;
      goto LABEL_106;
    }
    v112 = 0;
  }
  else
  {
    v112 = 1;
  }
  v16 = 0;
  v17 = *(_QWORD *)(a1 + 32);
  v108 = a1 + 24;
  if ( v17 != a1 + 24 )
  {
    while ( 1 )
    {
      v18 = v17 - 56;
      if ( !v17 )
        v18 = 0;
      if ( sub_15E4F60(v18) || !(unsigned __int8)sub_1C2F070(v18) )
        goto LABEL_27;
      ++v16;
      if ( v103 == v102
        || ((v19 = v116 + 8LL * (unsigned int)v117,
             v20 = (unsigned __int8 *)sub_1649960(v18),
             v22 = sub_16D1B30((__int64 *)&v116, v20, v21),
             v22 == -1)
          ? (v23 = v116 + 8LL * (unsigned int)v117)
          : (v23 = v116 + 8LL * v22),
            v23 == v19) )
      {
        if ( !v112 || v16 < v99 || v16 > v98 )
          goto LABEL_27;
      }
      v56 = sub_16E8CB0();
      v57 = (_QWORD *)v56[3];
      v58 = (__int64)v56;
      if ( v56[2] - (_QWORD)v57 <= 7u )
      {
        v58 = sub_16E7EE0((__int64)v56, "Select: ", 8u);
        v59 = *(_BYTE **)(v58 + 24);
        if ( *(_BYTE **)(v58 + 16) == v59 )
        {
LABEL_87:
          v58 = sub_16E7EE0(v58, "#", 1u);
          goto LABEL_88;
        }
      }
      else
      {
        *v57 = 0x203A7463656C6553LL;
        v59 = (_BYTE *)(v56[3] + 8LL);
        *(_QWORD *)(v58 + 24) = v59;
        if ( *(_BYTE **)(v58 + 16) == v59 )
          goto LABEL_87;
      }
      *v59 = 35;
      ++*(_QWORD *)(v58 + 24);
LABEL_88:
      v60 = sub_16E7A90(v58, v16);
      v61 = *(_QWORD *)(v60 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(v60 + 16) - v61) <= 2 )
      {
        v60 = sub_16E7EE0(v60, " : ", 3u);
      }
      else
      {
        *(_BYTE *)(v61 + 2) = 32;
        *(_WORD *)v61 = 14880;
        *(_QWORD *)(v60 + 24) += 3LL;
      }
      v105 = v60;
      v62 = sub_1649960(v18);
      v64 = v105;
      v65 = v62;
      v66 = v63;
      if ( v62 )
      {
        v115 = v63;
        v67 = v63;
        v119 = (char *)v121;
        if ( v63 > 0xF )
        {
          n = v63;
          src = v65;
          v95 = sub_22409D0(&v119, &v115, 0);
          v64 = v105;
          v65 = src;
          v119 = (char *)v95;
          v96 = (_QWORD *)v95;
          v66 = n;
          v121[0] = v115;
        }
        else
        {
          if ( v63 == 1 )
          {
            LOBYTE(v121[0]) = *v65;
            v68 = (char *)v121;
LABEL_147:
            v120 = (char *)v67;
            v68[v67] = 0;
            v86 = (size_t)v120;
            v87 = v119;
            goto LABEL_131;
          }
          if ( !v63 )
          {
            v68 = (char *)v121;
            goto LABEL_147;
          }
          v96 = v121;
        }
        v106 = v64;
        memcpy(v96, v65, v66);
        v67 = v115;
        v68 = v119;
        v64 = v106;
        goto LABEL_147;
      }
      LOBYTE(v121[0]) = 0;
      v86 = 0;
      v119 = (char *)v121;
      v87 = (char *)v121;
      v120 = 0;
LABEL_131:
      v88 = sub_16E7EE0(v64, v87, v86);
      v89 = *(_BYTE **)(v88 + 24);
      if ( *(_BYTE **)(v88 + 16) == v89 )
      {
        sub_16E7EE0(v88, "\n", 1u);
      }
      else
      {
        *v89 = 10;
        ++*(_QWORD *)(v88 + 24);
      }
      if ( v119 != (char *)v121 )
        j_j___libc_free_0(v119, v121[0] + 1LL);
      v119 = (char *)v18;
      sub_1CC42C0(&v128, &v119);
      v119 = (char *)v18;
      v90 = sub_1CC3D00((__int64)v122, (unsigned __int64 *)&v119);
      v92 = v91;
      if ( v91 )
      {
        v93 = v90 || (int *)v91 == &v123 || v18 < *(_QWORD *)(v91 + 32);
        v94 = sub_22077B0(40);
        *(_QWORD *)(v94 + 32) = v119;
        sub_220F040(v93, v94, v92, &v123);
        ++v127;
        v17 = *(_QWORD *)(v17 + 8);
        if ( v108 == v17 )
          break;
      }
      else
      {
LABEL_27:
        v17 = *(_QWORD *)(v17 + 8);
        if ( v108 == v17 )
          break;
      }
    }
  }
  for ( i = v134; v134 != v130; i = v134 )
  {
    if ( v135 == i )
    {
      v25 = *(_QWORD *)(*(v137 - 1) + 504);
      j_j___libc_free_0(i, 512);
      v69 = *--v137 + 512;
      v135 = (_QWORD *)*v137;
      v136 = v69;
      v134 = v135 + 63;
    }
    else
    {
      v25 = *(i - 1);
      v134 = i - 1;
    }
    v26 = v25 + 72;
    v27 = *(_QWORD *)(v25 + 80);
    if ( v26 != v27 )
    {
      while ( 1 )
      {
        if ( !v27 )
LABEL_156:
          BUG();
        j = *(_QWORD *)(v27 + 24);
        if ( j != v27 + 16 )
          break;
        v27 = *(_QWORD *)(v27 + 8);
        if ( v26 == v27 )
          goto LABEL_35;
      }
      while ( v26 != v27 )
      {
        if ( !j )
          BUG();
        if ( *(_BYTE *)(j - 8) == 78 )
        {
          v49 = *(_QWORD *)(j - 48);
          if ( !*(_BYTE *)(v49 + 16) )
          {
            v119 = *(char **)(j - 48);
            v50 = sub_1CC3D00((__int64)v122, (unsigned __int64 *)&v119);
            v52 = v51;
            if ( v51 )
            {
              v53 = v50 || (int *)v51 == &v123 || v49 < *(_QWORD *)(v51 + 32);
              v114 = v53;
              v54 = sub_22077B0(40);
              *(_QWORD *)(v54 + 32) = v119;
              sub_220F040(v114, v54, v52, &v123);
              v55 = v134;
              ++v127;
              if ( v134 == (_QWORD *)(v136 - 8) )
              {
                v84 = v137;
                if ( ((v132 - (__int64)v130) >> 3) + ((((__int64)((__int64)v137 - v133) >> 3) - 1) << 6) + v134 - v135 == 0xFFFFFFFFFFFFFFFLL )
                  sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
                if ( (unsigned __int64)(v129 - (((__int64)v137 - v128) >> 3)) <= 1 )
                {
                  sub_1CC4140(&v128, 1u, 0);
                  v84 = v137;
                }
                v84[1] = sub_22077B0(512);
                if ( v134 )
                  *v134 = v119;
                v85 = *++v137 + 512;
                v135 = (_QWORD *)*v137;
                v136 = v85;
                v134 = v135;
              }
              else
              {
                if ( v134 )
                {
                  *v134 = v119;
                  v55 = v134;
                }
                v134 = v55 + 1;
              }
            }
          }
        }
        for ( j = *(_QWORD *)(j + 8); j == v27 - 24 + 40; j = *(_QWORD *)(v27 + 24) )
        {
          v27 = *(_QWORD *)(v27 + 8);
          if ( v26 == v27 )
            goto LABEL_35;
          if ( !v27 )
            goto LABEL_156;
        }
      }
    }
LABEL_35:
    ;
  }
  v29 = 0;
  v119 = 0;
  v120 = 0;
  v121[0] = 0;
  v30 = *(_QWORD *)(a1 + 32);
  if ( v108 != v30 )
  {
    do
    {
      v31 = v124;
      v32 = v30 - 56;
      if ( !v30 )
        v32 = 0;
      if ( !v124 )
        goto LABEL_46;
      v33 = &v123;
      do
      {
        while ( 1 )
        {
          v34 = *((_QWORD *)v31 + 2);
          v35 = *((_QWORD *)v31 + 3);
          if ( *((_QWORD *)v31 + 4) >= v32 )
            break;
          v31 = (int *)*((_QWORD *)v31 + 3);
          if ( !v35 )
            goto LABEL_44;
        }
        v33 = v31;
        v31 = (int *)*((_QWORD *)v31 + 2);
      }
      while ( v34 );
LABEL_44:
      if ( v33 == &v123 || *((_QWORD *)v33 + 4) > v32 )
      {
LABEL_46:
        v115 = v32;
        if ( (char *)v121[0] == v29 )
        {
          sub_17E9700((__int64)&v119, v29, &v115);
          v29 = v120;
        }
        else
        {
          if ( v29 )
          {
            *(_QWORD *)v29 = v32;
            v29 = v120;
          }
          v29 += 8;
          v120 = v29;
        }
      }
      v30 = *(_QWORD *)(v30 + 8);
    }
    while ( v108 != v30 );
    v36 = (unsigned __int64)v119;
    v37 = (v29 - v119) >> 3;
    LODWORD(v38) = v37;
    if ( v29 != v119 )
    {
      while ( (_DWORD)v38 )
      {
        v39 = 0;
        v40 = 0;
        v41 = 0;
        v42 = 8LL * (unsigned int)v38;
        do
        {
          while ( 1 )
          {
            v43 = *(_QWORD *)(v36 + v39);
            if ( !*(_QWORD *)(v43 + 8) )
              break;
            ++v40;
            v39 += 8;
            *(_QWORD *)(v36 + 8 * v41) = v43;
            v36 = (unsigned __int64)v119;
            v41 = v40;
            if ( v39 == v42 )
              goto LABEL_57;
          }
          v44 = *(_QWORD *)(v36 + v39);
          v39 += 8;
          sub_15E3D00(v44);
          v36 = (unsigned __int64)v119;
        }
        while ( v39 != v42 );
LABEL_57:
        v45 = (unsigned __int64)v120;
        v37 = (__int64)&v120[-v36] >> 3;
        LODWORD(v38) = v37;
        if ( !v40 || v37 == v41 )
          goto LABEL_98;
        if ( v37 < v41 )
        {
          sub_1CC3B50((__int64)&v119, v41 - v37);
          v45 = (unsigned __int64)v120;
          v36 = (unsigned __int64)v119;
          v38 = (v120 - v119) >> 3;
        }
        else if ( v37 > v41 )
        {
          v46 = 8 * v41;
          if ( v120 != (char *)(v36 + v46) )
          {
            v120 = (char *)(v36 + v46);
            v45 = v36 + v46;
            LODWORD(v38) = v46 >> 3;
          }
        }
        if ( v36 == v45 )
        {
          LODWORD(v37) = v38;
          goto LABEL_98;
        }
      }
      v37 = (__int64)&v120[-v36] >> 3;
    }
LABEL_98:
    if ( (_DWORD)v37 )
    {
      v70 = 0;
      v71 = 8LL * (unsigned int)v37;
      do
      {
        v72 = *(_QWORD *)(v36 + v70);
        sub_15E0C30(v72);
        v73 = *(_BYTE *)(v72 + 32);
        *(_BYTE *)(v72 + 32) = v73 & 0xF0;
        if ( (v73 & 0x30) != 0 )
          *(_BYTE *)(v72 + 33) |= 0x40u;
        v70 += 8;
        v36 = (unsigned __int64)v119;
      }
      while ( v71 != v70 );
    }
    if ( v36 )
      j_j___libc_free_0(v36, v121[0] - v36);
  }
  v74 = 1;
LABEL_106:
  v75 = v116;
  if ( HIDWORD(v117) && (_DWORD)v117 )
  {
    v76 = 8LL * (unsigned int)v117;
    v77 = 0;
    do
    {
      v78 = *(_QWORD *)(v75 + v77);
      if ( v78 != -8 && v78 )
      {
        _libc_free(v78);
        v75 = v116;
      }
      v77 += 8;
    }
    while ( v77 != v76 );
  }
  _libc_free(v75);
  sub_1CC3510((__int64)v124);
  v79 = v128;
  if ( v128 )
  {
    v80 = (__int64 *)v133;
    v81 = v137 + 1;
    if ( (unsigned __int64)(v137 + 1) > v133 )
    {
      do
      {
        v82 = *v80++;
        j_j___libc_free_0(v82, 512);
      }
      while ( v81 > v80 );
      v79 = v128;
    }
    j_j___libc_free_0(v79, 8 * v129);
  }
  return v74;
}
