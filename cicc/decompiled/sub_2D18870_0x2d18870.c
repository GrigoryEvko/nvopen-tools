// Function: sub_2D18870
// Address: 0x2d18870
//
__int64 __fastcall sub_2D18870(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rbx
  __int64 v8; // rax
  const void *v9; // r12
  size_t v10; // r14
  int v11; // eax
  unsigned int v12; // r13d
  _QWORD *v13; // r15
  __int64 v14; // rax
  _QWORD *v15; // rcx
  __int64 v16; // rbx
  unsigned __int64 v17; // r12
  const char *v18; // r14
  size_t v19; // rdx
  size_t v20; // r15
  int v21; // eax
  int v22; // eax
  __int64 v23; // rax
  unsigned __int64 j; // rdi
  __int64 v25; // rbx
  __int64 v26; // r14
  __int64 v27; // rbx
  __int64 k; // r12
  _BYTE *v29; // r8
  __int64 v30; // rbx
  __int64 *v31; // rax
  unsigned __int64 v32; // rsi
  __int64 *v33; // rdi
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
  _QWORD *v44; // rdi
  unsigned __int64 v45; // rdx
  __int64 v46; // r14
  unsigned __int64 v47; // r15
  _QWORD *v48; // rax
  __int64 *v49; // rdx
  char v50; // r15
  __int64 v51; // rax
  _QWORD *v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r13
  __int64 v55; // rbx
  __int64 v56; // r12
  char v57; // al
  unsigned int v58; // r13d
  unsigned __int64 v59; // r8
  __int64 v60; // r12
  __int64 v61; // rbx
  _QWORD *v62; // rdi
  unsigned __int64 v63; // rdi
  unsigned __int64 *v64; // rbx
  unsigned __int64 *v65; // r12
  unsigned __int64 v66; // rdi
  __int64 *v68; // r15
  __int64 v69; // rdx
  _QWORD *v70; // rax
  _QWORD *v71; // rdx
  __int64 v72; // rdi
  _BYTE *v73; // rax
  __int64 v74; // r8
  __int64 v75; // rax
  const char *v76; // rax
  unsigned __int64 v77; // rdx
  __int64 v78; // r8
  const char *v79; // r10
  size_t v80; // r9
  unsigned __int64 v81; // rax
  _QWORD *v82; // rdx
  size_t v83; // rdx
  unsigned __int8 *v84; // rsi
  __int64 v85; // rdi
  _BYTE *v86; // rax
  _QWORD *v87; // rax
  __int64 *v88; // rdx
  __int64 *v89; // r14
  char v90; // r12
  __int64 v91; // rax
  _QWORD *v92; // rax
  _QWORD *v93; // rdx
  __int64 v94; // rax
  _QWORD *v95; // rdi
  size_t n; // [rsp+8h] [rbp-158h]
  const char *src; // [rsp+10h] [rbp-150h]
  unsigned int v98; // [rsp+18h] [rbp-148h]
  unsigned int v99; // [rsp+1Ch] [rbp-144h]
  unsigned __int64 v101; // [rsp+28h] [rbp-138h]
  __int64 v102; // [rsp+28h] [rbp-138h]
  __int64 v103; // [rsp+28h] [rbp-138h]
  char v104; // [rsp+38h] [rbp-128h]
  _QWORD *v105; // [rsp+38h] [rbp-128h]
  __int64 v106; // [rsp+40h] [rbp-120h]
  __int64 v107; // [rsp+48h] [rbp-118h]
  __int64 v108; // [rsp+50h] [rbp-110h]
  __int64 i; // [rsp+50h] [rbp-110h]
  unsigned int v110; // [rsp+58h] [rbp-108h]
  __int64 *v111; // [rsp+58h] [rbp-108h]
  unsigned __int64 v112; // [rsp+68h] [rbp-F8h] BYREF
  unsigned __int64 v113; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v114; // [rsp+78h] [rbp-E8h]
  __int64 v115; // [rsp+80h] [rbp-E0h]
  _QWORD *v116; // [rsp+90h] [rbp-D0h] BYREF
  _BYTE *v117; // [rsp+98h] [rbp-C8h]
  _QWORD v118[2]; // [rsp+A0h] [rbp-C0h] BYREF
  char v119[8]; // [rsp+B0h] [rbp-B0h] BYREF
  __int64 v120; // [rsp+B8h] [rbp-A8h] BYREF
  __int64 *v121; // [rsp+C0h] [rbp-A0h]
  __int64 *v122; // [rsp+C8h] [rbp-98h]
  __int64 *v123; // [rsp+D0h] [rbp-90h]
  __int64 v124; // [rsp+D8h] [rbp-88h]
  __int64 v125; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v126; // [rsp+E8h] [rbp-78h]
  _QWORD *v127; // [rsp+F0h] [rbp-70h]
  __int64 v128; // [rsp+F8h] [rbp-68h]
  __int64 v129; // [rsp+100h] [rbp-60h]
  unsigned __int64 v130; // [rsp+108h] [rbp-58h]
  _QWORD *v131; // [rsp+110h] [rbp-50h]
  _QWORD *v132; // [rsp+118h] [rbp-48h]
  __int64 v133; // [rsp+120h] [rbp-40h]
  __int64 *v134; // [rsp+128h] [rbp-38h]

  v125 = 0;
  v127 = 0;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v126 = 8;
  v125 = sub_22077B0(0x40u);
  v1 = (__int64 *)(v125 + ((4 * v126 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v2 = sub_22077B0(0x200u);
  v3 = qword_50160A8;
  v130 = (unsigned __int64)v1;
  *v1 = v2;
  v128 = v2;
  v132 = (_QWORD *)v2;
  v127 = (_QWORD *)v2;
  v131 = (_QWORD *)v2;
  v115 = 0x800000000LL;
  v129 = v2 + 512;
  v107 = qword_50160B0;
  v4 = (qword_50160B0 - v3) >> 5;
  v134 = v1;
  v133 = v129;
  LODWORD(v120) = 0;
  v121 = 0;
  v122 = &v120;
  v123 = &v120;
  v124 = 0;
  v113 = 0;
  v114 = 0;
  v106 = v3;
  if ( (_DWORD)v4 )
  {
    v5 = (unsigned int)(v4 - 1);
    v6 = v3;
    v7 = 0;
    v108 = 32 * v5;
    while ( 1 )
    {
      v8 = v7 + v6;
      v9 = *(const void **)v8;
      v10 = *(_QWORD *)(v8 + 8);
      v11 = sub_C92610();
      v12 = sub_C92740((__int64)&v113, v9, v10, v11);
      v13 = (_QWORD *)(v113 + 8LL * v12);
      if ( *v13 )
      {
        if ( *v13 != -8 )
        {
          if ( v108 == v7 )
            break;
          goto LABEL_4;
        }
        LODWORD(v115) = v115 - 1;
      }
      v14 = sub_C7D670(v10 + 9, 8);
      v15 = (_QWORD *)v14;
      if ( v10 )
      {
        v105 = (_QWORD *)v14;
        memcpy((void *)(v14 + 8), v9, v10);
        v15 = v105;
      }
      *((_BYTE *)v15 + v10 + 8) = 0;
      *v15 = v10;
      *v13 = v15;
      ++HIDWORD(v114);
      sub_C929D0((__int64 *)&v113, v12);
      if ( v108 == v7 )
        break;
LABEL_4:
      v6 = qword_50160A8;
      v7 += 32;
    }
  }
  if ( (unsigned __int64)(qword_5015FB0 - qword_5015FA8) <= 4
    || (v99 = *(_DWORD *)qword_5015FA8, v98 = *(_DWORD *)(qword_5015FA8 + 4), *(_DWORD *)qword_5015FA8 > v98) )
  {
    if ( v107 == v106 )
    {
      v58 = 0;
      goto LABEL_95;
    }
    v104 = 0;
  }
  else
  {
    v104 = 1;
  }
  v110 = 0;
  v16 = *(_QWORD *)(a1 + 32);
  for ( i = a1 + 24; i != v16; v16 = *(_QWORD *)(v16 + 8) )
  {
    while ( 1 )
    {
      v17 = v16 - 56;
      if ( !v16 )
        v17 = 0;
      if ( sub_B2FC80(v17) || !(unsigned __int8)sub_CE9220(v17) )
        goto LABEL_26;
      ++v110;
      if ( v107 == v106 )
        break;
      v101 = v113 + 8LL * (unsigned int)v114;
      v18 = sub_BD5D20(v17);
      v20 = v19;
      v21 = sub_C92610();
      v22 = sub_C92860((__int64 *)&v113, v18, v20, v21);
      v23 = v22 == -1 ? v113 + 8LL * (unsigned int)v114 : v113 + 8LL * v22;
      if ( v23 == v101 )
        break;
      v70 = sub_CB72A0();
      v71 = (_QWORD *)v70[4];
      v72 = (__int64)v70;
      if ( v70[3] - (_QWORD)v71 <= 7u )
      {
        v72 = sub_CB6200((__int64)v70, "Select: ", 8u);
        v73 = *(_BYTE **)(v72 + 32);
      }
      else
      {
        *v71 = 0x203A7463656C6553LL;
        v73 = (_BYTE *)(v70[4] + 8LL);
        *(_QWORD *)(v72 + 32) = v73;
      }
      if ( v73 != *(_BYTE **)(v72 + 24) )
      {
LABEL_119:
        *v73 = 35;
        ++*(_QWORD *)(v72 + 32);
        goto LABEL_120;
      }
LABEL_143:
      v72 = sub_CB6200(v72, (unsigned __int8 *)"#", 1u);
LABEL_120:
      v74 = sub_CB59D0(v72, v110);
      v75 = *(_QWORD *)(v74 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v74 + 24) - v75) <= 2 )
      {
        v74 = sub_CB6200(v74, (unsigned __int8 *)" : ", 3u);
      }
      else
      {
        *(_BYTE *)(v75 + 2) = 32;
        *(_WORD *)v75 = 14880;
        *(_QWORD *)(v74 + 32) += 3LL;
      }
      v102 = v74;
      v76 = sub_BD5D20(v17);
      v78 = v102;
      v79 = v76;
      v80 = v77;
      if ( !v76 )
      {
        LOBYTE(v118[0]) = 0;
        v83 = 0;
        v116 = v118;
        v84 = (unsigned __int8 *)v118;
        v117 = 0;
        goto LABEL_128;
      }
      v112 = v77;
      v81 = v77;
      v116 = v118;
      if ( v77 > 0xF )
      {
        n = v77;
        src = v79;
        v94 = sub_22409D0((__int64)&v116, &v112, 0);
        v78 = v102;
        v79 = src;
        v116 = (_QWORD *)v94;
        v95 = (_QWORD *)v94;
        v80 = n;
        v118[0] = v112;
LABEL_153:
        v103 = v78;
        memcpy(v95, v79, v80);
        v81 = v112;
        v82 = v116;
        v78 = v103;
        goto LABEL_149;
      }
      if ( v77 == 1 )
      {
        LOBYTE(v118[0]) = *v79;
        v82 = v118;
      }
      else
      {
        if ( v77 )
        {
          v95 = v118;
          goto LABEL_153;
        }
        v82 = v118;
      }
LABEL_149:
      v117 = (_BYTE *)v81;
      *((_BYTE *)v82 + v81) = 0;
      v83 = (size_t)v117;
      v84 = (unsigned __int8 *)v116;
LABEL_128:
      v85 = sub_CB6200(v78, v84, v83);
      v86 = *(_BYTE **)(v85 + 32);
      if ( *(_BYTE **)(v85 + 24) == v86 )
      {
        sub_CB6200(v85, (unsigned __int8 *)"\n", 1u);
      }
      else
      {
        *v86 = 10;
        ++*(_QWORD *)(v85 + 32);
      }
      if ( v116 != v118 )
        j_j___libc_free_0((unsigned __int64)v116);
      v116 = (_QWORD *)v17;
      sub_2D18750((unsigned __int64 *)&v125, &v116);
      v116 = (_QWORD *)v17;
      v87 = sub_2D18170((__int64)v119, (unsigned __int64 *)&v116);
      v89 = v88;
      if ( !v88 )
        goto LABEL_26;
      v90 = v87 || v88 == &v120 || v17 < v88[4];
      v91 = sub_22077B0(0x28u);
      *(_QWORD *)(v91 + 32) = v116;
      sub_220F040(v90, v91, v89, &v120);
      ++v124;
      v16 = *(_QWORD *)(v16 + 8);
      if ( i == v16 )
        goto LABEL_27;
    }
    if ( v104 && v110 >= v99 && v110 <= v98 )
    {
      v92 = sub_CB72A0();
      v93 = (_QWORD *)v92[4];
      v72 = (__int64)v92;
      if ( v92[3] - (_QWORD)v93 <= 7u )
      {
        v72 = sub_CB6200((__int64)v92, "Select: ", 8u);
        v73 = *(_BYTE **)(v72 + 32);
      }
      else
      {
        *v93 = 0x203A7463656C6553LL;
        v73 = (_BYTE *)(v92[4] + 8LL);
        *(_QWORD *)(v72 + 32) = v73;
      }
      if ( *(_BYTE **)(v72 + 24) != v73 )
        goto LABEL_119;
      goto LABEL_143;
    }
LABEL_26:
    ;
  }
LABEL_27:
  for ( j = (unsigned __int64)v131; v131 != v127; j = (unsigned __int64)v131 )
  {
    if ( v132 == (_QWORD *)j )
    {
      v25 = *(_QWORD *)(*(v134 - 1) + 504);
      j_j___libc_free_0(j);
      v53 = *--v134 + 512;
      v132 = (_QWORD *)*v134;
      v133 = v53;
      v131 = v132 + 63;
    }
    else
    {
      v25 = *(_QWORD *)(j - 8);
      v131 = (_QWORD *)(j - 8);
    }
    v26 = v25 + 72;
    v27 = *(_QWORD *)(v25 + 80);
    if ( v26 != v27 )
    {
      while ( 1 )
      {
        if ( !v27 )
LABEL_157:
          BUG();
        k = *(_QWORD *)(v27 + 32);
        if ( k != v27 + 24 )
          break;
        v27 = *(_QWORD *)(v27 + 8);
        if ( v26 == v27 )
          goto LABEL_34;
      }
      while ( v26 != v27 )
      {
        if ( !k )
          BUG();
        if ( *(_BYTE *)(k - 24) == 85 )
        {
          v47 = *(_QWORD *)(k - 56);
          if ( v47 )
          {
            if ( !*(_BYTE *)v47 && *(_QWORD *)(v47 + 24) == *(_QWORD *)(k + 56) )
            {
              v116 = *(_QWORD **)(k - 56);
              v48 = sub_2D18170((__int64)v119, (unsigned __int64 *)&v116);
              if ( v49 )
              {
                v50 = v48 || v49 == &v120 || v47 < v49[4];
                v111 = v49;
                v51 = sub_22077B0(0x28u);
                *(_QWORD *)(v51 + 32) = v116;
                sub_220F040(v50, v51, v111, &v120);
                v52 = v131;
                ++v124;
                if ( v131 == (_QWORD *)(v133 - 8) )
                {
                  v68 = v134;
                  if ( ((v129 - (__int64)v127) >> 3) + ((((__int64)((__int64)v134 - v130) >> 3) - 1) << 6) + v131 - v132 == 0xFFFFFFFFFFFFFFFLL )
                    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
                  if ( (unsigned __int64)(v126 - (((__int64)v134 - v125) >> 3)) <= 1 )
                  {
                    sub_2D185D0((unsigned __int64 *)&v125, 1u, 0);
                    v68 = v134;
                  }
                  v68[1] = sub_22077B0(0x200u);
                  if ( v131 )
                    *v131 = v116;
                  v69 = *++v134 + 512;
                  v132 = (_QWORD *)*v134;
                  v133 = v69;
                  v131 = v132;
                }
                else
                {
                  if ( v131 )
                  {
                    *v131 = v116;
                    v52 = v131;
                  }
                  v131 = v52 + 1;
                }
              }
            }
          }
        }
        for ( k = *(_QWORD *)(k + 8); k == v27 - 24 + 48; k = *(_QWORD *)(v27 + 32) )
        {
          v27 = *(_QWORD *)(v27 + 8);
          if ( v26 == v27 )
            goto LABEL_34;
          if ( !v27 )
            goto LABEL_157;
        }
      }
    }
LABEL_34:
    ;
  }
  v29 = 0;
  v116 = 0;
  v117 = 0;
  v118[0] = 0;
  v30 = *(_QWORD *)(a1 + 32);
  if ( i != v30 )
  {
    do
    {
      v31 = v121;
      v32 = v30 - 56;
      if ( !v30 )
        v32 = 0;
      if ( !v121 )
        goto LABEL_45;
      v33 = &v120;
      do
      {
        while ( 1 )
        {
          v34 = v31[2];
          v35 = v31[3];
          if ( v31[4] >= v32 )
            break;
          v31 = (__int64 *)v31[3];
          if ( !v35 )
            goto LABEL_43;
        }
        v33 = v31;
        v31 = (__int64 *)v31[2];
      }
      while ( v34 );
LABEL_43:
      if ( v33 == &v120 || v33[4] > v32 )
      {
LABEL_45:
        v112 = v32;
        if ( (_BYTE *)v118[0] == v29 )
        {
          sub_24147A0((__int64)&v116, v29, &v112);
          v29 = v117;
        }
        else
        {
          if ( v29 )
          {
            *(_QWORD *)v29 = v32;
            v29 = v117;
          }
          v29 += 8;
          v117 = v29;
        }
      }
      v30 = *(_QWORD *)(v30 + 8);
    }
    while ( i != v30 );
    v36 = (unsigned __int64)v116;
    v37 = (v29 - (_BYTE *)v116) >> 3;
    LODWORD(v38) = v37;
    if ( v116 != (_QWORD *)v29 )
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
            if ( !*(_QWORD *)(v43 + 16) )
              break;
            ++v40;
            v39 += 8;
            *(_QWORD *)(v36 + 8 * v41) = v43;
            v36 = (unsigned __int64)v116;
            v41 = v40;
            if ( v39 == v42 )
              goto LABEL_56;
          }
          v44 = *(_QWORD **)(v36 + v39);
          v39 += 8;
          sub_B2E860(v44);
          v36 = (unsigned __int64)v116;
        }
        while ( v39 != v42 );
LABEL_56:
        v45 = (unsigned __int64)v117;
        v37 = (__int64)&v117[-v36] >> 3;
        LODWORD(v38) = v37;
        if ( !v40 || v37 == v41 )
          goto LABEL_87;
        if ( v37 < v41 )
        {
          sub_2D17FC0((__int64)&v116, v41 - v37);
          v45 = (unsigned __int64)v117;
          v36 = (unsigned __int64)v116;
          v38 = (v117 - (_BYTE *)v116) >> 3;
        }
        else if ( v37 > v41 )
        {
          v46 = 8 * v41;
          if ( v117 != (_BYTE *)(v36 + v46) )
          {
            v117 = (_BYTE *)(v36 + v46);
            v45 = v36 + v46;
            LODWORD(v38) = v46 >> 3;
          }
        }
        if ( v45 == v36 )
        {
          LODWORD(v37) = v38;
          goto LABEL_87;
        }
      }
      v37 = (__int64)&v117[-v36] >> 3;
    }
LABEL_87:
    if ( (_DWORD)v37 )
    {
      v54 = 0;
      v55 = 8LL * (unsigned int)v37;
      do
      {
        v56 = *(_QWORD *)(v36 + v54);
        sub_B2CA40(v56, 0);
        v57 = *(_BYTE *)(v56 + 32);
        *(_BYTE *)(v56 + 32) = v57 & 0xF0;
        if ( (v57 & 0x30) != 0 )
          *(_BYTE *)(v56 + 33) |= 0x40u;
        v54 += 8;
        v36 = (unsigned __int64)v116;
      }
      while ( v55 != v54 );
    }
    if ( v36 )
      j_j___libc_free_0(v36);
  }
  v58 = 1;
LABEL_95:
  v59 = v113;
  if ( HIDWORD(v114) && (_DWORD)v114 )
  {
    v60 = 8LL * (unsigned int)v114;
    v61 = 0;
    do
    {
      v62 = *(_QWORD **)(v59 + v61);
      if ( v62 != (_QWORD *)-8LL && v62 )
      {
        sub_C7D6A0((__int64)v62, *v62 + 9LL, 8);
        v59 = v113;
      }
      v61 += 8;
    }
    while ( v61 != v60 );
  }
  _libc_free(v59);
  sub_2D17A80((unsigned __int64)v121);
  v63 = v125;
  if ( v125 )
  {
    v64 = (unsigned __int64 *)v130;
    v65 = (unsigned __int64 *)(v134 + 1);
    if ( (unsigned __int64)(v134 + 1) > v130 )
    {
      do
      {
        v66 = *v64++;
        j_j___libc_free_0(v66);
      }
      while ( v65 > v64 );
      v63 = v125;
    }
    j_j___libc_free_0(v63);
  }
  return v58;
}
