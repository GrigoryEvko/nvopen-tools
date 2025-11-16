// Function: sub_1495EB0
// Address: 0x1495eb0
//
void __fastcall sub_1495EB0(_QWORD *a1, __int64 a2, __m128i a3, __m128i a4)
{
  __int64 v4; // rax
  __int64 v5; // r13
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // r12
  __int64 i; // r13
  __int64 v10; // rbx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 *v13; // r12
  __int64 *j; // rbx
  __int64 v15; // rdx
  _BYTE *v16; // rax
  __int64 v17; // rax
  int v18; // ecx
  __int64 v19; // rsi
  __int64 v20; // rdi
  int v21; // ecx
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r10
  __int64 v25; // r15
  _QWORD **v26; // r14
  __int64 v27; // r12
  unsigned __int64 v28; // r12
  __int64 v29; // r13
  int v30; // eax
  unsigned __int64 v31; // rdx
  const char *v32; // rsi
  _QWORD *v33; // rax
  int v34; // eax
  __int64 *v35; // r12
  __int64 *v36; // r15
  __int64 *v37; // r12
  __int64 *v38; // r15
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // rax
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // rax
  _BYTE *v43; // rax
  unsigned __int64 v44; // rax
  unsigned __int64 v45; // rax
  unsigned __int64 v46; // rax
  unsigned __int64 v47; // rax
  __int64 *v48; // rdi
  _BYTE *v49; // rsi
  _BYTE *v50; // r9
  unsigned __int64 v51; // rdx
  __int64 v52; // rax
  unsigned __int64 v53; // rcx
  unsigned __int64 v54; // rdx
  _BYTE *v55; // rax
  char v56; // di
  unsigned __int64 v57; // rcx
  unsigned __int64 v58; // r9
  unsigned __int64 v59; // r12
  __int64 v60; // rax
  unsigned __int64 v61; // rdi
  unsigned __int64 v62; // rdx
  unsigned __int64 v63; // rax
  char v64; // si
  unsigned __int64 v65; // rax
  unsigned __int64 v66; // r13
  __int64 v67; // r12
  _QWORD *v68; // r15
  __int64 *v69; // rax
  char v70; // dl
  __int64 v71; // r12
  _QWORD *v72; // rax
  _QWORD *v73; // rsi
  _QWORD *v74; // rdi
  unsigned __int64 v75; // rdx
  char v76; // si
  char v77; // cl
  __int64 v78; // r13
  int v79; // eax
  unsigned __int64 v80; // rdx
  const char *v81; // rsi
  _QWORD *v82; // rax
  unsigned __int64 v83; // rdi
  char *v84; // rax
  const char *v85; // rsi
  unsigned int v86; // ecx
  unsigned int v87; // eax
  __int64 v88; // r8
  unsigned __int64 v89; // rdi
  char *v90; // rax
  const char *v91; // rsi
  unsigned int v92; // ecx
  unsigned int v93; // eax
  __int64 v94; // r8
  int v95; // r8d
  __int64 v96; // [rsp+18h] [rbp-368h]
  _QWORD *v97; // [rsp+20h] [rbp-360h]
  unsigned __int64 v98; // [rsp+28h] [rbp-358h]
  __int64 v99; // [rsp+30h] [rbp-350h]
  __int64 v100; // [rsp+38h] [rbp-348h]
  _QWORD v103[16]; // [rsp+50h] [rbp-330h] BYREF
  __int64 v104; // [rsp+D0h] [rbp-2B0h] BYREF
  _BYTE *v105; // [rsp+D8h] [rbp-2A8h]
  _BYTE *v106; // [rsp+E0h] [rbp-2A0h]
  __int64 v107; // [rsp+E8h] [rbp-298h]
  int v108; // [rsp+F0h] [rbp-290h]
  _BYTE v109[64]; // [rsp+F8h] [rbp-288h] BYREF
  unsigned __int64 v110; // [rsp+138h] [rbp-248h] BYREF
  unsigned __int64 v111; // [rsp+140h] [rbp-240h]
  unsigned __int64 v112; // [rsp+148h] [rbp-238h]
  __int64 v113; // [rsp+150h] [rbp-230h] BYREF
  _QWORD *v114; // [rsp+158h] [rbp-228h]
  _QWORD *v115; // [rsp+160h] [rbp-220h]
  unsigned int v116; // [rsp+168h] [rbp-218h]
  unsigned int v117; // [rsp+16Ch] [rbp-214h]
  int v118; // [rsp+170h] [rbp-210h]
  _BYTE v119[64]; // [rsp+178h] [rbp-208h] BYREF
  unsigned __int64 v120; // [rsp+1B8h] [rbp-1C8h] BYREF
  unsigned __int64 v121; // [rsp+1C0h] [rbp-1C0h]
  unsigned __int64 v122; // [rsp+1C8h] [rbp-1B8h]
  char v123[8]; // [rsp+1D0h] [rbp-1B0h] BYREF
  __int64 v124; // [rsp+1D8h] [rbp-1A8h]
  unsigned __int64 v125; // [rsp+1E0h] [rbp-1A0h]
  _BYTE v126[64]; // [rsp+1F8h] [rbp-188h] BYREF
  unsigned __int64 v127; // [rsp+238h] [rbp-148h]
  unsigned __int64 v128; // [rsp+240h] [rbp-140h]
  unsigned __int64 v129; // [rsp+248h] [rbp-138h]
  _QWORD *v130; // [rsp+250h] [rbp-130h] BYREF
  __int64 v131; // [rsp+258h] [rbp-128h]
  unsigned __int64 v132; // [rsp+260h] [rbp-120h] BYREF
  unsigned int v133; // [rsp+268h] [rbp-118h]
  char v134[64]; // [rsp+278h] [rbp-108h] BYREF
  unsigned __int64 v135; // [rsp+2B8h] [rbp-C8h]
  _BYTE *v136; // [rsp+2C0h] [rbp-C0h]
  unsigned __int64 v137; // [rsp+2C8h] [rbp-B8h]
  char v138[8]; // [rsp+2D0h] [rbp-B0h] BYREF
  __int64 v139; // [rsp+2D8h] [rbp-A8h]
  unsigned __int64 v140; // [rsp+2E0h] [rbp-A0h]
  char v141[64]; // [rsp+2F8h] [rbp-88h] BYREF
  unsigned __int64 v142; // [rsp+338h] [rbp-48h]
  unsigned __int64 v143; // [rsp+340h] [rbp-40h]
  unsigned __int64 v144; // [rsp+348h] [rbp-38h]

  sub_1263B40(a2, "Classifying expressions for: ");
  sub_15537D0(a1[3], a2, 0);
  sub_1263B40(a2, "\n");
  v4 = a1[3];
  v5 = *(_QWORD *)(v4 + 80);
  v100 = v4 + 72;
  if ( v4 + 72 == v5 )
  {
    v6 = 0;
  }
  else
  {
    do
    {
      if ( !v5 )
LABEL_197:
        BUG();
      v6 = *(_QWORD *)(v5 + 24);
      if ( v6 != v5 + 16 )
        break;
      v5 = *(_QWORD *)(v5 + 8);
    }
    while ( v4 + 72 != v5 );
  }
  v7 = v6;
  v8 = v5;
  i = v7;
LABEL_6:
  if ( v8 == v100 )
    goto LABEL_19;
  v10 = v8;
  do
  {
    if ( !i )
      BUG();
    if ( !sub_1456C80((__int64)a1, *(_QWORD *)(i - 24)) || (unsigned __int8)(*(_BYTE *)(i - 8) - 75) <= 1u )
      goto LABEL_11;
    sub_155C2B0(i - 24, a2, 0);
    v16 = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)v16 >= *(_QWORD *)(a2 + 16) )
    {
      sub_16E7DE0(a2, 10);
    }
    else
    {
      *(_QWORD *)(a2 + 24) = v16 + 1;
      *v16 = 10;
    }
    sub_1263B40(a2, "  -->  ");
    v99 = sub_146F1B0((__int64)a1, i - 24);
    sub_1456620(v99, a2);
    if ( !sub_14562D0(v99) )
    {
      sub_1263B40(a2, " U: ");
      v37 = sub_1477920((__int64)a1, v99, 0);
      LODWORD(v131) = *((_DWORD *)v37 + 2);
      if ( (unsigned int)v131 > 0x40 )
        sub_16A4FD0(&v130, v37);
      else
        v130 = (_QWORD *)*v37;
      v133 = *((_DWORD *)v37 + 6);
      if ( v133 > 0x40 )
        sub_16A4FD0(&v132, v37 + 2);
      else
        v132 = v37[2];
      sub_1592EE0(&v130, a2);
      sub_135E100((__int64 *)&v132);
      sub_135E100((__int64 *)&v130);
      sub_1263B40(a2, " S: ");
      v38 = sub_1477920((__int64)a1, v99, 1u);
      LODWORD(v131) = *((_DWORD *)v38 + 2);
      if ( (unsigned int)v131 > 0x40 )
        sub_16A4FD0(&v130, v38);
      else
        v130 = (_QWORD *)*v38;
      v133 = *((_DWORD *)v38 + 6);
      if ( v133 > 0x40 )
        sub_16A4FD0(&v132, v38 + 2);
      else
        v132 = v38[2];
      sub_1592EE0(&v130, a2);
      sub_135E100((__int64 *)&v132);
      sub_135E100((__int64 *)&v130);
    }
    v17 = a1[8];
    v18 = *(_DWORD *)(v17 + 24);
    if ( !v18 )
      goto LABEL_49;
    v19 = *(_QWORD *)(i + 16);
    v20 = *(_QWORD *)(v17 + 8);
    v21 = v18 - 1;
    v22 = v21 & (((unsigned int)v19 >> 9) ^ ((unsigned int)v19 >> 4));
    v23 = (__int64 *)(v20 + 16LL * v22);
    v24 = *v23;
    if ( v19 != *v23 )
    {
      v34 = 1;
      while ( v24 != -8 )
      {
        v95 = v34 + 1;
        v22 = v21 & (v34 + v22);
        v23 = (__int64 *)(v20 + 16LL * v22);
        v24 = *v23;
        if ( v19 == *v23 )
          goto LABEL_28;
        v34 = v95;
      }
LABEL_49:
      v25 = sub_1472270((__int64)a1, v99, 0);
      if ( v99 == v25 )
        goto LABEL_31;
      sub_1263B40(a2, "  -->  ");
      sub_1456620(v25, a2);
      if ( sub_14562D0(v25) )
        goto LABEL_31;
      v97 = 0;
LABEL_52:
      sub_1263B40(a2, " U: ");
      v35 = sub_1477920((__int64)a1, v25, 0);
      LODWORD(v131) = *((_DWORD *)v35 + 2);
      if ( (unsigned int)v131 > 0x40 )
        sub_16A4FD0(&v130, v35);
      else
        v130 = (_QWORD *)*v35;
      v133 = *((_DWORD *)v35 + 6);
      if ( v133 > 0x40 )
        sub_16A4FD0(&v132, v35 + 2);
      else
        v132 = v35[2];
      sub_1592EE0(&v130, a2);
      sub_135E100((__int64 *)&v132);
      sub_135E100((__int64 *)&v130);
      sub_1263B40(a2, " S: ");
      v36 = sub_1477920((__int64)a1, v25, 1u);
      LODWORD(v131) = *((_DWORD *)v36 + 2);
      if ( (unsigned int)v131 > 0x40 )
        sub_16A4FD0(&v130, v36);
      else
        v130 = (_QWORD *)*v36;
      v133 = *((_DWORD *)v36 + 6);
      if ( v133 > 0x40 )
        sub_16A4FD0(&v132, v36 + 2);
      else
        v132 = v36[2];
      sub_1592EE0(&v130, a2);
      sub_135E100((__int64 *)&v132);
      sub_135E100((__int64 *)&v130);
      v26 = (_QWORD **)v97;
      if ( !v97 )
        goto LABEL_31;
      goto LABEL_32;
    }
LABEL_28:
    v97 = (_QWORD *)v23[1];
    v25 = sub_1472270((__int64)a1, v99, v97);
    if ( v99 != v25 )
    {
      sub_1263B40(a2, "  -->  ");
      sub_1456620(v25, a2);
      if ( !sub_14562D0(v25) )
        goto LABEL_52;
    }
    v26 = (_QWORD **)v97;
    if ( !v97 )
      goto LABEL_31;
LABEL_32:
    sub_1263B40(a2, "\t\tExits: ");
    v27 = sub_1472270((__int64)a1, v99, *v26);
    if ( sub_146CEE0((__int64)a1, v27, (__int64)v26) )
      sub_1456620(v27, a2);
    else
      sub_1263B40(a2, "<<Unknown>>");
    sub_1263B40(a2, "\t\tLoopDispositions: { ");
    v96 = i;
    v28 = (unsigned __int64)v97;
    while ( 1 )
    {
      sub_15537D0(**(_QWORD **)(v28 + 32), a2, 0);
      v29 = sub_1263B40(a2, ": ");
      v30 = sub_146CB30((__int64)a1, v99, v28);
      v31 = 9;
      v32 = "Invariant";
      if ( v30 != 1 )
      {
        v32 = "Computable";
        if ( v30 != 2 )
          v32 = "Variant";
        v31 = 3LL * (v30 == 2) + 7;
      }
      v33 = *(_QWORD **)(v29 + 24);
      if ( *(_QWORD *)(v29 + 16) - (_QWORD)v33 >= v31 )
        break;
      sub_16E7EE0(v29, v32);
      v28 = *(_QWORD *)v28;
      if ( !v28 )
        goto LABEL_76;
LABEL_46:
      sub_1263B40(a2, ", ");
    }
    if ( (unsigned int)v31 >= 8 )
    {
      v83 = (unsigned __int64)(v33 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      *v33 = *(_QWORD *)v32;
      *(_QWORD *)((char *)v33 + v31 - 8) = *(_QWORD *)&v32[v31 - 8];
      v84 = (char *)v33 - v83;
      v85 = (const char *)(v32 - v84);
      if ( (((_DWORD)v31 + (_DWORD)v84) & 0xFFFFFFF8) >= 8 )
      {
        v86 = 0;
        v87 = (v31 + (_DWORD)v84) & 0xFFFFFFF8;
        do
        {
          v88 = v86;
          v86 += 8;
          *(_QWORD *)(v83 + v88) = *(_QWORD *)&v85[v88];
        }
        while ( v86 < v87 );
      }
    }
    else if ( (v31 & 4) != 0 )
    {
      *(_DWORD *)v33 = *(_DWORD *)v32;
      *(_DWORD *)((char *)v33 + (unsigned int)v31 - 4) = *(_DWORD *)&v32[(unsigned int)v31 - 4];
    }
    else if ( (_DWORD)v31 )
    {
      *(_BYTE *)v33 = *v32;
      if ( (v31 & 2) != 0 )
        *(_WORD *)((char *)v33 + (unsigned int)v31 - 2) = *(_WORD *)&v32[(unsigned int)v31 - 2];
    }
    *(_QWORD *)(v29 + 24) += v31;
    v28 = *(_QWORD *)v28;
    if ( v28 )
      goto LABEL_46;
LABEL_76:
    memset64(v103, v28, 0x10u);
    v103[1] = &v103[5];
    v103[2] = &v103[5];
    LODWORD(v103[3]) = 8;
    v104 = 0;
    v105 = v109;
    v106 = v109;
    v107 = 8;
    v108 = 0;
    v110 = 0;
    v111 = 0;
    v112 = 0;
    sub_1412190((__int64)&v104, (__int64)v97);
    v130 = v97;
    LOBYTE(v132) = 0;
    sub_1466380(&v110, (__int64)&v130);
    sub_16CCEE0(v123, v126, 8, v103);
    v127 = v103[13];
    memset(&v103[13], 0, 24);
    v128 = v103[14];
    v129 = v103[15];
    sub_16CCEE0(&v113, v119, 8, &v104);
    v39 = v110;
    v110 = 0;
    v120 = v39;
    v40 = v111;
    v111 = 0;
    v121 = v40;
    v41 = v112;
    v112 = 0;
    v122 = v41;
    sub_16CCEE0(&v130, v134, 8, &v113);
    v42 = v120;
    v120 = 0;
    v135 = v42;
    v43 = (_BYTE *)v121;
    v121 = 0;
    v136 = v43;
    v44 = v122;
    v122 = 0;
    v137 = v44;
    sub_16CCEE0(v138, v141, 8, v123);
    v45 = v127;
    v127 = 0;
    v142 = v45;
    v46 = v128;
    v128 = 0;
    v143 = v46;
    v47 = v129;
    v129 = 0;
    v144 = v47;
    if ( v120 )
      j_j___libc_free_0(v120, v122 - v120);
    if ( v115 != v114 )
      _libc_free((unsigned __int64)v115);
    if ( v127 )
      j_j___libc_free_0(v127, v129 - v127);
    if ( v125 != v124 )
      _libc_free(v125);
    if ( v110 )
      j_j___libc_free_0(v110, v112 - v110);
    if ( v106 != v105 )
      _libc_free((unsigned __int64)v106);
    if ( v103[13] )
      j_j___libc_free_0(v103[13], v103[15] - v103[13]);
    if ( v103[2] != v103[1] )
      _libc_free(v103[2]);
    v48 = &v113;
    sub_16CCCB0(&v113, v119, &v130);
    v49 = v136;
    v50 = (_BYTE *)v135;
    v120 = 0;
    v121 = 0;
    v122 = 0;
    v51 = (unsigned __int64)&v136[-v135];
    if ( v136 != (_BYTE *)v135 )
    {
      if ( v51 <= 0x7FFFFFFFFFFFFFF8LL )
      {
        v98 = (unsigned __int64)&v136[-v135];
        v52 = sub_22077B0(&v136[-v135]);
        v49 = v136;
        v50 = (_BYTE *)v135;
        v51 = v98;
        v53 = v52;
        goto LABEL_95;
      }
LABEL_195:
      sub_4261EA(v48, v49, v51);
    }
    v53 = 0;
LABEL_95:
    v120 = v53;
    v121 = v53;
    v122 = v53 + v51;
    if ( v50 != v49 )
    {
      v54 = v53;
      v55 = v50;
      do
      {
        if ( v54 )
        {
          *(_QWORD *)v54 = *(_QWORD *)v55;
          v56 = v55[16];
          *(_BYTE *)(v54 + 16) = v56;
          if ( v56 )
            *(_QWORD *)(v54 + 8) = *((_QWORD *)v55 + 1);
        }
        v55 += 24;
        v54 += 24LL;
      }
      while ( v55 != v49 );
      v53 += 8 * ((unsigned __int64)(v55 - 24 - v50) >> 3) + 24;
    }
    v49 = v126;
    v48 = (__int64 *)v123;
    v121 = v53;
    sub_16CCCB0(v123, v126, v138);
    v57 = v143;
    v58 = v142;
    v127 = 0;
    v128 = 0;
    v129 = 0;
    v59 = v143 - v142;
    if ( v143 == v142 )
    {
      v61 = 0;
    }
    else
    {
      if ( v59 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_195;
      v60 = sub_22077B0(v143 - v142);
      v57 = v143;
      v58 = v142;
      v61 = v60;
    }
    v127 = v61;
    v129 = v61 + v59;
    v62 = v61;
    v128 = v61;
    if ( v57 != v58 )
    {
      v63 = v58;
      do
      {
        if ( v62 )
        {
          *(_QWORD *)v62 = *(_QWORD *)v63;
          v64 = *(_BYTE *)(v63 + 16);
          *(_BYTE *)(v62 + 16) = v64;
          if ( v64 )
            *(_QWORD *)(v62 + 8) = *(_QWORD *)(v63 + 8);
        }
        v63 += 24LL;
        v62 += 24LL;
      }
      while ( v63 != v57 );
      v62 = v61 + 8 * ((v63 - 24 - v58) >> 3) + 24;
    }
    v65 = v120;
    v128 = v62;
    v66 = v121;
    if ( v121 - v120 == v62 - v61 )
      goto LABEL_129;
LABEL_113:
    while ( 2 )
    {
      while ( 2 )
      {
        v67 = *(_QWORD *)(v66 - 24);
        v68 = v97;
        if ( (_QWORD *)v67 != v97 )
        {
          sub_1263B40(a2, ", ");
          sub_15537D0(**(_QWORD **)(v67 + 32), a2, 0);
          v78 = sub_1263B40(a2, ": ");
          v79 = sub_146CB30((__int64)a1, v99, v67);
          v80 = 9;
          v81 = "Invariant";
          if ( v79 != 1 )
          {
            v81 = "Computable";
            if ( v79 != 2 )
              v81 = "Variant";
            v80 = 3LL * (v79 == 2) + 7;
          }
          v82 = *(_QWORD **)(v78 + 24);
          if ( v80 > *(_QWORD *)(v78 + 16) - (_QWORD)v82 )
          {
            sub_16E7EE0(v78, v81);
          }
          else
          {
            if ( (unsigned int)v80 >= 8 )
            {
              v89 = (unsigned __int64)(v82 + 1) & 0xFFFFFFFFFFFFFFF8LL;
              *v82 = *(_QWORD *)v81;
              *(_QWORD *)((char *)v82 + v80 - 8) = *(_QWORD *)&v81[v80 - 8];
              v90 = (char *)v82 - v89;
              v91 = (const char *)(v81 - v90);
              if ( (((_DWORD)v80 + (_DWORD)v90) & 0xFFFFFFF8) >= 8 )
              {
                v92 = 0;
                v93 = (v80 + (_DWORD)v90) & 0xFFFFFFF8;
                do
                {
                  v94 = v92;
                  v92 += 8;
                  *(_QWORD *)(v89 + v94) = *(_QWORD *)&v91[v94];
                }
                while ( v92 < v93 );
              }
            }
            else if ( (v80 & 4) != 0 )
            {
              *(_DWORD *)v82 = *(_DWORD *)v81;
              *(_DWORD *)((char *)v82 + (unsigned int)v80 - 4) = *(_DWORD *)&v81[(unsigned int)v80 - 4];
            }
            else if ( (_DWORD)v80 )
            {
              *(_BYTE *)v82 = *v81;
              if ( (v80 & 2) != 0 )
                *(_WORD *)((char *)v82 + (unsigned int)v80 - 2) = *(_WORD *)&v81[(unsigned int)v80 - 2];
            }
            *(_QWORD *)(v78 + 24) += v80;
          }
          v66 = v121;
LABEL_158:
          v68 = *(_QWORD **)(v66 - 24);
        }
        if ( !*(_BYTE *)(v66 - 8) )
        {
          v69 = (__int64 *)v68[1];
          *(_BYTE *)(v66 - 8) = 1;
          *(_QWORD *)(v66 - 16) = v69;
          goto LABEL_118;
        }
        while ( 1 )
        {
          v69 = *(__int64 **)(v66 - 16);
LABEL_118:
          if ( v69 == (__int64 *)v68[2] )
            break;
          *(_QWORD *)(v66 - 16) = v69 + 1;
          v71 = *v69;
          v72 = v114;
          if ( v115 == v114 )
          {
            v73 = &v114[v117];
            if ( v114 != v73 )
            {
              v74 = 0;
              while ( v71 != *v72 )
              {
                if ( *v72 == -2 )
                {
                  v74 = v72;
                  if ( v72 + 1 == v73 )
                    goto LABEL_126;
                  ++v72;
                }
                else if ( v73 == ++v72 )
                {
                  if ( !v74 )
                    goto LABEL_155;
LABEL_126:
                  *v74 = v71;
                  --v118;
                  ++v113;
                  goto LABEL_127;
                }
              }
              continue;
            }
LABEL_155:
            if ( v117 < v116 )
            {
              ++v117;
              *v73 = v71;
              ++v113;
LABEL_127:
              v104 = v71;
              LOBYTE(v106) = 0;
              sub_1466380(&v120, (__int64)&v104);
              v65 = v120;
              v66 = v121;
              goto LABEL_128;
            }
          }
          sub_16CCBA0(&v113, v71);
          if ( v70 )
            goto LABEL_127;
        }
        v121 -= 24LL;
        v65 = v120;
        v66 = v121;
        if ( v121 != v120 )
          goto LABEL_158;
LABEL_128:
        v61 = v127;
        if ( v66 - v65 != v128 - v127 )
          continue;
        break;
      }
LABEL_129:
      if ( v65 != v66 )
      {
        v75 = v61;
        while ( *(_QWORD *)v65 == *(_QWORD *)v75 )
        {
          v76 = *(_BYTE *)(v65 + 16);
          v77 = *(_BYTE *)(v75 + 16);
          if ( v76 && v77 )
          {
            if ( *(_QWORD *)(v65 + 8) != *(_QWORD *)(v75 + 8) )
              goto LABEL_113;
          }
          else if ( v76 != v77 )
          {
            goto LABEL_113;
          }
          v65 += 24LL;
          v75 += 24LL;
          if ( v65 == v66 )
            goto LABEL_136;
        }
        continue;
      }
      break;
    }
LABEL_136:
    i = v96;
    if ( v61 )
      j_j___libc_free_0(v61, v129 - v61);
    if ( v125 != v124 )
      _libc_free(v125);
    if ( v120 )
      j_j___libc_free_0(v120, v122 - v120);
    if ( v115 != v114 )
      _libc_free((unsigned __int64)v115);
    if ( v142 )
      j_j___libc_free_0(v142, v144 - v142);
    if ( v140 != v139 )
      _libc_free(v140);
    if ( v135 )
      j_j___libc_free_0(v135, v137 - v135);
    if ( v132 != v131 )
      _libc_free(v132);
    sub_1263B40(a2, " }");
LABEL_31:
    sub_1263B40(a2, "\n");
LABEL_11:
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v10 + 24) )
    {
      v11 = v10 - 24;
      if ( !v10 )
        v11 = 0;
      if ( i != v11 + 40 )
        break;
      v10 = *(_QWORD *)(v10 + 8);
      if ( v100 == v10 )
      {
        v8 = v10;
        goto LABEL_6;
      }
      if ( !v10 )
        goto LABEL_197;
    }
  }
  while ( v100 != v10 );
LABEL_19:
  sub_1263B40(a2, "Determining loop execution counts for: ");
  sub_15537D0(a1[3], a2, 0);
  sub_1263B40(a2, "\n");
  v12 = a1[8];
  v13 = *(__int64 **)(v12 + 40);
  for ( j = *(__int64 **)(v12 + 32); v13 != j; ++j )
  {
    v15 = *j;
    sub_14959E0(a2, a1, v15, a3, a4);
  }
}
