// Function: sub_19D7EE0
// Address: 0x19d7ee0
//
__int64 __fastcall sub_19D7EE0(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // rdi
  __int64 *v5; // r13
  const char *v6; // rax
  __int64 v7; // rdi
  const char *v8; // r12
  size_t v9; // rdx
  size_t v10; // rbx
  size_t v11; // rdx
  const char *v12; // rsi
  size_t v13; // r14
  int v14; // eax
  __int64 *v15; // r14
  __int64 *v16; // r15
  __int64 *v17; // r12
  __int64 *v18; // r12
  unsigned __int64 v19; // r13
  int v20; // eax
  __int64 v21; // rdi
  const char *v22; // r15
  size_t v23; // rdx
  size_t v24; // rbx
  __int64 v25; // rdi
  size_t v26; // rdx
  const char *v27; // rsi
  size_t v28; // r14
  __int64 v29; // r14
  unsigned __int64 v30; // rbx
  int v31; // eax
  unsigned __int64 v32; // rcx
  const char *v33; // r8
  const char *v34; // rax
  __int64 v35; // rdi
  size_t v36; // rdx
  size_t v37; // r14
  const char *v38; // rax
  size_t v39; // rdx
  size_t v40; // r15
  __int64 v41; // rdi
  int v42; // eax
  int v43; // r15d
  __int64 v44; // r11
  __int64 v45; // r10
  __int64 v46; // r9
  char v47; // r8
  __int64 v48; // rsi
  __int64 v49; // rax
  __int64 v50; // rcx
  __int64 v51; // rdx
  int v52; // r14d
  __int64 v53; // rax
  __int64 v54; // rdi
  bool v55; // cc
  __int64 v56; // rdi
  int v57; // edi
  __int64 v58; // rdi
  __int64 v59; // rdi
  __int64 v60; // rdx
  __int64 v61; // rsi
  unsigned int v62; // eax
  unsigned __int64 v63; // rdx
  __int64 v64; // rsi
  unsigned int v65; // eax
  __int64 v66; // r12
  __int64 v67; // r13
  __int64 *v68; // rbx
  __int64 v69; // rsi
  unsigned int v70; // eax
  unsigned int v71; // ecx
  __int64 v72; // rdi
  __int64 v73; // rdx
  __int64 v74; // r11
  __int64 v75; // rsi
  __int64 v76; // r10
  void *v77; // rax
  __int64 v78; // r9
  __int64 v79; // rsi
  char v80; // r8
  __int64 v81; // rbx
  __int64 *v82; // r14
  void *v83; // rdi
  unsigned int v84; // edx
  unsigned int v85; // r12d
  __int64 v86; // r11
  __int64 v87; // r10
  __int64 v88; // r9
  char v89; // r8
  __int64 v90; // rsi
  __int64 v91; // rcx
  __int64 v92; // rax
  __int64 v93; // r15
  __int64 v94; // r13
  __int64 v95; // rdi
  __int64 v96; // rdi
  __int64 v97; // rdi
  __int64 v98; // r12
  __int64 v99; // rdx
  unsigned __int64 v100; // rcx
  __int64 v101; // rdi
  unsigned int v102; // eax
  __int64 v103; // [rsp+8h] [rbp-128h]
  __int64 *v104; // [rsp+10h] [rbp-120h]
  __int64 *v105; // [rsp+18h] [rbp-118h]
  __int64 v106; // [rsp+20h] [rbp-110h]
  __int64 v107; // [rsp+28h] [rbp-108h]
  __int64 v108; // [rsp+30h] [rbp-100h]
  __int64 v109; // [rsp+38h] [rbp-F8h]
  __int64 v110; // [rsp+38h] [rbp-F8h]
  unsigned int v111; // [rsp+40h] [rbp-F0h]
  char v112; // [rsp+48h] [rbp-E8h]
  __int64 v113; // [rsp+48h] [rbp-E8h]
  __int64 v114; // [rsp+50h] [rbp-E0h]
  __int64 v115; // [rsp+58h] [rbp-D8h]
  __int64 v116; // [rsp+58h] [rbp-D8h]
  char v117; // [rsp+58h] [rbp-D8h]
  __int64 v118; // [rsp+60h] [rbp-D0h]
  __int64 v119; // [rsp+60h] [rbp-D0h]
  __int64 v120; // [rsp+60h] [rbp-D0h]
  __int64 v121; // [rsp+68h] [rbp-C8h]
  __int64 v122; // [rsp+68h] [rbp-C8h]
  __int64 *v123; // [rsp+70h] [rbp-C0h]
  __int64 v124; // [rsp+70h] [rbp-C0h]
  __int64 v125; // [rsp+70h] [rbp-C0h]
  __int64 *v126; // [rsp+78h] [rbp-B8h]
  int v128; // [rsp+80h] [rbp-B0h]
  const char *s1; // [rsp+88h] [rbp-A8h]
  int s1a; // [rsp+88h] [rbp-A8h]
  void *s1b; // [rsp+88h] [rbp-A8h]
  __int64 v132; // [rsp+90h] [rbp-A0h] BYREF
  __int64 v133; // [rsp+98h] [rbp-98h]
  __int64 v134; // [rsp+A0h] [rbp-90h]
  char v135; // [rsp+A8h] [rbp-88h]
  __int64 v136; // [rsp+B0h] [rbp-80h]
  __int64 v137; // [rsp+B8h] [rbp-78h]
  __int64 v138; // [rsp+C0h] [rbp-70h]
  unsigned int v139; // [rsp+C8h] [rbp-68h]
  __int64 v140; // [rsp+D0h] [rbp-60h]
  __int64 v141; // [rsp+D8h] [rbp-58h]
  void *v142; // [rsp+E0h] [rbp-50h]
  unsigned int v143; // [rsp+E8h] [rbp-48h]
  int v144; // [rsp+F0h] [rbp-40h]

  result = (char *)a2 - (char *)a1;
  v106 = a3;
  if ( (char *)a2 - (char *)a1 <= 1664 )
    return result;
  if ( !a3 )
  {
    v123 = a2;
    goto LABEL_96;
  }
  v126 = a2;
  v105 = a1 + 13;
  v104 = a1 + 17;
  v103 = (__int64)(a1 + 19);
  while ( 2 )
  {
    --v106;
    v4 = a1[17];
    v5 = &a1[4 * (0x4EC4EC4EC4EC4EC5LL * (v126 - a1) / 2)
           + 4
           * ((0x4EC4EC4EC4EC4EC5LL * (v126 - a1) + ((unsigned __int64)(0x4EC4EC4EC4EC4EC5LL * (v126 - a1)) >> 63))
            & 0xFFFFFFFFFFFFFFFELL)
           + 0x4EC4EC4EC4EC4EC5LL * (v126 - a1) / 2];
    if ( v4 )
      v4 = *(_QWORD *)(v4 - 24LL * (*(_DWORD *)(v4 + 20) & 0xFFFFFFF));
    v6 = sub_1649960(v4);
    v7 = v5[4];
    v8 = v6;
    v10 = v9;
    if ( v7 )
      v7 = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
    v12 = sub_1649960(v7);
    v13 = v11;
    if ( v10 > v11 )
    {
      if ( !v11 )
        goto LABEL_61;
      v14 = memcmp(v8, v12, v11);
      if ( !v14 )
      {
LABEL_12:
        if ( v10 < v13 )
        {
LABEL_13:
          v15 = v5 + 4;
          v16 = v126 - 13;
          v17 = v126 - 9;
          goto LABEL_14;
        }
        goto LABEL_61;
      }
LABEL_60:
      if ( v14 < 0 )
        goto LABEL_13;
      goto LABEL_61;
    }
    if ( v10 )
    {
      v14 = memcmp(v8, v12, v10);
      if ( v14 )
        goto LABEL_60;
    }
    if ( v10 != v13 )
      goto LABEL_12;
    v99 = v5[4];
    v100 = a1[17];
    if ( v100 )
    {
      v101 = *(_QWORD *)(v100 - 24LL * (*(_DWORD *)(v100 + 20) & 0xFFFFFFF));
      if ( v99 )
      {
        if ( v101 != *(_QWORD *)(v99 - 24LL * (*(_DWORD *)(v99 + 20) & 0xFFFFFFF)) )
          goto LABEL_120;
      }
      else if ( v101 )
      {
LABEL_120:
        v100 = *(_QWORD *)(v100 - 24LL * (*(_DWORD *)(v100 + 20) & 0xFFFFFFF));
        if ( v99 )
        {
          v102 = *(_DWORD *)(v99 + 20);
LABEL_122:
          LOBYTE(v102) = *(_QWORD *)(v99 - 24LL * (v102 & 0xFFFFFFF)) > v100;
          goto LABEL_123;
        }
LABEL_61:
        v15 = v5 + 4;
        v16 = v126 - 13;
        v17 = v126 - 9;
        goto LABEL_62;
      }
    }
    else if ( v99 )
    {
      v102 = *(_DWORD *)(v99 + 20);
      if ( *(_QWORD *)(v99 - 24LL * (v102 & 0xFFFFFFF)) )
        goto LABEL_122;
    }
    v102 = (unsigned int)sub_16AEA10(v103, (__int64)(v5 + 6)) >> 31;
LABEL_123:
    v15 = v5 + 4;
    v16 = v126 - 13;
    v17 = v126 - 9;
    if ( !(_BYTE)v102 )
    {
LABEL_62:
      if ( sub_19D6260(v104, v17) )
        goto LABEL_16;
      if ( sub_19D6260(v15, v17) )
      {
LABEL_64:
        sub_19D69A0(a1, v16);
        goto LABEL_17;
      }
LABEL_91:
      sub_19D69A0(a1, v5);
      goto LABEL_17;
    }
LABEL_14:
    if ( sub_19D6260(v15, v17) )
      goto LABEL_91;
    if ( sub_19D6260(v104, v17) )
      goto LABEL_64;
LABEL_16:
    sub_19D69A0(a1, v105);
LABEL_17:
    v18 = v105;
    v19 = (unsigned __int64)v126;
    while ( 1 )
    {
      v21 = v18[4];
      v123 = v18;
      if ( v21 )
        v21 = *(_QWORD *)(v21 - 24LL * (*(_DWORD *)(v21 + 20) & 0xFFFFFFF));
      v22 = sub_1649960(v21);
      v24 = v23;
      v25 = a1[4];
      if ( v25 )
        v25 = *(_QWORD *)(v25 - 24LL * (*(_DWORD *)(v25 + 20) & 0xFFFFFFF));
      v27 = sub_1649960(v25);
      v28 = v26;
      if ( v24 > v26 )
        break;
      if ( v24 )
      {
        v20 = memcmp(v22, v27, v24);
        if ( v20 )
          goto LABEL_30;
      }
      if ( v24 == v28 )
      {
        v63 = v18[4];
        v29 = a1[4];
        if ( v63 )
        {
          v64 = *(_QWORD *)(v63 - 24LL * (*(_DWORD *)(v63 + 20) & 0xFFFFFFF));
          if ( v29 )
          {
            if ( v64 != *(_QWORD *)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF)) )
              goto LABEL_78;
LABEL_90:
            v65 = (unsigned int)sub_16AEA10((__int64)(v18 + 6), (__int64)(a1 + 6)) >> 31;
LABEL_81:
            if ( !(_BYTE)v65 )
              goto LABEL_32;
            goto LABEL_22;
          }
          if ( !v64 )
            goto LABEL_90;
LABEL_78:
          v63 = *(_QWORD *)(v63 - 24LL * (*(_DWORD *)(v63 + 20) & 0xFFFFFFF));
          if ( !v29 )
            goto LABEL_32;
          v65 = *(_DWORD *)(v29 + 20);
        }
        else
        {
          if ( !v29 )
            goto LABEL_90;
          v65 = *(_DWORD *)(v29 + 20);
          if ( !*(_QWORD *)(v29 - 24LL * (v65 & 0xFFFFFFF)) )
            goto LABEL_90;
        }
        LOBYTE(v65) = v63 < *(_QWORD *)(v29 - 24LL * (v65 & 0xFFFFFFF));
        goto LABEL_81;
      }
LABEL_21:
      if ( v24 >= v28 )
        goto LABEL_31;
LABEL_22:
      v18 += 13;
    }
    if ( !v26 )
      goto LABEL_31;
    v20 = memcmp(v22, v27, v26);
    if ( !v20 )
      goto LABEL_21;
LABEL_30:
    if ( v20 < 0 )
      goto LABEL_22;
LABEL_31:
    v29 = a1[4];
LABEL_32:
    v30 = v19 - 104;
    while ( 2 )
    {
      v19 = v30;
      if ( v29 )
        v29 = *(_QWORD *)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF));
      v34 = sub_1649960(v29);
      v35 = *(_QWORD *)(v30 + 32);
      v37 = v36;
      if ( v35 )
        v35 = *(_QWORD *)(v35 - 24LL * (*(_DWORD *)(v35 + 20) & 0xFFFFFFF));
      s1 = v34;
      v38 = sub_1649960(v35);
      v33 = s1;
      v40 = v39;
      if ( v37 > v39 )
      {
        if ( !v39 )
          goto LABEL_47;
        v31 = memcmp(s1, v38, v39);
        if ( v31 )
          goto LABEL_46;
LABEL_36:
        if ( v37 >= v40 )
          goto LABEL_47;
LABEL_37:
        v29 = a1[4];
LABEL_38:
        v30 -= 104LL;
        continue;
      }
      break;
    }
    if ( v37 )
    {
      v31 = memcmp(s1, v38, v37);
      if ( v31 )
      {
LABEL_46:
        if ( v31 >= 0 )
          goto LABEL_47;
        goto LABEL_37;
      }
    }
    if ( v37 != v40 )
      goto LABEL_36;
    v60 = *(_QWORD *)(v30 + 32);
    v29 = a1[4];
    if ( !v29 )
    {
      if ( !v60 )
        goto LABEL_74;
      v62 = *(_DWORD *)(v60 + 20);
      if ( !*(_QWORD *)(v60 - 24LL * (v62 & 0xFFFFFFF)) )
        goto LABEL_74;
      v32 = 0;
      goto LABEL_70;
    }
    v61 = *(_QWORD *)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF));
    if ( !v60 )
    {
      if ( !v61 )
        goto LABEL_74;
LABEL_68:
      v32 = *(_QWORD *)(v29 - 24LL * (*(_DWORD *)(v29 + 20) & 0xFFFFFFF));
      if ( !v60 )
        goto LABEL_47;
      v62 = *(_DWORD *)(v60 + 20);
LABEL_70:
      LOBYTE(v62) = v32 < *(_QWORD *)(v60 - 24LL * (v62 & 0xFFFFFFF));
      goto LABEL_71;
    }
    if ( v61 != *(_QWORD *)(v60 - 24LL * (*(_DWORD *)(v60 + 20) & 0xFFFFFFF)) )
      goto LABEL_68;
LABEL_74:
    v62 = (unsigned int)sub_16AEA10((__int64)(a1 + 6), v30 + 48) >> 31;
LABEL_71:
    if ( (_BYTE)v62 )
      goto LABEL_38;
LABEL_47:
    if ( (unsigned __int64)v18 < v30 )
    {
      v41 = v18[10];
      v42 = *((_DWORD *)v18 + 14);
      *((_DWORD *)v18 + 14) = 0;
      v43 = *((_DWORD *)v18 + 22);
      v44 = *v18;
      *((_DWORD *)v18 + 22) = 0;
      v45 = v18[1];
      v121 = v41;
      v46 = v18[2];
      s1a = v42;
      v47 = *((_BYTE *)v18 + 24);
      v48 = v18[4];
      v49 = v18[8];
      v50 = v18[5];
      *v18 = *(_QWORD *)v30;
      v51 = v18[6];
      v124 = v49;
      v52 = *((_DWORD *)v18 + 24);
      v18[1] = *(_QWORD *)(v30 + 8);
      v53 = v18[9];
      v18[2] = *(_QWORD *)(v30 + 16);
      *((_BYTE *)v18 + 24) = *(_BYTE *)(v30 + 24);
      v18[4] = *(_QWORD *)(v30 + 32);
      v18[5] = *(_QWORD *)(v30 + 40);
      v18[6] = *(_QWORD *)(v30 + 48);
      *((_DWORD *)v18 + 14) = *(_DWORD *)(v30 + 56);
      v54 = *(_QWORD *)(v30 + 64);
      *(_DWORD *)(v30 + 56) = 0;
      v55 = *((_DWORD *)v18 + 22) <= 0x40u;
      v18[8] = v54;
      v18[9] = *(_QWORD *)(v30 + 72);
      if ( !v55 )
      {
        v56 = v18[10];
        if ( v56 )
        {
          v107 = v53;
          v108 = v51;
          v109 = v50;
          v112 = v47;
          v114 = v46;
          v115 = v45;
          v118 = v44;
          j_j___libc_free_0_0(v56);
          v53 = v107;
          v51 = v108;
          v50 = v109;
          v47 = v112;
          v46 = v114;
          v45 = v115;
          v44 = v118;
        }
      }
      v18[10] = *(_QWORD *)(v30 + 80);
      *((_DWORD *)v18 + 22) = *(_DWORD *)(v30 + 88);
      v57 = *(_DWORD *)(v30 + 96);
      *(_DWORD *)(v30 + 88) = 0;
      *((_DWORD *)v18 + 24) = v57;
      v55 = *(_DWORD *)(v30 + 56) <= 0x40u;
      *(_QWORD *)v30 = v44;
      *(_QWORD *)(v30 + 8) = v45;
      *(_QWORD *)(v30 + 16) = v46;
      *(_BYTE *)(v30 + 24) = v47;
      *(_QWORD *)(v30 + 32) = v48;
      *(_QWORD *)(v30 + 40) = v50;
      if ( !v55 )
      {
        v58 = *(_QWORD *)(v30 + 48);
        if ( v58 )
        {
          v116 = v53;
          v119 = v51;
          j_j___libc_free_0_0(v58);
          v53 = v116;
          v51 = v119;
        }
      }
      v55 = *(_DWORD *)(v30 + 88) <= 0x40u;
      *(_QWORD *)(v30 + 48) = v51;
      *(_QWORD *)(v30 + 72) = v53;
      *(_DWORD *)(v30 + 56) = s1a;
      *(_QWORD *)(v30 + 64) = v124;
      if ( !v55 )
      {
        v59 = *(_QWORD *)(v30 + 80);
        if ( v59 )
          j_j___libc_free_0_0(v59);
      }
      *(_DWORD *)(v30 + 88) = v43;
      *(_DWORD *)(v30 + 96) = v52;
      *(_QWORD *)(v30 + 80) = v121;
      goto LABEL_22;
    }
    sub_19D7EE0(v18, v126, v106, v32, v33);
    result = (char *)v18 - (char *)a1;
    if ( (char *)v18 - (char *)a1 > 1664 )
    {
      if ( v106 )
      {
        v126 = v18;
        continue;
      }
LABEL_96:
      v66 = 0x4EC4EC4EC4EC4EC5LL * (result >> 3);
      v67 = (v66 - 2) >> 1;
      v68 = &a1[4 * v67 + 4 * ((v66 - 2) & 0xFFFFFFFFFFFFFFFELL) + v67];
      while ( 1 )
      {
        v69 = v68[5];
        v70 = *((_DWORD *)v68 + 22);
        *((_DWORD *)v68 + 22) = 0;
        v71 = *((_DWORD *)v68 + 14);
        v72 = v68[4];
        *((_DWORD *)v68 + 14) = 0;
        v73 = v68[6];
        v74 = *v68;
        v137 = v69;
        v75 = v68[8];
        v76 = v68[1];
        v143 = v70;
        v77 = (void *)v68[10];
        v78 = v68[2];
        v136 = v72;
        v140 = v75;
        v79 = v68[9];
        v142 = v77;
        v80 = *((_BYTE *)v68 + 24);
        LODWORD(v77) = *((_DWORD *)v68 + 24);
        v139 = v71;
        v138 = v73;
        v141 = v79;
        v132 = v74;
        v133 = v76;
        v134 = v78;
        v135 = v80;
        v144 = (int)v77;
        sub_19D7620((__int64)a1, v67, v66, &v132);
        if ( v143 > 0x40 && v142 )
          j_j___libc_free_0_0(v142);
        if ( v139 > 0x40 && v138 )
          j_j___libc_free_0_0(v138);
        v68 -= 13;
        if ( !v67 )
          break;
        --v67;
      }
      v81 = (__int64)a1;
      v82 = v123 - 13;
      do
      {
        v83 = (void *)v82[10];
        v84 = *((_DWORD *)v82 + 14);
        *((_DWORD *)v82 + 14) = 0;
        v85 = *((_DWORD *)v82 + 22);
        v86 = *v82;
        *((_DWORD *)v82 + 22) = 0;
        v87 = v82[1];
        v88 = v82[2];
        s1b = v83;
        v89 = *((_BYTE *)v82 + 24);
        v90 = v82[4];
        v91 = v82[5];
        v92 = v82[6];
        v93 = v82[8];
        v128 = *((_DWORD *)v82 + 24);
        v94 = v82[9];
        *v82 = *(_QWORD *)v81;
        v82[1] = *(_QWORD *)(v81 + 8);
        v82[2] = *(_QWORD *)(v81 + 16);
        *((_BYTE *)v82 + 24) = *(_BYTE *)(v81 + 24);
        v82[4] = *(_QWORD *)(v81 + 32);
        v82[5] = *(_QWORD *)(v81 + 40);
        v82[6] = *(_QWORD *)(v81 + 48);
        *((_DWORD *)v82 + 14) = *(_DWORD *)(v81 + 56);
        v95 = *(_QWORD *)(v81 + 64);
        *(_DWORD *)(v81 + 56) = 0;
        v55 = *((_DWORD *)v82 + 22) <= 0x40u;
        v82[8] = v95;
        v82[9] = *(_QWORD *)(v81 + 72);
        if ( !v55 )
        {
          v96 = v82[10];
          if ( v96 )
          {
            v110 = v92;
            v111 = v84;
            v113 = v91;
            v117 = v89;
            v120 = v88;
            v122 = v87;
            v125 = v86;
            j_j___libc_free_0_0(v96);
            v92 = v110;
            v84 = v111;
            v91 = v113;
            v89 = v117;
            v88 = v120;
            v87 = v122;
            v86 = v125;
          }
        }
        v97 = *(_QWORD *)(v81 + 80);
        v138 = v92;
        v143 = v85;
        v82[10] = v97;
        LODWORD(v97) = *(_DWORD *)(v81 + 88);
        v98 = (__int64)v82 - v81;
        v142 = s1b;
        *((_DWORD *)v82 + 22) = v97;
        LODWORD(v97) = *(_DWORD *)(v81 + 96);
        v139 = v84;
        v144 = v128;
        *(_DWORD *)(v81 + 88) = 0;
        *((_DWORD *)v82 + 24) = v97;
        v137 = v91;
        v136 = v90;
        v132 = v86;
        v133 = v87;
        v134 = v88;
        v135 = v89;
        v140 = v93;
        v141 = v94;
        result = sub_19D7620(v81, 0, 0x4EC4EC4EC4EC4EC5LL * (((__int64)v82 - v81) >> 3), &v132);
        if ( v143 > 0x40 && v142 )
          result = j_j___libc_free_0_0(v142);
        if ( v139 > 0x40 && v138 )
          result = j_j___libc_free_0_0(v138);
        v82 -= 13;
      }
      while ( v98 > 104 );
    }
    return result;
  }
}
