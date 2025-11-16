// Function: sub_2A64290
// Address: 0x2a64290
//
__int64 __fastcall sub_2A64290(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4)
{
  unsigned __int64 v4; // rax
  __int64 v5; // r14
  unsigned int v7; // r15d
  __int64 v8; // rbx
  __int64 *v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 *v13; // rax
  unsigned __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rcx
  unsigned int v17; // esi
  __int64 v18; // r15
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // r15
  unsigned __int64 v29; // rdx
  unsigned __int64 *v30; // rax
  __int64 v31; // r15
  __int64 v32; // rax
  unsigned __int64 v33; // r15
  unsigned __int64 *v34; // rax
  __int64 v35; // rcx
  __int64 v36; // r14
  __int64 v37; // rbx
  __int64 v38; // rdx
  __int64 v39; // r15
  __int64 *v40; // rax
  __int64 *v41; // rdx
  unsigned __int64 *v42; // rdi
  __int64 v44; // rdi
  __int64 v45; // r15
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  unsigned __int64 *v50; // rax
  __int64 *v51; // rax
  __int64 *v52; // rcx
  __int64 v53; // r14
  __int64 *v54; // rsi
  unsigned __int64 v55; // rbx
  unsigned __int8 v56; // r13
  __int64 v57; // rax
  unsigned int v58; // ebx
  __int64 v59; // r15
  __int64 v60; // r8
  __int64 v61; // rdx
  unsigned __int64 v62; // r9
  __int64 v63; // rax
  unsigned __int64 *v64; // rdx
  __int64 v65; // rax
  __int64 v66; // r12
  __int64 v67; // rax
  __int64 v68; // r13
  int v69; // ebx
  unsigned int v70; // r14d
  __int64 v71; // r15
  __int64 *v72; // rdx
  __int64 v73; // rcx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 *v76; // rax
  __int64 v77; // r13
  __int64 v78; // rcx
  _QWORD *v79; // rdi
  __int64 v80; // r8
  __int64 v81; // r9
  char v82; // dl
  __int64 v83; // rax
  __int64 v84; // r15
  unsigned __int64 v85; // rdx
  unsigned __int64 *v86; // rax
  __int64 v87; // r15
  unsigned __int16 v88; // r12
  _QWORD *v89; // rax
  _QWORD *v90; // rbx
  __int64 *v91; // r12
  __int64 v92; // rsi
  __int64 v93; // rcx
  __int64 v94; // r8
  __int64 v95; // r9
  __int64 v96; // rcx
  __int64 v97; // rax
  __int64 v98; // r14
  __int64 v99; // r14
  _QWORD *v100; // rdi
  __int64 v101; // rsi
  unsigned __int8 *v102; // rsi
  __int64 v103; // [rsp+8h] [rbp-1E8h]
  __int64 v106; // [rsp+20h] [rbp-1D0h]
  __int64 v107; // [rsp+28h] [rbp-1C8h]
  int v108; // [rsp+28h] [rbp-1C8h]
  unsigned __int8 v109; // [rsp+37h] [rbp-1B9h]
  int v110; // [rsp+38h] [rbp-1B8h]
  _QWORD *v111; // [rsp+38h] [rbp-1B8h]
  __int64 v112; // [rsp+38h] [rbp-1B8h]
  __int64 v113; // [rsp+38h] [rbp-1B8h]
  const char *v114; // [rsp+40h] [rbp-1B0h] BYREF
  __int64 v115; // [rsp+48h] [rbp-1A8h]
  char v116; // [rsp+60h] [rbp-190h]
  char v117; // [rsp+61h] [rbp-18Fh]
  __int64 v118; // [rsp+70h] [rbp-180h] BYREF
  __int64 *v119; // [rsp+78h] [rbp-178h]
  __int64 v120; // [rsp+80h] [rbp-170h]
  int v121; // [rsp+88h] [rbp-168h]
  char v122; // [rsp+8Ch] [rbp-164h]
  char v123; // [rsp+90h] [rbp-160h] BYREF
  _QWORD *v124; // [rsp+D0h] [rbp-120h] BYREF
  __int64 *v125; // [rsp+D8h] [rbp-118h]
  __int64 v126; // [rsp+E0h] [rbp-110h]
  int v127; // [rsp+E8h] [rbp-108h] BYREF
  char v128; // [rsp+ECh] [rbp-104h]
  char v129; // [rsp+F0h] [rbp-100h] BYREF
  char v130; // [rsp+108h] [rbp-E8h]
  char v131; // [rsp+110h] [rbp-E0h]
  unsigned __int64 *v132; // [rsp+130h] [rbp-C0h] BYREF
  __int64 v133; // [rsp+138h] [rbp-B8h]
  _BYTE v134[176]; // [rsp+140h] [rbp-B0h] BYREF

  v119 = (__int64 *)&v123;
  v4 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  v118 = 0;
  v120 = 8;
  v121 = 0;
  v122 = 1;
  v107 = a2 + 48;
  if ( a2 + 48 == v4 )
    return 0;
  if ( !v4 )
    goto LABEL_82;
  v5 = v4 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
    return 0;
  v110 = sub_B46E30(v5);
  if ( !v110 )
    return 0;
  v109 = 0;
  v7 = 0;
  do
  {
    while ( 1 )
    {
      v8 = sub_B46EC0(v5, v7);
      if ( (unsigned __int8)sub_2A64280(a1, a2, v8) )
        break;
      v109 = 1;
LABEL_7:
      if ( v110 == ++v7 )
        goto LABEL_15;
    }
    if ( !v122 )
      goto LABEL_58;
    v13 = v119;
    v9 = &v119[HIDWORD(v120)];
    if ( v119 != v9 )
    {
      while ( v8 != *v13 )
      {
        if ( v9 == ++v13 )
          goto LABEL_13;
      }
      goto LABEL_7;
    }
LABEL_13:
    if ( HIDWORD(v120) >= (unsigned int)v120 )
    {
LABEL_58:
      sub_C8CC70((__int64)&v118, v8, (__int64)v9, v10, v11, v12);
      goto LABEL_7;
    }
    ++v7;
    ++HIDWORD(v120);
    *v9 = v8;
    ++v118;
  }
  while ( v110 != v7 );
LABEL_15:
  if ( !v109 )
    goto LABEL_55;
  v14 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v107 == v14 )
  {
    v111 = 0;
    goto LABEL_21;
  }
  if ( !v14 )
LABEL_127:
    BUG();
  v15 = 0;
  if ( (unsigned int)*(unsigned __int8 *)(v14 - 24) - 30 < 0xB )
    v15 = (_QWORD *)(v14 - 24);
  v111 = v15;
LABEL_21:
  v16 = HIDWORD(v120);
  v17 = HIDWORD(v120) - v121;
  if ( v121 == HIDWORD(v120) )
  {
    v124 = 0;
    v125 = (__int64 *)&v129;
    v132 = (unsigned __int64 *)v134;
    v126 = 8;
    v127 = 0;
    v128 = 1;
    v133 = 0x800000000LL;
    if ( v107 == v14 )
      goto LABEL_94;
    if ( !v14 )
      goto LABEL_82;
    v68 = v14 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v14 - 24) - 30 > 0xA || (v69 = sub_B46E30(v14 - 24)) == 0 )
    {
LABEL_94:
      sub_B43D60(v111);
      v77 = sub_AA48A0(a2);
      sub_B43C20((__int64)&v114, a2);
      v79 = sub_BD2C40(72, unk_3F148B8);
      if ( v79 )
        sub_B4C8A0((__int64)v79, v77, (__int64)v114, v115);
      sub_FFDB80(a3, v132, (unsigned int)v133, v78, v80, v81);
      if ( v132 != (unsigned __int64 *)v134 )
        _libc_free((unsigned __int64)v132);
      if ( !v128 )
        _libc_free((unsigned __int64)v125);
      goto LABEL_55;
    }
    v70 = 0;
    while ( 1 )
    {
      v71 = sub_B46EC0(v68, v70);
      sub_AA5980(v71, a2, 0);
      if ( v128 )
      {
        v76 = v125;
        v73 = HIDWORD(v126);
        v72 = &v125[HIDWORD(v126)];
        if ( v125 != v72 )
        {
          while ( v71 != *v76 )
          {
            if ( v72 == ++v76 )
              goto LABEL_104;
          }
          goto LABEL_93;
        }
LABEL_104:
        if ( HIDWORD(v126) < (unsigned int)v126 )
          break;
      }
      sub_C8CC70((__int64)&v124, v71, (__int64)v72, v73, v74, v75);
      if ( v82 )
        goto LABEL_101;
LABEL_93:
      if ( v69 == ++v70 )
        goto LABEL_94;
    }
    ++HIDWORD(v126);
    *v72 = v71;
    v124 = (_QWORD *)((char *)v124 + 1);
LABEL_101:
    v83 = (unsigned int)v133;
    v84 = v71 | 4;
    v85 = (unsigned int)v133 + 1LL;
    if ( v85 > HIDWORD(v133) )
    {
      sub_C8D5F0((__int64)&v132, v134, v85, 0x10u, v74, v75);
      v83 = (unsigned int)v133;
    }
    v86 = &v132[2 * v83];
    *v86 = a2;
    v86[1] = v84;
    LODWORD(v133) = v133 + 1;
    goto LABEL_93;
  }
  if ( v17 == 1 )
  {
    v51 = v119;
    if ( !v122 )
      v16 = (unsigned int)v120;
    v52 = &v119[v16];
    v53 = *v119;
    if ( v119 != v52 )
    {
      while ( 1 )
      {
        v53 = *v51;
        v54 = v51;
        if ( (unsigned __int64)*v51 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v52 == ++v51 )
        {
          v53 = v54[1];
          break;
        }
      }
    }
    v132 = (unsigned __int64 *)v134;
    v133 = 0x800000000LL;
    if ( v107 == v14 )
    {
LABEL_106:
      sub_B43C20((__int64)&v124, a2);
      v87 = (__int64)v124;
      v88 = (unsigned __int16)v125;
      v89 = sub_BD2C40(72, 1u);
      v90 = v89;
      if ( v89 )
        sub_B4C8F0((__int64)v89, v53, 1u, v87, v88);
      v91 = v90 + 6;
      v92 = v111[6];
      v124 = (_QWORD *)v92;
      if ( v92 )
      {
        sub_B96E90((__int64)&v124, v92, 1);
        if ( v91 == (__int64 *)&v124 )
        {
          if ( v124 )
            sub_B91220((__int64)&v124, (__int64)v124);
          goto LABEL_112;
        }
        v101 = v90[6];
        if ( !v101 )
        {
LABEL_122:
          v102 = (unsigned __int8 *)v124;
          v90[6] = v124;
          if ( v102 )
            sub_B976B0((__int64)&v124, v102, (__int64)(v90 + 6));
LABEL_112:
          sub_B43D60(v111);
          sub_FFDB80(a3, v132, (unsigned int)v133, v93, v94, v95);
          v42 = v132;
          if ( v132 != (unsigned __int64 *)v134 )
            goto LABEL_54;
          goto LABEL_55;
        }
      }
      else
      {
        if ( v91 == (__int64 *)&v124 )
          goto LABEL_112;
        v101 = v90[6];
        if ( !v101 )
          goto LABEL_112;
      }
      sub_B91220((__int64)(v90 + 6), v101);
      goto LABEL_122;
    }
    if ( v14 )
    {
      v55 = v14 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v14 - 24) - 30 <= 0xA )
      {
        v108 = sub_B46E30(v14 - 24);
        if ( v108 )
        {
          v56 = 0;
          v57 = v55;
          v58 = 0;
          v59 = v57;
          do
          {
            v65 = sub_B46EC0(v59, v58);
            if ( ((v56 ^ 1) & (v65 == v53)) != 0 )
            {
              v56 = (v56 ^ 1) & (v65 == v53);
            }
            else
            {
              v106 = v65;
              sub_AA5980(v65, a2, 0);
              v61 = (unsigned int)v133;
              v62 = (unsigned int)v133 + 1LL;
              v63 = v106 | 4;
              if ( v62 > HIDWORD(v133) )
              {
                sub_C8D5F0((__int64)&v132, v134, (unsigned int)v133 + 1LL, 0x10u, v60, v62);
                v61 = (unsigned int)v133;
                v63 = v106 | 4;
              }
              v64 = &v132[2 * v61];
              *v64 = a2;
              v64[1] = v63;
              LODWORD(v133) = v133 + 1;
            }
            ++v58;
          }
          while ( v108 != v58 );
        }
      }
      goto LABEL_106;
    }
LABEL_82:
    BUG();
  }
  if ( v17 <= 1 )
    goto LABEL_127;
  v130 = 0;
  v131 = 0;
  v124 = v111;
  sub_B540B0(&v124);
  v132 = (unsigned __int64 *)v134;
  v133 = 0x800000000LL;
  v18 = *(_QWORD *)(*(v124 - 1) + 32LL);
  if ( !(unsigned __int8)sub_B19060((__int64)&v118, v18, v19, v20) )
  {
    if ( !*a4 )
    {
      v96 = *(_QWORD *)(v18 + 72);
      v117 = 1;
      v114 = "default.unreachable";
      v103 = v96;
      v116 = 3;
      v113 = sub_AA48A0(v18);
      v97 = sub_22077B0(0x50u);
      v98 = v97;
      if ( v97 )
        sub_AA4D50(v97, v113, (__int64)&v114, v103, v18);
      *a4 = v98;
      v99 = sub_AA48A0(v18);
      sub_B43C20((__int64)&v114, *a4);
      v100 = sub_BD2C40(72, unk_3F148B8);
      if ( v100 )
        sub_B4C8A0((__int64)v100, v99, (__int64)v114, v115);
    }
    sub_AA5980(v18, a2, 0);
    v23 = *a4;
    v24 = *(v124 - 1);
    if ( *(_QWORD *)(v24 + 32) )
    {
      v25 = *(_QWORD *)(v24 + 40);
      **(_QWORD **)(v24 + 48) = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 16) = *(_QWORD *)(v24 + 48);
    }
    *(_QWORD *)(v24 + 32) = v23;
    if ( v23 )
    {
      v26 = *(_QWORD *)(v23 + 16);
      *(_QWORD *)(v24 + 40) = v26;
      if ( v26 )
        *(_QWORD *)(v26 + 16) = v24 + 40;
      *(_QWORD *)(v24 + 48) = v23 + 16;
      *(_QWORD *)(v23 + 16) = v24 + 32;
    }
    v27 = (unsigned int)v133;
    v28 = v18 | 4;
    v29 = (unsigned int)v133 + 1LL;
    if ( v29 > HIDWORD(v133) )
    {
      sub_C8D5F0((__int64)&v132, v134, v29, 0x10u, v21, v22);
      v27 = (unsigned int)v133;
    }
    v30 = &v132[2 * v27];
    *v30 = a2;
    v30[1] = v28;
    v31 = *a4;
    LODWORD(v133) = v133 + 1;
    v32 = (unsigned int)v133;
    v33 = v31 & 0xFFFFFFFFFFFFFFFBLL;
    if ( (unsigned __int64)(unsigned int)v133 + 1 > HIDWORD(v133) )
    {
      sub_C8D5F0((__int64)&v132, v134, (unsigned int)v133 + 1LL, 0x10u, v21, v22);
      v32 = (unsigned int)v133;
    }
    v34 = &v132[2 * v32];
    *v34 = a2;
    v34[1] = v33;
    LODWORD(v133) = v133 + 1;
  }
  v35 = (__int64)v124;
  v36 = 0;
  v37 = (__int64)v124;
  if ( (*((_DWORD *)v124 + 1) & 0x7FFFFFFu) >> 1 != 1 )
  {
    do
    {
      v38 = 32;
      if ( (_DWORD)v36 != -2 )
        v38 = 32LL * (unsigned int)(2 * v36 + 3);
      v39 = *(_QWORD *)(*(_QWORD *)(v37 - 8) + v38);
      if ( v122 )
      {
        v40 = v119;
        v41 = &v119[HIDWORD(v120)];
        if ( v119 == v41 )
          goto LABEL_60;
        while ( *v40 != v39 )
        {
          if ( v41 == ++v40 )
            goto LABEL_60;
        }
      }
      else
      {
        v112 = v38;
        if ( !sub_C8CA60((__int64)&v118, v39) )
        {
          v39 = *(_QWORD *)(*(_QWORD *)(v37 - 8) + v112);
LABEL_60:
          v44 = v39;
          v45 = v39 | 4;
          sub_AA5980(v44, a2, 0);
          v48 = (unsigned int)v133;
          v49 = (unsigned int)v133 + 1LL;
          if ( v49 > HIDWORD(v133) )
          {
            sub_C8D5F0((__int64)&v132, v134, v49, 0x10u, v46, v47);
            v48 = (unsigned int)v133;
          }
          v50 = &v132[2 * v48];
          *v50 = a2;
          v50[1] = v45;
          LODWORD(v133) = v133 + 1;
          sub_B541A0((__int64)&v124, v37, v36);
          v35 = (__int64)v124;
          continue;
        }
        v35 = (__int64)v124;
      }
      ++v36;
    }
    while ( ((*(_DWORD *)(v35 + 4) & 0x7FFFFFFu) >> 1) - 1 != v36 );
  }
  sub_FFDB80(a3, v132, (unsigned int)v133, v35, v21, v22);
  if ( v132 != (unsigned __int64 *)v134 )
    _libc_free((unsigned __int64)v132);
  if ( v131 )
  {
    v66 = (__int64)v124;
    v67 = sub_B53F50((__int64)&v124);
    sub_B99FD0(v66, 2u, v67);
  }
  if ( v130 )
  {
    v42 = (unsigned __int64 *)v125;
    if ( v125 != (__int64 *)&v127 )
LABEL_54:
      _libc_free((unsigned __int64)v42);
  }
LABEL_55:
  if ( !v122 )
    _libc_free((unsigned __int64)v119);
  return v109;
}
