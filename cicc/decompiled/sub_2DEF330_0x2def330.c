// Function: sub_2DEF330
// Address: 0x2def330
//
__int64 __fastcall sub_2DEF330(unsigned __int64 a1, _QWORD *a2, __int64 a3)
{
  unsigned __int8 *v4; // rdi
  unsigned __int8 v5; // al
  unsigned __int8 *v6; // rdi
  unsigned __int8 v7; // al
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r15
  _QWORD *k; // r14
  char v12; // r12
  __int64 v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // rdx
  char v16; // al
  __int64 v17; // r13
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // r12
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // r15
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // r12
  unsigned __int64 v30; // rbx
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rbx
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rbx
  unsigned __int64 v35; // rdi
  __int64 v37; // r15
  _QWORD *v38; // r13
  char v39; // r12
  __int64 v40; // rax
  _QWORD *v41; // rax
  _QWORD *v42; // rdx
  _QWORD *v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r9
  _QWORD *v46; // r13
  char v47; // r14
  __int64 v48; // rax
  __int64 v49; // r14
  int *v50; // r13
  __int64 v51; // rcx
  __int64 v52; // rax
  __int64 v53; // r15
  __int64 v54; // rax
  __int64 v55; // r12
  __int64 v56; // r8
  int v57; // ecx
  unsigned __int64 v58; // rdi
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // r12
  unsigned __int64 v61; // rdi
  __int64 v62; // r12
  unsigned __int64 v63; // r15
  unsigned __int64 v64; // rdi
  unsigned __int64 v65; // rdi
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // r12
  unsigned __int64 v68; // rdi
  __int64 v69; // r12
  unsigned __int64 v70; // rdi
  unsigned __int64 v71; // rdi
  unsigned __int64 v72; // rax
  unsigned __int64 v73; // r8
  unsigned __int64 v74; // rdi
  __int64 v75; // rax
  unsigned __int64 v76; // rdi
  __int64 v77; // r15
  char v78; // r12
  __int64 v79; // rax
  _QWORD *v80; // rax
  _QWORD *v81; // rdx
  __int64 j; // r15
  char v83; // r12
  __int64 v84; // rax
  _QWORD *v85; // rax
  _QWORD *v86; // rdx
  __int64 v87; // rdx
  char v88; // al
  _QWORD *v89; // [rsp+8h] [rbp-2A8h]
  unsigned __int64 v90; // [rsp+8h] [rbp-2A8h]
  _QWORD *v91; // [rsp+8h] [rbp-2A8h]
  _QWORD *v92; // [rsp+10h] [rbp-2A0h]
  _QWORD *v93; // [rsp+10h] [rbp-2A0h]
  unsigned __int64 v95; // [rsp+18h] [rbp-298h]
  unsigned __int64 v96; // [rsp+18h] [rbp-298h]
  unsigned __int64 v97; // [rsp+18h] [rbp-298h]
  __int64 v98; // [rsp+18h] [rbp-298h]
  _QWORD *m; // [rsp+20h] [rbp-290h]
  int *v100; // [rsp+20h] [rbp-290h]
  _QWORD *i; // [rsp+20h] [rbp-290h]
  __int64 v102; // [rsp+28h] [rbp-288h]
  unsigned __int8 v103; // [rsp+28h] [rbp-288h]
  __int64 (__fastcall **v104)(); // [rsp+30h] [rbp-280h] BYREF
  __int64 v105; // [rsp+38h] [rbp-278h]
  __int64 v106; // [rsp+40h] [rbp-270h]
  char v107; // [rsp+50h] [rbp-260h] BYREF
  unsigned __int64 v108; // [rsp+58h] [rbp-258h]
  __int64 v109; // [rsp+60h] [rbp-250h]
  char v110; // [rsp+80h] [rbp-230h] BYREF
  unsigned __int64 v111; // [rsp+88h] [rbp-228h]
  __int64 v112; // [rsp+90h] [rbp-220h]
  __int64 v113; // [rsp+B0h] [rbp-200h]
  __int64 (__fastcall **v114)(); // [rsp+C0h] [rbp-1F0h] BYREF
  __int64 v115; // [rsp+C8h] [rbp-1E8h]
  __int64 v116; // [rsp+D0h] [rbp-1E0h]
  char v117; // [rsp+E0h] [rbp-1D0h] BYREF
  unsigned __int64 v118; // [rsp+E8h] [rbp-1C8h]
  __int64 v119; // [rsp+F0h] [rbp-1C0h]
  char v120; // [rsp+110h] [rbp-1A0h] BYREF
  unsigned __int64 v121; // [rsp+118h] [rbp-198h]
  __int64 v122; // [rsp+120h] [rbp-190h]
  __int64 v123; // [rsp+140h] [rbp-170h]
  _QWORD v124[18]; // [rsp+150h] [rbp-160h] BYREF
  unsigned __int64 v125; // [rsp+1E0h] [rbp-D0h] BYREF
  __int64 v126; // [rsp+1E8h] [rbp-C8h]
  _BYTE *v127; // [rsp+1F0h] [rbp-C0h] BYREF
  __int64 v128; // [rsp+1F8h] [rbp-B8h]
  _BYTE v129[96]; // [rsp+200h] [rbp-B0h] BYREF
  __int64 v130; // [rsp+260h] [rbp-50h]
  int v131; // [rsp+268h] [rbp-48h]
  __int64 v132; // [rsp+270h] [rbp-40h]

  v102 = *(_QWORD *)(*(_QWORD *)(a1 - 64) + 8LL);
  sub_2DEB2E0((__int64)&v104, v102);
  v4 = *(unsigned __int8 **)(a1 - 64);
  v5 = *v4;
  if ( *v4 <= 0x1Cu )
    goto LABEL_5;
  switch ( v5 )
  {
    case '\\':
      v88 = sub_2DEF330(v4, &v104, a3);
      break;
    case '=':
      v88 = sub_2DEC8F0(v4, &v104, a3);
      break;
    case 'N':
      v88 = sub_2DF0160(v4, &v104, a3);
      break;
    default:
LABEL_5:
      v105 = 0;
      goto LABEL_6;
  }
  if ( !v88 )
    goto LABEL_5;
LABEL_6:
  sub_2DEB2E0((__int64)&v114, v102);
  v6 = *(unsigned __int8 **)(a1 - 32);
  v7 = *v6;
  if ( *v6 <= 0x1Cu )
  {
    v8 = v105;
    goto LABEL_8;
  }
  if ( v7 == 92 )
  {
    v16 = sub_2DEF330(v6, &v114, a3);
  }
  else if ( v7 == 61 )
  {
    v16 = sub_2DEC8F0(v6, &v114, a3);
  }
  else
  {
    v8 = v105;
    if ( v7 != 78 )
      goto LABEL_8;
    v16 = sub_2DF0160(v6, &v114, a3);
  }
  v8 = v105;
  if ( !v16 )
  {
LABEL_8:
    v115 = 0;
    if ( v8 )
    {
LABEL_9:
      v9 = v106;
      a2[1] = v8;
      a2[2] = v9;
      goto LABEL_10;
    }
    goto LABEL_23;
  }
  if ( !v105 )
  {
    if ( v115 )
    {
      a2[1] = v115;
      v38 = a2 + 9;
      k = a2 + 3;
      a2[2] = v116;
LABEL_155:
      v77 = v119;
      for ( i = a2 + 4; (char *)v77 != &v117; v77 = sub_220EF30(v77) )
      {
        v80 = sub_2DEDBB0(k, (__int64)i, (unsigned __int64 *)(v77 + 32));
        if ( v81 )
        {
          v78 = v80 || v81 == i || *(_QWORD *)(v77 + 32) < v81[4];
          v91 = v81;
          v79 = sub_22077B0(0x28u);
          *(_QWORD *)(v79 + 32) = *(_QWORD *)(v77 + 32);
          sub_220F040(v78, v79, v91, i);
          ++a2[8];
        }
      }
      for ( j = v122; (char *)j != &v120; j = sub_220EF30(j) )
      {
        v85 = sub_2D12A40(v38, (__int64)(a2 + 10), (unsigned __int64 *)(j + 32));
        if ( v86 )
        {
          v83 = v85 || v86 == a2 + 10 || *(_QWORD *)(j + 32) < v86[4];
          v93 = v86;
          v84 = sub_22077B0(0x28u);
          *(_QWORD *)(v84 + 32) = *(_QWORD *)(j + 32);
          sub_220F040(v83, v84, v93, a2 + 10);
          ++a2[14];
        }
      }
      goto LABEL_73;
    }
    goto LABEL_23;
  }
  if ( !v115 )
    goto LABEL_9;
  if ( v115 != v105 || (v87 = v106, v106 != v116) )
  {
LABEL_23:
    v103 = 0;
    goto LABEL_24;
  }
  a2[1] = v115;
  a2[2] = v87;
LABEL_10:
  v10 = v109;
  for ( k = a2 + 3; (char *)v10 != &v107; v10 = sub_220EF30(v10) )
  {
    v14 = sub_2DEDBB0(a2 + 3, (__int64)(a2 + 4), (unsigned __int64 *)(v10 + 32));
    if ( v15 )
    {
      v12 = v14 || a2 + 4 == v15 || *(_QWORD *)(v10 + 32) < v15[4];
      v92 = v15;
      v13 = sub_22077B0(0x28u);
      *(_QWORD *)(v13 + 32) = *(_QWORD *)(v10 + 32);
      sub_220F040(v12, v13, v92, a2 + 4);
      ++a2[8];
    }
  }
  v37 = v112;
  v38 = a2 + 9;
  for ( m = a2 + 10; (char *)v37 != &v110; v37 = sub_220EF30(v37) )
  {
    v41 = sub_2D12A40(a2 + 9, (__int64)m, (unsigned __int64 *)(v37 + 32));
    if ( v42 )
    {
      v39 = v41 || v42 == m || *(_QWORD *)(v37 + 32) < v42[4];
      v89 = v42;
      v40 = sub_22077B0(0x28u);
      *(_QWORD *)(v40 + 32) = *(_QWORD *)(v37 + 32);
      sub_220F040(v39, v40, v89, m);
      ++a2[14];
    }
  }
  if ( v115 )
    goto LABEL_155;
LABEL_73:
  v125 = a1;
  v43 = sub_2D11AF0((__int64)v38, &v125);
  v46 = (_QWORD *)v44;
  if ( v44 )
  {
    v47 = 1;
    if ( !v43 && (_QWORD *)v44 != a2 + 10 )
      v47 = a1 < *(_QWORD *)(v44 + 32);
    v48 = sub_22077B0(0x28u);
    *(_QWORD *)(v48 + 32) = v125;
    sub_220F040(v47, v48, v46, a2 + 10);
    ++a2[14];
  }
  v49 = 0;
  a2[15] = a1;
  v50 = *(int **)(a1 + 72);
  v100 = &v50[*(unsigned int *)(a1 + 80)];
  if ( v100 != v50 )
  {
    while ( 1 )
    {
      v54 = *v50;
      v55 = v49 + a2[16];
      v56 = v55 + 16;
      if ( (int)v54 < 0 )
        break;
      v57 = *(_DWORD *)(v102 + 32);
      if ( v57 > (int)v54 )
      {
        if ( !v105 )
        {
          v126 = 0;
          memset(v124, 0, sizeof(v124));
          LODWORD(v124[0]) = -1;
          v124[2] = &v124[4];
          v124[3] = 0x400000000LL;
          v128 = 0x400000000LL;
          LODWORD(v124[17]) = 1;
          LODWORD(v125) = -1;
          v127 = v129;
          v131 = 1;
          v130 = 0;
          v132 = 0;
          *(_DWORD *)v55 = -1;
          *(_QWORD *)(v55 + 8) = v126;
          sub_2DEB400(v55 + 16, (unsigned __int64 *)&v127, v44, 0, v56, v45);
          if ( *(_DWORD *)(v55 + 136) > 0x40u )
          {
            v65 = *(_QWORD *)(v55 + 128);
            if ( v65 )
              j_j___libc_free_0_0(v65);
          }
          *(_QWORD *)(v55 + 128) = v130;
          *(_DWORD *)(v55 + 136) = v131;
          *(_QWORD *)(v55 + 144) = v132;
          v66 = (unsigned __int64)v127;
          v67 = (unsigned __int64)&v127[24 * (unsigned int)v128];
          if ( v127 != (_BYTE *)v67 )
          {
            do
            {
              v67 -= 24LL;
              if ( *(_DWORD *)(v67 + 16) > 0x40u )
              {
                v68 = *(_QWORD *)(v67 + 8);
                if ( v68 )
                {
                  v96 = v66;
                  j_j___libc_free_0_0(v68);
                  v66 = v96;
                }
              }
            }
            while ( v66 != v67 );
            v67 = (unsigned __int64)v127;
          }
          if ( (_BYTE *)v67 != v129 )
            _libc_free(v67);
          if ( LODWORD(v124[17]) > 0x40 && v124[16] )
            j_j___libc_free_0_0(v124[16]);
          v69 = v124[2];
          v44 = 3LL * LODWORD(v124[3]);
          v63 = v124[2] + 24LL * LODWORD(v124[3]);
          if ( v124[2] == v63 )
            goto LABEL_109;
          do
          {
            v63 -= 24LL;
            if ( *(_DWORD *)(v63 + 16) > 0x40u )
            {
              v70 = *(_QWORD *)(v63 + 8);
              if ( v70 )
                j_j___libc_free_0_0(v70);
            }
          }
          while ( v69 != v63 );
          goto LABEL_108;
        }
        v51 = 19 * v54;
        v52 = v113;
      }
      else
      {
        if ( !v115 )
        {
          v126 = 0;
          memset(v124, 0, sizeof(v124));
          LODWORD(v124[0]) = -1;
          v124[2] = &v124[4];
          v124[3] = 0x400000000LL;
          v128 = 0x400000000LL;
          LODWORD(v124[17]) = 1;
          LODWORD(v125) = -1;
          v127 = v129;
          v131 = 1;
          v130 = 0;
          v132 = 0;
          *(_DWORD *)v55 = -1;
          *(_QWORD *)(v55 + 8) = v126;
          sub_2DEB400(v55 + 16, (unsigned __int64 *)&v127, v44, 0, v56, v45);
          if ( *(_DWORD *)(v55 + 136) > 0x40u )
          {
            v71 = *(_QWORD *)(v55 + 128);
            if ( v71 )
              j_j___libc_free_0_0(v71);
          }
          *(_QWORD *)(v55 + 128) = v130;
          *(_DWORD *)(v55 + 136) = v131;
          *(_QWORD *)(v55 + 144) = v132;
          v72 = (unsigned __int64)v127;
          v73 = (unsigned __int64)&v127[24 * (unsigned int)v128];
          if ( v127 != (_BYTE *)v73 )
          {
            do
            {
              v73 -= 24LL;
              if ( *(_DWORD *)(v73 + 16) > 0x40u )
              {
                v74 = *(_QWORD *)(v73 + 8);
                if ( v74 )
                {
                  v90 = v73;
                  v97 = v72;
                  j_j___libc_free_0_0(v74);
                  v73 = v90;
                  v72 = v97;
                }
              }
            }
            while ( v72 != v73 );
            v73 = (unsigned __int64)v127;
          }
          if ( (_BYTE *)v73 != v129 )
            _libc_free(v73);
          if ( LODWORD(v124[17]) > 0x40 && v124[16] )
            j_j___libc_free_0_0(v124[16]);
          v75 = v124[2];
          v44 = 3LL * LODWORD(v124[3]);
          v63 = v124[2] + 24LL * LODWORD(v124[3]);
          if ( v124[2] != v63 )
          {
            do
            {
              v63 -= 24LL;
              if ( *(_DWORD *)(v63 + 16) > 0x40u )
              {
                v76 = *(_QWORD *)(v63 + 8);
                if ( v76 )
                {
                  v98 = v75;
                  j_j___libc_free_0_0(v76);
                  v75 = v98;
                }
              }
            }
            while ( v75 != v63 );
            goto LABEL_108;
          }
LABEL_109:
          if ( (_QWORD *)v63 != &v124[4] )
            _libc_free(v63);
          goto LABEL_84;
        }
        v51 = 19LL * (unsigned int)(v54 - v57);
        v52 = v123;
      }
      v53 = v52 + 8 * v51;
      *(_DWORD *)v55 = *(_DWORD *)v53;
      *(_QWORD *)(v55 + 8) = *(_QWORD *)(v53 + 8);
      sub_2DEB050(v55 + 16, (__int64 *)(v53 + 16), v44, v51, v56, v45);
      if ( *(_DWORD *)(v55 + 136) <= 0x40u && *(_DWORD *)(v53 + 136) <= 0x40u )
      {
        *(_QWORD *)(v55 + 128) = *(_QWORD *)(v53 + 128);
        *(_DWORD *)(v55 + 136) = *(_DWORD *)(v53 + 136);
      }
      else
      {
        sub_C43990(v55 + 128, v53 + 128);
      }
      *(_QWORD *)(v55 + 144) = *(_QWORD *)(v53 + 144);
LABEL_84:
      ++v50;
      v49 += 152;
      if ( v100 == v50 )
        goto LABEL_111;
    }
    v126 = 0;
    memset(v124, 0, sizeof(v124));
    LODWORD(v124[0]) = -1;
    v124[2] = &v124[4];
    v124[3] = 0x400000000LL;
    v128 = 0x400000000LL;
    LODWORD(v124[17]) = 1;
    LODWORD(v125) = -1;
    v127 = v129;
    v131 = 1;
    v130 = 0;
    v132 = 0;
    *(_DWORD *)v55 = -1;
    *(_QWORD *)(v55 + 8) = v126;
    sub_2DEB400(v55 + 16, (unsigned __int64 *)&v127, v44, 0, v56, v45);
    if ( *(_DWORD *)(v55 + 136) > 0x40u )
    {
      v58 = *(_QWORD *)(v55 + 128);
      if ( v58 )
        j_j___libc_free_0_0(v58);
    }
    *(_QWORD *)(v55 + 128) = v130;
    *(_DWORD *)(v55 + 136) = v131;
    *(_QWORD *)(v55 + 144) = v132;
    v59 = (unsigned __int64)v127;
    v60 = (unsigned __int64)&v127[24 * (unsigned int)v128];
    if ( v127 != (_BYTE *)v60 )
    {
      do
      {
        v60 -= 24LL;
        if ( *(_DWORD *)(v60 + 16) > 0x40u )
        {
          v61 = *(_QWORD *)(v60 + 8);
          if ( v61 )
          {
            v95 = v59;
            j_j___libc_free_0_0(v61);
            v59 = v95;
          }
        }
      }
      while ( v59 != v60 );
      v60 = (unsigned __int64)v127;
    }
    if ( (_BYTE *)v60 != v129 )
      _libc_free(v60);
    if ( LODWORD(v124[17]) > 0x40 && v124[16] )
      j_j___libc_free_0_0(v124[16]);
    v62 = v124[2];
    v44 = 3LL * LODWORD(v124[3]);
    v63 = v124[2] + 24LL * LODWORD(v124[3]);
    if ( v124[2] == v63 )
      goto LABEL_109;
    do
    {
      v63 -= 24LL;
      if ( *(_DWORD *)(v63 + 16) > 0x40u )
      {
        v64 = *(_QWORD *)(v63 + 8);
        if ( v64 )
          j_j___libc_free_0_0(v64);
      }
    }
    while ( v62 != v63 );
LABEL_108:
    v63 = v124[2];
    goto LABEL_109;
  }
LABEL_111:
  v103 = 1;
LABEL_24:
  v114 = off_49D4228;
  if ( v123 )
  {
    v17 = v123 + 152LL * *(_QWORD *)(v123 - 8);
    while ( v123 != v17 )
    {
      v17 -= 152;
      if ( *(_DWORD *)(v17 + 136) > 0x40u )
      {
        v18 = *(_QWORD *)(v17 + 128);
        if ( v18 )
          j_j___libc_free_0_0(v18);
      }
      v19 = *(_QWORD *)(v17 + 16);
      v20 = v19 + 24LL * *(unsigned int *)(v17 + 24);
      if ( v19 != v20 )
      {
        do
        {
          v20 -= 24LL;
          if ( *(_DWORD *)(v20 + 16) > 0x40u )
          {
            v21 = *(_QWORD *)(v20 + 8);
            if ( v21 )
              j_j___libc_free_0_0(v21);
          }
        }
        while ( v19 != v20 );
        v19 = *(_QWORD *)(v17 + 16);
      }
      if ( v19 != v17 + 32 )
        _libc_free(v19);
    }
    j_j_j___libc_free_0_0(v17 - 8);
  }
  v22 = v121;
  while ( v22 )
  {
    sub_2DEAE80(*(_QWORD *)(v22 + 24));
    v23 = v22;
    v22 = *(_QWORD *)(v22 + 16);
    j_j___libc_free_0(v23);
  }
  v24 = v118;
  while ( v24 )
  {
    sub_2DEACB0(*(_QWORD *)(v24 + 24));
    v25 = v24;
    v24 = *(_QWORD *)(v24 + 16);
    j_j___libc_free_0(v25);
  }
  v104 = off_49D4228;
  if ( v113 )
  {
    v26 = 152LL * *(_QWORD *)(v113 - 8);
    v27 = v113 + v26;
    while ( v113 != v27 )
    {
      v27 -= 152;
      if ( *(_DWORD *)(v27 + 136) > 0x40u )
      {
        v28 = *(_QWORD *)(v27 + 128);
        if ( v28 )
          j_j___libc_free_0_0(v28);
      }
      v29 = *(_QWORD *)(v27 + 16);
      v30 = v29 + 24LL * *(unsigned int *)(v27 + 24);
      if ( v29 != v30 )
      {
        do
        {
          v30 -= 24LL;
          if ( *(_DWORD *)(v30 + 16) > 0x40u )
          {
            v31 = *(_QWORD *)(v30 + 8);
            if ( v31 )
              j_j___libc_free_0_0(v31);
          }
        }
        while ( v29 != v30 );
        v29 = *(_QWORD *)(v27 + 16);
      }
      if ( v29 != v27 + 32 )
        _libc_free(v29);
    }
    j_j_j___libc_free_0_0(v27 - 8);
  }
  v32 = v111;
  while ( v32 )
  {
    sub_2DEAE80(*(_QWORD *)(v32 + 24));
    v33 = v32;
    v32 = *(_QWORD *)(v32 + 16);
    j_j___libc_free_0(v33);
  }
  v34 = v108;
  while ( v34 )
  {
    sub_2DEACB0(*(_QWORD *)(v34 + 24));
    v35 = v34;
    v34 = *(_QWORD *)(v34 + 16);
    j_j___libc_free_0(v35);
  }
  return v103;
}
