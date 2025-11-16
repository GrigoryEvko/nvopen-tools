// Function: sub_14602B0
// Address: 0x14602b0
//
__int64 __fastcall sub_14602B0(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // rdi
  __int64 v4; // rax
  int v5; // r14d
  _QWORD *v6; // rbx
  unsigned int v7; // eax
  __int64 v8; // rdx
  _QWORD *v9; // r12
  __int64 v10; // rdi
  int v11; // eax
  __int64 v12; // rdx
  _QWORD *v13; // rax
  _QWORD *j; // rdx
  __int64 v15; // rax
  _QWORD *v16; // rbx
  _QWORD *v17; // r12
  unsigned __int64 v18; // rdi
  __int64 v19; // rax
  _QWORD *v20; // rbx
  _QWORD *v21; // r12
  unsigned __int64 v22; // rdi
  unsigned __int64 *v23; // rbx
  unsigned __int64 *v24; // r12
  unsigned __int64 v25; // rdi
  unsigned __int64 *v26; // rbx
  unsigned __int64 v27; // r12
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rbx
  __int64 v32; // r12
  __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rbx
  __int64 v37; // r12
  __int64 v38; // rdi
  __int64 v39; // rdi
  __int64 v40; // rax
  _QWORD *v41; // rbx
  _QWORD *v42; // r12
  unsigned __int64 v43; // rdi
  __int64 v44; // rax
  _QWORD *v45; // rbx
  _QWORD *v46; // r12
  unsigned __int64 v47; // rdi
  __int64 v48; // rax
  _QWORD *v49; // rbx
  _QWORD *v50; // r12
  unsigned __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rbx
  unsigned __int64 v54; // r12
  __int64 v55; // r15
  __int64 v56; // rdx
  _QWORD *v57; // rax
  _QWORD *v58; // r14
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // rbx
  unsigned __int64 v63; // r12
  __int64 v64; // r15
  __int64 v65; // rdx
  _QWORD *v66; // rax
  _QWORD *v67; // r14
  unsigned __int64 v68; // rdi
  unsigned __int64 v69; // rdi
  unsigned __int64 v70; // rdi
  unsigned __int64 v71; // rdi
  unsigned __int64 v72; // rdi
  __int64 v73; // rax
  _QWORD *v74; // rbx
  _QWORD *v75; // r12
  __int64 v76; // rdi
  __int64 result; // rax
  __int64 v78; // rdi
  unsigned int v79; // ecx
  _QWORD *v80; // rdi
  unsigned int v81; // eax
  int v82; // r12d
  unsigned int v83; // eax
  _QWORD *v84; // rax
  __int64 v85; // rdx
  _QWORD *k; // rdx
  __int64 v87; // rdi
  _QWORD *v88; // rbx
  _QWORD *m; // r12
  __int64 v90; // rax
  _QWORD *v91; // rax
  _QWORD *v92; // r12
  _QWORD *v93; // rbx
  __int64 *v94; // rdi
  _QWORD *v95; // rax
  _QWORD *v96; // r12
  _QWORD *v97; // rbx
  __int64 *v98; // rdi
  unsigned int v99; // edx
  int v100; // r12d
  unsigned int v101; // eax
  _QWORD *v102; // rdi
  unsigned int v103; // eax
  _QWORD *v104; // rax
  __int64 v105; // rdx
  _QWORD *i; // rdx
  _QWORD *v107; // rax
  _QWORD *v108; // rax
  __int64 v109; // [rsp+8h] [rbp-A8h]
  __int64 v110; // [rsp+8h] [rbp-A8h]
  __int64 v111; // [rsp+10h] [rbp-A0h]
  __int64 v112; // [rsp+10h] [rbp-A0h]
  _QWORD *v113; // [rsp+18h] [rbp-98h]
  _QWORD *v114; // [rsp+18h] [rbp-98h]
  void *v115; // [rsp+20h] [rbp-90h] BYREF
  _BYTE v116[40]; // [rsp+28h] [rbp-88h] BYREF
  void *v117; // [rsp+50h] [rbp-60h] BYREF
  _BYTE v118[88]; // [rsp+58h] [rbp-58h] BYREF

  v2 = *(_QWORD **)(a1 + 1032);
  while ( v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)v2[9];
    v4 = v3[3];
    *v3 = &unk_49EE2B0;
    if ( v4 != -8 && v4 != 0 && v4 != -16 )
      sub_1649B30(v3 + 1);
  }
  v5 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  *(_QWORD *)(a1 + 1032) = 0;
  if ( !v5 && !*(_DWORD *)(a1 + 132) )
    goto LABEL_20;
  v6 = *(_QWORD **)(a1 + 120);
  v7 = 4 * v5;
  v8 = *(unsigned int *)(a1 + 136);
  v9 = &v6[8 * v8];
  if ( (unsigned int)(4 * v5) < 0x40 )
    v7 = 64;
  if ( (unsigned int)v8 <= v7 )
  {
    for ( ; v6 != v9; v6 += 8 )
    {
      if ( *v6 != -8 )
      {
        if ( *v6 != -16 )
        {
          v10 = v6[5];
          if ( v10 )
            j_j___libc_free_0(v10, v6[7] - v10);
          j___libc_free_0(v6[2]);
        }
        *v6 = -8;
      }
    }
    goto LABEL_19;
  }
  do
  {
    while ( *v6 == -16 )
    {
LABEL_173:
      v6 += 8;
      if ( v6 == v9 )
        goto LABEL_207;
    }
    if ( *v6 != -8 )
    {
      v87 = v6[5];
      if ( v87 )
        j_j___libc_free_0(v87, v6[7] - v87);
      j___libc_free_0(v6[2]);
      goto LABEL_173;
    }
    v6 += 8;
  }
  while ( v6 != v9 );
LABEL_207:
  v99 = *(_DWORD *)(a1 + 136);
  if ( !v5 )
  {
    if ( v99 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 120));
      *(_DWORD *)(a1 + 136) = 0;
      goto LABEL_220;
    }
LABEL_19:
    *(_QWORD *)(a1 + 128) = 0;
    goto LABEL_20;
  }
  v100 = 64;
  if ( v5 != 1 )
  {
    _BitScanReverse(&v101, v5 - 1);
    v100 = 1 << (33 - (v101 ^ 0x1F));
    if ( v100 < 64 )
      v100 = 64;
  }
  v102 = *(_QWORD **)(a1 + 120);
  if ( v99 != v100 )
  {
    j___libc_free_0(v102);
    v103 = sub_14548E0(v100);
    *(_DWORD *)(a1 + 136) = v103;
    if ( v103 )
    {
      v104 = (_QWORD *)sub_22077B0((unsigned __int64)v103 << 6);
      v105 = *(unsigned int *)(a1 + 136);
      *(_QWORD *)(a1 + 128) = 0;
      *(_QWORD *)(a1 + 120) = v104;
      for ( i = &v104[8 * v105]; i != v104; v104 += 8 )
      {
        if ( v104 )
          *v104 = -8;
      }
      goto LABEL_20;
    }
LABEL_220:
    *(_QWORD *)(a1 + 120) = 0;
    *(_QWORD *)(a1 + 128) = 0;
    goto LABEL_20;
  }
  *(_QWORD *)(a1 + 128) = 0;
  v108 = &v102[8 * (unsigned __int64)v99];
  do
  {
    if ( v102 )
      *v102 = -8;
    v102 += 8;
  }
  while ( v108 != v102 );
LABEL_20:
  sub_145FF90(a1 + 144);
  v11 = *(_DWORD *)(a1 + 96);
  ++*(_QWORD *)(a1 + 80);
  if ( v11 )
  {
    v79 = 4 * v11;
    v12 = *(unsigned int *)(a1 + 104);
    if ( (unsigned int)(4 * v11) < 0x40 )
      v79 = 64;
    if ( (unsigned int)v12 <= v79 )
    {
LABEL_23:
      v13 = *(_QWORD **)(a1 + 88);
      for ( j = &v13[2 * v12]; j != v13; v13 += 2 )
        *v13 = -8;
      goto LABEL_25;
    }
    v80 = *(_QWORD **)(a1 + 88);
    v81 = v11 - 1;
    if ( v81 )
    {
      _BitScanReverse(&v81, v81);
      v82 = 1 << (33 - (v81 ^ 0x1F));
      if ( v82 < 64 )
        v82 = 64;
      if ( (_DWORD)v12 == v82 )
      {
        *(_QWORD *)(a1 + 96) = 0;
        v107 = &v80[2 * (unsigned int)v12];
        do
        {
          if ( v80 )
            *v80 = -8;
          v80 += 2;
        }
        while ( v107 != v80 );
        goto LABEL_26;
      }
    }
    else
    {
      v82 = 64;
    }
    j___libc_free_0(v80);
    v83 = sub_14548E0(v82);
    *(_DWORD *)(a1 + 104) = v83;
    if ( !v83 )
      goto LABEL_222;
    v84 = (_QWORD *)sub_22077B0(16LL * v83);
    v85 = *(unsigned int *)(a1 + 104);
    *(_QWORD *)(a1 + 96) = 0;
    *(_QWORD *)(a1 + 88) = v84;
    for ( k = &v84[2 * v85]; k != v84; v84 += 2 )
    {
      if ( v84 )
        *v84 = -8;
    }
  }
  else if ( *(_DWORD *)(a1 + 100) )
  {
    v12 = *(unsigned int *)(a1 + 104);
    if ( (unsigned int)v12 <= 0x40 )
      goto LABEL_23;
    j___libc_free_0(*(_QWORD *)(a1 + 88));
    *(_DWORD *)(a1 + 104) = 0;
LABEL_222:
    *(_QWORD *)(a1 + 88) = 0;
LABEL_25:
    *(_QWORD *)(a1 + 96) = 0;
  }
LABEL_26:
  if ( *(_DWORD *)(a1 + 544) )
  {
    v95 = *(_QWORD **)(a1 + 536);
    v96 = &v95[8 * (unsigned __int64)*(unsigned int *)(a1 + 552)];
    if ( v95 != v96 )
    {
      while ( 1 )
      {
        v97 = v95;
        if ( *v95 != -16 && *v95 != -8 )
          break;
        v95 += 8;
        if ( v96 == v95 )
          goto LABEL_27;
      }
      while ( v96 != v97 )
      {
        v98 = v97 + 1;
        v97 += 8;
        sub_14575E0(v98);
        if ( v97 == v96 )
          break;
        while ( *v97 == -8 || *v97 == -16 )
        {
          v97 += 8;
          if ( v96 == v97 )
            goto LABEL_27;
        }
      }
    }
  }
LABEL_27:
  if ( *(_DWORD *)(a1 + 576) )
  {
    v91 = *(_QWORD **)(a1 + 568);
    v92 = &v91[8 * (unsigned __int64)*(unsigned int *)(a1 + 584)];
    if ( v91 != v92 )
    {
      while ( 1 )
      {
        v93 = v91;
        if ( *v91 != -8 && *v91 != -16 )
          break;
        v91 += 8;
        if ( v92 == v91 )
          goto LABEL_28;
      }
      while ( v92 != v93 )
      {
        v94 = v93 + 1;
        v93 += 8;
        sub_14575E0(v94);
        if ( v93 == v92 )
          break;
        while ( *v93 == -8 || *v93 == -16 )
        {
          v93 += 8;
          if ( v92 == v93 )
            goto LABEL_28;
        }
      }
    }
  }
LABEL_28:
  v15 = *(unsigned int *)(a1 + 1024);
  if ( (_DWORD)v15 )
  {
    v16 = *(_QWORD **)(a1 + 1008);
    v17 = &v16[8 * v15];
    do
    {
      while ( *v16 == -8 )
      {
        if ( v16[1] != -8 )
          goto LABEL_31;
        v16 += 8;
        if ( v17 == v16 )
          goto LABEL_37;
      }
      if ( *v16 != -16 || v16[1] != -16 )
      {
LABEL_31:
        v18 = v16[3];
        if ( (_QWORD *)v18 != v16 + 5 )
          _libc_free(v18);
      }
      v16 += 8;
    }
    while ( v17 != v16 );
  }
LABEL_37:
  j___libc_free_0(*(_QWORD *)(a1 + 1008));
  v19 = *(unsigned int *)(a1 + 992);
  if ( (_DWORD)v19 )
  {
    v20 = *(_QWORD **)(a1 + 976);
    v21 = &v20[7 * v19];
    do
    {
      if ( *v20 != -8 && *v20 != -16 )
      {
        v22 = v20[1];
        if ( (_QWORD *)v22 != v20 + 3 )
          _libc_free(v22);
      }
      v20 += 7;
    }
    while ( v21 != v20 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 976));
  v23 = *(unsigned __int64 **)(a1 + 880);
  v24 = &v23[*(unsigned int *)(a1 + 888)];
  while ( v24 != v23 )
  {
    v25 = *v23++;
    _libc_free(v25);
  }
  v26 = *(unsigned __int64 **)(a1 + 928);
  v27 = (unsigned __int64)&v26[2 * *(unsigned int *)(a1 + 936)];
  if ( v26 != (unsigned __int64 *)v27 )
  {
    do
    {
      v28 = *v26;
      v26 += 2;
      _libc_free(v28);
    }
    while ( v26 != (unsigned __int64 *)v27 );
    v27 = *(_QWORD *)(a1 + 928);
  }
  if ( v27 != a1 + 944 )
    _libc_free(v27);
  v29 = *(_QWORD *)(a1 + 880);
  if ( v29 != a1 + 896 )
    _libc_free(v29);
  *(_QWORD *)(a1 + 840) = &unk_49EC540;
  sub_16BD9D0(a1 + 840);
  *(_QWORD *)(a1 + 816) = &unk_49EC4E0;
  sub_16BD9D0(a1 + 816);
  v30 = *(unsigned int *)(a1 + 808);
  if ( (_DWORD)v30 )
  {
    v31 = *(_QWORD *)(a1 + 792);
    v32 = v31 + 40 * v30;
    do
    {
      if ( *(_QWORD *)v31 != -16 && *(_QWORD *)v31 != -8 )
      {
        if ( *(_DWORD *)(v31 + 32) > 0x40u )
        {
          v33 = *(_QWORD *)(v31 + 24);
          if ( v33 )
            j_j___libc_free_0_0(v33);
        }
        if ( *(_DWORD *)(v31 + 16) > 0x40u )
        {
          v34 = *(_QWORD *)(v31 + 8);
          if ( v34 )
            j_j___libc_free_0_0(v34);
        }
      }
      v31 += 40;
    }
    while ( v32 != v31 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 792));
  v35 = *(unsigned int *)(a1 + 776);
  if ( (_DWORD)v35 )
  {
    v36 = *(_QWORD *)(a1 + 760);
    v37 = v36 + 40 * v35;
    do
    {
      if ( *(_QWORD *)v36 != -8 && *(_QWORD *)v36 != -16 )
      {
        if ( *(_DWORD *)(v36 + 32) > 0x40u )
        {
          v38 = *(_QWORD *)(v36 + 24);
          if ( v38 )
            j_j___libc_free_0_0(v38);
        }
        if ( *(_DWORD *)(v36 + 16) > 0x40u )
        {
          v39 = *(_QWORD *)(v36 + 8);
          if ( v39 )
            j_j___libc_free_0_0(v39);
        }
      }
      v36 += 40;
    }
    while ( v37 != v36 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 760));
  v40 = *(unsigned int *)(a1 + 744);
  if ( (_DWORD)v40 )
  {
    v41 = *(_QWORD **)(a1 + 728);
    v42 = &v41[5 * v40];
    do
    {
      if ( *v41 != -16 && *v41 != -8 )
      {
        v43 = v41[1];
        if ( (_QWORD *)v43 != v41 + 3 )
          _libc_free(v43);
      }
      v41 += 5;
    }
    while ( v42 != v41 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 728));
  j___libc_free_0(*(_QWORD *)(a1 + 696));
  v44 = *(unsigned int *)(a1 + 680);
  if ( (_DWORD)v44 )
  {
    v45 = *(_QWORD **)(a1 + 664);
    v46 = &v45[5 * v44];
    do
    {
      if ( *v45 != -8 && *v45 != -16 )
      {
        v47 = v45[1];
        if ( (_QWORD *)v47 != v45 + 3 )
          _libc_free(v47);
      }
      v45 += 5;
    }
    while ( v46 != v45 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 664));
  v48 = *(unsigned int *)(a1 + 648);
  if ( (_DWORD)v48 )
  {
    v49 = *(_QWORD **)(a1 + 632);
    v50 = &v49[7 * v48];
    do
    {
      if ( *v49 != -16 && *v49 != -8 )
      {
        v51 = v49[1];
        if ( (_QWORD *)v51 != v49 + 3 )
          _libc_free(v51);
      }
      v49 += 7;
    }
    while ( v50 != v49 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 632));
  j___libc_free_0(*(_QWORD *)(a1 + 600));
  v52 = *(unsigned int *)(a1 + 584);
  if ( (_DWORD)v52 )
  {
    v53 = *(_QWORD *)(a1 + 568);
    v109 = v53 + (v52 << 6);
    do
    {
      if ( *(_QWORD *)v53 != -16 && *(_QWORD *)v53 != -8 )
      {
        v111 = *(_QWORD *)(v53 + 8);
        v54 = v111 + 24LL * *(unsigned int *)(v53 + 16);
        if ( v111 != v54 )
        {
          do
          {
            v55 = *(_QWORD *)(v54 - 8);
            v54 -= 24LL;
            if ( v55 )
            {
              v56 = *(unsigned int *)(v55 + 208);
              *(_QWORD *)v55 = &unk_49EC708;
              if ( (_DWORD)v56 )
              {
                v57 = *(_QWORD **)(v55 + 192);
                v58 = &v57[7 * v56];
                do
                {
                  if ( *v57 != -8 && *v57 != -16 )
                  {
                    v59 = v57[1];
                    if ( (_QWORD *)v59 != v57 + 3 )
                    {
                      v113 = v57;
                      _libc_free(v59);
                      v57 = v113;
                    }
                  }
                  v57 += 7;
                }
                while ( v58 != v57 );
              }
              j___libc_free_0(*(_QWORD *)(v55 + 192));
              v60 = *(_QWORD *)(v55 + 40);
              if ( v60 != v55 + 56 )
                _libc_free(v60);
              j_j___libc_free_0(v55, 216);
            }
          }
          while ( v111 != v54 );
          v54 = *(_QWORD *)(v53 + 8);
        }
        if ( v54 != v53 + 24 )
          _libc_free(v54);
      }
      v53 += 64;
    }
    while ( v109 != v53 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 568));
  v61 = *(unsigned int *)(a1 + 552);
  if ( (_DWORD)v61 )
  {
    v62 = *(_QWORD *)(a1 + 536);
    v110 = v62 + (v61 << 6);
    do
    {
      if ( *(_QWORD *)v62 != -16 && *(_QWORD *)v62 != -8 )
      {
        v112 = *(_QWORD *)(v62 + 8);
        v63 = v112 + 24LL * *(unsigned int *)(v62 + 16);
        if ( v112 != v63 )
        {
          do
          {
            v64 = *(_QWORD *)(v63 - 8);
            v63 -= 24LL;
            if ( v64 )
            {
              v65 = *(unsigned int *)(v64 + 208);
              *(_QWORD *)v64 = &unk_49EC708;
              if ( (_DWORD)v65 )
              {
                v66 = *(_QWORD **)(v64 + 192);
                v67 = &v66[7 * v65];
                do
                {
                  if ( *v66 != -16 && *v66 != -8 )
                  {
                    v68 = v66[1];
                    if ( (_QWORD *)v68 != v66 + 3 )
                    {
                      v114 = v66;
                      _libc_free(v68);
                      v66 = v114;
                    }
                  }
                  v66 += 7;
                }
                while ( v67 != v66 );
              }
              j___libc_free_0(*(_QWORD *)(v64 + 192));
              v69 = *(_QWORD *)(v64 + 40);
              if ( v69 != v64 + 56 )
                _libc_free(v69);
              j_j___libc_free_0(v64, 216);
            }
          }
          while ( v112 != v63 );
          v63 = *(_QWORD *)(v62 + 8);
        }
        if ( v63 != v62 + 24 )
          _libc_free(v63);
      }
      v62 += 64;
    }
    while ( v110 != v62 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 536));
  j___libc_free_0(*(_QWORD *)(a1 + 504));
  v70 = *(_QWORD *)(a1 + 400);
  if ( v70 != *(_QWORD *)(a1 + 392) )
    _libc_free(v70);
  v71 = *(_QWORD *)(a1 + 296);
  if ( v71 != *(_QWORD *)(a1 + 288) )
    _libc_free(v71);
  v72 = *(_QWORD *)(a1 + 192);
  if ( v72 != *(_QWORD *)(a1 + 184) )
    _libc_free(v72);
  if ( *(_DWORD *)(a1 + 168) )
  {
    sub_1457D90(&v115, -8, 0);
    sub_1457D90(&v117, -16, 0);
    v88 = *(_QWORD **)(a1 + 152);
    for ( m = &v88[6 * *(unsigned int *)(a1 + 168)]; m != v88; v88 += 6 )
    {
      v90 = v88[3];
      *v88 = &unk_49EE2B0;
      if ( v90 != -8 && v90 != 0 && v90 != -16 )
        sub_1649B30(v88 + 1);
    }
    v117 = &unk_49EE2B0;
    sub_1455FA0((__int64)v118);
    v115 = &unk_49EE2B0;
    sub_1455FA0((__int64)v116);
  }
  j___libc_free_0(*(_QWORD *)(a1 + 152));
  v73 = *(unsigned int *)(a1 + 136);
  if ( (_DWORD)v73 )
  {
    v74 = *(_QWORD **)(a1 + 120);
    v75 = &v74[8 * v73];
    do
    {
      if ( *v74 != -8 && *v74 != -16 )
      {
        v76 = v74[5];
        if ( v76 )
          j_j___libc_free_0(v76, v74[7] - v76);
        j___libc_free_0(v74[2]);
      }
      v74 += 8;
    }
    while ( v75 != v74 );
  }
  j___libc_free_0(*(_QWORD *)(a1 + 120));
  result = j___libc_free_0(*(_QWORD *)(a1 + 88));
  v78 = *(_QWORD *)(a1 + 72);
  if ( v78 )
    return j_j___libc_free_0(v78, 32);
  return result;
}
