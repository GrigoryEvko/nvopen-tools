// Function: sub_DA11D0
// Address: 0xda11d0
//
__int64 __fastcall sub_DA11D0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r12
  _QWORD *v4; // rdi
  __int64 v5; // rax
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  _QWORD *i; // rdx
  __int64 v10; // rax
  _QWORD *v11; // r12
  _QWORD *v12; // r13
  _QWORD *v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rax
  _QWORD *v16; // r12
  _QWORD *v17; // r13
  _QWORD *v18; // rdi
  __int64 v19; // rsi
  __int64 *v20; // r14
  __int64 *v21; // r12
  __int64 k; // rax
  __int64 v23; // rdi
  unsigned int v24; // ecx
  __int64 *v25; // r12
  __int64 *v26; // r13
  __int64 v27; // rdi
  __int64 v28; // rdi
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // r13
  __int64 v32; // rdi
  __int64 v33; // rdi
  __int64 v34; // rax
  __int64 v35; // r12
  __int64 v36; // r13
  __int64 v37; // rdi
  __int64 v38; // rdi
  __int64 v39; // rsi
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // r13
  __int64 v43; // rdi
  __int64 v44; // rsi
  __int64 v45; // rax
  _QWORD *v46; // r12
  _QWORD *v47; // r13
  _QWORD *v48; // rdi
  __int64 v49; // rsi
  __int64 v50; // rax
  _QWORD *v51; // r12
  _QWORD *v52; // r13
  _QWORD *v53; // rdi
  __int64 v54; // rsi
  __int64 v55; // rax
  _QWORD *v56; // r12
  _QWORD *v57; // r13
  _QWORD *v58; // rdi
  __int64 v59; // rsi
  __int64 v60; // rax
  _QWORD *v61; // r12
  _QWORD *v62; // r13
  _QWORD *v63; // rdi
  __int64 v64; // rsi
  __int64 v65; // rax
  __int64 v66; // r12
  __int64 v67; // r13
  __int64 v68; // rdi
  __int64 v69; // rsi
  __int64 v70; // rax
  __int64 v71; // r12
  __int64 v72; // r13
  __int64 v73; // r15
  __int64 v74; // rdi
  __int64 v75; // rax
  __int64 v76; // rsi
  __int64 v77; // rax
  __int64 v78; // r12
  __int64 v79; // r13
  __int64 v80; // r15
  __int64 v81; // rdi
  __int64 v82; // rax
  __int64 v83; // rax
  __int64 v84; // r12
  __int64 v85; // r13
  __int64 v86; // rdi
  __int64 v87; // rsi
  __int64 v88; // rax
  _QWORD *v89; // r12
  _QWORD *v90; // r13
  _QWORD *v91; // rdi
  __int64 v92; // rax
  __int64 v93; // rsi
  __int64 v94; // rax
  __int64 v95; // r12
  __int64 v96; // r13
  __int64 v97; // rdi
  __int64 result; // rax
  __int64 v99; // rdi
  unsigned int v100; // ecx
  unsigned int v101; // eax
  _QWORD *v102; // rdi
  int v103; // r12d
  _QWORD *v104; // rax
  _QWORD *v105; // r12
  _QWORD *m; // r13
  __int64 v107; // rax
  unsigned __int64 v108; // rax
  unsigned __int64 v109; // rdi
  _QWORD *v110; // rax
  __int64 v111; // rdx
  _QWORD *j; // rdx
  __int64 v113; // [rsp+8h] [rbp-98h]
  __int64 v114; // [rsp+8h] [rbp-98h]
  void *v115; // [rsp+10h] [rbp-90h] BYREF
  __int64 v116; // [rsp+18h] [rbp-88h] BYREF
  __int64 v117; // [rsp+28h] [rbp-78h]
  void *v118; // [rsp+40h] [rbp-60h] BYREF
  __int64 v119; // [rsp+48h] [rbp-58h] BYREF
  __int64 v120; // [rsp+58h] [rbp-48h]

  v3 = *(_QWORD **)(a1 + 1544);
  while ( v3 )
  {
    v4 = v3;
    v3 = (_QWORD *)v3[9];
    v5 = v4[3];
    *v4 = &unk_49DB368;
    if ( v5 != -4096 && v5 != 0 && v5 != -8192 )
      sub_BD60C0(v4 + 1);
  }
  *(_QWORD *)(a1 + 1544) = 0;
  sub_D9DE90(a1 + 96, a2);
  sub_D9FB70(a1 + 128);
  v6 = *(_DWORD *)(a1 + 80);
  ++*(_QWORD *)(a1 + 64);
  if ( !v6 )
  {
    if ( !*(_DWORD *)(a1 + 84) )
      goto LABEL_12;
    v7 = *(unsigned int *)(a1 + 88);
    if ( (unsigned int)v7 > 0x40 )
    {
      a2 = 16LL * (unsigned int)v7;
      sub_C7D6A0(*(_QWORD *)(a1 + 72), a2, 8);
      *(_QWORD *)(a1 + 72) = 0;
      *(_QWORD *)(a1 + 80) = 0;
      *(_DWORD *)(a1 + 88) = 0;
      goto LABEL_12;
    }
    goto LABEL_9;
  }
  v100 = 4 * v6;
  a2 = 64;
  v7 = *(unsigned int *)(a1 + 88);
  if ( (unsigned int)(4 * v6) < 0x40 )
    v100 = 64;
  if ( (unsigned int)v7 <= v100 )
  {
LABEL_9:
    v8 = *(_QWORD **)(a1 + 72);
    for ( i = &v8[2 * v7]; i != v8; v8 += 2 )
      *v8 = -4096;
    *(_QWORD *)(a1 + 80) = 0;
    goto LABEL_12;
  }
  v101 = v6 - 1;
  if ( v101 )
  {
    _BitScanReverse(&v101, v101);
    v102 = *(_QWORD **)(a1 + 72);
    v103 = 1 << (33 - (v101 ^ 0x1F));
    if ( v103 < 64 )
      v103 = 64;
    if ( (_DWORD)v7 == v103 )
    {
      *(_QWORD *)(a1 + 80) = 0;
      v104 = &v102[2 * (unsigned int)v7];
      do
      {
        if ( v102 )
          *v102 = -4096;
        v102 += 2;
      }
      while ( v104 != v102 );
      goto LABEL_12;
    }
  }
  else
  {
    v102 = *(_QWORD **)(a1 + 72);
    v103 = 64;
  }
  sub_C7D6A0((__int64)v102, 16LL * (unsigned int)v7, 8);
  a2 = 8;
  v108 = ((((((((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
            | (4 * v103 / 3u + 1)
            | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 4)
          | (((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
          | (4 * v103 / 3u + 1)
          | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
          | (4 * v103 / 3u + 1)
          | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 4)
        | (((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
        | (4 * v103 / 3u + 1)
        | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 16;
  v109 = (v108
        | (((((((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
            | (4 * v103 / 3u + 1)
            | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 4)
          | (((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
          | (4 * v103 / 3u + 1)
          | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 8)
        | (((((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
          | (4 * v103 / 3u + 1)
          | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 4)
        | (((4 * v103 / 3u + 1) | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1)) >> 2)
        | (4 * v103 / 3u + 1)
        | ((unsigned __int64)(4 * v103 / 3u + 1) >> 1))
       + 1;
  *(_DWORD *)(a1 + 88) = v109;
  v110 = (_QWORD *)sub_C7D670(16 * v109, 8);
  v111 = *(unsigned int *)(a1 + 88);
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 72) = v110;
  for ( j = &v110[2 * v111]; j != v110; v110 += 2 )
  {
    if ( v110 )
      *v110 = -4096;
  }
LABEL_12:
  sub_D9CC70(a1 + 648);
  sub_D9CC70(a1 + 680);
  if ( !*(_BYTE *)(a1 + 1412) )
    _libc_free(*(_QWORD *)(a1 + 1392), a2);
  if ( !*(_BYTE *)(a1 + 1252) )
    _libc_free(*(_QWORD *)(a1 + 1232), a2);
  v10 = *(unsigned int *)(a1 + 1216);
  if ( (_DWORD)v10 )
  {
    v11 = *(_QWORD **)(a1 + 1200);
    v12 = &v11[8 * v10];
    while ( 1 )
    {
      while ( *v11 == -4096 )
      {
        if ( v11[1] != -4096 )
          goto LABEL_19;
        v11 += 8;
        if ( v12 == v11 )
        {
LABEL_25:
          LODWORD(v10) = *(_DWORD *)(a1 + 1216);
          goto LABEL_26;
        }
      }
      if ( *v11 != -8192 || v11[1] != -8192 )
      {
LABEL_19:
        v13 = (_QWORD *)v11[3];
        if ( v13 != v11 + 5 )
          _libc_free(v13, a2);
      }
      v11 += 8;
      if ( v12 == v11 )
        goto LABEL_25;
    }
  }
LABEL_26:
  v14 = (unsigned __int64)(unsigned int)v10 << 6;
  sub_C7D6A0(*(_QWORD *)(a1 + 1200), v14, 8);
  v15 = *(unsigned int *)(a1 + 1184);
  if ( (_DWORD)v15 )
  {
    v16 = *(_QWORD **)(a1 + 1168);
    v17 = &v16[7 * v15];
    do
    {
      if ( *v16 != -8192 && *v16 != -4096 )
      {
        v18 = (_QWORD *)v16[1];
        if ( v18 != v16 + 3 )
          _libc_free(v18, v14);
      }
      v16 += 7;
    }
    while ( v17 != v16 );
    v15 = *(unsigned int *)(a1 + 1184);
  }
  v19 = 56 * v15;
  sub_C7D6A0(*(_QWORD *)(a1 + 1168), 56 * v15, 8);
  v20 = *(__int64 **)(a1 + 1080);
  v21 = &v20[*(unsigned int *)(a1 + 1088)];
  if ( v20 != v21 )
  {
    for ( k = *(_QWORD *)(a1 + 1080); ; k = *(_QWORD *)(a1 + 1080) )
    {
      v23 = *v20;
      v24 = (unsigned int)(((__int64)v20 - k) >> 3) >> 7;
      v19 = 4096LL << v24;
      if ( v24 >= 0x1E )
        v19 = 0x40000000000LL;
      ++v20;
      sub_C7D6A0(v23, v19, 16);
      if ( v21 == v20 )
        break;
    }
  }
  v25 = *(__int64 **)(a1 + 1128);
  v26 = &v25[2 * *(unsigned int *)(a1 + 1136)];
  if ( v25 != v26 )
  {
    do
    {
      v19 = v25[1];
      v27 = *v25;
      v25 += 2;
      sub_C7D6A0(v27, v19, 16);
    }
    while ( v26 != v25 );
    v26 = *(__int64 **)(a1 + 1128);
  }
  if ( v26 != (__int64 *)(a1 + 1144) )
    _libc_free(v26, v19);
  v28 = *(_QWORD *)(a1 + 1080);
  if ( v28 != a1 + 1096 )
    _libc_free(v28, v19);
  sub_C65770((_QWORD *)(a1 + 1048), v19);
  sub_C65770((_QWORD *)(a1 + 1032), v19);
  v29 = *(unsigned int *)(a1 + 1024);
  if ( (_DWORD)v29 )
  {
    v30 = *(_QWORD *)(a1 + 1008);
    v31 = v30 + 40 * v29;
    do
    {
      if ( *(_QWORD *)v30 != -8192 && *(_QWORD *)v30 != -4096 )
      {
        if ( *(_DWORD *)(v30 + 32) > 0x40u )
        {
          v32 = *(_QWORD *)(v30 + 24);
          if ( v32 )
            j_j___libc_free_0_0(v32);
        }
        if ( *(_DWORD *)(v30 + 16) > 0x40u )
        {
          v33 = *(_QWORD *)(v30 + 8);
          if ( v33 )
            j_j___libc_free_0_0(v33);
        }
      }
      v30 += 40;
    }
    while ( v31 != v30 );
    v29 = *(unsigned int *)(a1 + 1024);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1008), 40 * v29, 8);
  v34 = *(unsigned int *)(a1 + 992);
  if ( (_DWORD)v34 )
  {
    v35 = *(_QWORD *)(a1 + 976);
    v36 = v35 + 40 * v34;
    do
    {
      if ( *(_QWORD *)v35 != -4096 && *(_QWORD *)v35 != -8192 )
      {
        if ( *(_DWORD *)(v35 + 32) > 0x40u )
        {
          v37 = *(_QWORD *)(v35 + 24);
          if ( v37 )
            j_j___libc_free_0_0(v37);
        }
        if ( *(_DWORD *)(v35 + 16) > 0x40u )
        {
          v38 = *(_QWORD *)(v35 + 8);
          if ( v38 )
            j_j___libc_free_0_0(v38);
        }
      }
      v35 += 40;
    }
    while ( v36 != v35 );
    v34 = *(unsigned int *)(a1 + 992);
  }
  v39 = 40 * v34;
  sub_C7D6A0(*(_QWORD *)(a1 + 976), 40 * v34, 8);
  v40 = *(unsigned int *)(a1 + 960);
  if ( (_DWORD)v40 )
  {
    v41 = *(_QWORD *)(a1 + 944);
    v42 = v41 + 104 * v40;
    do
    {
      while ( *(_QWORD *)v41 == -4096 || *(_QWORD *)v41 == -8192 || *(_BYTE *)(v41 + 36) )
      {
        v41 += 104;
        if ( v42 == v41 )
          goto LABEL_78;
      }
      v43 = *(_QWORD *)(v41 + 16);
      v41 += 104;
      _libc_free(v43, v39);
    }
    while ( v42 != v41 );
LABEL_78:
    v40 = *(unsigned int *)(a1 + 960);
  }
  v44 = 104 * v40;
  sub_C7D6A0(*(_QWORD *)(a1 + 944), 104 * v40, 8);
  v45 = *(unsigned int *)(a1 + 928);
  if ( (_DWORD)v45 )
  {
    v46 = *(_QWORD **)(a1 + 912);
    v47 = &v46[5 * v45];
    do
    {
      if ( *v46 != -4096 && *v46 != -8192 )
      {
        v48 = (_QWORD *)v46[1];
        if ( v48 != v46 + 3 )
          _libc_free(v48, v44);
      }
      v46 += 5;
    }
    while ( v47 != v46 );
    v45 = *(unsigned int *)(a1 + 928);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 912), 40 * v45, 8);
  v49 = 16LL * *(unsigned int *)(a1 + 896);
  sub_C7D6A0(*(_QWORD *)(a1 + 880), v49, 8);
  v50 = *(unsigned int *)(a1 + 864);
  if ( (_DWORD)v50 )
  {
    v51 = *(_QWORD **)(a1 + 848);
    v52 = &v51[5 * v50];
    do
    {
      if ( *v51 != -4096 && *v51 != -8192 )
      {
        v53 = (_QWORD *)v51[1];
        if ( v53 != v51 + 3 )
          _libc_free(v53, v49);
      }
      v51 += 5;
    }
    while ( v52 != v51 );
    v50 = *(unsigned int *)(a1 + 864);
  }
  v54 = 40 * v50;
  sub_C7D6A0(*(_QWORD *)(a1 + 848), 40 * v50, 8);
  v55 = *(unsigned int *)(a1 + 832);
  if ( (_DWORD)v55 )
  {
    v56 = *(_QWORD **)(a1 + 816);
    v57 = &v56[7 * v55];
    do
    {
      if ( *v56 != -8192 && *v56 != -4096 )
      {
        v58 = (_QWORD *)v56[1];
        if ( v58 != v56 + 3 )
          _libc_free(v58, v54);
      }
      v56 += 7;
    }
    while ( v57 != v56 );
    v55 = *(unsigned int *)(a1 + 832);
  }
  v59 = 56 * v55;
  sub_C7D6A0(*(_QWORD *)(a1 + 816), 56 * v55, 8);
  v60 = *(unsigned int *)(a1 + 800);
  if ( (_DWORD)v60 )
  {
    v61 = *(_QWORD **)(a1 + 784);
    v62 = &v61[7 * v60];
    do
    {
      if ( *v61 != -8192 && *v61 != -4096 )
      {
        v63 = (_QWORD *)v61[1];
        if ( v63 != v61 + 3 )
          _libc_free(v63, v59);
      }
      v61 += 7;
    }
    while ( v62 != v61 );
    v60 = *(unsigned int *)(a1 + 800);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 784), 56 * v60, 8);
  v64 = 16LL * *(unsigned int *)(a1 + 768);
  sub_C7D6A0(*(_QWORD *)(a1 + 752), v64, 8);
  v65 = *(unsigned int *)(a1 + 736);
  if ( (_DWORD)v65 )
  {
    v66 = *(_QWORD *)(a1 + 720);
    v67 = v66 + 72 * v65;
    do
    {
      while ( *(_QWORD *)v66 == -8192 || *(_QWORD *)v66 == -4096 || *(_BYTE *)(v66 + 36) )
      {
        v66 += 72;
        if ( v67 == v66 )
          goto LABEL_118;
      }
      v68 = *(_QWORD *)(v66 + 16);
      v66 += 72;
      _libc_free(v68, v64);
    }
    while ( v67 != v66 );
LABEL_118:
    v65 = *(unsigned int *)(a1 + 736);
  }
  v69 = 72 * v65;
  sub_C7D6A0(*(_QWORD *)(a1 + 720), 72 * v65, 8);
  v70 = *(unsigned int *)(a1 + 704);
  if ( (_DWORD)v70 )
  {
    v71 = *(_QWORD *)(a1 + 688);
    v113 = v71 + 168 * v70;
    do
    {
      if ( *(_QWORD *)v71 != -8192 && *(_QWORD *)v71 != -4096 )
      {
        v72 = *(_QWORD *)(v71 + 8);
        v73 = v72 + 112LL * *(unsigned int *)(v71 + 16);
        if ( v72 != v73 )
        {
          do
          {
            v73 -= 112;
            v74 = *(_QWORD *)(v73 + 64);
            if ( v74 != v73 + 80 )
              _libc_free(v74, v69);
            if ( *(_BYTE *)(v73 + 32) )
              *(_QWORD *)(v73 + 24) = 0;
            v75 = *(_QWORD *)(v73 + 24);
            *(_QWORD *)v73 = &unk_49DB368;
            if ( v75 != 0 && v75 != -4096 && v75 != -8192 )
              sub_BD60C0((_QWORD *)(v73 + 8));
          }
          while ( v72 != v73 );
          v73 = *(_QWORD *)(v71 + 8);
        }
        if ( v73 != v71 + 24 )
          _libc_free(v73, v69);
      }
      v71 += 168;
    }
    while ( v113 != v71 );
    v70 = *(unsigned int *)(a1 + 704);
  }
  v76 = 168 * v70;
  sub_C7D6A0(*(_QWORD *)(a1 + 688), 168 * v70, 8);
  v77 = *(unsigned int *)(a1 + 672);
  if ( (_DWORD)v77 )
  {
    v78 = *(_QWORD *)(a1 + 656);
    v114 = v78 + 168 * v77;
    do
    {
      if ( *(_QWORD *)v78 != -8192 && *(_QWORD *)v78 != -4096 )
      {
        v79 = *(_QWORD *)(v78 + 8);
        v80 = v79 + 112LL * *(unsigned int *)(v78 + 16);
        if ( v79 != v80 )
        {
          do
          {
            v80 -= 112;
            v81 = *(_QWORD *)(v80 + 64);
            if ( v81 != v80 + 80 )
              _libc_free(v81, v76);
            if ( *(_BYTE *)(v80 + 32) )
              *(_QWORD *)(v80 + 24) = 0;
            v82 = *(_QWORD *)(v80 + 24);
            *(_QWORD *)v80 = &unk_49DB368;
            if ( v82 != 0 && v82 != -4096 && v82 != -8192 )
              sub_BD60C0((_QWORD *)(v80 + 8));
          }
          while ( v79 != v80 );
          v80 = *(_QWORD *)(v78 + 8);
        }
        if ( v80 != v78 + 24 )
          _libc_free(v80, v76);
      }
      v78 += 168;
    }
    while ( v114 != v78 );
    v77 = *(unsigned int *)(a1 + 672);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 656), 168 * v77, 8);
  v83 = *(unsigned int *)(a1 + 640);
  if ( (_DWORD)v83 )
  {
    v84 = *(_QWORD *)(a1 + 624);
    v85 = v84 + 24 * v83;
    do
    {
      if ( *(_QWORD *)v84 != -4096 && *(_QWORD *)v84 != -8192 && *(_DWORD *)(v84 + 16) > 0x40u )
      {
        v86 = *(_QWORD *)(v84 + 8);
        if ( v86 )
          j_j___libc_free_0_0(v86);
      }
      v84 += 24;
    }
    while ( v85 != v84 );
    v83 = *(unsigned int *)(a1 + 640);
  }
  v87 = 24 * v83;
  sub_C7D6A0(*(_QWORD *)(a1 + 624), 24 * v83, 8);
  if ( !*(_BYTE *)(a1 + 540) )
    _libc_free(*(_QWORD *)(a1 + 520), v87);
  if ( !*(_BYTE *)(a1 + 444) )
    _libc_free(*(_QWORD *)(a1 + 424), v87);
  if ( !*(_BYTE *)(a1 + 348) )
    _libc_free(*(_QWORD *)(a1 + 328), v87);
  if ( !*(_BYTE *)(a1 + 252) )
    _libc_free(*(_QWORD *)(a1 + 232), v87);
  v88 = *(unsigned int *)(a1 + 216);
  if ( (_DWORD)v88 )
  {
    v89 = *(_QWORD **)(a1 + 200);
    v90 = &v89[9 * v88];
    do
    {
      if ( *v89 != -4096 && *v89 != -8192 )
      {
        v91 = (_QWORD *)v89[1];
        if ( v91 != v89 + 3 )
          _libc_free(v91, v87);
      }
      v89 += 9;
    }
    while ( v90 != v89 );
    v88 = *(unsigned int *)(a1 + 216);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 200), 72 * v88, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 168), 32LL * *(unsigned int *)(a1 + 184), 8);
  v92 = *(unsigned int *)(a1 + 152);
  if ( (_DWORD)v92 )
  {
    sub_D982A0(&v115, -4096, 0);
    sub_D982A0(&v118, -8192, 0);
    v105 = *(_QWORD **)(a1 + 136);
    for ( m = &v105[6 * *(unsigned int *)(a1 + 152)]; m != v105; v105 += 6 )
    {
      v107 = v105[3];
      *v105 = &unk_49DB368;
      if ( v107 != -4096 && v107 != 0 && v107 != -8192 )
        sub_BD60C0(v105 + 1);
    }
    v118 = &unk_49DB368;
    if ( v120 != -4096 && v120 != 0 && v120 != -8192 )
      sub_BD60C0(&v119);
    v115 = &unk_49DB368;
    if ( v117 != -4096 && v117 != 0 && v117 != -8192 )
      sub_BD60C0(&v116);
    v92 = *(unsigned int *)(a1 + 152);
  }
  v93 = 48 * v92;
  sub_C7D6A0(*(_QWORD *)(a1 + 136), 48 * v92, 8);
  v94 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v94 )
  {
    v95 = *(_QWORD *)(a1 + 104);
    v96 = v95 + 88 * v94;
    do
    {
      if ( *(_QWORD *)v95 != -4096 && *(_QWORD *)v95 != -8192 )
      {
        v97 = *(_QWORD *)(v95 + 40);
        if ( v97 != v95 + 56 )
          _libc_free(v97, v93);
        v93 = 8LL * *(unsigned int *)(v95 + 32);
        sub_C7D6A0(*(_QWORD *)(v95 + 16), v93, 8);
      }
      v95 += 88;
    }
    while ( v96 != v95 );
    v94 = *(unsigned int *)(a1 + 120);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 104), 88 * v94, 8);
  result = sub_C7D6A0(*(_QWORD *)(a1 + 72), 16LL * *(unsigned int *)(a1 + 88), 8);
  v99 = *(_QWORD *)(a1 + 56);
  if ( v99 )
    return j_j___libc_free_0(v99, 32);
  return result;
}
