// Function: sub_17157C0
// Address: 0x17157c0
//
__int64 __fastcall sub_17157C0(__int64 a1, __int64 a2, __int64 a3)
{
  int v4; // eax
  __int64 v5; // rdx
  _QWORD *v6; // rax
  _QWORD *i; // rdx
  _QWORD *v8; // r15
  _QWORD *v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r13
  char v14; // r14
  __int64 v15; // rdx
  __int64 (__fastcall ***v16)(); // rsi
  __int64 (__fastcall ***v17)(); // rdi
  __int64 (__fastcall ***v18)(); // r14
  _DWORD *v19; // r12
  _DWORD *v20; // rbx
  __int64 v21; // rdi
  __int64 v22; // rsi
  __int64 (__fastcall ***v24)(); // r13
  int v25; // r15d
  unsigned int v26; // ecx
  _DWORD *v27; // rax
  int v28; // edx
  _BYTE *v29; // rsi
  int v30; // eax
  int v31; // ebx
  __int64 v32; // rcx
  int v33; // eax
  _DWORD *v34; // rdi
  int v35; // esi
  _QWORD *v36; // rdi
  int v37; // r14d
  _DWORD *v38; // rax
  _DWORD *v39; // rbx
  _DWORD *v40; // r13
  unsigned int v41; // eax
  __int64 v42; // rdi
  __int64 v43; // rdi
  int v44; // r10d
  _DWORD *v45; // rsi
  __int64 v46; // r15
  int v47; // r8d
  int v48; // ecx
  _DWORD *v49; // r13
  _DWORD *v50; // rdx
  _DWORD *v51; // rbx
  char *v52; // r14
  char *v53; // rdi
  char *v54; // r13
  __int64 v55; // r8
  unsigned int v56; // edx
  _QWORD *v57; // rdi
  __int64 v58; // rax
  unsigned int v59; // eax
  _QWORD *v60; // r15
  char v61; // al
  __int64 v62; // rax
  bool v63; // zf
  unsigned int v64; // esi
  int v65; // ecx
  int v66; // ecx
  __int64 v67; // r8
  __int64 v68; // rdx
  _QWORD *v69; // r10
  __int64 v70; // rdi
  int v71; // eax
  int v72; // r11d
  int v73; // eax
  int v74; // ecx
  int v75; // ecx
  __int64 v76; // r8
  _QWORD *v77; // r11
  int v78; // r15d
  __int64 v79; // rdx
  __int64 v80; // rdi
  unsigned int v81; // ecx
  _QWORD *v82; // rdi
  unsigned int v83; // eax
  int v84; // eax
  unsigned __int64 v85; // rax
  unsigned __int64 v86; // rax
  int v87; // r15d
  __int64 v88; // r14
  _QWORD *v89; // rax
  __int64 v90; // rdx
  _QWORD *j; // rdx
  int v92; // ebx
  unsigned int v93; // eax
  _DWORD *v94; // rdi
  unsigned __int64 v95; // rdx
  _DWORD *v96; // rax
  _DWORD *k; // rdx
  int v98; // r15d
  int v99; // r9d
  _DWORD *v100; // r8
  _DWORD *v101; // rax
  _QWORD *v102; // rax
  __int64 v103; // [rsp+10h] [rbp-1B0h]
  __int64 v104; // [rsp+20h] [rbp-1A0h]
  _DWORD *v105; // [rsp+30h] [rbp-190h]
  __int64 v107; // [rsp+38h] [rbp-188h]
  __int64 (__fastcall ***v108)(); // [rsp+40h] [rbp-180h] BYREF
  __int64 (__fastcall ***v109)(); // [rsp+48h] [rbp-178h]
  __int64 (__fastcall ***v110)(); // [rsp+50h] [rbp-170h]
  __int64 v111; // [rsp+60h] [rbp-160h] BYREF
  _DWORD *v112; // [rsp+68h] [rbp-158h]
  __int64 v113; // [rsp+70h] [rbp-150h]
  unsigned int v114; // [rsp+78h] [rbp-148h]
  __int64 (__fastcall **v115)(); // [rsp+80h] [rbp-140h] BYREF
  _QWORD v116[2]; // [rsp+88h] [rbp-138h] BYREF
  __int64 v117; // [rsp+98h] [rbp-128h]
  __int64 v118; // [rsp+A0h] [rbp-120h]
  _BYTE v119[184]; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v120; // [rsp+168h] [rbp-58h]
  __int64 v121; // [rsp+170h] [rbp-50h]
  __int64 v122; // [rsp+178h] [rbp-48h]

  v103 = a1 + 24;
  v4 = *(_DWORD *)(a1 + 40);
  ++*(_QWORD *)(a1 + 24);
  if ( !v4 )
  {
    if ( !*(_DWORD *)(a1 + 44) )
      goto LABEL_7;
    v5 = *(unsigned int *)(a1 + 48);
    if ( (unsigned int)v5 > 0x40 )
    {
      j___libc_free_0(*(_QWORD *)(a1 + 32));
      *(_QWORD *)(a1 + 32) = 0;
      *(_QWORD *)(a1 + 40) = 0;
      *(_DWORD *)(a1 + 48) = 0;
      goto LABEL_7;
    }
    goto LABEL_4;
  }
  v81 = 4 * v4;
  v5 = *(unsigned int *)(a1 + 48);
  if ( (unsigned int)(4 * v4) < 0x40 )
    v81 = 64;
  if ( (unsigned int)v5 <= v81 )
  {
LABEL_4:
    v6 = *(_QWORD **)(a1 + 32);
    for ( i = &v6[v5]; i != v6; ++v6 )
      *v6 = -8;
    *(_QWORD *)(a1 + 40) = 0;
    goto LABEL_7;
  }
  v82 = *(_QWORD **)(a1 + 32);
  v83 = v4 - 1;
  if ( !v83 )
  {
    v88 = 1024;
    v87 = 128;
LABEL_149:
    j___libc_free_0(v82);
    *(_DWORD *)(a1 + 48) = v87;
    v89 = (_QWORD *)sub_22077B0(v88);
    v90 = *(unsigned int *)(a1 + 48);
    *(_QWORD *)(a1 + 40) = 0;
    *(_QWORD *)(a1 + 32) = v89;
    for ( j = &v89[v90]; j != v89; ++v89 )
    {
      if ( v89 )
        *v89 = -8;
    }
    goto LABEL_7;
  }
  _BitScanReverse(&v83, v83);
  v84 = 1 << (33 - (v83 ^ 0x1F));
  if ( v84 < 64 )
    v84 = 64;
  if ( (_DWORD)v5 != v84 )
  {
    v85 = (4 * v84 / 3u + 1) | ((unsigned __int64)(4 * v84 / 3u + 1) >> 1);
    v86 = ((v85 | (v85 >> 2)) >> 4) | v85 | (v85 >> 2) | ((((v85 | (v85 >> 2)) >> 4) | v85 | (v85 >> 2)) >> 8);
    v87 = (v86 | (v86 >> 16)) + 1;
    v88 = 8 * ((v86 | (v86 >> 16)) + 1);
    goto LABEL_149;
  }
  *(_QWORD *)(a1 + 40) = 0;
  v102 = &v82[v5];
  do
  {
    if ( v82 )
      *v82 = -8;
    ++v82;
  }
  while ( v102 != v82 );
LABEL_7:
  v8 = *(_QWORD **)(a1 + 56);
  v9 = &v8[5 * *(unsigned int *)(a1 + 64)];
  while ( v8 != v9 )
  {
    v10 = *(v9 - 2);
    v9 -= 5;
    *v9 = &unk_49EE2B0;
    if ( v10 != -8 && v10 != 0 && v10 != -16 )
      sub_1649B30(v9 + 1);
  }
  *(_DWORD *)(a1 + 64) = 0;
  sub_190A6C0(v119);
  v121 = 0;
  v120 = a3;
  v11 = *(_QWORD *)(a2 + 80);
  v122 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v107 = v11;
  v104 = a2 + 72;
  if ( v11 == a2 + 72 )
  {
    v19 = 0;
    goto LABEL_38;
  }
  do
  {
    v108 = 0;
    v109 = 0;
    v110 = 0;
    if ( !v107 )
      BUG();
    v12 = *(_QWORD *)(v107 + 24);
    v13 = v107 + 16;
    v14 = 0;
    if ( v12 == v107 + 16 )
      goto LABEL_30;
    do
    {
      while ( 1 )
      {
        if ( !v12 )
        {
          v115 = 0;
          BUG();
        }
        v115 = (__int64 (__fastcall **)())(v12 - 24);
        if ( *(_BYTE *)(v12 - 8) != 35 )
          goto LABEL_16;
        v15 = *(_QWORD *)(v12 - 16);
        if ( v15 && !*(_QWORD *)(v15 + 8) )
          v14 = 1;
        v16 = v109;
        if ( v109 != v110 )
          break;
        sub_170B610((__int64)&v108, v109, &v115);
LABEL_16:
        v12 = *(_QWORD *)(v12 + 8);
        if ( v13 == v12 )
          goto LABEL_26;
      }
      if ( v109 )
      {
        *v109 = (__int64 (__fastcall **)())(v12 - 24);
        v16 = v109;
      }
      v109 = v16 + 1;
      v12 = *(_QWORD *)(v12 + 8);
    }
    while ( v13 != v12 );
LABEL_26:
    v17 = v108;
    if ( !v14 || (v18 = v109, (unsigned __int64)((char *)v109 - (char *)v108) <= 8) )
    {
      if ( !v108 )
        goto LABEL_30;
      goto LABEL_29;
    }
    v24 = v108;
    if ( v109 != v108 )
    {
      while ( 1 )
      {
        v115 = *v24;
        v30 = sub_1911FD0(v119);
        v31 = v30;
        if ( !v114 )
          break;
        v25 = 37 * v30;
        v26 = (v114 - 1) & (37 * v30);
        v27 = &v112[8 * v26];
        v28 = *v27;
        if ( v31 != *v27 )
        {
          v44 = 1;
          v34 = 0;
          while ( v28 != -1 )
          {
            if ( !v34 && v28 == -2 )
              v34 = v27;
            v26 = (v114 - 1) & (v44 + v26);
            v27 = &v112[8 * v26];
            v28 = *v27;
            if ( v31 == *v27 )
              goto LABEL_43;
            ++v44;
          }
          if ( !v34 )
            v34 = v27;
          ++v111;
          v33 = v113 + 1;
          if ( 4 * ((int)v113 + 1) < 3 * v114 )
          {
            if ( v114 - HIDWORD(v113) - v33 <= v114 >> 3 )
            {
              sub_17155A0((__int64)&v111, v114);
              if ( !v114 )
              {
LABEL_216:
                LODWORD(v113) = v113 + 1;
                BUG();
              }
              v45 = 0;
              LODWORD(v46) = (v114 - 1) & v25;
              v47 = 1;
              v33 = v113 + 1;
              v34 = &v112[8 * (unsigned int)v46];
              v48 = *v34;
              if ( v31 != *v34 )
              {
                while ( v48 != -1 )
                {
                  if ( !v45 && v48 == -2 )
                    v45 = v34;
                  v46 = (v114 - 1) & ((_DWORD)v46 + v47);
                  v34 = &v112[8 * v46];
                  v48 = *v34;
                  if ( v31 == *v34 )
                    goto LABEL_51;
                  ++v47;
                }
                if ( v45 )
                  v34 = v45;
              }
            }
            goto LABEL_51;
          }
LABEL_49:
          sub_17155A0((__int64)&v111, 2 * v114);
          if ( !v114 )
            goto LABEL_216;
          LODWORD(v32) = (v114 - 1) & (37 * v31);
          v33 = v113 + 1;
          v34 = &v112[8 * (unsigned int)v32];
          v35 = *v34;
          if ( v31 != *v34 )
          {
            v99 = 1;
            v100 = 0;
            while ( v35 != -1 )
            {
              if ( !v100 && v35 == -2 )
                v100 = v34;
              v32 = (v114 - 1) & ((_DWORD)v32 + v99);
              v34 = &v112[8 * v32];
              v35 = *v34;
              if ( v31 == *v34 )
                goto LABEL_51;
              ++v99;
            }
            if ( v100 )
              v34 = v100;
          }
LABEL_51:
          LODWORD(v113) = v33;
          if ( *v34 != -1 )
            --HIDWORD(v113);
          *v34 = v31;
          v29 = 0;
          v36 = v34 + 2;
          *v36 = 0;
          v36[1] = 0;
          v36[2] = 0;
LABEL_54:
          ++v24;
          sub_170B610((__int64)v36, v29, &v115);
          if ( v18 == v24 )
            goto LABEL_55;
          continue;
        }
LABEL_43:
        v29 = (_BYTE *)*((_QWORD *)v27 + 2);
        if ( v29 == *((_BYTE **)v27 + 3) )
        {
          v36 = v27 + 2;
          goto LABEL_54;
        }
        if ( v29 )
        {
          *(_QWORD *)v29 = v115;
          v29 = (_BYTE *)*((_QWORD *)v27 + 2);
        }
        ++v24;
        *((_QWORD *)v27 + 2) = v29 + 8;
        if ( v18 == v24 )
          goto LABEL_55;
      }
      ++v111;
      goto LABEL_49;
    }
LABEL_55:
    v37 = v113;
    if ( !(_DWORD)v113 )
      goto LABEL_56;
    v38 = v112;
    v49 = &v112[8 * v114];
    if ( v112 == v49 )
      goto LABEL_95;
    v50 = v112;
    while ( 1 )
    {
      v51 = v50;
      if ( *v50 <= 0xFFFFFFFD )
        break;
      v50 += 8;
      if ( v49 == v50 )
        goto LABEL_95;
    }
    if ( v49 == v50 )
    {
LABEL_95:
      ++v111;
      goto LABEL_57;
    }
    while ( 2 )
    {
      v52 = (char *)*((_QWORD *)v51 + 1);
      v53 = (char *)*((_QWORD *)v51 + 2);
      if ( (unsigned __int64)(v53 - v52) <= 8 || v53 == v52 )
        goto LABEL_98;
      v105 = v49;
      v54 = (char *)*((_QWORD *)v51 + 2);
      while ( 2 )
      {
        v64 = *(_DWORD *)(a1 + 48);
        if ( !v64 )
        {
          ++*(_QWORD *)(a1 + 24);
          goto LABEL_123;
        }
        v55 = *(_QWORD *)(a1 + 32);
        v56 = (v64 - 1) & (((unsigned int)*(_QWORD *)v52 >> 9) ^ ((unsigned int)*(_QWORD *)v52 >> 4));
        v57 = (_QWORD *)(v55 + 8LL * v56);
        v58 = *v57;
        if ( *(_QWORD *)v52 != *v57 )
        {
          v72 = 1;
          v69 = 0;
          while ( v58 != -8 )
          {
            if ( v58 != -16 || v69 )
              v57 = v69;
            v56 = (v64 - 1) & (v72 + v56);
            v58 = *(_QWORD *)(v55 + 8LL * v56);
            if ( *(_QWORD *)v52 == v58 )
              goto LABEL_107;
            ++v72;
            v69 = v57;
            v57 = (_QWORD *)(v55 + 8LL * v56);
          }
          v73 = *(_DWORD *)(a1 + 40);
          if ( !v69 )
            v69 = v57;
          ++*(_QWORD *)(a1 + 24);
          v71 = v73 + 1;
          if ( 4 * v71 >= 3 * v64 )
          {
LABEL_123:
            sub_1467110(v103, 2 * v64);
            v65 = *(_DWORD *)(a1 + 48);
            if ( !v65 )
              goto LABEL_214;
            v66 = v65 - 1;
            v67 = *(_QWORD *)(a1 + 32);
            LODWORD(v68) = v66 & (((unsigned int)*(_QWORD *)v52 >> 9) ^ ((unsigned int)*(_QWORD *)v52 >> 4));
            v69 = (_QWORD *)(v67 + 8LL * (unsigned int)v68);
            v70 = *v69;
            v71 = *(_DWORD *)(a1 + 40) + 1;
            if ( *(_QWORD *)v52 != *v69 )
            {
              v98 = 1;
              v77 = 0;
              while ( v70 != -8 )
              {
                if ( v70 == -16 && !v77 )
                  v77 = v69;
                v68 = v66 & (unsigned int)(v68 + v98);
                v69 = (_QWORD *)(v67 + 8 * v68);
                v70 = *v69;
                if ( *(_QWORD *)v52 == *v69 )
                  goto LABEL_125;
                ++v98;
              }
              goto LABEL_138;
            }
          }
          else if ( v64 - *(_DWORD *)(a1 + 44) - v71 <= v64 >> 3 )
          {
            sub_1467110(v103, v64);
            v74 = *(_DWORD *)(a1 + 48);
            if ( !v74 )
            {
LABEL_214:
              ++*(_DWORD *)(a1 + 40);
              BUG();
            }
            v75 = v74 - 1;
            v76 = *(_QWORD *)(a1 + 32);
            v77 = 0;
            v78 = 1;
            LODWORD(v79) = v75 & (((unsigned int)*(_QWORD *)v52 >> 9) ^ ((unsigned int)*(_QWORD *)v52 >> 4));
            v69 = (_QWORD *)(v76 + 8LL * (unsigned int)v79);
            v80 = *v69;
            v71 = *(_DWORD *)(a1 + 40) + 1;
            if ( *v69 != *(_QWORD *)v52 )
            {
              while ( v80 != -8 )
              {
                if ( v80 == -16 && !v77 )
                  v77 = v69;
                v79 = v75 & (unsigned int)(v79 + v78);
                v69 = (_QWORD *)(v76 + 8 * v79);
                v80 = *v69;
                if ( *(_QWORD *)v52 == *v69 )
                  goto LABEL_125;
                ++v78;
              }
LABEL_138:
              if ( v77 )
                v69 = v77;
            }
          }
LABEL_125:
          *(_DWORD *)(a1 + 40) = v71;
          if ( *v69 != -8 )
            --*(_DWORD *)(a1 + 44);
          v58 = *(_QWORD *)v52;
          *v69 = *(_QWORD *)v52;
        }
LABEL_107:
        v117 = v58;
        v116[0] = 2;
        v116[1] = 0;
        if ( v58 != -8 && v58 != 0 && v58 != -16 )
          sub_164C220((__int64)v116);
        v118 = v103;
        v115 = off_49EFFB0;
        v59 = *(_DWORD *)(a1 + 64);
        if ( v59 >= *(_DWORD *)(a1 + 68) )
        {
          sub_170B7A0(a1 + 56, 0);
          v59 = *(_DWORD *)(a1 + 64);
        }
        v60 = (_QWORD *)(*(_QWORD *)(a1 + 56) + 40LL * v59);
        if ( v60 )
        {
          v61 = v116[0];
          v60[2] = 0;
          v60[1] = v61 & 6;
          v62 = v117;
          v63 = v117 == -8;
          v60[3] = v117;
          if ( v62 != 0 && !v63 && v62 != -16 )
            sub_1649AC0(v60 + 1, v116[0] & 0xFFFFFFFFFFFFFFF8LL);
          *v60 = off_49EFFB0;
          v60[4] = v118;
          v59 = *(_DWORD *)(a1 + 64);
        }
        *(_DWORD *)(a1 + 64) = v59 + 1;
        v115 = (__int64 (__fastcall **)())&unk_49EE2B0;
        if ( v117 != -8 && v117 != 0 && v117 != -16 )
          sub_1649B30(v116);
        v52 += 8;
        if ( v54 != v52 )
          continue;
        break;
      }
      v49 = v105;
LABEL_98:
      v51 += 8;
      if ( v51 != v49 )
      {
        while ( *v51 > 0xFFFFFFFD )
        {
          v51 += 8;
          if ( v49 == v51 )
            goto LABEL_101;
        }
        if ( v49 != v51 )
          continue;
      }
      break;
    }
LABEL_101:
    v37 = v113;
LABEL_56:
    ++v111;
    v38 = v112;
    if ( HIDWORD(v113) | v37 )
    {
LABEL_57:
      v39 = v38;
      v40 = &v38[8 * v114];
      v41 = 4 * v37;
      if ( (unsigned int)(4 * v37) < 0x40 )
        v41 = 64;
      if ( v114 <= v41 )
      {
        while ( v39 != v40 )
        {
          if ( *v39 != -1 )
          {
            if ( *v39 != -2 )
            {
              v42 = *((_QWORD *)v39 + 1);
              if ( v42 )
                j_j___libc_free_0(v42, *((_QWORD *)v39 + 3) - v42);
            }
            *v39 = -1;
          }
          v39 += 8;
        }
        goto LABEL_68;
      }
      do
      {
        while ( *v39 > 0xFFFFFFFD )
        {
          v39 += 8;
          if ( v39 == v40 )
            goto LABEL_76;
        }
        v43 = *((_QWORD *)v39 + 1);
        if ( v43 )
          j_j___libc_free_0(v43, *((_QWORD *)v39 + 3) - v43);
        v39 += 8;
      }
      while ( v39 != v40 );
LABEL_76:
      if ( !v37 )
      {
        if ( v114 )
        {
          j___libc_free_0(v112);
          v112 = 0;
          v113 = 0;
          v114 = 0;
          goto LABEL_69;
        }
LABEL_68:
        v113 = 0;
        goto LABEL_69;
      }
      v92 = 64;
      if ( v37 != 1 )
      {
        _BitScanReverse(&v93, v37 - 1);
        v92 = 1 << (33 - (v93 ^ 0x1F));
        if ( v92 < 64 )
          v92 = 64;
      }
      v94 = v112;
      if ( v92 == v114 )
      {
        v113 = 0;
        v101 = &v112[8 * v92];
        do
        {
          if ( v94 )
            *v94 = -1;
          v94 += 8;
        }
        while ( v101 != v94 );
      }
      else
      {
        j___libc_free_0(v112);
        v95 = ((((((((4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 2)
                 | (4 * v92 / 3u + 1)
                 | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 4)
               | (((4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 2)
               | (4 * v92 / 3u + 1)
               | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 2)
               | (4 * v92 / 3u + 1)
               | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 4)
             | (((4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 2)
             | (4 * v92 / 3u + 1)
             | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 16;
        v114 = (v95
              | (((((((4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 2)
                  | (4 * v92 / 3u + 1)
                  | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 4)
                | (((4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 2)
                | (4 * v92 / 3u + 1)
                | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 8)
              | (((((4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 2)
                | (4 * v92 / 3u + 1)
                | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 4)
              | (((4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 2)
              | (4 * v92 / 3u + 1)
              | ((4 * v92 / 3u + 1) >> 1))
             + 1;
        v96 = (_DWORD *)sub_22077B0(
                          32
                        * ((v95
                          | (((((((4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 2)
                              | (4 * v92 / 3u + 1)
                              | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 4)
                            | (((4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 2)
                            | (4 * v92 / 3u + 1)
                            | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 8)
                          | (((((4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 2)
                            | (4 * v92 / 3u + 1)
                            | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 4)
                          | (((4 * v92 / 3u + 1) | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1)) >> 2)
                          | (4 * v92 / 3u + 1)
                          | ((unsigned __int64)(4 * v92 / 3u + 1) >> 1))
                         + 1));
        v113 = 0;
        v112 = v96;
        for ( k = &v96[8 * v114]; k != v96; v96 += 8 )
        {
          if ( v96 )
            *v96 = -1;
        }
      }
    }
LABEL_69:
    v17 = v108;
    if ( v108 )
LABEL_29:
      j_j___libc_free_0(v17, (char *)v110 - (char *)v17);
LABEL_30:
    v107 = *(_QWORD *)(v107 + 8);
  }
  while ( v104 != v107 );
  v19 = v112;
  if ( v114 )
  {
    v20 = &v112[8 * v114];
    do
    {
      while ( 1 )
      {
        if ( *v19 <= 0xFFFFFFFD )
        {
          v21 = *((_QWORD *)v19 + 1);
          if ( v21 )
            break;
        }
        v19 += 8;
        if ( v20 == v19 )
          goto LABEL_37;
      }
      v22 = *((_QWORD *)v19 + 3);
      v19 += 8;
      j_j___libc_free_0(v21, v22 - v21);
    }
    while ( v20 != v19 );
LABEL_37:
    v19 = v112;
  }
LABEL_38:
  j___libc_free_0(v19);
  return sub_190A790(v119);
}
