// Function: sub_1B3EC40
// Address: 0x1b3ec40
//
__int64 __fastcall sub_1B3EC40(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r9
  __int64 v5; // rax
  __int64 *v6; // r8
  __int64 *v7; // r15
  __int64 v8; // r11
  __int64 i; // r14
  __int64 m; // rbx
  __int64 v11; // rax
  __int64 *v12; // r11
  char v13; // bl
  __int64 v14; // rdi
  int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // r10
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // esi
  __int64 v21; // r10
  __int64 v22; // rdx
  __int64 v23; // r11
  unsigned int v24; // ecx
  __int64 *v25; // rax
  __int64 v26; // r9
  __int64 v27; // rax
  __int64 v28; // r12
  __int64 v30; // rax
  __int64 v31; // r13
  unsigned int v32; // r12d
  _QWORD *v33; // r10
  unsigned __int64 v34; // rsi
  __int64 *v35; // rax
  __int64 v36; // rdi
  int v37; // eax
  int v38; // edi
  int v39; // edx
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // rax
  _QWORD *v42; // rax
  __int64 *v43; // r10
  __int64 *v44; // r9
  _QWORD *k; // rdx
  __int64 *v46; // rax
  __int64 v47; // rbx
  __int64 v48; // r12
  int v49; // esi
  int v50; // esi
  __int64 v51; // r8
  int v52; // r10d
  _QWORD *v53; // r11
  unsigned int v54; // ecx
  _QWORD *v55; // rdx
  __int64 v56; // rdi
  int v57; // r12d
  int v58; // r12d
  unsigned int v59; // esi
  int v60; // r8d
  __int64 *v61; // rdi
  unsigned __int64 v62; // rax
  __int64 v63; // rax
  _QWORD *v64; // rax
  __int64 *v65; // r9
  _QWORD *j; // rdx
  __int64 *v67; // rax
  __int64 v68; // rbx
  __int64 v69; // r12
  int v70; // esi
  int v71; // esi
  __int64 v72; // r11
  int v73; // r10d
  _QWORD *v74; // r8
  unsigned int v75; // ecx
  _QWORD *v76; // rdx
  __int64 v77; // rdi
  __int64 v78; // r11
  int v79; // r12d
  int v80; // r12d
  int v81; // r8d
  unsigned int v82; // esi
  __int64 v83; // rcx
  __int64 v84; // rcx
  int v85; // r8d
  __int64 *v86; // rdi
  int v87; // ecx
  int v88; // r9d
  int v89; // r9d
  __int64 v90; // r10
  __int64 *v91; // r11
  int v92; // edi
  unsigned int v93; // esi
  __int64 v94; // r8
  int v95; // r9d
  int v96; // r9d
  __int64 v97; // r10
  unsigned int v98; // esi
  __int64 v99; // r8
  int v100; // edi
  _QWORD *v101; // [rsp+0h] [rbp-3B0h]
  __int64 v102; // [rsp+18h] [rbp-398h]
  __int64 v103; // [rsp+20h] [rbp-390h]
  __int64 v104; // [rsp+20h] [rbp-390h]
  __int64 v105; // [rsp+20h] [rbp-390h]
  __int64 v106; // [rsp+20h] [rbp-390h]
  __int64 v107; // [rsp+28h] [rbp-388h]
  int v108; // [rsp+28h] [rbp-388h]
  __int64 *v109; // [rsp+28h] [rbp-388h]
  __int64 v110; // [rsp+28h] [rbp-388h]
  _QWORD *v111; // [rsp+28h] [rbp-388h]
  __int64 v112; // [rsp+28h] [rbp-388h]
  unsigned __int64 v114; // [rsp+38h] [rbp-378h]
  char v115; // [rsp+43h] [rbp-36Dh]
  unsigned int v116; // [rsp+44h] [rbp-36Ch]
  __int64 v117; // [rsp+48h] [rbp-368h] BYREF
  __int64 *v118; // [rsp+50h] [rbp-360h] BYREF
  __int64 v119; // [rsp+58h] [rbp-358h]
  _BYTE v120[848]; // [rsp+60h] [rbp-350h] BYREF

  v118 = (__int64 *)v120;
  v117 = a2;
  v119 = 0x6400000000LL;
  v102 = sub_1B3DC90(a1, a2, (__int64)&v118);
  v5 = (unsigned int)v119;
  if ( !(_DWORD)v119 )
  {
    v28 = sub_1599EF0(*(__int64 ***)(*(_QWORD *)a1 + 8LL));
    sub_1A703E0(*(_QWORD *)(a1 + 8), &v117)[1] = v28;
    goto LABEL_43;
  }
  do
  {
    v6 = &v118[v5];
    v114 = (unsigned __int64)v118;
    if ( v118 == v6 )
      goto LABEL_39;
    v115 = 0;
    v7 = &v118[v5];
    do
    {
      v8 = *(v7 - 1);
      v2 = *(unsigned int *)(v8 + 40);
      if ( !(_DWORD)v2 )
        goto LABEL_23;
      v116 = 0;
      for ( i = 0; ; i = m )
      {
        v2 = *(_QWORD *)(v8 + 48);
        m = *(_QWORD *)(v2 + 8LL * v116);
        if ( !*(_DWORD *)(m + 24) )
        {
          v107 = v8;
          v30 = sub_1599EF0(*(__int64 ***)(*(_QWORD *)a1 + 8LL));
          v8 = v107;
          *(_QWORD *)(m + 8) = v30;
          v31 = *(_QWORD *)(a1 + 8);
          v6 = (__int64 *)v30;
          v32 = *(_DWORD *)(v31 + 24);
          v33 = *(_QWORD **)(v31 + 8);
          if ( v32 )
          {
            v3 = *(_QWORD *)m;
            v34 = v32 - 1;
            v2 = (unsigned int)v34 & (((unsigned int)*(_QWORD *)m >> 9) ^ ((unsigned int)*(_QWORD *)m >> 4));
            v35 = &v33[2 * v2];
            v36 = *v35;
            if ( *(_QWORD *)m == *v35 )
            {
LABEL_49:
              v35[1] = (__int64)v6;
              *(_QWORD *)(m + 16) = m;
              v37 = *(_DWORD *)(v102 + 24);
              *(_DWORD *)(m + 24) = v37;
              *(_DWORD *)(v102 + 24) = v37 + 1;
              goto LABEL_7;
            }
            v108 = 1;
            v4 = 0;
            while ( v36 != -8 )
            {
              if ( v36 == -16 && !v4 )
                v4 = (__int64)v35;
              v2 = (unsigned int)v34 & (v108 + (_DWORD)v2);
              v3 = (unsigned int)(v108 + 1);
              v35 = &v33[2 * (unsigned int)v2];
              v36 = *v35;
              if ( *(_QWORD *)m == *v35 )
                goto LABEL_49;
              ++v108;
            }
            v38 = *(_DWORD *)(v31 + 16);
            if ( v4 )
              v35 = (__int64 *)v4;
            ++*(_QWORD *)v31;
            v39 = v38 + 1;
            if ( 4 * (v38 + 1) < 3 * v32 )
            {
              v3 = v32 - *(_DWORD *)(v31 + 20) - v39;
              if ( (unsigned int)v3 > v32 >> 3 )
                goto LABEL_56;
              v105 = v8;
              v111 = v33;
              v62 = ((((v34 | (v34 >> 1)) >> 2) | v34 | (v34 >> 1)) >> 4) | ((v34 | (v34 >> 1)) >> 2) | v34 | (v34 >> 1);
              v63 = ((((v62 >> 8) | v62) >> 16) | (v62 >> 8) | v62) + 1;
              if ( (unsigned int)v63 < 0x40 )
                LODWORD(v63) = 64;
              *(_DWORD *)(v31 + 24) = v63;
              v64 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v63);
              v8 = v105;
              *(_QWORD *)(v31 + 8) = v64;
              if ( v111 )
              {
                *(_QWORD *)(v31 + 16) = 0;
                v65 = &v111[2 * v32];
                for ( j = &v64[2 * *(unsigned int *)(v31 + 24)]; j != v64; v64 += 2 )
                {
                  if ( v64 )
                    *v64 = -8;
                }
                v101 = v111;
                v67 = v111;
                v106 = m;
                v68 = v8;
                do
                {
                  v69 = *v67;
                  if ( *v67 != -16 && v69 != -8 )
                  {
                    v70 = *(_DWORD *)(v31 + 24);
                    if ( !v70 )
                    {
                      MEMORY[0] = *v67;
                      BUG();
                    }
                    v71 = v70 - 1;
                    v72 = *(_QWORD *)(v31 + 8);
                    v73 = 1;
                    v74 = 0;
                    v75 = v71 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
                    v76 = (_QWORD *)(v72 + 16LL * v75);
                    v77 = *v76;
                    if ( v69 != *v76 )
                    {
                      while ( v77 != -8 )
                      {
                        if ( !v74 && v77 == -16 )
                          v74 = v76;
                        v75 = v71 & (v73 + v75);
                        v76 = (_QWORD *)(v72 + 16LL * v75);
                        v77 = *v76;
                        if ( v69 == *v76 )
                          goto LABEL_98;
                        ++v73;
                      }
                      if ( v74 )
                        v76 = v74;
                    }
LABEL_98:
                    *v76 = v69;
                    v76[1] = v67[1];
                    ++*(_DWORD *)(v31 + 16);
                  }
                  v67 += 2;
                }
                while ( v65 != v67 );
                v78 = v68;
                m = v106;
                v112 = v78;
                j___libc_free_0(v101);
                v3 = *(_QWORD *)(v31 + 8);
                v79 = *(_DWORD *)(v31 + 24);
                v8 = v112;
                v39 = *(_DWORD *)(v31 + 16) + 1;
              }
              else
              {
                v83 = *(unsigned int *)(v31 + 24);
                *(_QWORD *)(v31 + 16) = 0;
                v79 = v83;
                v3 = (__int64)&v64[2 * v83];
                if ( v64 == (_QWORD *)v3 )
                {
                  v39 = 1;
                }
                else
                {
                  do
                  {
                    if ( v64 )
                      *v64 = -8;
                    v64 += 2;
                  }
                  while ( (_QWORD *)v3 != v64 );
                  v3 = *(_QWORD *)(v31 + 8);
                  v79 = *(_DWORD *)(v31 + 24);
                  v39 = *(_DWORD *)(v31 + 16) + 1;
                }
              }
              if ( !v79 )
                goto LABEL_189;
              v80 = v79 - 1;
              v81 = 1;
              v61 = 0;
              v82 = v80 & (((unsigned int)*(_QWORD *)m >> 9) ^ ((unsigned int)*(_QWORD *)m >> 4));
              v35 = (__int64 *)(v3 + 16LL * v82);
              v4 = *v35;
              if ( *v35 == *(_QWORD *)m )
                goto LABEL_56;
              while ( v4 != -8 )
              {
                if ( !v61 && v4 == -16 )
                  v61 = v35;
                v82 = v80 & (v81 + v82);
                v35 = (__int64 *)(v3 + 16LL * v82);
                v4 = *v35;
                if ( *(_QWORD *)m == *v35 )
                  goto LABEL_56;
                ++v81;
              }
LABEL_104:
              if ( v61 )
                v35 = v61;
LABEL_56:
              *(_DWORD *)(v31 + 16) = v39;
              if ( *v35 != -8 )
                --*(_DWORD *)(v31 + 20);
              v2 = *(_QWORD *)m;
              v35[1] = 0;
              *v35 = v2;
              v6 = *(__int64 **)(m + 8);
              goto LABEL_49;
            }
          }
          else
          {
            ++*(_QWORD *)v31;
          }
          v103 = v8;
          v109 = v33;
          v40 = ((((((((2 * v32 - 1) | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 2)
                   | (2 * v32 - 1)
                   | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 4)
                 | (((2 * v32 - 1) | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 2)
                 | (2 * v32 - 1)
                 | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 8)
               | (((((2 * v32 - 1) | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 2)
                 | (2 * v32 - 1)
                 | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 4)
               | (((2 * v32 - 1) | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 2)
               | (2 * v32 - 1)
               | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 16;
          v41 = (v40
               | (((((((2 * v32 - 1) | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 2)
                   | (2 * v32 - 1)
                   | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 4)
                 | (((2 * v32 - 1) | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 2)
                 | (2 * v32 - 1)
                 | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 8)
               | (((((2 * v32 - 1) | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 2)
                 | (2 * v32 - 1)
                 | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 4)
               | (((2 * v32 - 1) | ((unsigned __int64)(2 * v32 - 1) >> 1)) >> 2)
               | (2 * v32 - 1)
               | ((unsigned __int64)(2 * v32 - 1) >> 1))
              + 1;
          if ( (unsigned int)v41 < 0x40 )
            LODWORD(v41) = 64;
          *(_DWORD *)(v31 + 24) = v41;
          v42 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v41);
          v43 = v109;
          v8 = v103;
          *(_QWORD *)(v31 + 8) = v42;
          if ( v109 )
          {
            *(_QWORD *)(v31 + 16) = 0;
            v44 = &v109[2 * v32];
            for ( k = &v42[2 * *(unsigned int *)(v31 + 24)]; k != v42; v42 += 2 )
            {
              if ( v42 )
                *v42 = -8;
            }
            v46 = v109;
            if ( v109 != v44 )
            {
              v104 = m;
              v47 = v8;
              do
              {
                v48 = *v46;
                if ( *v46 != -16 && v48 != -8 )
                {
                  v49 = *(_DWORD *)(v31 + 24);
                  if ( !v49 )
                  {
                    MEMORY[0] = *v46;
                    BUG();
                  }
                  v50 = v49 - 1;
                  v51 = *(_QWORD *)(v31 + 8);
                  v52 = 1;
                  v53 = 0;
                  v54 = v50 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
                  v55 = (_QWORD *)(v51 + 16LL * v54);
                  v56 = *v55;
                  if ( v48 != *v55 )
                  {
                    while ( v56 != -8 )
                    {
                      if ( v56 == -16 && !v53 )
                        v53 = v55;
                      v54 = v50 & (v52 + v54);
                      v55 = (_QWORD *)(v51 + 16LL * v54);
                      v56 = *v55;
                      if ( v48 == *v55 )
                        goto LABEL_73;
                      ++v52;
                    }
                    if ( v53 )
                      v55 = v53;
                  }
LABEL_73:
                  *v55 = v48;
                  v55[1] = v46[1];
                  ++*(_DWORD *)(v31 + 16);
                }
                v46 += 2;
              }
              while ( v44 != v46 );
              v8 = v47;
              v43 = v109;
              m = v104;
            }
            v110 = v8;
            j___libc_free_0(v43);
            v3 = *(_QWORD *)(v31 + 8);
            v57 = *(_DWORD *)(v31 + 24);
            v8 = v110;
            v39 = *(_DWORD *)(v31 + 16) + 1;
          }
          else
          {
            v84 = *(unsigned int *)(v31 + 24);
            *(_QWORD *)(v31 + 16) = 0;
            v57 = v84;
            v3 = (__int64)&v42[2 * v84];
            if ( v42 == (_QWORD *)v3 )
            {
              v39 = 1;
            }
            else
            {
              do
              {
                if ( v42 )
                  *v42 = -8;
                v42 += 2;
              }
              while ( (_QWORD *)v3 != v42 );
              v3 = *(_QWORD *)(v31 + 8);
              v57 = *(_DWORD *)(v31 + 24);
              v39 = *(_DWORD *)(v31 + 16) + 1;
            }
          }
          if ( !v57 )
          {
LABEL_189:
            ++*(_DWORD *)(v31 + 16);
            BUG();
          }
          v58 = v57 - 1;
          v59 = v58 & (((unsigned int)*(_QWORD *)m >> 9) ^ ((unsigned int)*(_QWORD *)m >> 4));
          v35 = (__int64 *)(v3 + 16LL * v59);
          v4 = *v35;
          if ( *(_QWORD *)m == *v35 )
            goto LABEL_56;
          v60 = 1;
          v61 = 0;
          while ( v4 != -8 )
          {
            if ( v4 == -16 && !v61 )
              v61 = v35;
            v59 = v58 & (v60 + v59);
            v35 = (__int64 *)(v3 + 16LL * v59);
            v4 = *v35;
            if ( *(_QWORD *)m == *v35 )
              goto LABEL_56;
            ++v60;
          }
          goto LABEL_104;
        }
LABEL_7:
        if ( i && m != i )
        {
          v2 = *(unsigned int *)(m + 24);
          v11 = m;
          v3 = *(unsigned int *)(i + 24);
          for ( m = i; ; v3 = *(unsigned int *)(m + 24) )
          {
            while ( (int)v2 <= (int)v3 )
            {
              while ( (int)v2 < (int)v3 )
              {
                v11 = *(_QWORD *)(v11 + 32);
                if ( !v11 )
                  goto LABEL_18;
                v2 = *(unsigned int *)(v11 + 24);
              }
              if ( m == v11 )
                goto LABEL_18;
            }
            m = *(_QWORD *)(m + 32);
            if ( !m )
              break;
          }
          m = v11;
        }
LABEL_18:
        if ( *(_DWORD *)(v8 + 40) == ++v116 )
          break;
      }
      if ( m && *(_QWORD *)(v8 + 32) != m )
      {
        *(_QWORD *)(v8 + 32) = m;
        v115 = 1;
      }
LABEL_23:
      --v7;
    }
    while ( (__int64 *)v114 != v7 );
    v5 = (unsigned int)v119;
  }
  while ( v115 );
  v4 = (__int64)v118;
  v12 = &v118[(unsigned int)v119];
  do
  {
    if ( (__int64 *)v4 == v12 )
      break;
    v13 = 0;
    v3 = (__int64)v12;
    do
    {
      v2 = *(_QWORD *)(v3 - 8);
      v14 = *(_QWORD *)(v2 + 16);
      if ( v2 == v14 )
        goto LABEL_37;
      v15 = *(_DWORD *)(v2 + 40);
      v16 = *(_QWORD *)(v2 + 32);
      if ( v15 )
      {
        v6 = *(__int64 **)(v2 + 48);
        v17 = (__int64)&v6[(unsigned int)(v15 - 1) + 1];
        while ( 1 )
        {
          v18 = *v6;
          if ( v16 != *v6 )
            break;
LABEL_143:
          if ( (__int64 *)v17 == ++v6 )
            goto LABEL_144;
        }
        while ( *(_QWORD *)(v18 + 16) != v18 )
        {
          v18 = *(_QWORD *)(v18 + 32);
          if ( v16 == v18 )
            goto LABEL_143;
        }
        v19 = *(_QWORD *)(v3 - 8);
LABEL_36:
        *(_QWORD *)(v2 + 16) = v19;
        v13 = 1;
        goto LABEL_37;
      }
LABEL_144:
      v19 = *(_QWORD *)(v16 + 16);
      if ( v14 != v19 )
        goto LABEL_36;
LABEL_37:
      v3 -= 8;
    }
    while ( v4 != v3 );
  }
  while ( v13 );
LABEL_39:
  sub_1B3CF30(a1, (__int64)&v118, v2, v3, (__int64)v6, v4);
  v20 = *(_DWORD *)(a1 + 48);
  v21 = a1 + 24;
  if ( !v20 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_167;
  }
  v22 = v117;
  v23 = *(_QWORD *)(a1 + 32);
  v24 = (v20 - 1) & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4));
  v25 = (__int64 *)(v23 + 16LL * v24);
  v26 = *v25;
  if ( v117 == *v25 )
  {
LABEL_41:
    v27 = v25[1];
    goto LABEL_42;
  }
  v85 = 1;
  v86 = 0;
  while ( v26 != -8 )
  {
    if ( v26 == -16 && !v86 )
      v86 = v25;
    v24 = (v20 - 1) & (v85 + v24);
    v25 = (__int64 *)(v23 + 16LL * v24);
    v26 = *v25;
    if ( v117 == *v25 )
      goto LABEL_41;
    ++v85;
  }
  if ( v86 )
    v25 = v86;
  ++*(_QWORD *)(a1 + 24);
  v87 = *(_DWORD *)(a1 + 40) + 1;
  if ( 4 * v87 >= 3 * v20 )
  {
LABEL_167:
    sub_1B3C650(v21, 2 * v20);
    v95 = *(_DWORD *)(a1 + 48);
    if ( v95 )
    {
      v22 = v117;
      v96 = v95 - 1;
      v97 = *(_QWORD *)(a1 + 32);
      v87 = *(_DWORD *)(a1 + 40) + 1;
      v98 = v96 & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4));
      v25 = (__int64 *)(v97 + 16LL * v98);
      v99 = *v25;
      if ( *v25 == v117 )
        goto LABEL_157;
      v100 = 1;
      v91 = 0;
      while ( v99 != -8 )
      {
        if ( !v91 && v99 == -16 )
          v91 = v25;
        v98 = v96 & (v100 + v98);
        v25 = (__int64 *)(v97 + 16LL * v98);
        v99 = *v25;
        if ( v117 == *v25 )
          goto LABEL_157;
        ++v100;
      }
      goto LABEL_163;
    }
LABEL_187:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
  if ( v20 - *(_DWORD *)(a1 + 44) - v87 <= v20 >> 3 )
  {
    sub_1B3C650(v21, v20);
    v88 = *(_DWORD *)(a1 + 48);
    if ( v88 )
    {
      v22 = v117;
      v89 = v88 - 1;
      v90 = *(_QWORD *)(a1 + 32);
      v91 = 0;
      v87 = *(_DWORD *)(a1 + 40) + 1;
      v92 = 1;
      v93 = v89 & (((unsigned int)v117 >> 9) ^ ((unsigned int)v117 >> 4));
      v25 = (__int64 *)(v90 + 16LL * v93);
      v94 = *v25;
      if ( *v25 == v117 )
        goto LABEL_157;
      while ( v94 != -8 )
      {
        if ( !v91 && v94 == -16 )
          v91 = v25;
        v93 = v89 & (v92 + v93);
        v25 = (__int64 *)(v90 + 16LL * v93);
        v94 = *v25;
        if ( v117 == *v25 )
          goto LABEL_157;
        ++v92;
      }
LABEL_163:
      if ( v91 )
        v25 = v91;
      goto LABEL_157;
    }
    goto LABEL_187;
  }
LABEL_157:
  *(_DWORD *)(a1 + 40) = v87;
  if ( *v25 != -8 )
    --*(_DWORD *)(a1 + 44);
  *v25 = v22;
  v25[1] = 0;
  v27 = 0;
LABEL_42:
  v28 = *(_QWORD *)(*(_QWORD *)(v27 + 16) + 8LL);
LABEL_43:
  if ( v118 != (__int64 *)v120 )
    _libc_free((unsigned __int64)v118);
  return v28;
}
