// Function: sub_11D5FF0
// Address: 0x11d5ff0
//
__int64 __fastcall sub_11D5FF0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // rax
  _BYTE *v7; // r15
  __int64 v8; // rbx
  __int64 *i; // r13
  __int64 *m; // r12
  __int64 *v11; // rax
  _BYTE *v12; // r11
  char v13; // bl
  __int64 v14; // rdi
  int v15; // eax
  __int64 v16; // rsi
  __int64 v17; // r10
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int64 v20; // rsi
  __int64 v21; // r10
  __int64 v22; // rdx
  __int64 *v23; // rdi
  int v24; // r8d
  __int64 v25; // r11
  unsigned int v26; // ecx
  __int64 *v27; // rax
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v32; // rax
  __int64 v33; // r14
  __int64 v34; // r15
  __int64 v35; // r10
  unsigned __int64 v36; // rdi
  __int64 *v37; // rax
  __int64 *v38; // rax
  int v39; // eax
  int v40; // eax
  int v41; // eax
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rax
  _QWORD *v45; // rax
  __int64 v46; // r10
  __int64 v47; // rdx
  __int64 *v48; // r15
  _QWORD *k; // rdx
  __int64 *v50; // rax
  __int64 v51; // r11
  int v52; // esi
  int v53; // esi
  __int64 v54; // r8
  int v55; // r10d
  _QWORD *v56; // r9
  unsigned int v57; // ecx
  _QWORD *v58; // rdx
  __int64 v59; // rdi
  int v60; // r15d
  int v61; // r15d
  unsigned int v62; // esi
  __int64 v63; // rdi
  unsigned __int64 v64; // rax
  __int64 v65; // rax
  _QWORD *v66; // rax
  __int64 *v67; // r10
  __int64 v68; // rdx
  __int64 *v69; // r9
  _QWORD *j; // rdx
  __int64 *v71; // rax
  __int64 v72; // r15
  __int64 v73; // r11
  int v74; // esi
  int v75; // esi
  __int64 v76; // rbx
  int v77; // r10d
  _QWORD *v78; // r8
  unsigned int v79; // ecx
  _QWORD *v80; // rdx
  __int64 v81; // rdi
  int v82; // r15d
  int v83; // r15d
  unsigned int v84; // esi
  __int64 v85; // rcx
  __int64 v86; // rcx
  int v87; // ecx
  int v88; // r9d
  int v89; // r9d
  __int64 v90; // r10
  __int64 *v91; // r11
  int v92; // edi
  __int64 v93; // r8
  int v94; // r9d
  int v95; // r9d
  __int64 v96; // r10
  __int64 v97; // r8
  int v98; // edi
  __int64 v99; // [rsp+0h] [rbp-3C0h]
  __int64 v100; // [rsp+0h] [rbp-3C0h]
  __int64 v101; // [rsp+20h] [rbp-3A0h]
  __int64 v102; // [rsp+20h] [rbp-3A0h]
  __int64 *v103; // [rsp+20h] [rbp-3A0h]
  __int64 v104; // [rsp+20h] [rbp-3A0h]
  __int64 v105; // [rsp+28h] [rbp-398h]
  __int64 v106; // [rsp+30h] [rbp-390h]
  _BYTE *v108; // [rsp+40h] [rbp-380h]
  char v109; // [rsp+4Bh] [rbp-375h]
  unsigned int v110; // [rsp+4Ch] [rbp-374h]
  _BYTE *v111; // [rsp+50h] [rbp-370h]
  __int64 v112; // [rsp+58h] [rbp-368h] BYREF
  _BYTE *v113; // [rsp+60h] [rbp-360h] BYREF
  __int64 v114; // [rsp+68h] [rbp-358h]
  _BYTE v115[848]; // [rsp+70h] [rbp-350h] BYREF

  v113 = v115;
  v112 = a2;
  v114 = 0x6400000000LL;
  v106 = sub_11D3A60(a1, a2, (__int64)&v113);
  v6 = (unsigned int)v114;
  if ( !(_DWORD)v114 )
  {
    v20 = (unsigned __int64)&v112;
    v30 = sub_ACADE0(*(__int64 ***)(*(_QWORD *)a1 + 8LL));
    *sub_11D31A0(*(_QWORD *)(a1 + 8), &v112) = v30;
    goto LABEL_43;
  }
  do
  {
    v7 = &v113[8 * v6];
    v108 = v113;
    if ( v113 == v7 )
      goto LABEL_39;
    v109 = 0;
    do
    {
      v8 = *((_QWORD *)v7 - 1);
      v2 = *(unsigned int *)(v8 + 40);
      if ( !(_DWORD)v2 )
        goto LABEL_23;
      v110 = 0;
      v5 = 0;
      v111 = v7;
      for ( i = 0; ; i = m )
      {
        v2 = *(_QWORD *)(v8 + 48);
        m = *(__int64 **)(v2 + 8LL * v110);
        if ( !*((_DWORD *)m + 6) )
        {
          v32 = sub_ACADE0(*(__int64 ***)(*(_QWORD *)a1 + 8LL));
          m[1] = v32;
          v33 = *(_QWORD *)(a1 + 8);
          v105 = v32;
          v34 = *(unsigned int *)(v33 + 24);
          v35 = *(_QWORD *)(v33 + 8);
          if ( (_DWORD)v34 )
          {
            v36 = (unsigned int)(v34 - 1);
            v5 = 1;
            v2 = 0;
            v3 = (unsigned int)v36 & (((unsigned int)*m >> 9) ^ ((unsigned int)*m >> 4));
            v37 = (__int64 *)(v35 + 16 * v3);
            v4 = *v37;
            if ( *m == *v37 )
            {
LABEL_49:
              v38 = v37 + 1;
LABEL_50:
              *v38 = v105;
              m[2] = (__int64)m;
              v39 = *(_DWORD *)(v106 + 24);
              *((_DWORD *)m + 6) = v39;
              *(_DWORD *)(v106 + 24) = v39 + 1;
              goto LABEL_7;
            }
            while ( v4 != -4096 )
            {
              if ( v4 == -8192 && !v2 )
                v2 = (__int64)v37;
              v3 = (unsigned int)v36 & ((_DWORD)v5 + (_DWORD)v3);
              v37 = (__int64 *)(v35 + 16LL * (unsigned int)v3);
              v4 = *v37;
              if ( *m == *v37 )
                goto LABEL_49;
              v5 = (unsigned int)(v5 + 1);
            }
            if ( !v2 )
              v2 = (__int64)v37;
            v40 = *(_DWORD *)(v33 + 16);
            ++*(_QWORD *)v33;
            v41 = v40 + 1;
            if ( 4 * v41 < (unsigned int)(3 * v34) )
            {
              v3 = (unsigned int)(v34 - *(_DWORD *)(v33 + 20) - v41);
              if ( (unsigned int)v3 > (unsigned int)v34 >> 3 )
              {
LABEL_61:
                *(_DWORD *)(v33 + 16) = v41;
                if ( *(_QWORD *)v2 != -4096 )
                  --*(_DWORD *)(v33 + 20);
                v42 = *m;
                *(_QWORD *)(v2 + 8) = 0;
                *(_QWORD *)v2 = v42;
                v38 = (__int64 *)(v2 + 8);
                goto LABEL_50;
              }
              v103 = (__int64 *)v35;
              v64 = ((((v36 | (v36 >> 1)) >> 2) | v36 | (v36 >> 1)) >> 4) | ((v36 | (v36 >> 1)) >> 2) | v36 | (v36 >> 1);
              v65 = ((((v64 >> 8) | v64) >> 16) | (v64 >> 8) | v64) + 1;
              if ( (unsigned int)v65 < 0x40 )
                LODWORD(v65) = 64;
              *(_DWORD *)(v33 + 24) = v65;
              v66 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v65, 8);
              v67 = v103;
              *(_QWORD *)(v33 + 8) = v66;
              if ( v103 )
              {
                v68 = *(unsigned int *)(v33 + 24);
                *(_QWORD *)(v33 + 16) = 0;
                v104 = 16 * v34;
                v69 = &v67[2 * v34];
                for ( j = &v66[2 * v68]; j != v66; v66 += 2 )
                {
                  if ( v66 )
                    *v66 = -4096;
                }
                v100 = (__int64)v67;
                v71 = v67;
                v72 = v8;
                do
                {
                  v73 = *v71;
                  if ( *v71 != -8192 && v73 != -4096 )
                  {
                    v74 = *(_DWORD *)(v33 + 24);
                    if ( !v74 )
                    {
                      MEMORY[0] = *v71;
                      BUG();
                    }
                    v75 = v74 - 1;
                    v76 = *(_QWORD *)(v33 + 8);
                    v77 = 1;
                    v78 = 0;
                    v79 = v75 & (((unsigned int)v73 >> 9) ^ ((unsigned int)v73 >> 4));
                    v80 = (_QWORD *)(v76 + 16LL * v79);
                    v81 = *v80;
                    if ( v73 != *v80 )
                    {
                      while ( v81 != -4096 )
                      {
                        if ( !v78 && v81 == -8192 )
                          v78 = v80;
                        v79 = v75 & (v77 + v79);
                        v80 = (_QWORD *)(v76 + 16LL * v79);
                        v81 = *v80;
                        if ( v73 == *v80 )
                          goto LABEL_104;
                        ++v77;
                      }
                      if ( v78 )
                        v80 = v78;
                    }
LABEL_104:
                    *v80 = v73;
                    v80[1] = v71[1];
                    ++*(_DWORD *)(v33 + 16);
                  }
                  v71 += 2;
                }
                while ( v69 != v71 );
                v8 = v72;
                sub_C7D6A0(v100, v104, 8);
              }
              else
              {
                v85 = *(unsigned int *)(v33 + 24);
                *(_QWORD *)(v33 + 16) = 0;
                v82 = v85;
                v3 = (__int64)&v66[2 * v85];
                if ( v66 == (_QWORD *)v3 )
                {
                  v41 = 1;
                  goto LABEL_108;
                }
                do
                {
                  if ( v66 )
                    *v66 = -4096;
                  v66 += 2;
                }
                while ( (_QWORD *)v3 != v66 );
              }
              v3 = *(_QWORD *)(v33 + 8);
              v82 = *(_DWORD *)(v33 + 24);
              v41 = *(_DWORD *)(v33 + 16) + 1;
LABEL_108:
              if ( !v82 )
              {
LABEL_189:
                ++*(_DWORD *)(v33 + 16);
                BUG();
              }
              v83 = v82 - 1;
              v4 = 1;
              v63 = 0;
              v84 = v83 & (((unsigned int)*m >> 9) ^ ((unsigned int)*m >> 4));
              v2 = v3 + 16LL * v84;
              v5 = *(_QWORD *)v2;
              if ( *(_QWORD *)v2 == *m )
                goto LABEL_61;
              while ( v5 != -4096 )
              {
                if ( !v63 && v5 == -8192 )
                  v63 = v2;
                v84 = v83 & (v4 + v84);
                v2 = v3 + 16LL * v84;
                v5 = *(_QWORD *)v2;
                if ( *m == *(_QWORD *)v2 )
                  goto LABEL_61;
                v4 = (unsigned int)(v4 + 1);
              }
LABEL_111:
              if ( v63 )
                v2 = v63;
              goto LABEL_61;
            }
          }
          else
          {
            ++*(_QWORD *)v33;
          }
          v101 = v35;
          v43 = ((((((((unsigned int)(2 * v34 - 1) | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v34 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 4)
                 | (((unsigned int)(2 * v34 - 1) | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v34 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 8)
               | (((((unsigned int)(2 * v34 - 1) | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v34 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 4)
               | (((unsigned int)(2 * v34 - 1) | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v34 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 16;
          v44 = (v43
               | (((((((unsigned int)(2 * v34 - 1) | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 2)
                   | (unsigned int)(2 * v34 - 1)
                   | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 4)
                 | (((unsigned int)(2 * v34 - 1) | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v34 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 8)
               | (((((unsigned int)(2 * v34 - 1) | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v34 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 4)
               | (((unsigned int)(2 * v34 - 1) | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v34 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v34 - 1) >> 1))
              + 1;
          if ( (unsigned int)v44 < 0x40 )
            LODWORD(v44) = 64;
          *(_DWORD *)(v33 + 24) = v44;
          v45 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v44, 8);
          v46 = v101;
          *(_QWORD *)(v33 + 8) = v45;
          if ( v101 )
          {
            v47 = *(unsigned int *)(v33 + 24);
            *(_QWORD *)(v33 + 16) = 0;
            v102 = 16 * v34;
            v48 = (__int64 *)(v46 + 16 * v34);
            for ( k = &v45[2 * v47]; k != v45; v45 += 2 )
            {
              if ( v45 )
                *v45 = -4096;
            }
            v50 = (__int64 *)v46;
            if ( (__int64 *)v46 != v48 )
            {
              v99 = v46;
              do
              {
                v51 = *v50;
                if ( *v50 != -8192 && v51 != -4096 )
                {
                  v52 = *(_DWORD *)(v33 + 24);
                  if ( !v52 )
                  {
                    MEMORY[0] = *v50;
                    BUG();
                  }
                  v53 = v52 - 1;
                  v54 = *(_QWORD *)(v33 + 8);
                  v55 = 1;
                  v56 = 0;
                  v57 = v53 & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
                  v58 = (_QWORD *)(v54 + 16LL * v57);
                  v59 = *v58;
                  if ( v51 != *v58 )
                  {
                    while ( v59 != -4096 )
                    {
                      if ( v59 == -8192 && !v56 )
                        v56 = v58;
                      v57 = v53 & (v55 + v57);
                      v58 = (_QWORD *)(v54 + 16LL * v57);
                      v59 = *v58;
                      if ( v51 == *v58 )
                        goto LABEL_78;
                      ++v55;
                    }
                    if ( v56 )
                      v58 = v56;
                  }
LABEL_78:
                  *v58 = v51;
                  v58[1] = v50[1];
                  ++*(_DWORD *)(v33 + 16);
                }
                v50 += 2;
              }
              while ( v48 != v50 );
              v46 = v99;
            }
            sub_C7D6A0(v46, v102, 8);
          }
          else
          {
            v86 = *(unsigned int *)(v33 + 24);
            *(_QWORD *)(v33 + 16) = 0;
            v60 = v86;
            v3 = (__int64)&v45[2 * v86];
            if ( v45 == (_QWORD *)v3 )
            {
              v41 = 1;
              goto LABEL_83;
            }
            do
            {
              if ( v45 )
                *v45 = -4096;
              v45 += 2;
            }
            while ( (_QWORD *)v3 != v45 );
          }
          v3 = *(_QWORD *)(v33 + 8);
          v60 = *(_DWORD *)(v33 + 24);
          v41 = *(_DWORD *)(v33 + 16) + 1;
LABEL_83:
          if ( !v60 )
            goto LABEL_189;
          v61 = v60 - 1;
          v62 = v61 & (((unsigned int)*m >> 9) ^ ((unsigned int)*m >> 4));
          v2 = v3 + 16LL * v62;
          v5 = *(_QWORD *)v2;
          if ( *m == *(_QWORD *)v2 )
            goto LABEL_61;
          v4 = 1;
          v63 = 0;
          while ( v5 != -4096 )
          {
            if ( v5 == -8192 && !v63 )
              v63 = v2;
            v62 = v61 & (v4 + v62);
            v2 = v3 + 16LL * v62;
            v5 = *(_QWORD *)v2;
            if ( *m == *(_QWORD *)v2 )
              goto LABEL_61;
            v4 = (unsigned int)(v4 + 1);
          }
          goto LABEL_111;
        }
LABEL_7:
        if ( i && m != i )
        {
          v2 = *((unsigned int *)m + 6);
          v11 = m;
          v3 = *((unsigned int *)i + 6);
          for ( m = i; ; v3 = *((unsigned int *)m + 6) )
          {
            while ( (int)v3 >= (int)v2 )
            {
              while ( (int)v3 > (int)v2 )
              {
                v11 = (__int64 *)v11[4];
                if ( !v11 )
                  goto LABEL_18;
                v2 = *((unsigned int *)v11 + 6);
              }
              if ( m == v11 )
                goto LABEL_18;
            }
            m = (__int64 *)m[4];
            if ( !m )
              break;
          }
          m = v11;
        }
LABEL_18:
        if ( *(_DWORD *)(v8 + 40) == ++v110 )
          break;
      }
      v7 = v111;
      if ( m && *(__int64 **)(v8 + 32) != m )
      {
        *(_QWORD *)(v8 + 32) = m;
        v109 = 1;
      }
LABEL_23:
      v7 -= 8;
    }
    while ( v108 != v7 );
    v6 = (unsigned int)v114;
  }
  while ( v109 );
  v5 = (__int64)v113;
  v12 = &v113[8 * (unsigned int)v114];
  do
  {
    if ( (_BYTE *)v5 == v12 )
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
        v4 = *(_QWORD *)(v2 + 48);
        v17 = v4 + 8LL * (unsigned int)(v15 - 1) + 8;
        while ( 1 )
        {
          v18 = *(_QWORD *)v4;
          if ( v16 != *(_QWORD *)v4 )
            break;
LABEL_145:
          v4 += 8;
          if ( v17 == v4 )
            goto LABEL_146;
        }
        while ( *(_QWORD *)(v18 + 16) != v18 )
        {
          v18 = *(_QWORD *)(v18 + 32);
          if ( v16 == v18 )
            goto LABEL_145;
        }
        v19 = *(_QWORD *)(v3 - 8);
LABEL_36:
        *(_QWORD *)(v2 + 16) = v19;
        v13 = 1;
        goto LABEL_37;
      }
LABEL_146:
      v19 = *(_QWORD *)(v16 + 16);
      if ( v14 != v19 )
        goto LABEL_36;
LABEL_37:
      v3 -= 8;
    }
    while ( v5 != v3 );
  }
  while ( v13 );
LABEL_39:
  sub_11D52F0(a1, (__int64)&v113, v2, v3, v4, v5);
  v20 = *(unsigned int *)(a1 + 48);
  v21 = a1 + 24;
  if ( !(_DWORD)v20 )
  {
    ++*(_QWORD *)(a1 + 24);
    goto LABEL_173;
  }
  v22 = v112;
  v23 = 0;
  v24 = 1;
  v25 = *(_QWORD *)(a1 + 32);
  v26 = (v20 - 1) & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
  v27 = (__int64 *)(v25 + 16LL * v26);
  v28 = *v27;
  if ( v112 == *v27 )
  {
LABEL_41:
    v29 = v27[1];
    goto LABEL_42;
  }
  while ( v28 != -4096 )
  {
    if ( v28 == -8192 && !v23 )
      v23 = v27;
    v26 = (v20 - 1) & (v24 + v26);
    v27 = (__int64 *)(v25 + 16LL * v26);
    v28 = *v27;
    if ( v112 == *v27 )
      goto LABEL_41;
    ++v24;
  }
  if ( v23 )
    v27 = v23;
  ++*(_QWORD *)(a1 + 24);
  v87 = *(_DWORD *)(a1 + 40) + 1;
  if ( 4 * v87 >= (unsigned int)(3 * v20) )
  {
LABEL_173:
    sub_11D3880(v21, 2 * v20);
    v94 = *(_DWORD *)(a1 + 48);
    if ( v94 )
    {
      v22 = v112;
      v95 = v94 - 1;
      v96 = *(_QWORD *)(a1 + 32);
      v87 = *(_DWORD *)(a1 + 40) + 1;
      v20 = v95 & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
      v27 = (__int64 *)(v96 + 16 * v20);
      v97 = *v27;
      if ( *v27 == v112 )
        goto LABEL_163;
      v98 = 1;
      v91 = 0;
      while ( v97 != -4096 )
      {
        if ( !v91 && v97 == -8192 )
          v91 = v27;
        v20 = v95 & (unsigned int)(v98 + v20);
        v27 = (__int64 *)(v96 + 16LL * (unsigned int)v20);
        v97 = *v27;
        if ( v112 == *v27 )
          goto LABEL_163;
        ++v98;
      }
      goto LABEL_169;
    }
LABEL_187:
    ++*(_DWORD *)(a1 + 40);
    BUG();
  }
  if ( (int)v20 - *(_DWORD *)(a1 + 44) - v87 <= (unsigned int)v20 >> 3 )
  {
    sub_11D3880(v21, v20);
    v88 = *(_DWORD *)(a1 + 48);
    if ( v88 )
    {
      v22 = v112;
      v89 = v88 - 1;
      v90 = *(_QWORD *)(a1 + 32);
      v91 = 0;
      v87 = *(_DWORD *)(a1 + 40) + 1;
      v92 = 1;
      v20 = v89 & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
      v27 = (__int64 *)(v90 + 16 * v20);
      v93 = *v27;
      if ( *v27 == v112 )
        goto LABEL_163;
      while ( v93 != -4096 )
      {
        if ( !v91 && v93 == -8192 )
          v91 = v27;
        v20 = v89 & (unsigned int)(v92 + v20);
        v27 = (__int64 *)(v90 + 16LL * (unsigned int)v20);
        v93 = *v27;
        if ( v112 == *v27 )
          goto LABEL_163;
        ++v92;
      }
LABEL_169:
      if ( v91 )
        v27 = v91;
      goto LABEL_163;
    }
    goto LABEL_187;
  }
LABEL_163:
  *(_DWORD *)(a1 + 40) = v87;
  if ( *v27 != -4096 )
    --*(_DWORD *)(a1 + 44);
  *v27 = v22;
  v27[1] = 0;
  v29 = 0;
LABEL_42:
  v30 = *(_QWORD *)(*(_QWORD *)(v29 + 16) + 8LL);
LABEL_43:
  if ( v113 != v115 )
    _libc_free(v113, v20);
  return v30;
}
