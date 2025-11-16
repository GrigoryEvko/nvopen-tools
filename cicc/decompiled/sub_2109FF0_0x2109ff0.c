// Function: sub_2109FF0
// Address: 0x2109ff0
//
__int64 __fastcall sub_2109FF0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  _BYTE *v3; // r14
  __int64 v4; // r10
  __int64 i; // r12
  __int64 k; // rbx
  int v7; // edx
  __int64 v8; // rax
  int v9; // ecx
  unsigned __int64 v10; // r9
  __int64 v11; // r15
  _BYTE *v12; // r11
  char v13; // bl
  _BYTE *v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdi
  int v17; // eax
  __int64 v18; // rsi
  __int64 *v19; // r8
  __int64 v20; // r10
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // r15
  __int64 *v25; // rax
  __int64 v26; // rdx
  int v27; // r8d
  __int64 v28; // r15
  unsigned int v29; // r13d
  __int64 *v30; // r9
  unsigned int v31; // edx
  __int64 *v32; // rax
  __int64 v33; // rsi
  int v34; // eax
  __int64 *v35; // r11
  int v36; // edi
  int v37; // edx
  __int64 v38; // rdx
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rax
  _QWORD *v41; // rax
  __int64 *v42; // r9
  __int64 *v43; // r13
  _QWORD *j; // rdx
  __int64 *v45; // rax
  __int64 v46; // r11
  int v47; // esi
  int v48; // esi
  __int64 v49; // r8
  int v50; // r9d
  __int64 *v51; // r10
  unsigned int v52; // ecx
  __int64 *v53; // rdx
  __int64 v54; // rdi
  _QWORD *v55; // rcx
  int v56; // r13d
  int v57; // r13d
  unsigned int v58; // esi
  __int64 v59; // r9
  int v60; // r8d
  __int64 *v61; // rdi
  int v62; // r13d
  int v63; // r13d
  __int64 v64; // r11
  __int64 *v65; // rsi
  int v66; // edi
  unsigned int v67; // ecx
  __int64 v68; // r8
  __int64 v69; // rcx
  unsigned int v70; // esi
  __int64 v71; // rdx
  __int64 v72; // r11
  unsigned int v73; // ecx
  __int64 *v74; // rax
  __int64 v75; // r9
  __int64 v76; // rax
  unsigned int v77; // r12d
  __int64 v79; // r14
  size_t v80; // rbx
  __int64 v81; // r13
  __int64 *v82; // rax
  __int64 v83; // rdx
  int v84; // r8d
  __int64 *v85; // rdi
  int v86; // edi
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
  __int64 v101; // [rsp+0h] [rbp-3B0h]
  __int64 v102; // [rsp+18h] [rbp-398h]
  __int64 v103; // [rsp+20h] [rbp-390h]
  size_t v104; // [rsp+28h] [rbp-388h]
  int v105; // [rsp+28h] [rbp-388h]
  __int64 *v106; // [rsp+28h] [rbp-388h]
  __int64 v107; // [rsp+28h] [rbp-388h]
  unsigned __int64 v109; // [rsp+38h] [rbp-378h]
  char v110; // [rsp+43h] [rbp-36Dh]
  unsigned int v111; // [rsp+44h] [rbp-36Ch]
  __int64 v112; // [rsp+48h] [rbp-368h] BYREF
  _BYTE *v113; // [rsp+50h] [rbp-360h] BYREF
  __int64 v114; // [rsp+58h] [rbp-358h]
  _BYTE v115[848]; // [rsp+60h] [rbp-350h] BYREF

  v113 = v115;
  v112 = a2;
  v114 = 0x6400000000LL;
  v102 = sub_2109140((__int64)a1, a2, (__int64)&v113);
  v2 = (unsigned int)v114;
  if ( !(_DWORD)v114 )
  {
    v79 = *(_QWORD *)(*a1 + 32);
    v80 = *(_QWORD *)(*a1 + 40);
    v81 = *(_QWORD *)(*a1 + 16);
    v82 = (__int64 *)sub_1DD5EE0(v112);
    sub_21072E0(9u, v112, v82, v81, v80, v79);
    v77 = *(_DWORD *)(*(_QWORD *)(v83 + 32) + 8LL);
    *((_DWORD *)sub_2107730(a1[1], &v112) + 2) = v77;
    goto LABEL_107;
  }
  do
  {
    v3 = &v113[8 * v2];
    v109 = (unsigned __int64)v113;
    if ( v113 == v3 )
    {
      v11 = (__int64)a1;
      goto LABEL_103;
    }
    v110 = 0;
    do
    {
      v4 = *((_QWORD *)v3 - 1);
      if ( !*(_DWORD *)(v4 + 40) )
        goto LABEL_23;
      v111 = 0;
      for ( i = 0; ; i = k )
      {
        k = *(_QWORD *)(*(_QWORD *)(v4 + 48) + 8LL * v111);
        if ( !*(_DWORD *)(k + 24) )
        {
          v23 = *(_QWORD *)k;
          v101 = v4;
          v24 = *(_QWORD *)(*a1 + 16);
          v103 = *(_QWORD *)(*a1 + 32);
          v104 = *(_QWORD *)(*a1 + 40);
          v25 = (__int64 *)sub_1DD5EE0(*(_QWORD *)k);
          sub_21072E0(9u, v23, v25, v24, v104, v103);
          v4 = v101;
          v27 = *(_DWORD *)(*(_QWORD *)(v26 + 32) + 8LL);
          *(_DWORD *)(k + 8) = v27;
          v28 = a1[1];
          v29 = *(_DWORD *)(v28 + 24);
          v30 = *(__int64 **)(v28 + 8);
          if ( v29 )
          {
            v31 = (v29 - 1) & (((unsigned int)*(_QWORD *)k >> 9) ^ ((unsigned int)*(_QWORD *)k >> 4));
            v32 = &v30[2 * v31];
            v33 = *v32;
            if ( *(_QWORD *)k == *v32 )
            {
LABEL_43:
              *((_DWORD *)v32 + 2) = v27;
              *(_QWORD *)(k + 16) = k;
              v34 = *(_DWORD *)(v102 + 24);
              *(_DWORD *)(k + 24) = v34;
              *(_DWORD *)(v102 + 24) = v34 + 1;
              goto LABEL_7;
            }
            v105 = 1;
            v35 = 0;
            while ( v33 != -8 )
            {
              if ( !v35 && v33 == -16 )
                v35 = v32;
              v31 = (v29 - 1) & (v105 + v31);
              v32 = &v30[2 * v31];
              v33 = *v32;
              if ( *(_QWORD *)k == *v32 )
                goto LABEL_43;
              ++v105;
            }
            v36 = *(_DWORD *)(v28 + 16);
            if ( v35 )
              v32 = v35;
            ++*(_QWORD *)v28;
            v37 = v36 + 1;
            if ( 4 * (v36 + 1) < 3 * v29 )
            {
              if ( v29 - *(_DWORD *)(v28 + 20) - v37 <= v29 >> 3 )
              {
                sub_1DA35E0(v28, v29);
                v62 = *(_DWORD *)(v28 + 24);
                if ( !v62 )
                  goto LABEL_161;
                v63 = v62 - 1;
                v64 = *(_QWORD *)(v28 + 8);
                v65 = 0;
                v4 = v101;
                v37 = *(_DWORD *)(v28 + 16) + 1;
                v66 = 1;
                v67 = v63 & (((unsigned int)*(_QWORD *)k >> 9) ^ ((unsigned int)*(_QWORD *)k >> 4));
                v32 = (__int64 *)(v64 + 16LL * v67);
                v68 = *v32;
                if ( *(_QWORD *)k != *v32 )
                {
                  while ( v68 != -8 )
                  {
                    if ( v68 == -16 && !v65 )
                      v65 = v32;
                    v67 = v63 & (v66 + v67);
                    v32 = (__int64 *)(v64 + 16LL * v67);
                    v68 = *v32;
                    if ( *(_QWORD *)k == *v32 )
                      goto LABEL_50;
                    ++v66;
                  }
                  if ( v65 )
                    v32 = v65;
                }
              }
LABEL_50:
              *(_DWORD *)(v28 + 16) = v37;
              if ( *v32 != -8 )
                --*(_DWORD *)(v28 + 20);
              v38 = *(_QWORD *)k;
              *((_DWORD *)v32 + 2) = 0;
              *v32 = v38;
              v27 = *(_DWORD *)(k + 8);
              goto LABEL_43;
            }
          }
          else
          {
            ++*(_QWORD *)v28;
          }
          v106 = v30;
          v39 = ((((((((2 * v29 - 1) | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 2)
                   | (2 * v29 - 1)
                   | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 4)
                 | (((2 * v29 - 1) | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 2)
                 | (2 * v29 - 1)
                 | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 8)
               | (((((2 * v29 - 1) | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 2)
                 | (2 * v29 - 1)
                 | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 4)
               | (((2 * v29 - 1) | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 2)
               | (2 * v29 - 1)
               | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 16;
          v40 = (v39
               | (((((((2 * v29 - 1) | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 2)
                   | (2 * v29 - 1)
                   | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 4)
                 | (((2 * v29 - 1) | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 2)
                 | (2 * v29 - 1)
                 | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 8)
               | (((((2 * v29 - 1) | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 2)
                 | (2 * v29 - 1)
                 | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 4)
               | (((2 * v29 - 1) | ((unsigned __int64)(2 * v29 - 1) >> 1)) >> 2)
               | (2 * v29 - 1)
               | ((unsigned __int64)(2 * v29 - 1) >> 1))
              + 1;
          if ( (unsigned int)v40 < 0x40 )
            LODWORD(v40) = 64;
          *(_DWORD *)(v28 + 24) = v40;
          v41 = (_QWORD *)sub_22077B0(16LL * (unsigned int)v40);
          v42 = v106;
          v4 = v101;
          *(_QWORD *)(v28 + 8) = v41;
          if ( v106 )
          {
            *(_QWORD *)(v28 + 16) = 0;
            v43 = &v106[2 * v29];
            for ( j = &v41[2 * *(unsigned int *)(v28 + 24)]; j != v41; v41 += 2 )
            {
              if ( v41 )
                *v41 = -8;
            }
            v45 = v106;
            if ( v106 != v43 )
            {
              do
              {
                v46 = *v45;
                if ( *v45 != -16 && v46 != -8 )
                {
                  v47 = *(_DWORD *)(v28 + 24);
                  if ( !v47 )
                  {
                    MEMORY[0] = *v45;
                    BUG();
                  }
                  v48 = v47 - 1;
                  v49 = *(_QWORD *)(v28 + 8);
                  v50 = 1;
                  v51 = 0;
                  v52 = v48 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
                  v53 = (__int64 *)(v49 + 16LL * v52);
                  v54 = *v53;
                  if ( v46 != *v53 )
                  {
                    while ( v54 != -8 )
                    {
                      if ( !v51 && v54 == -16 )
                        v51 = v53;
                      v52 = v48 & (v50 + v52);
                      v53 = (__int64 *)(v49 + 16LL * v52);
                      v54 = *v53;
                      if ( v46 == *v53 )
                        goto LABEL_66;
                      ++v50;
                    }
                    if ( v51 )
                      v53 = v51;
                  }
LABEL_66:
                  *v53 = v46;
                  *((_DWORD *)v53 + 2) = *((_DWORD *)v45 + 2);
                  ++*(_DWORD *)(v28 + 16);
                }
                v45 += 2;
              }
              while ( v43 != v45 );
              v4 = v101;
              v42 = v106;
            }
            v107 = v4;
            j___libc_free_0(v42);
            v55 = *(_QWORD **)(v28 + 8);
            v56 = *(_DWORD *)(v28 + 24);
            v4 = v107;
            v37 = *(_DWORD *)(v28 + 16) + 1;
          }
          else
          {
            v69 = *(unsigned int *)(v28 + 24);
            *(_QWORD *)(v28 + 16) = 0;
            v56 = v69;
            v55 = &v41[2 * v69];
            if ( v41 == v55 )
            {
              v37 = 1;
            }
            else
            {
              do
              {
                if ( v41 )
                  *v41 = -8;
                v41 += 2;
              }
              while ( v55 != v41 );
              v55 = *(_QWORD **)(v28 + 8);
              v56 = *(_DWORD *)(v28 + 24);
              v37 = *(_DWORD *)(v28 + 16) + 1;
            }
          }
          if ( !v56 )
          {
LABEL_161:
            ++*(_DWORD *)(v28 + 16);
            BUG();
          }
          v57 = v56 - 1;
          v58 = v57 & (((unsigned int)*(_QWORD *)k >> 9) ^ ((unsigned int)*(_QWORD *)k >> 4));
          v32 = &v55[2 * v58];
          v59 = *v32;
          if ( *(_QWORD *)k != *v32 )
          {
            v60 = 1;
            v61 = 0;
            while ( v59 != -8 )
            {
              if ( !v61 && v59 == -16 )
                v61 = v32;
              v58 = v57 & (v60 + v58);
              v32 = &v55[2 * v58];
              v59 = *v32;
              if ( *(_QWORD *)k == *v32 )
                goto LABEL_50;
              ++v60;
            }
            if ( v61 )
              v32 = v61;
          }
          goto LABEL_50;
        }
LABEL_7:
        if ( i && k != i )
        {
          v7 = *(_DWORD *)(k + 24);
          v8 = k;
          v9 = *(_DWORD *)(i + 24);
          for ( k = i; ; v9 = *(_DWORD *)(k + 24) )
          {
            while ( v9 >= v7 )
            {
              while ( v7 < v9 )
              {
                v8 = *(_QWORD *)(v8 + 32);
                if ( !v8 )
                  goto LABEL_18;
                v7 = *(_DWORD *)(v8 + 24);
              }
              if ( k == v8 )
                goto LABEL_18;
            }
            k = *(_QWORD *)(k + 32);
            if ( !k )
              break;
          }
          k = v8;
        }
LABEL_18:
        if ( *(_DWORD *)(v4 + 40) == ++v111 )
          break;
      }
      if ( k && *(_QWORD *)(v4 + 32) != k )
      {
        *(_QWORD *)(v4 + 32) = k;
        v110 = 1;
      }
LABEL_23:
      v3 -= 8;
    }
    while ( (_BYTE *)v109 != v3 );
    v2 = (unsigned int)v114;
  }
  while ( v110 );
  v10 = (unsigned __int64)v113;
  v11 = (__int64)a1;
  v12 = &v113[8 * (unsigned int)v114];
  do
  {
    if ( (_BYTE *)v10 == v12 )
      break;
    v13 = 0;
    v14 = v12;
    do
    {
      v15 = *((_QWORD *)v14 - 1);
      v16 = *(_QWORD *)(v15 + 16);
      if ( v15 == v16 )
        goto LABEL_37;
      v17 = *(_DWORD *)(v15 + 40);
      v18 = *(_QWORD *)(v15 + 32);
      if ( v17 )
      {
        v19 = *(__int64 **)(v15 + 48);
        v20 = (__int64)&v19[(unsigned int)(v17 - 1) + 1];
        while ( 1 )
        {
          v21 = *v19;
          if ( v18 != *v19 )
            break;
LABEL_112:
          if ( (__int64 *)v20 == ++v19 )
            goto LABEL_113;
        }
        while ( *(_QWORD *)(v21 + 16) != v21 )
        {
          v21 = *(_QWORD *)(v21 + 32);
          if ( v18 == v21 )
            goto LABEL_112;
        }
        v22 = *((_QWORD *)v14 - 1);
LABEL_36:
        *(_QWORD *)(v15 + 16) = v22;
        v13 = 1;
        goto LABEL_37;
      }
LABEL_113:
      v22 = *(_QWORD *)(v18 + 16);
      if ( v16 != v22 )
        goto LABEL_36;
LABEL_37:
      v14 -= 8;
    }
    while ( (_BYTE *)v10 != v14 );
  }
  while ( v13 );
LABEL_103:
  sub_21084A0(v11, &v113);
  v70 = *(_DWORD *)(v11 + 48);
  if ( !v70 )
  {
    ++*(_QWORD *)(v11 + 24);
    goto LABEL_141;
  }
  v71 = v112;
  v72 = *(_QWORD *)(v11 + 32);
  v73 = (v70 - 1) & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
  v74 = (__int64 *)(v72 + 16LL * v73);
  v75 = *v74;
  if ( *v74 == v112 )
  {
LABEL_105:
    v76 = v74[1];
    goto LABEL_106;
  }
  v84 = 1;
  v85 = 0;
  while ( v75 != -8 )
  {
    if ( v75 == -16 && !v85 )
      v85 = v74;
    v73 = (v70 - 1) & (v84 + v73);
    v74 = (__int64 *)(v72 + 16LL * v73);
    v75 = *v74;
    if ( v112 == *v74 )
      goto LABEL_105;
    ++v84;
  }
  if ( v85 )
    v74 = v85;
  v86 = *(_DWORD *)(v11 + 40);
  ++*(_QWORD *)(v11 + 24);
  v87 = v86 + 1;
  if ( 4 * (v86 + 1) >= 3 * v70 )
  {
LABEL_141:
    sub_2107BB0(v11 + 24, 2 * v70);
    v95 = *(_DWORD *)(v11 + 48);
    if ( v95 )
    {
      v71 = v112;
      v96 = v95 - 1;
      v97 = *(_QWORD *)(v11 + 32);
      v87 = *(_DWORD *)(v11 + 40) + 1;
      v98 = v96 & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
      v74 = (__int64 *)(v97 + 16LL * v98);
      v99 = *v74;
      if ( *v74 == v112 )
        goto LABEL_131;
      v100 = 1;
      v91 = 0;
      while ( v99 != -8 )
      {
        if ( v99 == -16 && !v91 )
          v91 = v74;
        v98 = v96 & (v100 + v98);
        v74 = (__int64 *)(v97 + 16LL * v98);
        v99 = *v74;
        if ( v112 == *v74 )
          goto LABEL_131;
        ++v100;
      }
      goto LABEL_137;
    }
LABEL_160:
    ++*(_DWORD *)(v11 + 40);
    BUG();
  }
  if ( v70 - *(_DWORD *)(v11 + 44) - v87 <= v70 >> 3 )
  {
    sub_2107BB0(v11 + 24, v70);
    v88 = *(_DWORD *)(v11 + 48);
    if ( v88 )
    {
      v89 = v88 - 1;
      v90 = *(_QWORD *)(v11 + 32);
      v91 = 0;
      v71 = v112;
      v87 = *(_DWORD *)(v11 + 40) + 1;
      v92 = 1;
      v93 = v89 & (((unsigned int)v112 >> 9) ^ ((unsigned int)v112 >> 4));
      v74 = (__int64 *)(v90 + 16LL * v93);
      v94 = *v74;
      if ( *v74 == v112 )
        goto LABEL_131;
      while ( v94 != -8 )
      {
        if ( v94 == -16 && !v91 )
          v91 = v74;
        v93 = v89 & (v92 + v93);
        v74 = (__int64 *)(v90 + 16LL * v93);
        v94 = *v74;
        if ( v112 == *v74 )
          goto LABEL_131;
        ++v92;
      }
LABEL_137:
      if ( v91 )
        v74 = v91;
      goto LABEL_131;
    }
    goto LABEL_160;
  }
LABEL_131:
  *(_DWORD *)(v11 + 40) = v87;
  if ( *v74 != -8 )
    --*(_DWORD *)(v11 + 44);
  *v74 = v71;
  v74[1] = 0;
  v76 = 0;
LABEL_106:
  v77 = *(_DWORD *)(*(_QWORD *)(v76 + 16) + 8LL);
LABEL_107:
  if ( v113 != v115 )
    _libc_free((unsigned __int64)v113);
  return v77;
}
