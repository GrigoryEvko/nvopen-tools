// Function: sub_2C3A0F0
// Address: 0x2c3a0f0
//
_QWORD *__fastcall sub_2C3A0F0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  __int64 v7; // rax
  __int64 *v8; // rbx
  unsigned int v9; // esi
  __int64 *v10; // r13
  int v11; // r11d
  char *v12; // r10
  unsigned int v13; // eax
  __int64 *v14; // rdi
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r14
  __int64 *v23; // rbx
  __int64 *v24; // r15
  unsigned __int64 v25; // rcx
  int v26; // eax
  __int64 *v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // r12
  __int64 v30; // r13
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rax
  _QWORD *v33; // rax
  __int64 v34; // r11
  __int64 *v35; // rsi
  _QWORD *i; // rdx
  __int64 *v37; // rax
  __int64 v38; // r12
  int v39; // edx
  _QWORD *v40; // rdi
  __int64 v41; // r8
  int v42; // ecx
  int v43; // edx
  int v44; // esi
  unsigned int v45; // ecx
  __int64 v46; // r12
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  __int64 v49; // rdx
  _QWORD *v50; // rax
  _QWORD *v51; // rdx
  unsigned __int64 v53; // rax
  __int64 v54; // rax
  _QWORD *v55; // rax
  __int64 v56; // r11
  __int64 *v57; // rsi
  _QWORD *j; // rdx
  __int64 *v59; // rax
  __int64 v60; // r12
  int v61; // edx
  _QWORD *v62; // rdi
  __int64 v63; // r8
  int v64; // ecx
  int v65; // esi
  int v66; // r11d
  __int64 v67; // r10
  unsigned int v68; // ecx
  _QWORD *v69; // rsi
  _QWORD *v70; // rdx
  _QWORD *v71; // r10
  _QWORD *v72; // rsi
  _QWORD *v73; // rdx
  _QWORD *v74; // r10
  int v75; // r11d
  __int64 v76; // r14
  __int64 v77; // rax
  unsigned __int64 v78; // rdx
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // r8
  __int64 v82; // r9
  unsigned int v83; // eax
  int v84; // r12d
  unsigned int v85; // eax
  int v87; // [rsp+20h] [rbp-80h]
  int v88; // [rsp+20h] [rbp-80h]
  int v89; // [rsp+24h] [rbp-7Ch]
  __int64 v90; // [rsp+28h] [rbp-78h]
  char *v91; // [rsp+30h] [rbp-70h] BYREF
  __int64 v92; // [rsp+38h] [rbp-68h]
  __int64 v93; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v94; // [rsp+48h] [rbp-58h]
  __int64 v95; // [rsp+50h] [rbp-50h]
  __int64 v96; // [rsp+58h] [rbp-48h]
  char *v97; // [rsp+60h] [rbp-40h] BYREF
  __int64 v98; // [rsp+68h] [rbp-38h]
  _BYTE v99[48]; // [rsp+70h] [rbp-30h] BYREF

  v6 = 0;
  v7 = *(unsigned int *)(a2 + 24);
  v8 = *(__int64 **)(a2 + 16);
  v93 = 0;
  v9 = 0;
  v10 = &v8[v7];
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = v99;
  v98 = 0;
  if ( v10 == v8 )
  {
    v93 = 1;
    goto LABEL_56;
  }
  while ( 1 )
  {
    if ( v9 )
    {
      a6 = v9 - 1;
      v11 = 1;
      v12 = 0;
      v13 = a6 & (((unsigned int)*v8 >> 9) ^ ((unsigned int)*v8 >> 4));
      v14 = (__int64 *)(v6 + 8LL * v13);
      a5 = *v14;
      if ( *v8 == *v14 )
        goto LABEL_4;
      while ( a5 != -4096 )
      {
        if ( a5 == -8192 && !v12 )
          v12 = (char *)v14;
        v13 = a6 & (v11 + v13);
        v14 = (__int64 *)(v6 + 8LL * v13);
        a5 = *v14;
        if ( *v8 == *v14 )
          goto LABEL_4;
        ++v11;
      }
      if ( !v12 )
        v12 = (char *)v14;
      ++v93;
      v15 = v95 + 1;
      v91 = v12;
      if ( 4 * ((int)v95 + 1) < 3 * v9 )
      {
        if ( v9 - (v15 + HIDWORD(v95)) > v9 >> 3 )
          goto LABEL_132;
        goto LABEL_9;
      }
    }
    else
    {
      ++v93;
      v91 = 0;
    }
    v9 *= 2;
LABEL_9:
    sub_2C39F20((__int64)&v93, v9);
    sub_2C2F640((__int64)&v93, v8, &v91);
    v12 = v91;
    v15 = v95 + 1;
LABEL_132:
    LODWORD(v95) = v15;
    if ( *(_QWORD *)v12 != -4096 )
      --HIDWORD(v95);
    v76 = *v8;
    *(_QWORD *)v12 = *v8;
    v77 = (unsigned int)v98;
    v78 = (unsigned int)v98 + 1LL;
    if ( v78 > HIDWORD(v98) )
    {
      sub_C8D5F0((__int64)&v97, v99, v78, 8u, a5, a6);
      v77 = (unsigned int)v98;
    }
    *(_QWORD *)&v97[8 * v77] = v76;
    LODWORD(v98) = v98 + 1;
LABEL_4:
    if ( v10 == ++v8 )
      break;
    v6 = (__int64)v94;
    v9 = v96;
  }
  if ( (_DWORD)v98 )
  {
    v89 = 0;
    v16 = 0;
    while ( 1 )
    {
      v17 = *(_QWORD *)&v97[8 * v16];
      if ( !v17 )
        BUG();
      if ( (unsigned int)*(unsigned __int8 *)(v17 - 32) - 29 > 7 )
      {
        v18 = *(_QWORD *)(v17 - 24);
        v19 = v18 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (v18 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          if ( (v18 & 4) != 0 )
          {
            v20 = *(_QWORD *)v19;
            v21 = 8LL * *(unsigned int *)(v19 + 8);
          }
          else
          {
            v20 = v17 - 24;
            v21 = 8;
          }
          v90 = v20 + v21;
          if ( v20 + v21 != v20 )
            break;
        }
      }
LABEL_48:
      v16 = (unsigned int)++v89;
      if ( (_DWORD)v98 == v89 )
        goto LABEL_49;
    }
    v22 = v20;
    while ( 1 )
    {
      v23 = *(__int64 **)(*(_QWORD *)v22 + 16LL);
      v24 = &v23[*(unsigned int *)(*(_QWORD *)v22 + 24LL)];
      if ( v23 != v24 )
        break;
LABEL_47:
      v22 += 8;
      if ( v90 == v22 )
        goto LABEL_48;
    }
    while ( 2 )
    {
      while ( 2 )
      {
        v29 = (unsigned int)v96;
        v30 = (__int64)v94;
        if ( !(_DWORD)v96 )
        {
          ++v93;
          goto LABEL_25;
        }
        v25 = (unsigned int)(v96 - 1);
        a6 = 1;
        a5 = 0;
        v26 = v25 & (((unsigned int)*v23 >> 9) ^ ((unsigned int)*v23 >> 4));
        v27 = &v94[v26];
        v28 = *v27;
        if ( *v27 == *v23 )
        {
LABEL_22:
          if ( ++v23 == v24 )
            goto LABEL_47;
          continue;
        }
        break;
      }
      while ( v28 != -4096 )
      {
        if ( v28 != -8192 || a5 )
          v27 = (__int64 *)a5;
        a5 = (unsigned int)(a6 + 1);
        v26 = v25 & (a6 + v26);
        v28 = v94[v26];
        if ( *v23 == v28 )
          goto LABEL_22;
        a6 = (unsigned int)a5;
        a5 = (__int64)v27;
        v27 = &v94[v26];
      }
      if ( !a5 )
        a5 = (__int64)v27;
      ++v93;
      v43 = v95 + 1;
      if ( 4 * ((int)v95 + 1) >= (unsigned int)(3 * v96) )
      {
LABEL_25:
        v31 = ((((((((unsigned int)(2 * v96 - 1) | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v96 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 4)
               | (((unsigned int)(2 * v96 - 1) | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v96 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 8)
             | (((((unsigned int)(2 * v96 - 1) | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v96 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 4)
             | (((unsigned int)(2 * v96 - 1) | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v96 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 16;
        v32 = (v31
             | (((((((unsigned int)(2 * v96 - 1) | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 2)
                 | (unsigned int)(2 * v96 - 1)
                 | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 4)
               | (((unsigned int)(2 * v96 - 1) | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v96 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 8)
             | (((((unsigned int)(2 * v96 - 1) | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 2)
               | (unsigned int)(2 * v96 - 1)
               | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 4)
             | (((unsigned int)(2 * v96 - 1) | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1)) >> 2)
             | (unsigned int)(2 * v96 - 1)
             | ((unsigned __int64)(unsigned int)(2 * v96 - 1) >> 1))
            + 1;
        if ( (unsigned int)v32 < 0x40 )
          LODWORD(v32) = 64;
        LODWORD(v96) = v32;
        v33 = (_QWORD *)sub_C7D670(8LL * (unsigned int)v32, 8);
        v94 = v33;
        if ( v30 )
        {
          v34 = 8 * v29;
          v95 = 0;
          v35 = (__int64 *)(v30 + 8 * v29);
          for ( i = &v33[(unsigned int)v96]; i != v33; ++v33 )
          {
            if ( v33 )
              *v33 = -4096;
          }
          v37 = (__int64 *)v30;
          if ( (__int64 *)v30 != v35 )
          {
            do
            {
              v38 = *v37;
              if ( *v37 != -4096 && v38 != -8192 )
              {
                if ( !(_DWORD)v96 )
                {
                  MEMORY[0] = *v37;
                  BUG();
                }
                v39 = (v96 - 1) & (((unsigned int)v38 >> 9) ^ ((unsigned int)v38 >> 4));
                v40 = &v94[v39];
                v41 = *v40;
                if ( v38 != *v40 )
                {
                  v87 = 1;
                  v71 = 0;
                  while ( v41 != -4096 )
                  {
                    if ( v41 != -8192 || v71 )
                      v40 = v71;
                    v39 = (v96 - 1) & (v87 + v39);
                    v41 = v94[v39];
                    if ( v38 == v41 )
                    {
                      v40 = &v94[v39];
                      goto LABEL_37;
                    }
                    ++v87;
                    v71 = v40;
                    v40 = &v94[v39];
                  }
                  if ( v71 )
                    v40 = v71;
                }
LABEL_37:
                *v40 = v38;
                LODWORD(v95) = v95 + 1;
              }
              ++v37;
            }
            while ( v35 != v37 );
          }
          sub_C7D6A0(v30, v34, 8);
          v33 = v94;
          v42 = v96;
          v43 = v95 + 1;
        }
        else
        {
          v95 = 0;
          v69 = &v33[(unsigned int)v96];
          v42 = v96;
          if ( v33 != v69 )
          {
            v70 = v33;
            do
            {
              if ( v70 )
                *v70 = -4096;
              ++v70;
            }
            while ( v69 != v70 );
          }
          v43 = 1;
        }
        if ( !v42 )
          goto LABEL_170;
        v44 = v42 - 1;
        v45 = (v42 - 1) & (((unsigned int)*v23 >> 9) ^ ((unsigned int)*v23 >> 4));
        a5 = (__int64)&v33[v45];
        a6 = *(_QWORD *)a5;
        if ( *(_QWORD *)a5 != *v23 )
        {
          v75 = 1;
          v67 = 0;
          while ( a6 != -4096 )
          {
            if ( !v67 && a6 == -8192 )
              v67 = a5;
            v45 = v44 & (v75 + v45);
            a5 = (__int64)&v33[v45];
            a6 = *(_QWORD *)a5;
            if ( *v23 == *(_QWORD *)a5 )
              goto LABEL_42;
            ++v75;
          }
          goto LABEL_88;
        }
      }
      else if ( (int)v96 - HIDWORD(v95) - v43 <= (unsigned int)v96 >> 3 )
      {
        v53 = (v25 >> 1) | v25 | (((v25 >> 1) | v25) >> 2);
        v54 = ((((((v53 >> 4) | v53) >> 8) | (v53 >> 4) | v53) >> 16) | (((v53 >> 4) | v53) >> 8) | (v53 >> 4) | v53)
            + 1;
        if ( (unsigned int)v54 < 0x40 )
          LODWORD(v54) = 64;
        LODWORD(v96) = v54;
        v55 = (_QWORD *)sub_C7D670(8LL * (unsigned int)v54, 8);
        v94 = v55;
        if ( v30 )
        {
          v56 = 8 * v29;
          v95 = 0;
          v57 = (__int64 *)(v30 + 8 * v29);
          for ( j = &v55[(unsigned int)v96]; j != v55; ++v55 )
          {
            if ( v55 )
              *v55 = -4096;
          }
          v59 = (__int64 *)v30;
          do
          {
            v60 = *v59;
            if ( *v59 != -8192 && v60 != -4096 )
            {
              if ( !(_DWORD)v96 )
              {
                MEMORY[0] = *v59;
                BUG();
              }
              v61 = (v96 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
              v62 = &v94[v61];
              v63 = *v62;
              if ( v60 != *v62 )
              {
                v88 = 1;
                v74 = 0;
                while ( v63 != -4096 )
                {
                  if ( !v74 && v63 == -8192 )
                    v74 = v62;
                  v61 = (v96 - 1) & (v88 + v61);
                  v62 = &v94[v61];
                  v63 = *v62;
                  if ( v60 == *v62 )
                    goto LABEL_82;
                  ++v88;
                }
                if ( v74 )
                  v62 = v74;
              }
LABEL_82:
              *v62 = v60;
              LODWORD(v95) = v95 + 1;
            }
            ++v59;
          }
          while ( v57 != v59 );
          sub_C7D6A0(v30, v56, 8);
          v55 = v94;
          v64 = v96;
          v43 = v95 + 1;
        }
        else
        {
          v95 = 0;
          v72 = &v55[(unsigned int)v96];
          v64 = v96;
          if ( v55 != v72 )
          {
            v73 = v55;
            do
            {
              if ( v73 )
                *v73 = -4096;
              ++v73;
            }
            while ( v72 != v73 );
          }
          v43 = 1;
        }
        if ( !v64 )
        {
LABEL_170:
          LODWORD(v95) = v95 + 1;
          BUG();
        }
        v65 = v64 - 1;
        v66 = 1;
        v67 = 0;
        v68 = (v64 - 1) & (((unsigned int)*v23 >> 9) ^ ((unsigned int)*v23 >> 4));
        a5 = (__int64)&v55[v68];
        a6 = *(_QWORD *)a5;
        if ( *v23 != *(_QWORD *)a5 )
        {
          while ( a6 != -4096 )
          {
            if ( !v67 && a6 == -8192 )
              v67 = a5;
            v68 = v65 & (v66 + v68);
            a5 = (__int64)&v55[v68];
            a6 = *(_QWORD *)a5;
            if ( *v23 == *(_QWORD *)a5 )
              goto LABEL_42;
            ++v66;
          }
LABEL_88:
          if ( v67 )
            a5 = v67;
        }
      }
LABEL_42:
      LODWORD(v95) = v43;
      if ( *(_QWORD *)a5 != -4096 )
        --HIDWORD(v95);
      v46 = *v23;
      *(_QWORD *)a5 = *v23;
      v47 = (unsigned int)v98;
      v48 = (unsigned int)v98 + 1LL;
      if ( v48 > HIDWORD(v98) )
      {
        sub_C8D5F0((__int64)&v97, v99, v48, 8u, a5, a6);
        v47 = (unsigned int)v98;
      }
      ++v23;
      *(_QWORD *)&v97[8 * v47] = v46;
      LODWORD(v98) = v98 + 1;
      if ( v23 == v24 )
        goto LABEL_47;
      continue;
    }
  }
LABEL_49:
  ++v93;
  if ( !(_DWORD)v95 )
  {
    v6 = HIDWORD(v95);
    if ( !HIDWORD(v95) )
      goto LABEL_56;
    v49 = (unsigned int)v96;
    if ( (unsigned int)v96 <= 0x40 )
      goto LABEL_53;
    sub_C7D6A0((__int64)v94, 8LL * (unsigned int)v96, 8);
    LODWORD(v96) = 0;
    goto LABEL_140;
  }
  v6 = (unsigned int)(4 * v95);
  v49 = (unsigned int)v96;
  if ( (unsigned int)v6 < 0x40 )
    v6 = 64;
  if ( (unsigned int)v6 < (unsigned int)v96 )
  {
    if ( (_DWORD)v95 == 1 )
    {
      v84 = 64;
    }
    else
    {
      _BitScanReverse(&v83, v95 - 1);
      v84 = 1 << (33 - (v83 ^ 0x1F));
      if ( v84 < 64 )
        v84 = 64;
      if ( v84 == (_DWORD)v96 )
        goto LABEL_165;
    }
    sub_C7D6A0((__int64)v94, 8LL * (unsigned int)v96, 8);
    v85 = sub_2C261E0(v84);
    LODWORD(v96) = v85;
    if ( v85 )
    {
      v94 = (_QWORD *)sub_C7D670(8LL * v85, 8);
LABEL_165:
      sub_2C2F600((__int64)&v93);
      goto LABEL_56;
    }
LABEL_140:
    v94 = 0;
    goto LABEL_55;
  }
LABEL_53:
  v50 = v94;
  v51 = &v94[v49];
  if ( v94 != v51 )
  {
    do
      *v50++ = -4096;
    while ( v51 != v50 );
  }
LABEL_55:
  v95 = 0;
LABEL_56:
  v91 = (char *)&v93;
  v92 = 0;
  if ( (_DWORD)v98 )
  {
    sub_2C258D0((__int64)&v91, &v97, (unsigned int)v98, v6, a5, a6);
    *a1 = a1 + 2;
    a1[1] = 0x600000000LL;
    if ( (_DWORD)v92 )
      sub_2C258D0((__int64)a1, &v91, v79, v80, v81, v82);
    if ( v91 != (char *)&v93 )
      _libc_free((unsigned __int64)v91);
  }
  else
  {
    *a1 = a1 + 2;
    a1[1] = 0x600000000LL;
  }
  if ( v97 != v99 )
    _libc_free((unsigned __int64)v97);
  sub_C7D6A0((__int64)v94, 8LL * (unsigned int)v96, 8);
  return a1;
}
