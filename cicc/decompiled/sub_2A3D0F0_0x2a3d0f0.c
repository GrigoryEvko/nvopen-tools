// Function: sub_2A3D0F0
// Address: 0x2a3d0f0
//
void __fastcall sub_2A3D0F0(_QWORD *a1, __int64 (__fastcall *a2)(__int64, __int64), __int64 a3)
{
  char *v3; // rax
  char *v4; // rsi
  unsigned int v5; // edx
  int v6; // ecx
  _QWORD *j; // r15
  __int64 v8; // r14
  unsigned __int8 *v9; // r12
  const char *v10; // rax
  unsigned __int64 v11; // rdx
  _QWORD *k; // r15
  __int64 v13; // r14
  unsigned __int8 *v14; // r12
  const char *v15; // rax
  unsigned __int64 v16; // rdx
  __int64 *v17; // r13
  __int64 *v18; // rbx
  unsigned __int64 v19; // r12
  __int64 v20; // r14
  __int64 v21; // rdx
  __int64 v22; // r15
  __int64 v23; // r14
  const char *v24; // rax
  unsigned __int64 v25; // rdx
  __int64 *v26; // rax
  __int64 v27; // rbx
  const char **v28; // rsi
  const char *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rbx
  __int64 v33; // r12
  __int64 m; // r12
  __int64 n; // rbx
  char *v36; // rax
  const char *v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  _QWORD *v40; // rbx
  __int64 v41; // rcx
  __int64 v42; // r13
  const char *v43; // rax
  unsigned __int64 v44; // rdx
  __int64 *v45; // rax
  __int64 v46; // r14
  __int64 v47; // r15
  __int64 v48; // r13
  __int64 v49; // r14
  __int64 v50; // rax
  __int64 v51; // r14
  __int64 i; // r15
  __int64 v53; // rdx
  __int64 v54; // rax
  char *v55; // rax
  const char *v56; // rsi
  size_t v57; // rdx
  unsigned __int64 v58; // [rsp+20h] [rbp-420h]
  _QWORD *v61; // [rsp+40h] [rbp-400h]
  _QWORD *v62; // [rsp+40h] [rbp-400h]
  __int64 v63; // [rsp+50h] [rbp-3F0h] BYREF
  __int64 v64; // [rsp+58h] [rbp-3E8h]
  _QWORD v65[4]; // [rsp+60h] [rbp-3E0h] BYREF
  __int16 v66; // [rsp+80h] [rbp-3C0h]
  _BYTE *v67; // [rsp+90h] [rbp-3B0h] BYREF
  __int64 v68; // [rsp+98h] [rbp-3A8h]
  _BYTE v69[128]; // [rsp+A0h] [rbp-3A0h] BYREF
  _BYTE *v70; // [rsp+120h] [rbp-320h] BYREF
  __int64 v71; // [rsp+128h] [rbp-318h]
  _BYTE v72[128]; // [rsp+130h] [rbp-310h] BYREF
  _BYTE *v73; // [rsp+1B0h] [rbp-290h] BYREF
  __int64 v74; // [rsp+1B8h] [rbp-288h]
  _BYTE v75[128]; // [rsp+1C0h] [rbp-280h] BYREF
  _BYTE *v76; // [rsp+240h] [rbp-200h] BYREF
  __int64 v77; // [rsp+248h] [rbp-1F8h]
  _BYTE v78[128]; // [rsp+250h] [rbp-1F0h] BYREF
  char *v79; // [rsp+2D0h] [rbp-170h] BYREF
  size_t v80; // [rsp+2D8h] [rbp-168h]
  __int64 v81; // [rsp+2E0h] [rbp-160h]
  char v82; // [rsp+2E8h] [rbp-158h] BYREF
  __int16 v83; // [rsp+2F0h] [rbp-150h]
  char *v84; // [rsp+370h] [rbp-D0h] BYREF
  unsigned __int64 v85; // [rsp+378h] [rbp-C8h]
  __int64 v86; // [rsp+380h] [rbp-C0h]
  __int64 v87; // [rsp+388h] [rbp-B8h]
  __int64 v88; // [rsp+390h] [rbp-B0h]
  __int64 v89; // [rsp+398h] [rbp-A8h]
  __int64 v90; // [rsp+3A0h] [rbp-A0h]
  __int64 v91; // [rsp+3A8h] [rbp-98h]
  __int64 v92; // [rsp+3B0h] [rbp-90h]
  __int64 v93; // [rsp+3B8h] [rbp-88h]
  __int64 v94; // [rsp+3C0h] [rbp-80h]
  __int64 v95; // [rsp+3C8h] [rbp-78h]
  __int64 v96; // [rsp+3D0h] [rbp-70h]
  __int64 v97; // [rsp+3D8h] [rbp-68h]
  __int64 v98; // [rsp+3E0h] [rbp-60h]
  __int64 v99; // [rsp+3E8h] [rbp-58h]
  __int64 *v100; // [rsp+3F0h] [rbp-50h]
  __int64 *v101; // [rsp+3F8h] [rbp-48h]
  __int64 v102; // [rsp+400h] [rbp-40h]
  char v103; // [rsp+408h] [rbp-38h]

  v3 = (char *)a1[21];
  v4 = &v3[a1[22]];
  if ( v3 == v4 )
  {
    v58 = 0;
  }
  else
  {
    v5 = 0;
    do
    {
      v6 = *v3++;
      v5 += v6;
    }
    while ( v4 != v3 );
    v58 = v5;
  }
  v70 = v72;
  v67 = v69;
  v73 = v75;
  v68 = 0x800000000LL;
  v71 = 0x800000000LL;
  v74 = 0x800000000LL;
  v77 = 0x800000000LL;
  v76 = v78;
  sub_2A3CDF0(qword_500AD08, qword_500AD10, (__int64)&v67);
  sub_2A3CDF0(qword_500AC08, qword_500AC10, (__int64)&v70);
  sub_2A3CDF0(qword_500AB08, qword_500AB10, (__int64)&v73);
  sub_2A3CDF0(qword_500AE08, qword_500AE10, (__int64)&v76);
  if ( (_BYTE)qword_500AA28 )
  {
    v40 = (_QWORD *)a1[4];
    v62 = a1 + 3;
    if ( v40 == a1 + 3 )
      goto LABEL_72;
    while ( 1 )
    {
      v41 = (__int64)(v40 - 7);
      if ( !v40 )
        v41 = 0;
      v42 = v41;
      v43 = sub_BD5D20(v41);
      v84 = (char *)v43;
      v85 = v44;
      if ( v44 <= 4 )
      {
        if ( v44 )
          goto LABEL_99;
LABEL_100:
        v45 = (__int64 *)a2(a3, v42);
        if ( sub_981210(*v45, v42, (unsigned int *)&v79) )
          goto LABEL_108;
        v46 = (__int64)&v76[16 * (unsigned int)v77];
        if ( v46 != sub_2A3CF00((__int64)v76, v46, (__int64)&v84) )
          goto LABEL_108;
        v47 = v42 + 72;
        v48 = *(_QWORD *)(v42 + 80);
        if ( v47 == v48 )
        {
          v49 = 0;
        }
        else
        {
          if ( !v48 )
            BUG();
          while ( 1 )
          {
            v49 = *(_QWORD *)(v48 + 32);
            if ( v49 != v48 + 24 )
              break;
            v48 = *(_QWORD *)(v48 + 8);
            if ( v47 == v48 )
              goto LABEL_108;
            if ( !v48 )
              BUG();
          }
        }
        if ( v47 == v48 )
          goto LABEL_108;
        v50 = v49;
        v51 = v47;
        i = v50;
        do
        {
          if ( !i )
            BUG();
          if ( *(_BYTE *)(*(_QWORD *)(i - 16) + 8LL) != 7 )
          {
            sub_BD5D20(i - 24);
            if ( !v53 )
            {
              v55 = sub_B458E0((unsigned int)*(unsigned __int8 *)(i - 24) - 29);
              LOWORD(v88) = 257;
              if ( *v55 )
              {
                v84 = v55;
                LOBYTE(v88) = 3;
              }
              sub_BD6B50((unsigned __int8 *)(i - 24), (const char **)&v84);
            }
          }
          for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v48 + 32) )
          {
            v54 = v48 - 24;
            if ( !v48 )
              v54 = 0;
            if ( i != v54 + 48 )
              break;
            v48 = *(_QWORD *)(v48 + 8);
            if ( v51 == v48 )
              goto LABEL_126;
            if ( !v48 )
              BUG();
          }
        }
        while ( v51 != v48 );
LABEL_126:
        v40 = (_QWORD *)v40[1];
        if ( v62 == v40 )
          goto LABEL_72;
      }
      else
      {
        if ( *(_DWORD *)v43 == 1836477548 && v43[4] == 46 )
          goto LABEL_108;
LABEL_99:
        if ( *v43 != 1 )
          goto LABEL_100;
LABEL_108:
        v40 = (_QWORD *)v40[1];
        if ( v62 == v40 )
          goto LABEL_72;
      }
    }
  }
  for ( j = (_QWORD *)a1[6]; a1 + 5 != j; j = (_QWORD *)j[1] )
  {
    v9 = (unsigned __int8 *)(j - 6);
    if ( !j )
      v9 = 0;
    v10 = sub_BD5D20((__int64)v9);
    v79 = (char *)v10;
    v80 = v11;
    if ( v11 > 4 )
    {
      if ( *(_DWORD *)v10 == 1836477548 && v10[4] == 46 )
        continue;
    }
    else if ( !v11 )
    {
LABEL_10:
      v8 = (__int64)&v67[16 * (unsigned int)v68];
      if ( v8 == sub_2A3CF00((__int64)v67, v8, (__int64)&v79) )
      {
        v84 = "alias";
        LOWORD(v88) = 259;
        sub_BD6B50(v9, (const char **)&v84);
      }
      continue;
    }
    if ( *v10 != 1 )
      goto LABEL_10;
  }
  for ( k = (_QWORD *)a1[2]; a1 + 1 != k; k = (_QWORD *)k[1] )
  {
    v14 = (unsigned __int8 *)(k - 7);
    if ( !k )
      v14 = 0;
    v15 = sub_BD5D20((__int64)v14);
    v79 = (char *)v15;
    v80 = v16;
    if ( v16 > 4 )
    {
      if ( *(_DWORD *)v15 == 1836477548 && v15[4] == 46 )
        continue;
    }
    else if ( !v16 )
    {
LABEL_22:
      v13 = (__int64)&v70[16 * (unsigned int)v71];
      if ( v13 == sub_2A3CF00((__int64)v70, v13, (__int64)&v79) )
      {
        v84 = "global";
        LOWORD(v88) = 259;
        sub_BD6B50(v14, (const char **)&v84);
      }
      continue;
    }
    if ( *v15 != 1 )
      goto LABEL_22;
  }
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  sub_BD22F0((__int64)&v84, a1, 1);
  v17 = v101;
  v18 = v100;
  if ( v100 != v101 )
  {
    v19 = v58;
    do
    {
      v20 = *v18;
      v63 = sub_BCB490(*v18);
      v64 = v21;
      if ( (*(_BYTE *)(v20 + 9) & 4) == 0 )
      {
        if ( v64 )
        {
          v22 = (__int64)&v73[16 * (unsigned int)v74];
          if ( v22 == sub_2A3CF00((__int64)v73, v22, (__int64)&v63) )
          {
            v80 = 0;
            v79 = &v82;
            v81 = 128;
            v19 = 1103515245 * v19 + 12345;
            if ( *off_49D3D20[((v19 >> 16) & 0x7FFF)
                            - ((((((v19 >> 16) & 0x7FFF) * (unsigned __int128)0xF0F0F0F0F0F0F0F1LL) >> 64)
                              & 0xFFFFFFFFFFFFFFF0LL)
                             + ((v19 >> 16) & 0x7FFF) / 0x11)] )
            {
              v65[2] = off_49D3D20[((v19 >> 16) & 0x7FFF)
                                 - ((((((v19 >> 16) & 0x7FFF) * (unsigned __int128)0xF0F0F0F0F0F0F0F1LL) >> 64)
                                   & 0xFFFFFFFFFFFFFFF0LL)
                                  + ((v19 >> 16) & 0x7FFF) / 0x11)];
              v66 = 771;
              v65[0] = "struct.";
              sub_CA0EC0((__int64)v65, (__int64)&v79);
              v57 = v80;
              v56 = v79;
            }
            else
            {
              v65[0] = "struct.";
              v66 = 259;
              v56 = "struct.";
              v57 = strlen("struct.");
            }
            sub_BCB4B0((__int64 **)v20, v56, v57);
            if ( v79 != &v82 )
              _libc_free((unsigned __int64)v79);
          }
        }
      }
      ++v18;
    }
    while ( v17 != v18 );
    v58 = v19;
  }
  v61 = (_QWORD *)a1[4];
  if ( v61 != a1 + 3 )
  {
    while ( 1 )
    {
      v23 = (__int64)(v61 - 7);
      if ( !v61 )
        v23 = 0;
      v24 = sub_BD5D20(v23);
      v79 = (char *)v24;
      v80 = v25;
      if ( v25 > 4 )
        break;
      if ( v25 )
        goto LABEL_47;
LABEL_48:
      v26 = (__int64 *)((__int64 (__fastcall *)(__int64, __int64, unsigned __int64, const char *))a2)(a3, v23, v25, v24);
      if ( !sub_981210(*v26, v23, (unsigned int *)v65) )
      {
        v27 = (__int64)&v76[16 * (unsigned int)v77];
        v28 = (const char **)v27;
        if ( v27 == sub_2A3CF00((__int64)v76, v27, (__int64)&v79) )
        {
          v29 = sub_BD5D20(v23);
          if ( v30 == 4 && *(_DWORD *)v29 == 1852399981 )
          {
            if ( (*(_BYTE *)(v23 + 2) & 1) == 0 )
              goto LABEL_53;
LABEL_86:
            sub_B2C6D0(v23, (__int64)v28, v30, v31);
            v32 = *(_QWORD *)(v23 + 96);
            v33 = v32 + 40LL * *(_QWORD *)(v23 + 104);
            if ( (*(_BYTE *)(v23 + 2) & 1) != 0 )
            {
              sub_B2C6D0(v23, (__int64)v28, v38, v39);
              v32 = *(_QWORD *)(v23 + 96);
            }
          }
          else
          {
            v58 = 1103515245 * v58 + 12345;
            v37 = off_49D3D20[((v58 >> 16) & 0x7FFF)
                            - ((((((v58 >> 16) & 0x7FFF) * (unsigned __int128)0xF0F0F0F0F0F0F0F1LL) >> 64)
                              & 0xFFFFFFFFFFFFFFF0LL)
                             + ((v58 >> 16) & 0x7FFF) / 0x11)];
            v83 = 257;
            if ( *v37 )
            {
              v79 = (char *)v37;
              LOBYTE(v83) = 3;
            }
            v28 = (const char **)&v79;
            sub_BD6B50((unsigned __int8 *)v23, (const char **)&v79);
            if ( (*(_BYTE *)(v23 + 2) & 1) != 0 )
              goto LABEL_86;
LABEL_53:
            v32 = *(_QWORD *)(v23 + 96);
            v33 = v32 + 40LL * *(_QWORD *)(v23 + 104);
          }
          for ( ; v33 != v32; v32 += 40 )
          {
            if ( *(_BYTE *)(*(_QWORD *)(v32 + 8) + 8LL) != 7 )
            {
              v79 = "arg";
              v83 = 259;
              sub_BD6B50((unsigned __int8 *)v32, (const char **)&v79);
            }
          }
          for ( m = *(_QWORD *)(v23 + 80); v23 + 72 != m; m = *(_QWORD *)(m + 8) )
          {
            v79 = "bb";
            v83 = 259;
            if ( !m )
            {
              sub_BD6B50(0, (const char **)&v79);
              BUG();
            }
            sub_BD6B50((unsigned __int8 *)(m - 24), (const char **)&v79);
            for ( n = *(_QWORD *)(m + 32); m + 24 != n; n = *(_QWORD *)(n + 8) )
            {
              if ( !n )
                BUG();
              if ( *(_BYTE *)(*(_QWORD *)(n - 16) + 8LL) != 7 )
              {
                v36 = sub_B458E0((unsigned int)*(unsigned __int8 *)(n - 24) - 29);
                v83 = 257;
                if ( *v36 )
                {
                  v79 = v36;
                  LOBYTE(v83) = 3;
                }
                sub_BD6B50((unsigned __int8 *)(n - 24), (const char **)&v79);
              }
            }
          }
        }
      }
LABEL_68:
      v61 = (_QWORD *)v61[1];
      if ( a1 + 3 == v61 )
        goto LABEL_69;
    }
    if ( *(_DWORD *)v24 == 1836477548 && v24[4] == 46 )
      goto LABEL_68;
LABEL_47:
    if ( *v24 == 1 )
      goto LABEL_68;
    goto LABEL_48;
  }
LABEL_69:
  if ( v100 )
    j_j___libc_free_0((unsigned __int64)v100);
  sub_C7D6A0(v97, 8LL * (unsigned int)v99, 8);
  sub_C7D6A0(v93, 8LL * (unsigned int)v95, 8);
  sub_C7D6A0(v89, 8LL * (unsigned int)v91, 8);
  sub_C7D6A0(v85, 8LL * (unsigned int)v87, 8);
LABEL_72:
  if ( v76 != v78 )
    _libc_free((unsigned __int64)v76);
  if ( v73 != v75 )
    _libc_free((unsigned __int64)v73);
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
  if ( v67 != v69 )
    _libc_free((unsigned __int64)v67);
}
