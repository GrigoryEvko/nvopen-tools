// Function: sub_F521F0
// Address: 0xf521f0
//
__int64 __fastcall sub_F521F0(__int64 a1)
{
  __int64 v1; // r8
  __int64 v2; // r9
  _BYTE *v3; // rsi
  _QWORD *i; // r15
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 *v10; // r12
  unsigned int v11; // ebx
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 v17; // r12
  _BYTE *v18; // rax
  __int64 v19; // r13
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rdx
  char *v24; // rax
  char v25; // cl
  _QWORD *v26; // rdx
  unsigned int v27; // eax
  __int64 v28; // rcx
  __int64 v29; // rdx
  __int64 v30; // rbx
  __int64 v31; // r15
  char v32; // al
  _QWORD *v33; // r15
  _BYTE *v34; // rax
  __int64 v35; // r13
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // rax
  __int64 j; // rdx
  char *v40; // rax
  char v41; // cl
  _QWORD *v42; // rdx
  unsigned int v43; // eax
  __int64 v44; // rcx
  __int64 v45; // rdx
  __int64 v46; // rbx
  __int64 v47; // r12
  char v48; // al
  _QWORD *m; // r13
  __int64 v50; // rdi
  __int64 v51; // rax
  __int64 v52; // r13
  __int64 v53; // r14
  __int64 v54; // r15
  __int64 v55; // r12
  __int64 v56; // rsi
  char *v57; // r13
  char *v58; // r12
  __int64 v59; // r13
  __int64 v60; // r12
  __int64 v61; // rdi
  __int64 v62; // rsi
  char *v63; // r13
  char *v64; // r12
  char *v65; // r13
  char *v66; // r12
  char *v67; // r13
  char *v68; // r12
  __int64 v70; // rax
  unsigned __int64 v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rdi
  __int64 v74; // r9
  __int64 v75; // rax
  unsigned __int64 v76; // rdx
  _QWORD *v77; // rax
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 v80; // rax
  __int64 v81; // r9
  __int64 v82; // [rsp+8h] [rbp-318h]
  __int64 *v84; // [rsp+20h] [rbp-300h]
  __int64 *v85; // [rsp+30h] [rbp-2F0h]
  __int64 v86; // [rsp+38h] [rbp-2E8h]
  __int64 v87; // [rsp+48h] [rbp-2D8h]
  _QWORD *v88; // [rsp+60h] [rbp-2C0h]
  __int64 *v89; // [rsp+60h] [rbp-2C0h]
  __int64 *v90; // [rsp+60h] [rbp-2C0h]
  _QWORD *v91; // [rsp+68h] [rbp-2B8h]
  _BYTE *v92; // [rsp+70h] [rbp-2B0h] BYREF
  __int64 v93; // [rsp+78h] [rbp-2A8h] BYREF
  __int64 *v94; // [rsp+80h] [rbp-2A0h] BYREF
  __int64 v95; // [rsp+88h] [rbp-298h]
  _BYTE v96[32]; // [rsp+90h] [rbp-290h] BYREF
  __int64 *v97; // [rsp+B0h] [rbp-270h] BYREF
  __int64 v98; // [rsp+B8h] [rbp-268h]
  _BYTE v99[48]; // [rsp+C0h] [rbp-260h] BYREF
  _QWORD *k; // [rsp+F0h] [rbp-230h] BYREF
  __int64 v101; // [rsp+F8h] [rbp-228h]
  _QWORD v102[8]; // [rsp+100h] [rbp-220h] BYREF
  __int64 v103[7]; // [rsp+140h] [rbp-1E0h] BYREF
  char *v104; // [rsp+178h] [rbp-1A8h]
  int v105; // [rsp+180h] [rbp-1A0h]
  char v106; // [rsp+188h] [rbp-198h] BYREF
  char *v107; // [rsp+1A8h] [rbp-178h]
  int v108; // [rsp+1B0h] [rbp-170h]
  char v109; // [rsp+1B8h] [rbp-168h] BYREF
  char *v110; // [rsp+1D8h] [rbp-148h]
  char v111; // [rsp+1E8h] [rbp-138h] BYREF
  char *v112; // [rsp+208h] [rbp-118h]
  char v113; // [rsp+218h] [rbp-108h] BYREF
  char *v114; // [rsp+238h] [rbp-E8h]
  int v115; // [rsp+240h] [rbp-E0h]
  char v116; // [rsp+248h] [rbp-D8h] BYREF
  __int64 v117; // [rsp+270h] [rbp-B0h]
  unsigned int v118; // [rsp+280h] [rbp-A0h]
  __int64 v119; // [rsp+288h] [rbp-98h]
  unsigned int v120; // [rsp+290h] [rbp-90h]
  char *v121; // [rsp+298h] [rbp-88h] BYREF
  int v122; // [rsp+2A0h] [rbp-80h]
  char v123; // [rsp+2A8h] [rbp-78h] BYREF
  __int64 v124; // [rsp+2D8h] [rbp-48h]
  unsigned int v125; // [rsp+2E8h] [rbp-38h]

  sub_AE0470((__int64)v103, *(__int64 **)(a1 + 40), 0, 0);
  v3 = (_BYTE *)(a1 + 72);
  v94 = (__int64 *)v96;
  v95 = 0x400000000LL;
  v97 = (__int64 *)v99;
  v98 = 0x600000000LL;
  v88 = (_QWORD *)(a1 + 72);
  v91 = *(_QWORD **)(a1 + 80);
  if ( v91 == (_QWORD *)(a1 + 72) )
  {
    v11 = 0;
    goto LABEL_101;
  }
  do
  {
    if ( !v91 )
      BUG();
    for ( i = (_QWORD *)v91[4]; v91 + 3 != i; i = (_QWORD *)i[1] )
    {
      if ( !i )
        BUG();
      if ( *((_BYTE *)i - 24) == 85 )
      {
        v14 = *(i - 7);
        if ( v14 )
        {
          if ( !*(_BYTE *)v14 )
          {
            v3 = (_BYTE *)i[7];
            if ( *(_BYTE **)(v14 + 24) == v3 && (*(_BYTE *)(v14 + 33) & 0x20) != 0 && *(_DWORD *)(v14 + 36) == 69 )
            {
              v15 = (unsigned int)v95;
              v16 = (unsigned int)v95 + 1LL;
              if ( v16 > HIDWORD(v95) )
              {
                v3 = v96;
                sub_C8D5F0((__int64)&v94, v96, v16, 8u, v1, v2);
                v15 = (unsigned int)v95;
              }
              v94[v15] = (__int64)(i - 3);
              LODWORD(v95) = v95 + 1;
            }
          }
        }
      }
      v5 = i[5];
      if ( v5 )
      {
        v6 = sub_B14240(v5);
        v8 = v7;
        v9 = v6;
        if ( v7 != v6 )
        {
          while ( *(_BYTE *)(v9 + 32) )
          {
            v9 = *(_QWORD *)(v9 + 8);
            if ( v9 == v7 )
              goto LABEL_16;
          }
LABEL_11:
          if ( v9 != v8 )
          {
            if ( !*(_BYTE *)(v9 + 64) )
            {
              v12 = (unsigned int)v98;
              v13 = (unsigned int)v98 + 1LL;
              if ( v13 > HIDWORD(v98) )
              {
                v3 = v99;
                sub_C8D5F0((__int64)&v97, v99, v13, 8u, v1, v2);
                v12 = (unsigned int)v98;
              }
              v97[v12] = v9;
              LODWORD(v98) = v98 + 1;
            }
            while ( 1 )
            {
              v9 = *(_QWORD *)(v9 + 8);
              if ( v9 == v8 )
                break;
              if ( !*(_BYTE *)(v9 + 32) )
                goto LABEL_11;
            }
          }
        }
      }
LABEL_16:
      ;
    }
    v91 = (_QWORD *)v91[1];
  }
  while ( v88 != v91 );
  if ( !(_DWORD)v95 )
  {
    v10 = v97;
    if ( (_DWORD)v98 )
    {
      v11 = 0;
      v85 = &v97[(unsigned int)v98];
LABEL_63:
      v90 = v10;
      while ( 1 )
      {
        v3 = 0;
        v33 = (_QWORD *)*v90;
        v34 = (_BYTE *)sub_B12A50(*v90, 0);
        v35 = (__int64)v34;
        if ( v34 )
        {
          if ( *v34 == 60 && !(unsigned __int8)sub_B4CE70((__int64)v34) )
          {
            v38 = *(_QWORD *)(v35 + 72);
            if ( !v38 || *(_BYTE *)(v38 + 8) != 16 && *(_BYTE *)(v38 + 8) != 15 )
              break;
          }
        }
LABEL_64:
        if ( ++v90 == v85 )
          goto LABEL_93;
      }
      for ( j = *(_QWORD *)(v35 + 16); j; j = *(_QWORD *)(j + 8) )
      {
        while ( 1 )
        {
          v40 = *(char **)(j + 24);
          v41 = *v40;
          if ( (unsigned __int8)*v40 > 0x1Cu && (v41 == 61 || v41 == 62) )
            break;
          j = *(_QWORD *)(j + 8);
          if ( !j )
            goto LABEL_79;
        }
        if ( (v40[2] & 1) != 0 )
          goto LABEL_64;
      }
LABEL_79:
      v42 = v102;
      v102[0] = v35;
      v101 = 0x800000001LL;
      v43 = 1;
      for ( k = v102; ; v42 = k )
      {
        v44 = v43--;
        v45 = v42[v44 - 1];
        LODWORD(v101) = v43;
        v46 = *(_QWORD *)(v45 + 16);
        if ( v46 )
          break;
LABEL_91:
        if ( !v43 )
        {
          sub_B14290(v33);
          if ( k != v102 )
            _libc_free(k, v3);
          v11 = 1;
          goto LABEL_64;
        }
      }
      while ( 1 )
      {
        v47 = *(_QWORD *)(v46 + 24);
        v48 = *(_BYTE *)v47;
        if ( *(_BYTE *)v47 <= 0x1Cu )
          goto LABEL_85;
        if ( v48 != 62 )
        {
          if ( v48 == 61 )
          {
            v3 = *(_BYTE **)(v46 + 24);
            sub_F51FC0((__int64)v33, (__int64)v3);
          }
          else if ( v48 == 85 )
          {
            if ( !sub_B46A10(*(_QWORD *)(v46 + 24)) )
            {
              sub_AE7AF0((__int64)&v92, (__int64)v33);
              v93 = 6;
              v77 = (_QWORD *)sub_B11F60((__int64)(v33 + 10));
              v78 = sub_B0DED0(v77, &v93, 1);
              v79 = v87;
              v82 = v78;
              LOWORD(v79) = 0;
              v87 = v79;
              v80 = sub_B12000((__int64)(v33 + 9));
              sub_F4EE60(v103, v35, v80, v82, (__int64)&v92, v81, v47 + 24, v87);
              v3 = v92;
              if ( v92 )
                sub_B91220((__int64)&v92, (__int64)v92);
            }
          }
          else if ( v48 == 78 && *(_BYTE *)(*(_QWORD *)(v47 + 8) + 8LL) == 14 )
          {
            v75 = (unsigned int)v101;
            v76 = (unsigned int)v101 + 1LL;
            if ( v76 > HIDWORD(v101) )
            {
              v3 = v102;
              sub_C8D5F0((__int64)&k, v102, v76, 8u, v36, v37);
              v75 = (unsigned int)v101;
            }
            k[v75] = v47;
            LODWORD(v101) = v101 + 1;
          }
          goto LABEL_85;
        }
        if ( (unsigned int)sub_BD2910(v46) == 1 )
        {
          v3 = (_BYTE *)v47;
          sub_F51C80((__int64)v33, v47, v103);
          v46 = *(_QWORD *)(v46 + 8);
          if ( !v46 )
          {
LABEL_90:
            v43 = v101;
            goto LABEL_91;
          }
        }
        else
        {
LABEL_85:
          v46 = *(_QWORD *)(v46 + 8);
          if ( !v46 )
            goto LABEL_90;
        }
      }
    }
    v11 = 0;
    goto LABEL_99;
  }
  v11 = 0;
  v89 = v94;
  v84 = &v94[(unsigned int)v95];
  do
  {
    v3 = 0;
    v17 = *v89;
    v18 = (_BYTE *)sub_B58EB0(*v89, 0);
    v19 = (__int64)v18;
    if ( !v18 )
      goto LABEL_33;
    if ( *v18 != 60 )
      goto LABEL_33;
    if ( (unsigned __int8)sub_B4CE70((__int64)v18) )
      goto LABEL_33;
    v22 = *(_QWORD *)(v19 + 72);
    if ( v22 )
    {
      if ( *(_BYTE *)(v22 + 8) == 16 || *(_BYTE *)(v22 + 8) == 15 )
        goto LABEL_33;
    }
    v23 = *(_QWORD *)(v19 + 16);
    if ( !v23 )
    {
LABEL_48:
      v26 = v102;
      v102[0] = v19;
      v101 = 0x800000001LL;
      v27 = 1;
      for ( k = v102; ; v26 = k )
      {
        v28 = v27--;
        v29 = v26[v28 - 1];
        LODWORD(v101) = v27;
        v30 = *(_QWORD *)(v29 + 16);
        if ( v30 )
          break;
LABEL_60:
        if ( !v27 )
        {
          sub_B43D60((_QWORD *)v17);
          if ( k != v102 )
            _libc_free(k, v3);
          v11 = 1;
          goto LABEL_33;
        }
      }
      while ( 1 )
      {
        v31 = *(_QWORD *)(v30 + 24);
        v32 = *(_BYTE *)v31;
        if ( *(_BYTE *)v31 <= 0x1Cu )
          goto LABEL_54;
        if ( v32 != 62 )
        {
          if ( v32 == 61 )
          {
            v3 = *(_BYTE **)(v30 + 24);
            sub_F51BB0(v17, (__int64)v3, v103);
          }
          else if ( v32 == 85 )
          {
            if ( !sub_B46A10(*(_QWORD *)(v30 + 24)) )
            {
              sub_AE7A80((__int64)&v92, v17);
              v93 = 6;
              v72 = sub_B0DED0(
                      *(_QWORD **)(*(_QWORD *)(v17 + 32 * (2LL - (*(_DWORD *)(v17 + 4) & 0x7FFFFFF))) + 24LL),
                      &v93,
                      1);
              v73 = v86;
              LOWORD(v73) = 0;
              v86 = v73;
              sub_F4EE60(
                v103,
                v19,
                *(_QWORD *)(*(_QWORD *)(v17 + 32 * (1LL - (*(_DWORD *)(v17 + 4) & 0x7FFFFFF))) + 24LL),
                v72,
                (__int64)&v92,
                v74,
                v31 + 24,
                v73);
              v3 = v92;
              if ( v92 )
                sub_B91220((__int64)&v92, (__int64)v92);
            }
          }
          else if ( v32 == 78 && *(_BYTE *)(*(_QWORD *)(v31 + 8) + 8LL) == 14 )
          {
            v70 = (unsigned int)v101;
            v71 = (unsigned int)v101 + 1LL;
            if ( v71 > HIDWORD(v101) )
            {
              v3 = v102;
              sub_C8D5F0((__int64)&k, v102, v71, 8u, v20, v21);
              v70 = (unsigned int)v101;
            }
            k[v70] = v31;
            LODWORD(v101) = v101 + 1;
          }
          goto LABEL_54;
        }
        if ( (unsigned int)sub_BD2910(v30) == 1 )
        {
          v3 = (_BYTE *)v31;
          sub_F519F0(v17, v31, v103);
          v30 = *(_QWORD *)(v30 + 8);
          if ( !v30 )
          {
LABEL_59:
            v27 = v101;
            goto LABEL_60;
          }
        }
        else
        {
LABEL_54:
          v30 = *(_QWORD *)(v30 + 8);
          if ( !v30 )
            goto LABEL_59;
        }
      }
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v24 = *(char **)(v23 + 24);
        v25 = *v24;
        if ( (unsigned __int8)*v24 > 0x1Cu && (v25 == 61 || v25 == 62) )
          break;
        v23 = *(_QWORD *)(v23 + 8);
        if ( !v23 )
          goto LABEL_48;
      }
      if ( (v24[2] & 1) != 0 )
        break;
      v23 = *(_QWORD *)(v23 + 8);
      if ( !v23 )
        goto LABEL_48;
    }
LABEL_33:
    ++v89;
  }
  while ( v84 != v89 );
  v10 = v97;
  v85 = &v97[(unsigned int)v98];
  if ( v97 != v85 )
    goto LABEL_63;
LABEL_93:
  if ( (_BYTE)v11 )
  {
    for ( m = *(_QWORD **)(a1 + 80); v91 != m; m = (_QWORD *)m[1] )
    {
      v50 = (__int64)(m - 3);
      if ( !m )
        v50 = 0;
      sub_F3F2F0(v50, (__int64)v3);
    }
  }
  v10 = v97;
LABEL_99:
  if ( v10 != (__int64 *)v99 )
    _libc_free(v10, v3);
LABEL_101:
  if ( v94 != (__int64 *)v96 )
    _libc_free(v94, v3);
  v51 = v125;
  if ( v125 )
  {
    v52 = v124;
    v53 = v124 + 56LL * v125;
    do
    {
      if ( *(_QWORD *)v52 != -8192 && *(_QWORD *)v52 != -4096 )
      {
        v54 = *(_QWORD *)(v52 + 8);
        v55 = v54 + 8LL * *(unsigned int *)(v52 + 16);
        if ( v54 != v55 )
        {
          do
          {
            v3 = *(_BYTE **)(v55 - 8);
            v55 -= 8;
            if ( v3 )
              sub_B91220(v55, (__int64)v3);
          }
          while ( v54 != v55 );
          v55 = *(_QWORD *)(v52 + 8);
        }
        if ( v55 != v52 + 24 )
          _libc_free(v55, v3);
      }
      v52 += 56;
    }
    while ( v53 != v52 );
    v51 = v125;
  }
  v56 = 56 * v51;
  sub_C7D6A0(v124, 56 * v51, 8);
  v57 = v121;
  v58 = &v121[8 * v122];
  if ( v121 != v58 )
  {
    do
    {
      v56 = *((_QWORD *)v58 - 1);
      v58 -= 8;
      if ( v56 )
        sub_B91220((__int64)v58, v56);
    }
    while ( v57 != v58 );
    v58 = v121;
  }
  if ( v58 != &v123 )
    _libc_free(v58, v56);
  v59 = v119;
  v60 = v119 + 56LL * v120;
  if ( v119 != v60 )
  {
    do
    {
      v60 -= 56;
      v61 = *(_QWORD *)(v60 + 40);
      if ( v61 != v60 + 56 )
        _libc_free(v61, v56);
      v56 = 8LL * *(unsigned int *)(v60 + 32);
      sub_C7D6A0(*(_QWORD *)(v60 + 16), v56, 8);
    }
    while ( v59 != v60 );
    v60 = v119;
  }
  if ( (char **)v60 != &v121 )
    _libc_free(v60, v56);
  v62 = 16LL * v118;
  sub_C7D6A0(v117, v62, 8);
  v63 = v114;
  v64 = &v114[8 * v115];
  if ( v114 != v64 )
  {
    do
    {
      v62 = *((_QWORD *)v64 - 1);
      v64 -= 8;
      if ( v62 )
        sub_B91220((__int64)v64, v62);
    }
    while ( v63 != v64 );
    v64 = v114;
  }
  if ( v64 != &v116 )
    _libc_free(v64, v62);
  if ( v112 != &v113 )
    _libc_free(v112, v62);
  if ( v110 != &v111 )
    _libc_free(v110, v62);
  v65 = v107;
  v66 = &v107[8 * v108];
  if ( v107 != v66 )
  {
    do
    {
      v62 = *((_QWORD *)v66 - 1);
      v66 -= 8;
      if ( v62 )
        sub_B91220((__int64)v66, v62);
    }
    while ( v65 != v66 );
    v66 = v107;
  }
  if ( v66 != &v109 )
    _libc_free(v66, v62);
  v67 = v104;
  v68 = &v104[8 * v105];
  if ( v104 != v68 )
  {
    do
    {
      v62 = *((_QWORD *)v68 - 1);
      v68 -= 8;
      if ( v62 )
        sub_B91220((__int64)v68, v62);
    }
    while ( v67 != v68 );
    v68 = v104;
  }
  if ( v68 != &v106 )
    _libc_free(v68, v62);
  return v11;
}
