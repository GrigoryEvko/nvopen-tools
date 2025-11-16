// Function: sub_29E1C50
// Address: 0x29e1c50
//
void __fastcall sub_29E1C50(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // rax
  __int64 v5; // rax
  int v6; // edx
  __int64 v7; // r13
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rbx
  int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  int v15; // r14d
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  int v24; // r13d
  const char *v25; // rbx
  int i; // r14d
  __int64 v27; // rsi
  _BYTE *v28; // r14
  __int64 v29; // rbx
  unsigned __int64 *v30; // r15
  unsigned __int64 *v31; // rax
  __int64 v32; // r15
  __int64 *v33; // r14
  __int64 v34; // r10
  __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rcx
  unsigned int v38; // ebx
  unsigned int v39; // edi
  _QWORD *v40; // rdx
  __int64 v41; // rsi
  unsigned __int8 *v42; // r13
  int v43; // eax
  unsigned __int64 v44; // rax
  __int64 v45; // rcx
  _BYTE *v46; // rbx
  unsigned __int64 v47; // rdi
  _BYTE *v48; // rbx
  unsigned __int64 v49; // r12
  unsigned __int64 v50; // rdi
  int v51; // edx
  int v52; // r9d
  __int64 v53; // rcx
  int v54; // eax
  int v55; // edx
  unsigned int v56; // ebx
  __int64 *v57; // rax
  __int64 v58; // rsi
  int v59; // edx
  __int64 v60; // r12
  __int64 v61; // rax
  __int64 v62; // rdx
  __int64 v63; // rbx
  int v64; // ebx
  __int64 v65; // rax
  __int64 v66; // rdx
  __int64 v67; // rdx
  __int64 v68; // rbx
  __int64 v69; // rax
  __int64 v70; // rax
  unsigned __int64 v71; // rax
  __int64 v72; // rdx
  unsigned __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rax
  __int64 v76; // rdx
  unsigned __int64 v77; // rax
  unsigned __int16 v78; // ax
  int v79; // r12d
  _BYTE *v80; // rdi
  __int64 v81; // rdi
  unsigned __int8 *v82; // rax
  int v83; // eax
  int v84; // edi
  _BYTE *v85; // rbx
  unsigned __int64 v86; // rdi
  _BYTE *v87; // rbx
  unsigned __int64 v88; // rdi
  __int64 v89; // [rsp+18h] [rbp-268h]
  __int64 v90; // [rsp+20h] [rbp-260h]
  __int64 v93; // [rsp+40h] [rbp-240h]
  unsigned __int64 v94; // [rsp+40h] [rbp-240h]
  unsigned __int64 v95; // [rsp+40h] [rbp-240h]
  unsigned __int8 v96; // [rsp+40h] [rbp-240h]
  __int64 v97; // [rsp+48h] [rbp-238h]
  int v98; // [rsp+58h] [rbp-228h]
  char v99; // [rsp+60h] [rbp-220h]
  __int64 v100; // [rsp+60h] [rbp-220h]
  __int64 *v101; // [rsp+68h] [rbp-218h]
  int v102; // [rsp+70h] [rbp-210h]
  __int64 v103; // [rsp+70h] [rbp-210h]
  unsigned __int64 v104; // [rsp+88h] [rbp-1F8h] BYREF
  unsigned __int64 v105; // [rsp+90h] [rbp-1F0h] BYREF
  unsigned int v106; // [rsp+98h] [rbp-1E8h]
  unsigned __int64 v107; // [rsp+A0h] [rbp-1E0h]
  unsigned int v108; // [rsp+A8h] [rbp-1D8h]
  unsigned __int64 v109; // [rsp+B0h] [rbp-1D0h] BYREF
  unsigned int v110; // [rsp+B8h] [rbp-1C8h]
  unsigned __int64 v111; // [rsp+C0h] [rbp-1C0h]
  unsigned int v112; // [rsp+C8h] [rbp-1B8h]
  char v113; // [rsp+D0h] [rbp-1B0h]
  unsigned __int64 v114; // [rsp+E0h] [rbp-1A0h] BYREF
  __int64 v115; // [rsp+E8h] [rbp-198h]
  unsigned __int64 v116; // [rsp+F0h] [rbp-190h]
  unsigned int v117; // [rsp+F8h] [rbp-188h]
  char v118; // [rsp+100h] [rbp-180h]
  __int64 v119; // [rsp+110h] [rbp-170h] BYREF
  _BYTE *v120; // [rsp+118h] [rbp-168h]
  __int64 v121; // [rsp+120h] [rbp-160h]
  _BYTE v122[72]; // [rsp+128h] [rbp-158h] BYREF
  _BYTE *v123; // [rsp+170h] [rbp-110h] BYREF
  __int64 v124; // [rsp+178h] [rbp-108h]
  _BYTE v125[96]; // [rsp+180h] [rbp-100h] BYREF
  _BYTE *v126; // [rsp+1E0h] [rbp-A0h] BYREF
  __int64 v127; // [rsp+1E8h] [rbp-98h]
  _BYTE v128[144]; // [rsp+1F0h] [rbp-90h] BYREF

  v3 = *((_QWORD *)a1 - 4);
  v93 = v3;
  if ( !v3 )
  {
LABEL_6:
    v5 = sub_B2BE50(v93);
    goto LABEL_7;
  }
  if ( !*(_BYTE *)v3 )
  {
    v4 = 0;
    if ( *((_QWORD *)a1 + 10) == *(_QWORD *)(v3 + 24) )
      v4 = *((_QWORD *)a1 - 4);
    v93 = v4;
    goto LABEL_6;
  }
  v93 = 0;
  v5 = sub_B2BE50(0);
LABEL_7:
  v6 = *a1;
  v101 = (__int64 *)v5;
  v123 = v125;
  v124 = 0x100000000LL;
  v127 = 0x100000000LL;
  v126 = v128;
  if ( v6 == 40 )
  {
    v7 = 32LL * (unsigned int)sub_B491D0((__int64)a1);
  }
  else
  {
    v7 = 0;
    if ( v6 != 85 )
    {
      v7 = 64;
      if ( v6 != 34 )
LABEL_183:
        BUG();
    }
  }
  if ( (a1[7] & 0x80u) != 0 )
  {
    v8 = sub_BD2BC0((__int64)a1);
    v10 = v8 + v9;
    if ( (a1[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v10 >> 4) )
        goto LABEL_184;
    }
    else if ( (unsigned int)((v10 - sub_BD2BC0((__int64)a1)) >> 4) )
    {
      if ( (a1[7] & 0x80u) != 0 )
      {
        v11 = *(_DWORD *)(sub_BD2BC0((__int64)a1) + 8);
        if ( (a1[7] & 0x80u) == 0 )
          BUG();
        v12 = sub_BD2BC0((__int64)a1);
        v14 = 32LL * (unsigned int)(*(_DWORD *)(v12 + v13 - 4) - v11);
        goto LABEL_17;
      }
LABEL_184:
      BUG();
    }
  }
  v14 = 0;
LABEL_17:
  v98 = (32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v7 - v14) >> 5;
  if ( !v98 )
  {
    v28 = v126;
    v29 = 88LL * (unsigned int)v127;
    goto LABEL_168;
  }
  v99 = 0;
  v15 = 0;
  while ( 2 )
  {
    v119 = sub_BD5C60((__int64)a1);
    v120 = v122;
    v121 = 0x800000000LL;
    sub_29E0080((__int64)&v123, (__int64)&v119, v16, v17, v18, v19);
    if ( v120 != v122 )
      _libc_free((unsigned __int64)v120);
    v119 = sub_BD5C60((__int64)a1);
    v120 = v122;
    v121 = 0x800000000LL;
    sub_29E0080((__int64)&v126, (__int64)&v119, v20, v21, v22, v23);
    if ( v120 != v122 )
      _libc_free((unsigned __int64)v120);
    if ( (unsigned __int8)sub_B49B80((__int64)a1, v15, 50) )
      sub_A77B20((__int64 **)&v123[88 * (unsigned int)v124 - 88], 50);
    if ( (unsigned __int8)sub_B49B80((__int64)a1, v15, 51) )
      sub_A77B20((__int64 **)&v123[88 * (unsigned int)v124 - 88], 51);
    v102 = v15;
    v24 = v15 + 1;
    v25 = "Z";
    for ( i = 90; ; i = *(_DWORD *)v25 )
    {
      v119 = *((_QWORD *)a1 + 9);
      v27 = sub_A747F0(&v119, v24, i);
      if ( !v27 )
      {
        v27 = sub_B49640((__int64)a1, v102, i);
        if ( !v27 )
          break;
      }
      v25 += 4;
      sub_A77670((__int64)&v126[88 * (unsigned int)v127 - 88], v27);
      if ( &unk_439B064 == (_UNKNOWN *)v25 )
        goto LABEL_33;
LABEL_30:
      ;
    }
    v25 += 4;
    if ( &unk_439B064 != (_UNKNOWN *)v25 )
      goto LABEL_30;
LABEL_33:
    v28 = v126;
    v29 = 88LL * (unsigned int)v127;
    v99 |= (*(_DWORD *)&v126[v29 - 72] | *(_DWORD *)&v123[88 * (unsigned int)v124 - 72]) != 0;
    if ( v98 != v24 )
    {
      v15 = v24;
      continue;
    }
    break;
  }
  if ( v99 )
  {
    v89 = v93 + 72;
    v90 = *(_QWORD *)(v93 + 80);
    if ( v93 + 72 == v90 )
      goto LABEL_63;
    v30 = &v104;
    while ( 2 )
    {
      if ( !v90 )
        BUG();
      if ( v90 + 24 == *(_QWORD *)(v90 + 32) )
        goto LABEL_61;
      v31 = v30;
      v32 = *(_QWORD *)(v90 + 32);
      v33 = (__int64 *)v31;
LABEL_43:
      if ( !v32 )
        BUG();
      v34 = v32 - 24;
      v103 = v32 - 24;
      if ( (unsigned __int8)(*(_BYTE *)(v32 - 24) - 34) <= 0x33u )
      {
        v35 = 0x8000000000041LL;
        if ( _bittest64(&v35, (unsigned int)*(unsigned __int8 *)(v32 - 24) - 34) )
        {
          v36 = *(unsigned int *)(a2 + 24);
          if ( (_DWORD)v36 )
          {
            v37 = *(_QWORD *)(a2 + 8);
            v38 = ((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4);
            v39 = (v36 - 1) & v38;
            v40 = (_QWORD *)(v37 + ((unsigned __int64)v39 << 6));
            v41 = v40[3];
            if ( v41 == v34 )
            {
LABEL_48:
              if ( v40 != (_QWORD *)(v37 + (v36 << 6)) )
              {
                v119 = 6;
                v120 = 0;
                v121 = v40[7];
                v42 = (unsigned __int8 *)v121;
                if ( v121 != -4096 && v121 != 0 && v121 != -8192 )
                {
                  sub_BD6050((unsigned __int64 *)&v119, v40[5] & 0xFFFFFFFFFFFFFFF8LL);
                  v42 = (unsigned __int8 *)v121;
                }
                if ( v42 )
                {
                  v43 = *v42;
                  if ( (unsigned __int8)v43 <= 0x1Cu
                    || (v44 = (unsigned int)(v43 - 34), (unsigned __int8)v44 > 0x33u)
                    || (v45 = 0x8000000000041LL, !_bittest64(&v45, v44)) )
                  {
                    if ( v42 != (unsigned __int8 *)-4096LL && v42 != (unsigned __int8 *)-8192LL )
                      sub_BD60C0(&v119);
                    goto LABEL_59;
                  }
                  if ( v42 != (unsigned __int8 *)-4096LL && v42 != (unsigned __int8 *)-8192LL )
                    sub_BD60C0(&v119);
                  v53 = *(_QWORD *)(a3 + 40);
                  v54 = *(_DWORD *)(a3 + 56);
                  if ( v54 )
                  {
                    v55 = v54 - 1;
                    v56 = (v54 - 1) & v38;
                    v57 = (__int64 *)(v53 + 16LL * v56);
                    v58 = *v57;
                    if ( v103 == *v57 )
                    {
LABEL_87:
                      if ( (unsigned __int8 *)v57[1] != v42 )
                        goto LABEL_59;
                      v104 = *((_QWORD *)v42 + 9);
                      v59 = *(unsigned __int8 *)(v32 - 24);
                      if ( v59 == 40 )
                      {
                        v60 = 32LL * (unsigned int)sub_B491D0(v103);
                      }
                      else
                      {
                        v60 = 0;
                        if ( v59 != 85 )
                        {
                          if ( v59 != 34 )
                            goto LABEL_183;
                          v60 = 64;
                        }
                      }
                      if ( *(char *)(v32 - 17) < 0 )
                      {
                        v61 = sub_BD2BC0(v103);
                        v63 = v61 + v62;
                        if ( *(char *)(v32 - 17) >= 0 )
                        {
                          if ( (unsigned int)(v63 >> 4) )
LABEL_187:
                            BUG();
                        }
                        else if ( (unsigned int)((v63 - sub_BD2BC0(v103)) >> 4) )
                        {
                          if ( *(char *)(v32 - 17) >= 0 )
                            goto LABEL_187;
                          v64 = *(_DWORD *)(sub_BD2BC0(v103) + 8);
                          if ( *(char *)(v32 - 17) >= 0 )
                            BUG();
                          v65 = sub_BD2BC0(v103);
                          v67 = 32LL * (unsigned int)(*(_DWORD *)(v65 + v66 - 4) - v64);
                          goto LABEL_98;
                        }
                      }
                      v67 = 0;
LABEL_98:
                      v68 = 0;
                      v69 = (32LL * (*(_DWORD *)(v32 - 20) & 0x7FFFFFF) - 32 - v60 - v67) >> 5;
                      v100 = (unsigned int)v69;
                      if ( !(_DWORD)v69 )
                      {
LABEL_133:
                        *((_QWORD *)v42 + 9) = v104;
                        goto LABEL_59;
                      }
                      while ( 1 )
                      {
LABEL_124:
                        v79 = v68 + 1;
                        if ( (unsigned __int8)sub_B49B80((__int64)v42, v68, 81) )
                          goto LABEL_123;
                        v80 = *(_BYTE **)&v42[32 * (v68 - (*((_DWORD *)v42 + 1) & 0x7FFFFFF))];
                        if ( *v80 > 0x15u || *v80 == 5 || (unsigned __int8)sub_AD6CA0((__int64)v80) )
                          break;
                        if ( ++v68 == v100 )
                          goto LABEL_133;
                      }
                      v81 = *(_QWORD *)(v103 + 32 * (v68 - (*(_DWORD *)(v32 - 20) & 0x7FFFFFF)));
                      if ( *(_BYTE *)v81 == 22 )
                      {
                        v97 = 88LL * *(unsigned int *)(v81 + 32);
                        v70 = sub_A7A280(v101, (__int64)&v126[v97]);
                        sub_A74940((__int64)&v119, (__int64)v101, v70);
                        v94 = sub_A745B0(v33, v68);
                        v114 = sub_A74DF0((__int64)&v119, 90);
                        v71 = 0;
                        v115 = v72;
                        if ( (_BYTE)v72 )
                          v71 = v114;
                        if ( v94 > v71 )
                          sub_A77390((__int64)&v119, 90);
                        v95 = sub_A745D0(v33, v68);
                        v114 = sub_A74DF0((__int64)&v119, 91);
                        v73 = 0;
                        v115 = v74;
                        if ( (_BYTE)v74 )
                          v73 = v114;
                        if ( v95 > v73 )
                          sub_A77390((__int64)&v119, 91);
                        v75 = sub_A74DF0((__int64)&v119, 86);
                        v115 = v76;
                        v114 = v75;
                        if ( (_BYTE)v76 && v114 )
                        {
                          _BitScanReverse64(&v77, v114);
                          v96 = 63 - (v77 ^ 0x3F);
                        }
                        else
                        {
                          v96 = 0;
                        }
                        v78 = sub_A74840(v33, v68);
                        if ( HIBYTE(v78) && (unsigned __int8)v78 > v96 )
                          sub_A77390((__int64)&v119, 86);
                        sub_A744F0((__int64)&v109, v33, v68);
                        if ( v113 )
                        {
                          sub_A74F90((__int64)&v114, (__int64)&v119);
                          if ( v118 )
                          {
                            sub_AB2160((__int64)&v105, (__int64)&v109, (__int64)&v114, 0);
                            sub_A77390((__int64)&v119, 97);
                            sub_A78C10((_QWORD **)&v119, (__int64)&v105);
                            if ( v108 > 0x40 && v107 )
                              j_j___libc_free_0_0(v107);
                            if ( v106 > 0x40 && v105 )
                              j_j___libc_free_0_0(v105);
                            if ( v118 )
                            {
                              v118 = 0;
                              if ( v117 > 0x40 && v116 )
                                j_j___libc_free_0_0(v116);
                              if ( (unsigned int)v115 > 0x40 && v114 )
                                j_j___libc_free_0_0(v114);
                            }
                          }
                          if ( v113 )
                          {
                            v113 = 0;
                            if ( v112 > 0x40 && v111 )
                              j_j___libc_free_0_0(v111);
                            if ( v110 > 0x40 && v109 )
                              j_j___libc_free_0_0(v109);
                          }
                        }
                        v104 = sub_A7B2C0(v33, v101, v79, (__int64)&v119);
                        if ( v120 != v122 )
                          _libc_free((unsigned __int64)v120);
LABEL_117:
                        v104 = sub_A7B2C0(v33, v101, v79, (__int64)&v123[v97]);
                        if ( (unsigned __int8)sub_A74710(v33, v79, 51) && (unsigned __int8)sub_A74710(v33, v79, 78) )
                          v104 = sub_A7A090(v33, v101, v79, 50);
                        if ( (unsigned __int8)sub_A74710(v33, v79, 50) )
                        {
                          v104 = sub_A7B980(v33, v101, v79, 51);
                          v104 = sub_A7B980(v33, v101, v79, 78);
                        }
                        if ( (unsigned __int8)sub_A74710(v33, v79, 51) || (unsigned __int8)sub_A74710(v33, v79, 50) )
                          v104 = sub_A7B980(v33, v101, v79, 77);
                      }
                      else if ( *(_BYTE *)(*(_QWORD *)(*(_QWORD *)&v42[32 * (v68 - (*((_DWORD *)v42 + 1) & 0x7FFFFFF))]
                                                     + 8LL)
                                         + 8LL) == 14 )
                      {
                        v82 = sub_98ACB0((unsigned __int8 *)v81, 6u);
                        if ( *v82 == 22 )
                        {
                          v97 = 88LL * *((unsigned int *)v82 + 8);
                          goto LABEL_117;
                        }
                      }
LABEL_123:
                      if ( ++v68 == v100 )
                        goto LABEL_133;
                      goto LABEL_124;
                    }
                    v83 = 1;
                    while ( v58 != -4096 )
                    {
                      v84 = v83 + 1;
                      v56 = v55 & (v83 + v56);
                      v57 = (__int64 *)(v53 + 16LL * v56);
                      v58 = *v57;
                      if ( v103 == *v57 )
                        goto LABEL_87;
                      v83 = v84;
                    }
                  }
                }
              }
            }
            else
            {
              v51 = 1;
              while ( v41 != -4096 )
              {
                v52 = v51 + 1;
                v39 = (v36 - 1) & (v51 + v39);
                v40 = (_QWORD *)(v37 + ((unsigned __int64)v39 << 6));
                v41 = v40[3];
                if ( v103 == v41 )
                  goto LABEL_48;
                v51 = v52;
              }
            }
          }
        }
      }
LABEL_59:
      v32 = *(_QWORD *)(v32 + 8);
      if ( v90 + 24 == v32 )
      {
        v30 = (unsigned __int64 *)v33;
LABEL_61:
        v90 = *(_QWORD *)(v90 + 8);
        if ( v89 == v90 )
        {
          v28 = v126;
          v29 = 88LL * (unsigned int)v127;
LABEL_63:
          v46 = &v28[v29];
          if ( v46 != v28 )
          {
            do
            {
              v46 -= 88;
              v47 = *((_QWORD *)v46 + 1);
              if ( (_BYTE *)v47 != v46 + 24 )
                _libc_free(v47);
            }
            while ( v46 != v28 );
            v28 = v126;
          }
          if ( v28 != v128 )
            _libc_free((unsigned __int64)v28);
          v48 = v123;
          v49 = (unsigned __int64)&v123[88 * (unsigned int)v124];
          if ( v123 != (_BYTE *)v49 )
          {
            do
            {
              v49 -= 88LL;
              v50 = *(_QWORD *)(v49 + 8);
              if ( v50 != v49 + 24 )
                _libc_free(v50);
            }
            while ( v48 != (_BYTE *)v49 );
LABEL_74:
            v49 = (unsigned __int64)v123;
            goto LABEL_75;
          }
          goto LABEL_75;
        }
        continue;
      }
      goto LABEL_43;
    }
  }
LABEL_168:
  v85 = &v28[v29];
  if ( v85 != v28 )
  {
    do
    {
      v85 -= 88;
      v86 = *((_QWORD *)v85 + 1);
      if ( (_BYTE *)v86 != v85 + 24 )
        _libc_free(v86);
    }
    while ( v85 != v28 );
    v28 = v126;
  }
  if ( v28 != v128 )
    _libc_free((unsigned __int64)v28);
  v87 = v123;
  v49 = (unsigned __int64)&v123[88 * (unsigned int)v124];
  if ( v123 != (_BYTE *)v49 )
  {
    do
    {
      v49 -= 88LL;
      v88 = *(_QWORD *)(v49 + 8);
      if ( v88 != v49 + 24 )
        _libc_free(v88);
    }
    while ( v87 != (_BYTE *)v49 );
    goto LABEL_74;
  }
LABEL_75:
  if ( (_BYTE *)v49 != v125 )
    _libc_free(v49);
}
