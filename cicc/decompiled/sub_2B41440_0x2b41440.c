// Function: sub_2B41440
// Address: 0x2b41440
//
void __fastcall sub_2B41440(__int64 a1)
{
  __int64 **v1; // r12
  int v2; // edx
  __int64 **v3; // r13
  unsigned __int64 v4; // rbx
  unsigned __int64 *v5; // r12
  __int64 v6; // rax
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // r13
  unsigned __int64 v9; // rdi
  __int64 v10; // rbx
  unsigned __int64 v11; // r12
  __int64 v12; // r15
  __int64 v13; // r14
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  __int64 v16; // rbx
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // r13
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
  _QWORD *v27; // r12
  __int64 v28; // rsi
  _QWORD *v29; // rbx
  unsigned __int64 v30; // rdi
  __int64 v31; // rax
  __int64 v32; // rdx
  _QWORD *v33; // r12
  __int64 v34; // rsi
  _QWORD *v35; // rbx
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rbx
  unsigned __int64 v38; // r13
  unsigned __int64 v39; // r14
  unsigned __int64 *v40; // r15
  unsigned __int64 *v41; // r12
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rdi
  __int64 v45; // rax
  __int64 **v46; // rbx
  __int64 v47; // r10
  __int64 *v48; // r12
  __int64 *v49; // r11
  __int64 v50; // rax
  __int64 *v51; // rdx
  __int64 *v52; // r9
  __int64 *v53; // r14
  __int64 *v54; // r12
  __int64 *v55; // r14
  __int64 *v56; // rbx
  __int64 v57; // r13
  int v58; // edx
  __int64 v59; // rcx
  unsigned int v60; // eax
  unsigned __int8 *v61; // rsi
  unsigned __int8 *v62; // r15
  int v63; // eax
  int v64; // eax
  unsigned __int64 *v65; // rdi
  __int64 *v66; // r12
  __int64 v67; // rax
  __int64 v68; // rdi
  unsigned __int64 v69; // rax
  unsigned __int64 v70; // rsi
  __int64 v71; // rax
  int v72; // edi
  __int64 v73; // rax
  char v74; // dh
  unsigned __int64 *v75; // r8
  char v76; // al
  __int64 v77; // rcx
  __int64 v78; // rsi
  unsigned __int64 *v79; // rdi
  int v80; // r15d
  unsigned __int64 *v81; // [rsp+8h] [rbp-D8h]
  __int64 v82; // [rsp+18h] [rbp-C8h]
  __int64 v83; // [rsp+28h] [rbp-B8h]
  __int64 **v84; // [rsp+30h] [rbp-B0h]
  __int64 v85; // [rsp+38h] [rbp-A8h]
  __int64 **v86; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v88; // [rsp+48h] [rbp-98h]
  __int64 *v89; // [rsp+48h] [rbp-98h]
  __int64 v90; // [rsp+48h] [rbp-98h]
  __int64 v91; // [rsp+48h] [rbp-98h]
  unsigned __int64 v92[2]; // [rsp+50h] [rbp-90h] BYREF
  void (__fastcall *v93)(unsigned __int64 *, unsigned __int64 *, __int64); // [rsp+60h] [rbp-80h]
  unsigned __int64 *v94; // [rsp+70h] [rbp-70h] BYREF
  __int64 v95; // [rsp+78h] [rbp-68h]
  _BYTE v96[96]; // [rsp+80h] [rbp-60h] BYREF

  v1 = *(__int64 ***)(a1 + 1984);
  v94 = (unsigned __int64 *)v96;
  v2 = *(_DWORD *)(a1 + 1992);
  v95 = 0x200000000LL;
  v3 = &v1[*(unsigned int *)(a1 + 2000)];
  if ( !v2 )
    goto LABEL_2;
  if ( v1 == v3 )
    goto LABEL_126;
  v46 = v1;
  while ( *v46 == (__int64 *)-8192LL || *v46 == (__int64 *)-4096LL )
  {
    if ( ++v46 == v3 )
      goto LABEL_126;
  }
  if ( v46 == v3 )
    goto LABEL_130;
  v47 = a1;
  do
  {
    v48 = *v46;
    if ( (*v46)[5] )
    {
      v49 = *v46;
      v50 = 32LL * (*((_DWORD *)v48 + 1) & 0x7FFFFFF);
      if ( (*((_BYTE *)v48 + 7) & 0x40) != 0 )
      {
        v51 = (__int64 *)*(v48 - 1);
        v52 = &v51[(unsigned __int64)v50 / 8];
        v53 = v51;
        if ( v51 == &v51[(unsigned __int64)v50 / 8] )
          goto LABEL_159;
      }
      else
      {
        v53 = &v48[v50 / 0xFFFFFFFFFFFFFFF8LL];
        if ( v48 == &v48[v50 / 0xFFFFFFFFFFFFFFF8LL] )
          goto LABEL_179;
        v52 = *v46;
      }
      v89 = *v46;
      v54 = v53;
      v55 = *v46;
      v86 = v46;
      v56 = v52;
      v84 = v3;
      v57 = v47;
      while ( 1 )
      {
        v62 = (unsigned __int8 *)*v54;
        if ( *(_BYTE *)*v54 <= 0x1Cu )
          goto LABEL_145;
        v63 = *(_DWORD *)(v57 + 2000);
        if ( !v63 )
          goto LABEL_148;
        v58 = v63 - 1;
        v59 = *(_QWORD *)(v57 + 1984);
        v60 = (v63 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
        v61 = *(unsigned __int8 **)(v59 + 8LL * v60);
        if ( v62 == v61 )
        {
LABEL_145:
          v54 += 4;
          if ( v56 == v54 )
            goto LABEL_157;
        }
        else
        {
          v72 = 1;
          while ( v61 != (unsigned __int8 *)-4096LL )
          {
            v60 = v58 & (v72 + v60);
            v61 = *(unsigned __int8 **)(v59 + 8LL * v60);
            if ( v62 == v61 )
              goto LABEL_145;
            ++v72;
          }
LABEL_148:
          if ( !(unsigned __int8)sub_BD36B0(*v54) || !sub_F509B0(v62, *(__int64 **)(v57 + 3304)) )
            goto LABEL_145;
          v64 = v95;
          if ( HIDWORD(v95) <= (unsigned int)v95 )
          {
            v81 = (unsigned __int64 *)sub_C8D7D0((__int64)&v94, (__int64)v96, 0, 0x18u, v92, (__int64)&v94);
            v79 = &v81[3 * (unsigned int)v95];
            if ( v79 )
            {
              *v79 = 6;
              v79[1] = 0;
              v79[2] = (unsigned __int64)v62;
              if ( v62 != (unsigned __int8 *)-4096LL && v62 != (unsigned __int8 *)-8192LL )
                sub_BD73F0((__int64)v79);
            }
            sub_F17F80((__int64)&v94, v81);
            v80 = v92[0];
            if ( v94 != (unsigned __int64 *)v96 )
              _libc_free((unsigned __int64)v94);
            LODWORD(v95) = v95 + 1;
            HIDWORD(v95) = v80;
            v94 = v81;
            goto LABEL_145;
          }
          v65 = &v94[3 * (unsigned int)v95];
          if ( v65 )
          {
            *v65 = 6;
            v65[1] = 0;
            v65[2] = (unsigned __int64)v62;
            if ( v62 != (unsigned __int8 *)-4096LL && v62 != (unsigned __int8 *)-8192LL )
              sub_BD73F0((__int64)v65);
            v64 = v95;
          }
          v54 += 4;
          LODWORD(v95) = v64 + 1;
          if ( v56 == v54 )
          {
LABEL_157:
            v48 = v89;
            v47 = v57;
            v46 = v86;
            v49 = v55;
            v3 = v84;
            v50 = 32LL * (*((_DWORD *)v89 + 1) & 0x7FFFFFF);
            if ( (*((_BYTE *)v89 + 7) & 0x40) != 0 )
            {
              v51 = (__int64 *)*(v89 - 1);
LABEL_159:
              v66 = v51;
              v49 = &v51[(unsigned __int64)v50 / 8];
LABEL_160:
              while ( v49 != v66 )
              {
                if ( *v66 )
                {
                  v67 = v66[1];
                  *(_QWORD *)v66[2] = v67;
                  if ( v67 )
                    *(_QWORD *)(v67 + 16) = v66[2];
                }
                *v66 = 0;
                v66 += 4;
              }
              goto LABEL_165;
            }
LABEL_179:
            v66 = &v48[v50 / 0xFFFFFFFFFFFFFFF8LL];
            goto LABEL_160;
          }
        }
      }
    }
    v68 = *(_QWORD *)(*(_QWORD *)(v47 + 3280) + 80LL);
    if ( *(_BYTE *)v48 == 84 )
    {
      v91 = v47;
      if ( v68 )
        v68 -= 24;
      v73 = sub_AA4FF0(v68);
      v77 = v82;
      v75 = (unsigned __int64 *)v73;
      v76 = 0;
      LOBYTE(v77) = 1;
      if ( v75 )
        v76 = v74;
      BYTE1(v77) = v76;
      v82 = v77;
      v78 = *(_QWORD *)(*(_QWORD *)(v91 + 3280) + 80LL);
      if ( v78 )
        v78 -= 24;
      sub_B44150(v48, v78, v75, v77);
      v47 = v91;
    }
    else
    {
      if ( !v68 )
        BUG();
      v69 = *(_QWORD *)(v68 + 24) & 0xFFFFFFFFFFFFFFF8LL;
      if ( v69 == v68 + 24 )
      {
        v70 = 0;
      }
      else
      {
        if ( !v69 )
          BUG();
        v70 = v69 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v69 - 24) - 30 >= 0xB )
          v70 = (*v46)[5];
      }
      v71 = v83;
      v90 = v47;
      LOWORD(v71) = 0;
      v83 = v71;
      sub_B44220(*v46, v70 + 24, v71);
      v47 = v90;
    }
    do
    {
LABEL_165:
      if ( ++v46 == v3 )
        goto LABEL_169;
    }
    while ( *v46 == (__int64 *)-8192LL || *v46 == (__int64 *)-4096LL );
  }
  while ( v46 != v3 );
LABEL_169:
  v1 = *(__int64 ***)(a1 + 1984);
  v2 = *(_DWORD *)(a1 + 1992);
  v3 = &v1[*(unsigned int *)(a1 + 2000)];
LABEL_126:
  if ( v2 && v1 != v3 )
  {
LABEL_130:
    while ( *v1 == (__int64 *)-8192LL || *v1 == (__int64 *)-4096LL )
    {
      if ( v3 == ++v1 )
        goto LABEL_2;
    }
LABEL_132:
    if ( v3 != v1 )
    {
      sub_B43D60(*v1);
      while ( v3 != ++v1 )
      {
        if ( *v1 != (__int64 *)-8192LL && *v1 != (__int64 *)-4096LL )
          goto LABEL_132;
      }
    }
  }
LABEL_2:
  v93 = 0;
  sub_F5C330((__int64)&v94, *(__int64 **)(a1 + 3304), 0, (__int64)v92);
  if ( v93 )
    v93(v92, v92, 3);
  v4 = (unsigned __int64)v94;
  v5 = &v94[3 * (unsigned int)v95];
  if ( v94 != v5 )
  {
    do
    {
      v6 = *(v5 - 1);
      v5 -= 3;
      if ( v6 != -4096 && v6 != 0 && v6 != -8192 )
        sub_BD60C0(v5);
    }
    while ( (unsigned __int64 *)v4 != v5 );
    v5 = v94;
  }
  if ( v5 != (unsigned __int64 *)v96 )
    _libc_free((unsigned __int64)v5);
  sub_C7D6A0(*(_QWORD *)(a1 + 3584), 4LL * *(unsigned int *)(a1 + 3600), 4);
  sub_C7D6A0(*(_QWORD *)(a1 + 3528), 24LL * *(unsigned int *)(a1 + 3544), 8);
  nullsub_61();
  *(_QWORD *)(a1 + 3496) = &unk_49D94D0;
  nullsub_63();
  v7 = *(_QWORD *)(a1 + 3368);
  if ( v7 != a1 + 3384 )
    _libc_free(v7);
  v85 = *(_QWORD *)(a1 + 3256);
  v88 = v85 + 16LL * *(unsigned int *)(a1 + 3264);
  if ( v85 != v88 )
  {
    do
    {
      v88 -= 16LL;
      v8 = *(_QWORD *)(v88 + 8);
      if ( v8 )
      {
        v9 = *(_QWORD *)(v8 + 144);
        if ( v9 != v8 + 160 )
          _libc_free(v9);
        sub_C7D6A0(*(_QWORD *)(v8 + 120), 8LL * *(unsigned int *)(v8 + 136), 8);
        sub_C7D6A0(*(_QWORD *)(v8 + 88), 16LL * *(unsigned int *)(v8 + 104), 8);
        v10 = *(_QWORD *)(v8 + 8);
        v11 = v10 + 8LL * *(unsigned int *)(v8 + 16);
        if ( v10 != v11 )
        {
          do
          {
            v12 = *(_QWORD *)(v11 - 8);
            v11 -= 8LL;
            if ( v12 )
            {
              v13 = v12 + 160LL * *(_QWORD *)(v12 - 8);
              while ( v12 != v13 )
              {
                v13 -= 160;
                v14 = *(_QWORD *)(v13 + 88);
                if ( v14 != v13 + 104 )
                  _libc_free(v14);
                v15 = *(_QWORD *)(v13 + 40);
                if ( v15 != v13 + 56 )
                  _libc_free(v15);
              }
              j_j_j___libc_free_0_0(v12 - 8);
            }
          }
          while ( v10 != v11 );
          v11 = *(_QWORD *)(v8 + 8);
        }
        if ( v11 != v8 + 24 )
          _libc_free(v11);
        j_j___libc_free_0(v8);
      }
    }
    while ( v85 != v88 );
    v88 = *(_QWORD *)(a1 + 3256);
  }
  if ( v88 != a1 + 3272 )
    _libc_free(v88);
  v16 = a1;
  sub_C7D6A0(*(_QWORD *)(a1 + 3232), 16LL * *(unsigned int *)(a1 + 3248), 8);
  sub_C7D6A0(*(_QWORD *)(v16 + 3200), 8LL * *(unsigned int *)(v16 + 3216), 8);
  sub_C7D6A0(*(_QWORD *)(v16 + 3168), 8LL * *(unsigned int *)(v16 + 3184), 8);
  v17 = *(_QWORD *)(a1 + 3144);
  if ( v17 != a1 + 3160 )
    _libc_free(v17);
  sub_C7D6A0(*(_QWORD *)(a1 + 3120), 8LL * *(unsigned int *)(a1 + 3136), 8);
  if ( !*(_BYTE *)(a1 + 2852) )
    _libc_free(*(_QWORD *)(a1 + 2832));
  if ( !*(_BYTE *)(a1 + 2788) )
    _libc_free(*(_QWORD *)(a1 + 2768));
  v18 = *(_QWORD *)(a1 + 2232);
  if ( v18 != a1 + 2248 )
    _libc_free(v18);
  sub_C7D6A0(*(_QWORD *)(a1 + 2208), 8LL * *(unsigned int *)(a1 + 2224), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 2176), 8LL * *(unsigned int *)(a1 + 2192), 8);
  if ( !*(_BYTE *)(a1 + 2036) )
    _libc_free(*(_QWORD *)(a1 + 2016));
  sub_C7D6A0(*(_QWORD *)(a1 + 1984), 8LL * *(unsigned int *)(a1 + 2000), 8);
  *(_QWORD *)(a1 + 1824) = &unk_49DDBE8;
  if ( (*(_BYTE *)(a1 + 1840) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 1848), 16LL * *(unsigned int *)(a1 + 1856), 8);
  nullsub_184();
  v19 = *(_QWORD *)(a1 + 1672);
  if ( v19 != a1 + 1688 )
    _libc_free(v19);
  if ( (*(_BYTE *)(a1 + 1320) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 1328), 40LL * *(unsigned int *)(a1 + 1336), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 1272), 24LL * *(unsigned int *)(a1 + 1288), 8);
  v20 = *(_QWORD *)(a1 + 1232);
  if ( v20 != a1 + 1248 )
    _libc_free(v20);
  sub_C7D6A0(*(_QWORD *)(a1 + 1208), 4LL * *(unsigned int *)(a1 + 1224), 4);
  v21 = *(unsigned int *)(a1 + 1192);
  if ( (_DWORD)v21 )
  {
    v22 = *(_QWORD *)(a1 + 1176);
    v23 = v22 + 88 * v21;
    do
    {
      if ( *(_QWORD *)v22 != -4096 && *(_QWORD *)v22 != -8192 )
      {
        v24 = *(_QWORD *)(v22 + 40);
        if ( v24 != v22 + 56 )
          _libc_free(v24);
        sub_C7D6A0(*(_QWORD *)(v22 + 16), 8LL * *(unsigned int *)(v22 + 32), 8);
      }
      v22 += 88;
    }
    while ( v23 != v22 );
    v21 = *(unsigned int *)(a1 + 1192);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 1176), 88 * v21, 8);
  v25 = *(_QWORD *)(a1 + 1152);
  if ( a1 + 1168 != v25 )
    _libc_free(v25);
  sub_C7D6A0(*(_QWORD *)(a1 + 1128), 8LL * *(unsigned int *)(a1 + 1144), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 1096), 16LL * *(unsigned int *)(a1 + 1112), 8);
  if ( !*(_BYTE *)(a1 + 956) )
    _libc_free(*(_QWORD *)(a1 + 936));
  if ( !*(_BYTE *)(a1 + 796) )
    _libc_free(*(_QWORD *)(a1 + 776));
  if ( (*(_BYTE *)(a1 + 696) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 704), 16LL * *(unsigned int *)(a1 + 712), 8);
  if ( (*(_BYTE *)(a1 + 392) & 1) != 0 )
  {
    v27 = (_QWORD *)(a1 + 400);
    v29 = (_QWORD *)(a1 + 688);
    goto LABEL_74;
  }
  v26 = *(unsigned int *)(a1 + 408);
  v27 = *(_QWORD **)(a1 + 400);
  v28 = 9 * v26;
  if ( !(_DWORD)v26 )
    goto LABEL_117;
  v29 = &v27[v28];
  if ( &v27[v28] == v27 )
    goto LABEL_117;
  do
  {
LABEL_74:
    if ( *v27 != -8192 && *v27 != -4096 )
    {
      v30 = v27[1];
      if ( (_QWORD *)v30 != v27 + 3 )
        _libc_free(v30);
    }
    v27 += 9;
  }
  while ( v27 != v29 );
  if ( (*(_BYTE *)(a1 + 392) & 1) == 0 )
  {
    v27 = *(_QWORD **)(a1 + 400);
    v28 = 9LL * *(unsigned int *)(a1 + 408);
LABEL_117:
    sub_C7D6A0((__int64)v27, v28 * 8, 8);
    v31 = a1;
    if ( (*(_BYTE *)(a1 + 88) & 1) == 0 )
      goto LABEL_81;
    goto LABEL_118;
  }
  v31 = a1;
  if ( (*(_BYTE *)(a1 + 88) & 1) != 0 )
  {
LABEL_118:
    v33 = (_QWORD *)(a1 + 96);
    v35 = (_QWORD *)(a1 + 384);
    goto LABEL_83;
  }
LABEL_81:
  v32 = *(unsigned int *)(v31 + 104);
  v33 = *(_QWORD **)(v31 + 96);
  v34 = 9 * v32;
  if ( !(_DWORD)v32 )
    goto LABEL_115;
  v35 = &v33[v34];
  if ( &v33[v34] == v33 )
    goto LABEL_115;
  do
  {
LABEL_83:
    if ( *v33 != -8192 && *v33 != -4096 )
    {
      v36 = v33[1];
      if ( (_QWORD *)v36 != v33 + 3 )
        _libc_free(v36);
    }
    v33 += 9;
  }
  while ( v33 != v35 );
  if ( (*(_BYTE *)(a1 + 88) & 1) == 0 )
  {
    v33 = *(_QWORD **)(a1 + 96);
    v34 = 9LL * *(unsigned int *)(a1 + 104);
LABEL_115:
    sub_C7D6A0((__int64)v33, v34 * 8, 8);
  }
  v37 = *(_QWORD *)a1;
  v38 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v38 )
  {
    do
    {
      v39 = *(_QWORD *)(v38 - 8);
      v38 -= 8LL;
      if ( v39 )
      {
        v40 = *(unsigned __int64 **)(v39 + 240);
        v41 = &v40[10 * *(unsigned int *)(v39 + 248)];
        if ( v40 != v41 )
        {
          do
          {
            v41 -= 10;
            if ( (unsigned __int64 *)*v41 != v41 + 2 )
              _libc_free(*v41);
          }
          while ( v40 != v41 );
          v41 = *(unsigned __int64 **)(v39 + 240);
        }
        if ( v41 != (unsigned __int64 *)(v39 + 256) )
          _libc_free((unsigned __int64)v41);
        v42 = *(_QWORD *)(v39 + 208);
        if ( v42 != v39 + 224 )
          _libc_free(v42);
        v43 = *(_QWORD *)(v39 + 144);
        if ( v43 != v39 + 160 )
          _libc_free(v43);
        v44 = *(_QWORD *)(v39 + 112);
        if ( v44 != v39 + 128 )
          _libc_free(v44);
        v45 = *(_QWORD *)(v39 + 96);
        if ( v45 != 0 && v45 != -4096 && v45 != -8192 )
          sub_BD60C0((_QWORD *)(v39 + 80));
        if ( *(_QWORD *)v39 != v39 + 16 )
          _libc_free(*(_QWORD *)v39);
        j_j___libc_free_0(v39);
      }
    }
    while ( v37 != v38 );
    v38 = *(_QWORD *)a1;
  }
  if ( v38 != a1 + 16 )
    _libc_free(v38);
}
