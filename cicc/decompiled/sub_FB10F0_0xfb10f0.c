// Function: sub_FB10F0
// Address: 0xfb10f0
//
__int64 __fastcall sub_FB10F0(__int64 a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  const char **v5; // r13
  __int64 v6; // r12
  __int64 *v7; // rdx
  int v8; // eax
  __int64 v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // rax
  __int64 v12; // rsi
  char v13; // al
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 *v16; // rdi
  __int64 *v17; // r14
  __int64 v18; // rdx
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  __int64 v21; // r14
  __int64 v22; // rax
  __int64 v23; // r9
  unsigned __int64 v24; // rdx
  __int64 v25; // rcx
  unsigned __int64 v26; // rsi
  int v27; // eax
  __int64 *v28; // rdi
  __int64 v29; // rcx
  __int64 *v30; // rdx
  __int64 *v31; // rbx
  __int64 v32; // rax
  __int64 *v33; // r12
  __int64 *v34; // rdi
  unsigned int v36; // eax
  _QWORD *v37; // rbx
  _QWORD *v38; // r13
  _BYTE *v39; // rdx
  __int64 v40; // rbx
  __int64 v41; // r14
  __int64 v42; // r15
  _BYTE *v43; // r12
  __int64 *v44; // rbx
  __int64 v45; // r15
  _QWORD *v46; // r14
  unsigned int v47; // eax
  __int64 v48; // rax
  _BYTE *v49; // r15
  _BYTE *v50; // rcx
  __int64 v51; // rax
  __int64 v52; // r13
  unsigned int v53; // eax
  unsigned int v54; // ebx
  __int64 v55; // r13
  __int64 v56; // rax
  __int64 v57; // rdi
  __int64 v58; // r14
  __int64 v59; // r15
  __int64 v60; // rsi
  __int64 v61; // r14
  __int64 v62; // rax
  __int64 v63; // rcx
  _QWORD *v64; // rax
  __int64 v65; // r13
  unsigned int *v66; // r15
  __int64 v67; // rbx
  __int64 v68; // rdx
  unsigned int v69; // esi
  signed __int64 v70; // rbx
  __int64 v71; // r13
  int v72; // eax
  int v73; // eax
  unsigned int v74; // edx
  __int64 v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rdx
  int v78; // eax
  unsigned int v79; // eax
  unsigned __int64 v80; // rbx
  __int64 v81; // rax
  __int64 v82; // r12
  unsigned __int64 v83; // rbx
  __int64 v84; // r13
  __int64 v85; // rax
  __int64 v86; // r15
  __int64 *v87; // rdx
  __int64 v88; // rcx
  __int64 v89; // r8
  __int64 v90; // r9
  _QWORD *v91; // rax
  char v92; // dl
  unsigned int *v93; // r13
  __int64 v94; // rbx
  __int64 v95; // rdx
  unsigned int v96; // esi
  __int32 v97; // edx
  __int64 v98; // rax
  __int64 v99; // rax
  unsigned __int64 v100; // rcx
  __int64 v101; // rax
  __int64 v102; // rsi
  __int64 v103; // rbx
  _BYTE *v104; // rax
  unsigned int *v105; // r13
  __int64 v106; // r12
  __int64 v107; // rdx
  __int64 v108; // [rsp+10h] [rbp-200h]
  __int64 v111; // [rsp+30h] [rbp-1E0h]
  __int64 *v112; // [rsp+38h] [rbp-1D8h]
  _BYTE **v113; // [rsp+38h] [rbp-1D8h]
  __int64 v114; // [rsp+40h] [rbp-1D0h]
  __int64 v115; // [rsp+40h] [rbp-1D0h]
  unsigned int v117; // [rsp+48h] [rbp-1C8h]
  __int64 v119; // [rsp+50h] [rbp-1C0h]
  _QWORD *v120; // [rsp+60h] [rbp-1B0h]
  _QWORD *v121; // [rsp+68h] [rbp-1A8h]
  __int64 v122; // [rsp+68h] [rbp-1A8h]
  __int64 v123; // [rsp+68h] [rbp-1A8h]
  __int64 v124; // [rsp+78h] [rbp-198h] BYREF
  signed __int64 v125; // [rsp+80h] [rbp-190h] BYREF
  unsigned int v126; // [rsp+88h] [rbp-188h]
  __int64 v127; // [rsp+90h] [rbp-180h] BYREF
  unsigned int v128; // [rsp+98h] [rbp-178h]
  __m128i v129; // [rsp+A0h] [rbp-170h] BYREF
  _QWORD v130[4]; // [rsp+B0h] [rbp-160h] BYREF
  __int64 v131; // [rsp+D0h] [rbp-140h] BYREF
  char *v132; // [rsp+D8h] [rbp-138h] BYREF
  __int64 v133; // [rsp+E0h] [rbp-130h]
  char v134[8]; // [rsp+E8h] [rbp-128h] BYREF
  char v135; // [rsp+F0h] [rbp-120h]
  char v136; // [rsp+F1h] [rbp-11Fh]
  __m128i v137; // [rsp+110h] [rbp-100h] BYREF
  __int64 v138; // [rsp+120h] [rbp-F0h] BYREF
  int v139; // [rsp+128h] [rbp-E8h]
  char v140; // [rsp+12Ch] [rbp-E4h]
  __int16 v141; // [rsp+130h] [rbp-E0h] BYREF
  __int64 *v142; // [rsp+160h] [rbp-B0h] BYREF
  __int64 v143; // [rsp+168h] [rbp-A8h]
  _BYTE v144[160]; // [rsp+170h] [rbp-A0h] BYREF

  v6 = a1;
  v7 = *(__int64 **)(a1 - 8);
  v108 = *v7;
  v142 = (__int64 *)v144;
  v143 = 0x200000000LL;
  v8 = *(_DWORD *)(a1 + 4);
  v124 = 0;
  v120 = 0;
  v114 = ((v8 & 0x7FFFFFFu) >> 1) - 1;
  if ( (v8 & 0x7FFFFFFu) >> 1 != 1 )
  {
    v9 = 0;
    do
    {
      v10 = (unsigned int)(2 * ++v9);
      v5 = (const char **)v7[4 * v10];
      v137.m128i_i64[0] = (__int64)&v138;
      v137.m128i_i64[1] = 0x400000000LL;
      v11 = 4;
      if ( (_DWORD)v9 != -1 )
        v11 = 4LL * (unsigned int)(v10 + 1);
      v12 = (__int64)v5;
      v13 = sub_FB0740(v6, (__int64)v5, v7[v11], &v124, (__int64)&v137, a4, a5);
      v16 = (__int64 *)v137.m128i_i64[0];
      if ( !v13 || v137.m128i_i32[2] > 1u )
        goto LABEL_37;
      v17 = v142;
      v18 = (__int64)&v142[7 * (unsigned int)v143];
      if ( v142 == (__int64 *)v18 )
      {
LABEL_25:
        v131 = *(_QWORD *)(v137.m128i_i64[0] + 8);
        v129.m128i_i64[1] = 0x400000001LL;
        v132 = v134;
        v133 = 0x400000000LL;
        v129.m128i_i64[0] = (__int64)v130;
        v130[0] = v5;
        sub_F8EFD0((__int64)&v132, (char **)&v129, v18, 0x400000001LL, (__int64)v134, v15);
        v24 = (unsigned int)v143;
        v25 = HIDWORD(v143);
        v26 = (unsigned int)v143 + 1LL;
        v27 = v143;
        if ( v26 > HIDWORD(v143) )
        {
          v5 = (const char **)&v131;
          if ( v142 > &v131
            || (v5 = (const char **)&v131,
                v25 = 7LL * (unsigned int)v143,
                v24 = (unsigned __int64)&v142[7 * (unsigned int)v143],
                (unsigned __int64)&v131 >= v24) )
          {
            sub_FA1E60((__int64)&v142, v26, v24, v25, (__int64)v134, v23);
            v24 = (unsigned int)v143;
            v28 = v142;
            v12 = (__int64)&v131;
            v27 = v143;
          }
          else
          {
            v112 = v142;
            sub_FA1E60((__int64)&v142, v26, v24, v25, (__int64)v134, v23);
            v28 = v142;
            v24 = (unsigned int)v143;
            v12 = (__int64)v142 + (char *)&v131 - (char *)v112;
            v27 = v143;
          }
        }
        else
        {
          v28 = v142;
          v12 = (__int64)&v131;
        }
        v29 = 7 * v24;
        v30 = &v28[7 * v24];
        if ( v30 )
        {
          *v30 = *(_QWORD *)v12;
          v30[1] = (__int64)(v30 + 3);
          v30[2] = 0x400000000LL;
          LODWORD(v5) = *(_DWORD *)(v12 + 16);
          if ( (_DWORD)v5 )
          {
            v12 += 8;
            sub_F8EFD0((__int64)(v30 + 1), (char **)v12, (__int64)v30, v29, (__int64)v134, v23);
          }
          v27 = v143;
        }
        LODWORD(v143) = v27 + 1;
        if ( v132 != v134 )
          _libc_free(v132, v12);
        if ( (_QWORD *)v129.m128i_i64[0] != v130 )
          _libc_free(v129.m128i_i64[0], v12);
        v20 = 1;
      }
      else
      {
        while ( *(_QWORD *)(v137.m128i_i64[0] + 8) != *v17 )
        {
          v17 += 7;
          if ( (__int64 *)v18 == v17 )
            goto LABEL_25;
        }
        v19 = *((unsigned int *)v17 + 4);
        if ( v19 + 1 > (unsigned __int64)*((unsigned int *)v17 + 5) )
        {
          v12 = (__int64)(v17 + 3);
          sub_C8D5F0((__int64)(v17 + 1), v17 + 3, v19 + 1, 8u, v14, v15);
          v19 = *((unsigned int *)v17 + 4);
        }
        *(_QWORD *)(v17[1] + 8 * v19) = v5;
        v20 = (unsigned int)(*((_DWORD *)v17 + 4) + 1);
        *((_DWORD *)v17 + 4) = v20;
      }
      v16 = (__int64 *)v137.m128i_i64[0];
      if ( (unsigned int)qword_4F8C548 < v20 || (unsigned int)v143 > 2 )
        goto LABEL_37;
      if ( v120 )
      {
        if ( *(_QWORD **)v137.m128i_i64[0] != v120 )
          goto LABEL_37;
      }
      else
      {
        v120 = *(_QWORD **)v137.m128i_i64[0];
      }
      if ( (__int64 *)v137.m128i_i64[0] != &v138 )
        _libc_free(v137.m128i_i64[0], v12);
      v7 = *(__int64 **)(v6 - 8);
    }
    while ( v114 != v9 );
  }
  v12 = 0;
  v137.m128i_i64[0] = (__int64)&v138;
  v137.m128i_i64[1] = 0x100000000LL;
  v21 = v7[4];
  sub_FB0740(v6, 0, v21, &v124, (__int64)&v137, a4, a5);
  if ( v137.m128i_i32[2] != 1 || (v16 = (__int64 *)v137.m128i_i64[0], (v119 = *(_QWORD *)(v137.m128i_i64[0] + 8)) == 0) )
  {
    v12 = 1;
    v22 = sub_AA5030(v21, 1);
    if ( !v22 )
      BUG();
    v16 = (__int64 *)v137.m128i_i64[0];
    if ( *(_BYTE *)(v22 - 24) != 36 )
    {
LABEL_37:
      if ( v16 != &v138 )
        _libc_free(v16, v12);
      goto LABEL_39;
    }
    v119 = 0;
  }
  if ( v16 != &v138 )
    _libc_free(v16, v12);
  v12 = v6;
  sub_D5F1F0(a2, v6);
  v32 = (unsigned int)v143;
  if ( (unsigned int)v143 == 2 )
  {
    v31 = v142;
    LODWORD(v5) = 0;
    if ( *((_DWORD *)v142 + 4) != 1 || *((_DWORD *)v142 + 18) != 1 )
      goto LABEL_40;
    v49 = *(_BYTE **)v142[1];
    if ( v119 )
    {
      v50 = *(_BYTE **)v142[8];
      v137.m128i_i64[0] = (__int64)"switch.selectcmp";
      v141 = 259;
      v51 = sub_92B530((unsigned int **)a2, 0x20u, v108, v50, (__int64)&v137);
      v137.m128i_i64[0] = (__int64)"switch.select";
      v141 = 259;
      v52 = sub_B36550((unsigned int **)a2, v51, v142[7], v119, (__int64)&v137, 0);
    }
    else
    {
      v52 = v142[7];
    }
    v137.m128i_i64[0] = (__int64)"switch.selectcmp";
    v141 = 259;
    v12 = sub_92B530((unsigned int **)a2, 0x20u, v108, v49, (__int64)&v137);
    v137.m128i_i64[0] = (__int64)"switch.select";
    v141 = 259;
    v122 = sub_B36550((unsigned int **)a2, v12, *v142, v52, (__int64)&v137, 0);
    goto LABEL_91;
  }
  v31 = v142;
  LOBYTE(v5) = (unsigned int)v143 == 1 && v119 != 0;
  if ( !(_BYTE)v5 )
    goto LABEL_40;
  v36 = *((_DWORD *)v142 + 4);
  v117 = v36;
  v12 = v36;
  if ( !v36 )
  {
    v32 = 1;
    LODWORD(v5) = 0;
    goto LABEL_40;
  }
  v37 = (_QWORD *)v142[1];
  v113 = (_BYTE **)v37;
  v38 = v37;
  if ( ((v36 - 1) & v36) == 0 )
  {
    v111 = v6;
    v121 = &v37[v36];
    v39 = (_BYTE *)*v37;
    v40 = *v37 + 24LL;
    v41 = (__int64)v39;
    v115 = v142[1];
    v42 = (__int64)(v39 + 24);
    v43 = v39;
    while ( 1 )
    {
      if ( (int)sub_C4C880(v40, v42) < 0 )
      {
        v42 = v40;
        v41 = (__int64)v43;
      }
      if ( ++v38 == v121 )
        break;
      v43 = (_BYTE *)*v38;
      v40 = *v38 + 24LL;
    }
    v44 = (__int64 *)v42;
    v45 = v41;
    v46 = (_QWORD *)v115;
    v126 = *(_DWORD *)(v45 + 32);
    if ( v126 > 0x40 )
      sub_C43690((__int64)&v125, 0, 0);
    else
      v125 = 0;
    do
    {
      v48 = *v46;
      LODWORD(v132) = *(_DWORD *)(*v46 + 32LL);
      if ( (unsigned int)v132 <= 0x40 )
        v131 = *(_QWORD *)(v48 + 24);
      else
        sub_C43780((__int64)&v131, (const void **)(v48 + 24));
      v12 = (__int64)v44;
      sub_C46B40((__int64)&v131, v44);
      v47 = (unsigned int)v132;
      LODWORD(v132) = 0;
      v137.m128i_i32[2] = v47;
      v137.m128i_i64[0] = v131;
      if ( v126 > 0x40 )
      {
        v12 = (__int64)&v137;
        sub_C43BD0(&v125, v137.m128i_i64);
        v47 = v137.m128i_u32[2];
      }
      else
      {
        v125 |= v131;
      }
      if ( v47 > 0x40 && v137.m128i_i64[0] )
        j_j___libc_free_0_0(v137.m128i_i64[0]);
      if ( (unsigned int)v132 > 0x40 && v131 )
        j_j___libc_free_0_0(v131);
      ++v46;
    }
    while ( v121 != v46 );
    _BitScanReverse(&v53, v117);
    v6 = v111;
    v54 = 31 - (v53 ^ 0x1F);
    if ( v126 > 0x40 )
    {
      if ( (unsigned int)sub_C44630((__int64)&v125) != v54 )
      {
        if ( v125 )
          j_j___libc_free_0_0(v125);
        goto LABEL_88;
      }
LABEL_141:
      if ( !sub_AC30F0(v45) )
      {
        v141 = 257;
        v108 = sub_929DE0((unsigned int **)a2, (_BYTE *)v108, (_BYTE *)v45, (__int64)&v137, 0, 0);
      }
      v97 = v126;
      v136 = 1;
      v131 = (__int64)"switch.and";
      v135 = 3;
      v128 = v126;
      if ( v126 > 0x40 )
      {
        sub_C43780((__int64)&v127, (const void **)&v125);
        v97 = v128;
        if ( v128 > 0x40 )
        {
          sub_C43D10((__int64)&v127);
          v97 = v128;
          v101 = v127;
LABEL_148:
          v129.m128i_i32[2] = v97;
          v129.m128i_i64[0] = v101;
          v128 = 0;
          v102 = 28;
          v123 = sub_AD8D80(*(_QWORD *)(v108 + 8), (__int64)&v129);
          v103 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64))(**(_QWORD **)(a2 + 80) + 16LL))(
                   *(_QWORD *)(a2 + 80),
                   28,
                   v108,
                   v123);
          if ( !v103 )
          {
            v141 = 257;
            v103 = sub_B504D0(28, v108, v123, (__int64)&v137, 0, 0);
            (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
              *(_QWORD *)(a2 + 88),
              v103,
              &v131,
              *(_QWORD *)(a2 + 56),
              *(_QWORD *)(a2 + 64));
            v102 = a2;
            if ( *(_QWORD *)a2 != *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8) )
            {
              v105 = *(unsigned int **)a2;
              v106 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
              do
              {
                v107 = *((_QWORD *)v105 + 1);
                v102 = *v105;
                v105 += 4;
                sub_B99FD0(v103, v102, v107);
              }
              while ( (unsigned int *)v106 != v105 );
              v6 = v111;
            }
          }
          sub_969240(v129.m128i_i64);
          sub_969240(&v127);
          v137.m128i_i64[0] = (__int64)"switch.selectcmp";
          v141 = 259;
          v104 = (_BYTE *)sub_AD6530(*(_QWORD *)(v103 + 8), v102);
          v12 = sub_92B530((unsigned int **)a2, 0x20u, v103, v104, (__int64)&v137);
          v141 = 257;
          v122 = sub_B36550((unsigned int **)a2, v12, *v142, v119, (__int64)&v137, 0);
          sub_969240(&v125);
          goto LABEL_91;
        }
        v98 = v127;
      }
      else
      {
        v98 = v125;
      }
      v99 = ~v98;
      v100 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v97;
      if ( !v97 )
        v100 = 0;
      v101 = v100 & v99;
      v127 = v101;
      goto LABEL_148;
    }
    if ( (unsigned int)sub_39FAC40(v125) == v54 )
      goto LABEL_141;
  }
LABEL_88:
  if ( v117 != 2 )
  {
LABEL_39:
    v31 = v142;
    v32 = (unsigned int)v143;
    LODWORD(v5) = 0;
    goto LABEL_40;
  }
  v137.m128i_i64[0] = (__int64)"switch.selectcmp.case1";
  v141 = 259;
  v55 = sub_92B530((unsigned int **)a2, 0x20u, v108, *v113, (__int64)&v137);
  v137.m128i_i64[0] = (__int64)"switch.selectcmp.case2";
  v141 = 259;
  v56 = sub_92B530((unsigned int **)a2, 0x20u, v108, v113[1], (__int64)&v137);
  v57 = *(_QWORD *)(a2 + 80);
  v58 = v56;
  v136 = 1;
  v131 = (__int64)"switch.selectcmp";
  v135 = 3;
  v59 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v57 + 16LL))(v57, 29, v55, v56);
  if ( !v59 )
  {
    v141 = 257;
    v59 = sub_B504D0(29, v55, v58, (__int64)&v137, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v59,
      &v131,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v93 = *(unsigned int **)a2;
    v94 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
    if ( *(_QWORD *)a2 != v94 )
    {
      do
      {
        v95 = *((_QWORD *)v93 + 1);
        v96 = *v93;
        v93 += 4;
        sub_B99FD0(v59, v96, v95);
      }
      while ( (unsigned int *)v94 != v93 );
    }
  }
  v141 = 257;
  v12 = v59;
  v122 = sub_B36550((unsigned int **)a2, v59, *v142, v119, (__int64)&v137, 0);
LABEL_91:
  if ( !v122 )
    goto LABEL_39;
  v60 = *(_QWORD *)(v6 + 40);
  v131 = 0;
  v132 = 0;
  v125 = (signed __int64)v120;
  v133 = 0;
  v127 = v60;
  v61 = v120[5];
  if ( a3 )
  {
    v62 = *(_QWORD *)(v61 + 16);
    if ( v62 )
    {
      while ( 1 )
      {
        v63 = *(_QWORD *)(v62 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v63 - 30) <= 0xAu )
          break;
        v62 = *(_QWORD *)(v62 + 8);
        if ( !v62 )
          goto LABEL_99;
      }
LABEL_97:
      if ( v60 == *(_QWORD *)(v63 + 40) )
        goto LABEL_100;
      while ( 1 )
      {
        v62 = *(_QWORD *)(v62 + 8);
        if ( !v62 )
          break;
        v63 = *(_QWORD *)(v62 + 24);
        if ( (unsigned __int8)(*(_BYTE *)v63 - 30) <= 0xAu )
          goto LABEL_97;
      }
    }
LABEL_99:
    v137.m128i_i64[0] = v60;
    v137.m128i_i64[1] = v61 & 0xFFFFFFFFFFFFFFFBLL;
    sub_F9E360((__int64)&v131, &v137);
  }
LABEL_100:
  v141 = 257;
  v64 = sub_BD2C40(72, 1u);
  v65 = (__int64)v64;
  if ( v64 )
    sub_B4C8F0((__int64)v64, v61, 1u, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v65,
    &v137,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v66 = *(unsigned int **)a2;
  v67 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v67 )
  {
    do
    {
      v68 = *((_QWORD *)v66 + 1);
      v69 = *v66;
      v66 += 4;
      sub_B99FD0(v65, v69, v68);
    }
    while ( (unsigned int *)v67 != v66 );
  }
  v137.m128i_i64[0] = (__int64)&v125;
  v137.m128i_i64[1] = (__int64)&v127;
  sub_B57920(v125, (unsigned __int8 (__fastcall *)(__int64, _QWORD))sub_F8F2B0, (__int64)&v137, 1);
  v70 = v125;
  v71 = v127;
  v72 = *(_DWORD *)(v125 + 4) & 0x7FFFFFF;
  if ( v72 == *(_DWORD *)(v125 + 72) )
  {
    sub_B48D90(v125);
    v72 = *(_DWORD *)(v70 + 4) & 0x7FFFFFF;
  }
  v73 = (v72 + 1) & 0x7FFFFFF;
  v74 = v73 | *(_DWORD *)(v70 + 4) & 0xF8000000;
  v75 = *(_QWORD *)(v70 - 8) + 32LL * (unsigned int)(v73 - 1);
  *(_DWORD *)(v70 + 4) = v74;
  if ( *(_QWORD *)v75 )
  {
    v76 = *(_QWORD *)(v75 + 8);
    **(_QWORD **)(v75 + 16) = v76;
    if ( v76 )
      *(_QWORD *)(v76 + 16) = *(_QWORD *)(v75 + 16);
  }
  *(_QWORD *)v75 = v122;
  v77 = *(_QWORD *)(v122 + 16);
  *(_QWORD *)(v75 + 8) = v77;
  if ( v77 )
    *(_QWORD *)(v77 + 16) = v75 + 8;
  v12 = v122;
  *(_QWORD *)(v75 + 16) = v122 + 16;
  *(_QWORD *)(v122 + 16) = v75;
  *(_QWORD *)(*(_QWORD *)(v70 - 8) + 32LL * *(unsigned int *)(v70 + 72)
                                   + 8LL * ((*(_DWORD *)(v70 + 4) & 0x7FFFFFFu) - 1)) = v71;
  v137.m128i_i64[1] = (__int64)&v141;
  v78 = *(_DWORD *)(v6 + 4);
  v137.m128i_i64[0] = 0;
  v140 = 1;
  v138 = 4;
  v79 = (v78 & 0x7FFFFFFu) >> 1;
  v139 = 0;
  if ( v79 )
  {
    v80 = (unsigned __int64)(v79 - 1) << 6;
    v81 = v6;
    v82 = 32;
    v83 = v80 + 96;
    v84 = v81;
    while ( 1 )
    {
      v85 = *(_QWORD *)(v84 - 8);
      v86 = *(_QWORD *)(v85 + v82);
      if ( v61 == v86 )
        goto LABEL_120;
      v12 = v127;
      sub_AA5980(*(_QWORD *)(v85 + v82), v127, 0);
      if ( !a3 )
        goto LABEL_120;
      if ( v140 )
      {
        v91 = (_QWORD *)v137.m128i_i64[1];
        v12 = HIDWORD(v138);
        v87 = (__int64 *)(v137.m128i_i64[1] + 8LL * HIDWORD(v138));
        if ( (__int64 *)v137.m128i_i64[1] != v87 )
        {
          while ( v86 != *v91 )
          {
            if ( v87 == ++v91 )
              goto LABEL_133;
          }
          goto LABEL_120;
        }
LABEL_133:
        if ( HIDWORD(v138) < (unsigned int)v138 )
          break;
      }
      v12 = v86;
      sub_C8CC70((__int64)&v137, v86, (__int64)v87, v88, v89, v90);
      if ( v92 )
        goto LABEL_132;
LABEL_120:
      v82 += 64;
      if ( v82 == v83 )
      {
        v6 = v84;
        goto LABEL_122;
      }
    }
    ++HIDWORD(v138);
    *v87 = v86;
    ++v137.m128i_i64[0];
LABEL_132:
    v12 = (__int64)&v129;
    v129.m128i_i64[0] = v127;
    v129.m128i_i64[1] = v86 | 4;
    sub_F9E360((__int64)&v131, &v129);
    goto LABEL_120;
  }
LABEL_122:
  sub_B43D60((_QWORD *)v6);
  if ( a3 )
  {
    v12 = v131;
    sub_FFB3D0(a3, v131, (__int64)&v132[-v131] >> 4);
  }
  if ( !v140 )
    _libc_free(v137.m128i_i64[1], v12);
  if ( v131 )
  {
    v12 = v133 - v131;
    j_j___libc_free_0(v131, v133 - v131);
  }
  v31 = v142;
  v32 = (unsigned int)v143;
  LODWORD(v5) = 1;
LABEL_40:
  v33 = &v31[7 * v32];
  if ( v31 != v33 )
  {
    do
    {
      v33 -= 7;
      v34 = (__int64 *)v33[1];
      if ( v34 != v33 + 3 )
        _libc_free(v34, v12);
    }
    while ( v31 != v33 );
    v33 = v142;
  }
  if ( v33 != (__int64 *)v144 )
    _libc_free(v33, v12);
  return (unsigned int)v5;
}
