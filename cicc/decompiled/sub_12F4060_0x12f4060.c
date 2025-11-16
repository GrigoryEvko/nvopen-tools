// Function: sub_12F4060
// Address: 0x12f4060
//
__int64 __fastcall sub_12F4060(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4, _QWORD *a5)
{
  __int64 v5; // rsi
  _DWORD *v6; // rax
  _DWORD *v7; // rax
  __int64 v8; // r13
  struct __jmp_buf_tag *v9; // r12
  int v10; // eax
  unsigned int v11; // r12d
  _QWORD *v13; // rbx
  _QWORD *v14; // r12
  _QWORD *v15; // rbx
  _QWORD *v16; // r12
  __int64 v17; // r12
  char *v18; // r13
  __int64 v19; // rax
  int v20; // eax
  __int64 v21; // r12
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  int v25; // r15d
  __int64 v26; // rcx
  _BYTE *v27; // rdi
  size_t v28; // rdx
  __int64 v29; // rax
  _QWORD *v30; // rsi
  __int64 v31; // r12
  __int64 v32; // rbx
  unsigned __int64 v33; // rsi
  _QWORD *v34; // rax
  _DWORD *v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // rdx
  bool v38; // dl
  __int64 v39; // rax
  _DWORD *v40; // r8
  _DWORD *v41; // rdi
  __int64 v42; // rcx
  __int64 v43; // rdx
  __int64 (__fastcall *v44)(__int64, _QWORD *, __int64, __int64, void *, size_t, _QWORD *, int *, __int64 *, __int64, _QWORD); // r13
  __int64 v45; // r11
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // r13
  __int64 v49; // rax
  __int64 v50; // r12
  int *v51; // rax
  int v52; // eax
  __int64 v53; // rax
  __int64 v54; // r12
  __int64 v55; // rax
  __int64 (*v56)(); // rax
  __int64 v57; // rbx
  __int64 v58; // rax
  unsigned __int64 v59; // r12
  unsigned __int64 v60; // rdx
  __int64 v61; // rbx
  __int64 v62; // r12
  __int64 v63; // rdi
  __m128i *v64; // rax
  __int64 v65; // rdx
  _QWORD *v66; // rbx
  _QWORD *v67; // r12
  _DWORD *v68; // rax
  __int64 v69; // r12
  __int64 v70; // rdi
  _DWORD *v71; // rax
  __int64 v72; // [rsp+0h] [rbp-600h]
  __int64 v73; // [rsp+8h] [rbp-5F8h]
  void *v74; // [rsp+10h] [rbp-5F0h]
  size_t v75; // [rsp+18h] [rbp-5E8h]
  int v77; // [rsp+28h] [rbp-5D8h]
  char v78; // [rsp+2Fh] [rbp-5D1h]
  _BYTE *v82; // [rsp+50h] [rbp-5B0h]
  __int64 v84; // [rsp+68h] [rbp-598h] BYREF
  int v85; // [rsp+70h] [rbp-590h] BYREF
  bool v86; // [rsp+74h] [rbp-58Ch]
  __int64 v87; // [rsp+78h] [rbp-588h] BYREF
  _QWORD v88[2]; // [rsp+80h] [rbp-580h] BYREF
  _BYTE v89[32]; // [rsp+90h] [rbp-570h] BYREF
  __int64 v90; // [rsp+B0h] [rbp-550h]
  __int16 v91; // [rsp+C0h] [rbp-540h]
  _QWORD *v92; // [rsp+D0h] [rbp-530h] BYREF
  _QWORD *v93; // [rsp+D8h] [rbp-528h]
  __int64 v94; // [rsp+E0h] [rbp-520h]
  __int64 v95; // [rsp+F0h] [rbp-510h]
  __int16 v96; // [rsp+100h] [rbp-500h]
  _QWORD *v97; // [rsp+110h] [rbp-4F0h]
  __int16 v98; // [rsp+120h] [rbp-4E0h]
  void *dest; // [rsp+130h] [rbp-4D0h]
  size_t v100; // [rsp+138h] [rbp-4C8h]
  _QWORD v101[2]; // [rsp+140h] [rbp-4C0h] BYREF
  _QWORD v102[2]; // [rsp+150h] [rbp-4B0h] BYREF
  __int64 v103; // [rsp+160h] [rbp-4A0h] BYREF
  _QWORD v104[2]; // [rsp+170h] [rbp-490h] BYREF
  __int64 v105; // [rsp+180h] [rbp-480h] BYREF
  _QWORD v106[2]; // [rsp+190h] [rbp-470h] BYREF
  __int64 v107; // [rsp+1A0h] [rbp-460h] BYREF
  void *src; // [rsp+1B0h] [rbp-450h] BYREF
  size_t n; // [rsp+1B8h] [rbp-448h]
  _QWORD v110[2]; // [rsp+1C0h] [rbp-440h] BYREF
  _QWORD v111[2]; // [rsp+1D0h] [rbp-430h] BYREF
  _QWORD v112[2]; // [rsp+1E0h] [rbp-420h] BYREF
  __int64 v113[2]; // [rsp+1F0h] [rbp-410h] BYREF
  _QWORD v114[2]; // [rsp+200h] [rbp-400h] BYREF
  _QWORD v115[2]; // [rsp+210h] [rbp-3F0h] BYREF
  _QWORD v116[2]; // [rsp+220h] [rbp-3E0h] BYREF
  _QWORD v117[4]; // [rsp+230h] [rbp-3D0h] BYREF
  int v118; // [rsp+250h] [rbp-3B0h]
  _QWORD *v119; // [rsp+258h] [rbp-3A8h]
  _BYTE v120[4]; // [rsp+260h] [rbp-3A0h] BYREF
  int v121; // [rsp+264h] [rbp-39Ch]
  _QWORD *v122; // [rsp+2A0h] [rbp-360h] BYREF
  _QWORD v123[6]; // [rsp+2B0h] [rbp-350h] BYREF
  _QWORD v124[2]; // [rsp+2E0h] [rbp-320h] BYREF
  __int64 v125; // [rsp+2F0h] [rbp-310h] BYREF
  int v126; // [rsp+30Ch] [rbp-2F4h]
  _QWORD v127[2]; // [rsp+320h] [rbp-2E0h] BYREF
  __int64 v128; // [rsp+330h] [rbp-2D0h] BYREF
  _QWORD v129[6]; // [rsp+360h] [rbp-2A0h] BYREF
  char v130; // [rsp+390h] [rbp-270h] BYREF
  char v131; // [rsp+391h] [rbp-26Fh]
  __int64 *v132; // [rsp+398h] [rbp-268h]
  __int64 v133; // [rsp+3A8h] [rbp-258h] BYREF
  __int64 *v134; // [rsp+3B8h] [rbp-248h]
  __int64 v135; // [rsp+3C8h] [rbp-238h] BYREF
  _QWORD *v136; // [rsp+3D8h] [rbp-228h]
  _QWORD *v137; // [rsp+3E0h] [rbp-220h]
  __int64 v138; // [rsp+3E8h] [rbp-218h]
  _BYTE v139[120]; // [rsp+3F0h] [rbp-210h] BYREF
  __int64 v140; // [rsp+468h] [rbp-198h]
  unsigned int v141; // [rsp+478h] [rbp-188h]
  __int64 v142; // [rsp+488h] [rbp-178h]
  __int64 v143; // [rsp+498h] [rbp-168h]
  __int64 v144; // [rsp+4A0h] [rbp-160h]
  __int64 v145; // [rsp+4B0h] [rbp-150h]
  _BYTE *v146; // [rsp+4C0h] [rbp-140h] BYREF
  __int64 v147; // [rsp+4C8h] [rbp-138h]
  _BYTE v148[304]; // [rsp+4D0h] [rbp-130h] BYREF

  v82 = (_BYTE *)a1[37];
  if ( !v82 )
  {
    v69 = a1[36];
    v82 = (_BYTE *)sub_22077B0(4480);
    if ( v82 )
      sub_12D6300((__int64)v82, v69);
    v70 = a1[37];
    a1[37] = v82;
    if ( v70 )
    {
      j_j___libc_free_0(v70, 4480);
      v82 = (_BYTE *)a1[37];
    }
  }
  if ( (unsigned __int8)v82[3448] + (unsigned __int8)v82[3408] == 2 )
  {
    v11 = 0;
    sub_223E0D0(qword_4FD4BE0, "Error: Cannot specify multiple -llcO#\n", 38);
    return v11;
  }
  v5 = 0;
  v146 = v148;
  v119 = &v146;
  v147 = 0x10000000000LL;
  v117[0] = &unk_49EFC48;
  v118 = 1;
  memset(&v117[1], 0, 24);
  sub_16E7A40(v117, 0, 0, 0);
  if ( !v82[3448] || BYTE4(qword_4FBB370[2]) )
  {
    if ( !v82[3744] )
      goto LABEL_6;
LABEL_9:
    v7 = (_DWORD *)sub_1C42D70(4, 4);
    *v7 = 1;
    v5 = (__int64)v7;
    sub_16D40E0(qword_4FBB410, v7);
    goto LABEL_10;
  }
  v68 = (_DWORD *)sub_1C42D70(4, 4);
  *v68 = 6;
  v5 = (__int64)v68;
  sub_16D40E0(qword_4FBB370, v68);
  if ( v82[3744] )
    goto LABEL_9;
LABEL_6:
  if ( v82[3784] )
  {
    v71 = (_DWORD *)sub_1C42D70(4, 4);
    *v71 = 2;
    v5 = (__int64)v71;
    sub_16D40E0(qword_4FBB410, v71);
  }
  else if ( v82[3824] )
  {
    v6 = (_DWORD *)sub_1C42D70(4, 4);
    *v6 = 3;
    v5 = (__int64)v6;
    sub_16D40E0(qword_4FBB410, v6);
  }
LABEL_10:
  sub_1611EE0(v89);
  v8 = sub_1C3E710();
  v9 = (struct __jmp_buf_tag *)sub_16D40F0(v8);
  if ( !v9 )
  {
    v5 = sub_1C42D70(200, 8);
    memset((void *)v5, 0, 0xC8u);
    sub_16D40E0(v8, v5);
    v9 = (struct __jmp_buf_tag *)sub_16D40F0(v8);
  }
  v10 = _setjmp(v9);
  if ( v10 )
  {
    if ( v10 == 1 )
    {
      sub_1C3E9C0(a4);
LABEL_15:
      v11 = 0;
      goto LABEL_16;
    }
    goto LABEL_45;
  }
  v17 = 0;
  v91 = 260;
  v18 = "nvptx";
  v90 = a2 + 240;
  sub_16E1010(&v122);
  v19 = sub_1632FA0(a2);
  v20 = sub_15A9520(v19, 0);
  v100 = 0;
  LOBYTE(v101[0]) = 0;
  if ( 8 * v20 == 64 )
    v18 = "nvptx64";
  LOBYTE(v17) = 8 * v20 == 64;
  dest = v101;
  v21 = 2 * v17 + 5;
  sub_16810B0(&v92, byte_3F871B3, 0);
  v22 = sub_1632FA0(a2);
  if ( 8 * (unsigned int)sub_15A9520(v22, 3) == 32 )
    sub_1681DA0(&v92, "sharedmem32bitptr", 17, 1);
  v24 = qword_4F95A60;
  v25 = 0;
  v26 = 0;
  if ( qword_4F95A68 != qword_4F95A60 )
  {
    do
    {
      sub_1681DA0(&v92, *(_QWORD *)(v24 + 32 * v26), *(_QWORD *)(v24 + 32 * v26 + 8), 1);
      v24 = qword_4F95A60;
      v26 = (unsigned int)++v25;
      v23 = (qword_4F95A68 - qword_4F95A60) >> 5;
    }
    while ( v25 != v23 );
  }
  if ( a1[16] )
  {
    sub_8FD6D0((__int64)v102, "fma-level=", a1 + 15);
    sub_1681DA0(&v92, v102[0], v102[1], 1);
    if ( (__int64 *)v102[0] != &v103 )
      j_j___libc_free_0(v102[0], v103 + 1);
  }
  if ( a1[20] )
  {
    sub_8FD6D0((__int64)v104, "prec-divf32=", a1 + 19);
    sub_1681DA0(&v92, v104[0], v104[1], 1);
    if ( (__int64 *)v104[0] != &v105 )
      j_j___libc_free_0(v104[0], v105 + 1);
  }
  if ( a1[24] )
  {
    sub_8FD6D0((__int64)v106, "prec-sqrtf32=", a1 + 23);
    sub_1681DA0(&v92, v106[0], v106[1], 1);
    if ( (__int64 *)v106[0] != &v107 )
      j_j___libc_free_0(v106[0], v107 + 1);
  }
  sub_16816B0(&src, &v92, v23, v26);
  v27 = dest;
  v28 = n;
  if ( src == v110 )
  {
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = v110[0];
      else
        memcpy(dest, src, n);
      v28 = n;
      v27 = dest;
    }
    v100 = v28;
    v27[v28] = 0;
    v27 = src;
    goto LABEL_61;
  }
  if ( dest == v101 )
  {
    dest = src;
    v100 = n;
    v101[0] = v110[0];
    goto LABEL_136;
  }
  v29 = v101[0];
  dest = src;
  v100 = n;
  v101[0] = v110[0];
  if ( !v27 )
  {
LABEL_136:
    v27 = v110;
    src = v110;
    goto LABEL_61;
  }
  src = v27;
  v110[0] = v29;
LABEL_61:
  n = 0;
  *v27 = 0;
  if ( src != v110 )
    j_j___libc_free_0(src, v110[0] + 1LL);
  v111[1] = 0;
  v111[0] = v112;
  LOBYTE(v112[0]) = 0;
  v113[0] = (__int64)v114;
  sub_12EFD20(v113, v18, (__int64)&v18[v21]);
  v30 = v111;
  v31 = sub_16D3AC0(v113, v111);
  if ( (_QWORD *)v113[0] != v114 )
  {
    v30 = (_QWORD *)(v114[0] + 1LL);
    j_j___libc_free_0(v113[0], v114[0] + 1LL);
  }
  if ( !v31 )
  {
    v84 = 30;
    v115[0] = v116;
    v64 = (__m128i *)sub_22409D0(v115, &v84, 0);
    v5 = 1;
    v115[0] = v64;
    v116[0] = v84;
    *v64 = _mm_load_si128((const __m128i *)&xmmword_4281B20);
    v65 = v115[0];
    qmemcpy(&v64[1], " nvptx target\n", 14);
    v115[1] = v84;
    *(_BYTE *)(v65 + v84) = 0;
    sub_1C3EFD0(v115, 1);
    if ( (_QWORD *)v115[0] != v116 )
    {
      v5 = v116[0] + 1LL;
      j_j___libc_free_0(v115[0], v116[0] + 1LL);
    }
    if ( (_QWORD *)v111[0] != v112 )
    {
      v5 = v112[0] + 1LL;
      j_j___libc_free_0(v111[0], v112[0] + 1LL);
    }
    v66 = v93;
    v67 = v92;
    if ( v93 != v92 )
    {
      do
      {
        if ( (_QWORD *)*v67 != v67 + 2 )
        {
          v5 = v67[2] + 1LL;
          j_j___libc_free_0(*v67, v5);
        }
        v67 += 4;
      }
      while ( v66 != v67 );
      v67 = v92;
    }
    if ( v67 )
    {
      v5 = v94 - (_QWORD)v67;
      j_j___libc_free_0(v67, v94 - (_QWORD)v67);
    }
    if ( dest != v101 )
    {
      v5 = v101[0] + 1LL;
      j_j___libc_free_0(dest, v101[0] + 1LL);
    }
    if ( v122 != v123 )
    {
      v5 = v123[0] + 1LL;
      j_j___libc_free_0(v122, v123[0] + 1LL);
    }
    goto LABEL_15;
  }
  v32 = 2;
  v129[0] = 0;
  v129[1] = 1;
  v129[2] = 8;
  v129[3] = 1;
  v129[4] = 1;
  v129[5] = 0;
  sub_167F890(&v130);
  v78 = v82[160];
  v131 = (16 * (v78 & 1)) | v131 & 0xEF;
  if ( !v82[3408] )
    v32 = v82[3448] != 0 ? 3 : 0;
  v33 = sub_16D5D50(&v130, v30, 16 * (v78 & 1u), v82);
  v34 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v35 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v36 = v34[2];
        v37 = v34[3];
        if ( v33 <= v34[4] )
          break;
        v34 = (_QWORD *)v34[3];
        if ( !v37 )
          goto LABEL_73;
      }
      v35 = v34;
      v34 = (_QWORD *)v34[2];
    }
    while ( v36 );
LABEL_73:
    v38 = 0;
    if ( v35 != dword_4FA0208 && v33 >= *((_QWORD *)v35 + 4) )
    {
      v39 = *((_QWORD *)v35 + 7);
      v40 = v35 + 12;
      if ( v39 )
      {
        v41 = v35 + 12;
        do
        {
          while ( 1 )
          {
            v42 = *(_QWORD *)(v39 + 16);
            v43 = *(_QWORD *)(v39 + 24);
            if ( *(_DWORD *)(v39 + 32) >= dword_4F95768 )
              break;
            v39 = *(_QWORD *)(v39 + 24);
            if ( !v43 )
              goto LABEL_80;
          }
          v41 = (_DWORD *)v39;
          v39 = *(_QWORD *)(v39 + 16);
        }
        while ( v42 );
LABEL_80:
        v38 = 0;
        if ( v40 != v41 && dword_4F95768 >= v41[8] )
        {
          v77 = dword_4F95800;
          v38 = v41[9] != 0;
        }
      }
    }
  }
  else
  {
    v38 = 0;
  }
  v44 = *(__int64 (__fastcall **)(__int64, _QWORD *, __int64, __int64, void *, size_t, _QWORD *, int *, __int64 *, __int64, _QWORD))(v31 + 88);
  v45 = a1[11];
  v46 = a1[12];
  v47 = *(_QWORD *)(a2 + 248);
  v88[0] = *(_QWORD *)(a2 + 240);
  v88[1] = v47;
  if ( v44 )
  {
    v86 = v38;
    v87 = 0x100000000LL;
    if ( v38 )
      v85 = v77;
    v72 = v45;
    v98 = 261;
    v73 = v46;
    v74 = dest;
    v75 = v100;
    v97 = v88;
    sub_16E1010(v127);
    v48 = v44(v31, v127, v72, v73, v74, v75, v129, &v85, &v87, v32, 0);
    if ( (__int64 *)v127[0] != &v128 )
      j_j___libc_free_0(v127[0], v128 + 1);
  }
  else
  {
    v48 = 0;
  }
  v95 = a2 + 240;
  v96 = 260;
  sub_16E1010(v124);
  sub_14A04B0(v139, v124);
  sub_149CBC0(v139);
  v49 = sub_22077B0(368);
  v50 = v49;
  if ( v49 )
    sub_149CCE0(v49, v139);
  sub_1619140(v89, v50, 0);
  sub_1BFB9A0(v120, a1[11], a1[12], v126 == 23);
  v51 = (int *)sub_16D40F0(qword_4FBB430);
  if ( v51 )
    v52 = *v51;
  else
    v52 = qword_4FBB430[2];
  v121 = v52;
  v53 = sub_22077B0(208);
  v54 = v53;
  if ( v53 )
    sub_1BFB520(v53, v120);
  sub_1619140(v89, v54, 1);
  v55 = sub_1C43550();
  sub_1619140(v89, v55, 1);
  v56 = *(__int64 (**)())(*(_QWORD *)v48 + 56LL);
  if ( v56 != sub_12D3B70 )
    ((void (__fastcall *)(__int64, _BYTE *, _QWORD *, _QWORD, _QWORD, _QWORD, _QWORD))v56)(
      v48,
      v89,
      v117,
      0,
      0,
      v82[3368] ^ 1u,
      0);
  sub_160FB70(v89, *a5, a5[1]);
  sub_1619BD0(v89, a2);
  sub_2241130(a3, 0, a3[1], v146, (unsigned int)v147);
  v5 = (__int64)a3;
  v57 = a3[1];
  v58 = *a3;
  v59 = v57 + 1;
  if ( *(_QWORD *)v5 == v5 + 16 )
    v60 = 15;
  else
    v60 = a3[2];
  if ( v59 > v60 )
  {
    v5 = a3[1];
    sub_2240BB0(a3, v57, 0, 0, 1);
    v58 = *a3;
  }
  *(_BYTE *)(v58 + v57) = 0;
  a3[1] = v59;
  *(_BYTE *)(*a3 + v57 + 1) = 0;
  if ( v144 )
  {
    v5 = v145 - v144;
    j_j___libc_free_0(v144, v145 - v144);
  }
  if ( v142 )
  {
    v5 = v143 - v142;
    j_j___libc_free_0(v142, v143 - v142);
  }
  if ( v141 )
  {
    v61 = v140;
    v62 = v140 + 40LL * v141;
    do
    {
      if ( *(_DWORD *)v61 <= 0xFFFFFFFD )
      {
        v63 = *(_QWORD *)(v61 + 8);
        if ( v63 != v61 + 24 )
        {
          v5 = *(_QWORD *)(v61 + 24) + 1LL;
          j_j___libc_free_0(v63, v5);
        }
      }
      v61 += 40;
    }
    while ( v62 != v61 );
  }
  j___libc_free_0(v140);
  if ( (__int64 *)v124[0] != &v125 )
  {
    v5 = v125 + 1;
    j_j___libc_free_0(v124[0], v125 + 1);
  }
  (*(void (__fastcall **)(__int64))(*(_QWORD *)v48 + 8LL))(v48);
  v13 = v137;
  v14 = v136;
  if ( v137 != v136 )
  {
    do
    {
      if ( (_QWORD *)*v14 != v14 + 2 )
      {
        v5 = v14[2] + 1LL;
        j_j___libc_free_0(*v14, v5);
      }
      v14 += 4;
    }
    while ( v13 != v14 );
    v14 = v136;
  }
  if ( v14 )
  {
    v5 = v138 - (_QWORD)v14;
    j_j___libc_free_0(v14, v138 - (_QWORD)v14);
  }
  if ( v134 != &v135 )
  {
    v5 = v135 + 1;
    j_j___libc_free_0(v134, v135 + 1);
  }
  if ( v132 != &v133 )
  {
    v5 = v133 + 1;
    j_j___libc_free_0(v132, v133 + 1);
  }
  if ( (_QWORD *)v111[0] != v112 )
  {
    v5 = v112[0] + 1LL;
    j_j___libc_free_0(v111[0], v112[0] + 1LL);
  }
  v15 = v93;
  v16 = v92;
  if ( v93 != v92 )
  {
    do
    {
      if ( (_QWORD *)*v16 != v16 + 2 )
      {
        v5 = v16[2] + 1LL;
        j_j___libc_free_0(*v16, v5);
      }
      v16 += 4;
    }
    while ( v15 != v16 );
    v16 = v92;
  }
  if ( v16 )
  {
    v5 = v94 - (_QWORD)v16;
    j_j___libc_free_0(v16, v94 - (_QWORD)v16);
  }
  if ( dest != v101 )
  {
    v5 = v101[0] + 1LL;
    j_j___libc_free_0(dest, v101[0] + 1LL);
  }
  if ( v122 != v123 )
  {
    v5 = v123[0] + 1LL;
    j_j___libc_free_0(v122, v123[0] + 1LL);
  }
LABEL_45:
  v11 = 1;
  sub_1C3E9C0(a4);
LABEL_16:
  sub_160FE50(v89);
  v117[0] = &unk_49EFD28;
  sub_16E7960(v117);
  if ( v146 != v148 )
    _libc_free(v146, v5);
  return v11;
}
