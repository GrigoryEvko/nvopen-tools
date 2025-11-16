// Function: sub_37E3110
// Address: 0x37e3110
//
__int64 __fastcall sub_37E3110(
        _QWORD *a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        __m128i a7,
        __int64 a8,
        _QWORD *a9,
        _QWORD *a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v13; // r12
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r15
  __int64 v17; // rax
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rdi
  __int64 (*v21)(void); // rdx
  __int64 v22; // rax
  __int64 v23; // rax
  unsigned int v24; // r13d
  bool v25; // zf
  _BYTE *v26; // rdi
  unsigned int v27; // r13d
  __int64 v28; // r8
  __int64 v29; // r9
  _QWORD *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rbx
  _QWORD *v34; // rax
  __int64 *v35; // rax
  __int64 v36; // r15
  __int64 v37; // rbx
  __int64 v38; // rcx
  __int64 v39; // rsi
  unsigned int v40; // eax
  __int64 *v41; // rdx
  __int64 v42; // rdi
  unsigned int v43; // edi
  __int64 v44; // rsi
  __int64 v45; // rdx
  __int64 v46; // rdi
  __int64 v47; // r8
  __int64 *v48; // rcx
  __int64 v49; // r9
  __int64 v50; // r10
  __int64 v51; // rdi
  __int64 v52; // rcx
  unsigned int v53; // eax
  __int64 *v54; // rdx
  __int64 v55; // r8
  __int64 v56; // rcx
  __int64 v57; // rdi
  __int64 *v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // rdi
  unsigned int v61; // esi
  __int64 *v62; // rax
  __int64 v63; // r8
  __int64 v64; // rsi
  char v65; // cl
  __int64 *v66; // rax
  __int64 *v67; // rdx
  __int64 *v68; // r15
  unsigned int v69; // eax
  __int64 *v70; // rax
  __int64 v71; // rdx
  __int64 *v72; // r14
  __int64 v73; // rsi
  __int64 *v74; // r13
  unsigned int v75; // eax
  int v77; // edx
  __int64 v78; // rcx
  __int64 v79; // r14
  __int64 v80; // r12
  __int64 *v81; // rbx
  __int64 *v82; // rax
  __int64 *v83; // rax
  int v84; // ecx
  int v85; // edx
  int v86; // eax
  __int64 v87; // rax
  int v88; // edx
  int v89; // r11d
  int v90; // r10d
  int v91; // r11d
  int v92; // r10d
  __int64 v93; // rax
  int v94; // r11d
  __int64 v97; // [rsp+28h] [rbp-188h]
  int v99; // [rsp+4Ch] [rbp-164h] BYREF
  _QWORD v100[6]; // [rsp+50h] [rbp-160h] BYREF
  _BYTE *v101; // [rsp+80h] [rbp-130h] BYREF
  __int64 v102; // [rsp+88h] [rbp-128h]
  _BYTE s[64]; // [rsp+90h] [rbp-120h] BYREF
  _QWORD *v104; // [rsp+D0h] [rbp-E0h] BYREF
  __int64 v105; // [rsp+D8h] [rbp-D8h]
  _QWORD v106[8]; // [rsp+E0h] [rbp-D0h] BYREF
  __int64 v107; // [rsp+120h] [rbp-90h] BYREF
  void *v108; // [rsp+128h] [rbp-88h]
  _BYTE v109[12]; // [rsp+130h] [rbp-80h]
  char v110; // [rsp+13Ch] [rbp-74h]
  char v111; // [rsp+140h] [rbp-70h] BYREF

  v13 = (__int64)a1;
  v16 = sub_22077B0(0xE30u);
  if ( v16 )
  {
    v17 = a1[51];
    v18 = a1[4];
    *(_QWORD *)(v16 + 24) = a11;
    v19 = a1[2];
    v20 = *(_QWORD *)(a11 + 16);
    *(_QWORD *)(v16 + 3408) = 0;
    *(_QWORD *)(v16 + 16) = v17;
    *(_QWORD *)(v16 + 32) = v13 + 2264;
    *(_QWORD *)(v16 + 48) = v16 + 64;
    *(_QWORD *)(v16 + 56) = 0x2000000000LL;
    *(_QWORD *)(v16 + 3144) = 0x2000000000LL;
    *(_QWORD *)(v16 + 3472) = v16 + 3488;
    *(_QWORD *)v16 = v18;
    *(_QWORD *)(v16 + 3480) = 0x400000000LL;
    *(_QWORD *)(v16 + 3616) = v19;
    *(_QWORD *)(v16 + 3136) = v16 + 3152;
    *(_QWORD *)(v16 + 3416) = 0;
    *(_QWORD *)(v16 + 3424) = 0;
    *(_DWORD *)(v16 + 3432) = 0;
    *(_QWORD *)(v16 + 3440) = 0;
    *(_QWORD *)(v16 + 3448) = 0;
    *(_QWORD *)(v16 + 3456) = 0;
    *(_DWORD *)(v16 + 3464) = 0;
    *(_QWORD *)(v16 + 3552) = 0;
    *(_QWORD *)(v16 + 3560) = 0;
    *(_QWORD *)(v16 + 3568) = 0;
    *(_DWORD *)(v16 + 3576) = 0;
    *(_QWORD *)(v16 + 3584) = 0;
    *(_QWORD *)(v16 + 3592) = 0;
    *(_QWORD *)(v16 + 3600) = 0;
    *(_DWORD *)(v16 + 3608) = 0;
    *(_QWORD *)(v16 + 3624) = v13 + 56;
    v21 = *(__int64 (**)(void))(*(_QWORD *)v20 + 144LL);
    v22 = 0;
    if ( v21 != sub_2C8F680 )
      v22 = v21();
    *(_QWORD *)(v16 + 8) = v22;
    *(_BYTE *)(v16 + 40) = sub_35DDE70(*(_QWORD *)(a12 + 256) + 856LL);
  }
  v23 = *(_QWORD *)(v13 + 408);
  *(_QWORD *)(v13 + 432) = v16;
  v24 = 0;
  v25 = *(_QWORD *)(v13 + 352) == 0;
  LODWORD(v23) = *(_DWORD *)(v23 + 40);
  *(_QWORD *)(v13 + 424) = 0;
  v99 = v23;
  if ( v25 )
    return v24;
  v26 = s;
  v101 = s;
  v102 = 0x1000000000LL;
  if ( a2 )
  {
    if ( a2 > 0x10uLL )
    {
      sub_C8D5F0((__int64)&v101, s, a2, 4u, v14, v15);
      v26 = &v101[4 * (unsigned int)v102];
    }
    memset(v26, 0, 4LL * a2);
    LODWORD(v102) = a2 + v102;
  }
  v27 = 0;
  sub_37D8820(v13, &v101, a3, a5, v14, v15);
  v110 = 1;
  v30 = v106;
  v100[0] = a10;
  v100[1] = v13;
  v100[2] = a9;
  v107 = 0;
  v100[3] = a6;
  v100[4] = &v99;
  *(_QWORD *)v109 = 8;
  v100[5] = a8;
  v108 = &v111;
  v31 = *(_QWORD *)(v13 + 352);
  *(_DWORD *)&v109[8] = 0;
  v106[0] = v31;
  v105 = 0x400000001LL;
  v32 = 1;
  v104 = v106;
  v106[1] = 0;
  while ( 1 )
  {
    v35 = &v30[2 * v32 - 2];
    v36 = v35[1];
    v37 = *v35;
    v35[1] = v36 + 1;
    v38 = *(unsigned int *)(a3 + 24);
    v39 = *(_QWORD *)(a3 + 8);
    if ( !(_DWORD)v38 )
      goto LABEL_58;
    v29 = (unsigned int)(v38 - 1);
    v40 = ((unsigned int)v37 >> 4) ^ ((unsigned int)v37 >> 9);
    v28 = (unsigned int)v29 & v40;
    v41 = (__int64 *)(v39 + 16 * v28);
    v42 = *v41;
    if ( v37 != *v41 )
    {
      v77 = 1;
      while ( v42 != -4096 )
      {
        v90 = v77 + 1;
        v28 = (unsigned int)v29 & (v77 + (_DWORD)v28);
        v41 = (__int64 *)(v39 + 16LL * (unsigned int)v28);
        v42 = *v41;
        if ( v37 == *v41 )
          goto LABEL_18;
        v77 = v90;
      }
LABEL_58:
      v43 = *(_DWORD *)(v37 + 176);
      goto LABEL_25;
    }
LABEL_18:
    v43 = *(_DWORD *)(v37 + 176);
    if ( v43 >= v27 && v41 != (__int64 *)(16 * v38 + v39) )
    {
      v44 = v41[1];
      v45 = *(unsigned int *)(a4 + 24);
      v46 = *(_QWORD *)(a4 + 8);
      if ( (_DWORD)v45 )
      {
        LODWORD(v47) = (v45 - 1) & v40;
        v48 = (__int64 *)(v46 + 88LL * (unsigned int)v47);
        v49 = *v48;
        if ( v37 == *v48 )
          goto LABEL_22;
        v84 = 1;
        while ( v49 != -4096 )
        {
          v94 = v84 + 1;
          v47 = ((_DWORD)v45 - 1) & (unsigned int)(v47 + v84);
          v48 = (__int64 *)(v46 + 88 * v47);
          v49 = *v48;
          if ( v37 == *v48 )
            goto LABEL_22;
          v84 = v94;
        }
      }
      v48 = (__int64 *)(v46 + 88 * v45);
LABEL_22:
      v50 = (__int64)(v48 + 1);
      v51 = *(_QWORD *)(a5 + 8);
      v52 = *(unsigned int *)(a5 + 24);
      if ( (_DWORD)v52 )
      {
        v53 = (v52 - 1) & v40;
        v54 = (__int64 *)(v51 + 72LL * v53);
        v55 = *v54;
        if ( v37 == *v54 )
        {
LABEL_24:
          sub_37D8C20(v13, v44, v50, (__int64)(v54 + 1), a6, a8, (int)a9, a10);
          v43 = *(_DWORD *)(v37 + 176);
          goto LABEL_25;
        }
        v85 = 1;
        while ( v55 != -4096 )
        {
          v91 = v85 + 1;
          v53 = (v52 - 1) & (v53 + v85);
          v54 = (__int64 *)(v51 + 72LL * v53);
          v55 = *v54;
          if ( v37 == *v54 )
            goto LABEL_24;
          v85 = v91;
        }
      }
      v54 = (__int64 *)(v51 + 72 * v52);
      goto LABEL_24;
    }
LABEL_25:
    v87 = (unsigned int)v105;
    if ( v27 < v43 )
      v27 = v43;
    if ( v36 < *(unsigned int *)(v37 + 40) )
    {
      v33 = *(_QWORD *)(*(_QWORD *)(v37 + 32) + 8 * v36);
      if ( (unsigned __int64)(unsigned int)v105 + 1 > HIDWORD(v105) )
      {
        sub_C8D5F0((__int64)&v104, v106, (unsigned int)v105 + 1LL, 0x10u, v28, v29);
        v87 = (unsigned int)v105;
      }
      v34 = &v104[2 * v87];
      *v34 = v33;
      v34[1] = 0;
      v32 = (unsigned int)(v105 + 1);
      LODWORD(v105) = v105 + 1;
      goto LABEL_14;
    }
    v56 = *(unsigned int *)(a3 + 24);
    v32 = (unsigned int)(v105 - 1);
    v57 = *(_QWORD *)(a3 + 8);
    LODWORD(v105) = v105 - 1;
    if ( !(_DWORD)v56 )
      goto LABEL_14;
    v29 = ((_DWORD)v56 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
    v58 = (__int64 *)(v57 + 16 * v29);
    v28 = *v58;
    if ( v37 == *v58 )
      break;
    v88 = 1;
    while ( v28 != -4096 )
    {
      v89 = v88 + 1;
      v29 = ((_DWORD)v56 - 1) & (unsigned int)(v88 + v29);
      v58 = (__int64 *)(v57 + 16LL * (unsigned int)v29);
      v28 = *v58;
      if ( v37 == *v58 )
        goto LABEL_30;
      v88 = v89;
    }
LABEL_14:
    if ( !(_DWORD)v32 )
      goto LABEL_44;
LABEL_15:
    v30 = v104;
  }
LABEL_30:
  if ( v58 == (__int64 *)(v57 + 16 * v56) )
    goto LABEL_14;
  v59 = *(unsigned int *)(a5 + 24);
  v60 = *(_QWORD *)(a5 + 8);
  if ( (_DWORD)v59 )
  {
    v61 = (v59 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
    v62 = (__int64 *)(v60 + 72LL * v61);
    v63 = *v62;
    if ( v37 == *v62 )
      goto LABEL_33;
    v86 = 1;
    while ( v63 != -4096 )
    {
      v92 = v86 + 1;
      v93 = ((_DWORD)v59 - 1) & (v61 + v86);
      v61 = v93;
      v62 = (__int64 *)(v60 + 72 * v93);
      v63 = *v62;
      if ( v37 == *v62 )
        goto LABEL_33;
      v86 = v92;
    }
  }
  v62 = (__int64 *)(v60 + 72 * v59);
LABEL_33:
  v64 = v58[1];
  sub_37D7E90(v13, v64, (__int64)&v107, (__int64)(v62 + 1));
  v65 = v110;
  v66 = (__int64 *)v108;
  if ( v110 )
  {
    v67 = (__int64 *)((char *)v108 + 8 * *(unsigned int *)&v109[4]);
    if ( v108 != v67 )
      goto LABEL_35;
    ++v107;
    goto LABEL_42;
  }
  v67 = (__int64 *)((char *)v108 + 8 * *(unsigned int *)v109);
  if ( v108 == v67 )
  {
    ++v107;
    goto LABEL_38;
  }
LABEL_35:
  while ( 1 )
  {
    v64 = *v66;
    v68 = v66;
    if ( (unsigned __int64)*v66 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v67 == ++v66 )
      goto LABEL_37;
  }
  if ( v66 != v67 )
  {
    v78 = a3;
    v79 = v13;
    v80 = v37;
    v81 = v67;
    do
    {
      if ( *(_DWORD *)&v101[4 * *(int *)(v64 + 24)] == *(_DWORD *)(v80 + 180) )
      {
        v97 = v78;
        sub_37E2870((__int64)v100, v64, a7);
        v78 = v97;
      }
      v82 = v68 + 1;
      if ( v68 + 1 == v81 )
        break;
      while ( 1 )
      {
        v64 = *v82;
        v68 = v82;
        if ( (unsigned __int64)*v82 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v81 == ++v82 )
          goto LABEL_66;
      }
    }
    while ( v81 != v82 );
LABEL_66:
    v13 = v79;
    a3 = v78;
    v65 = v110;
  }
LABEL_37:
  ++v107;
  if ( v65 )
  {
LABEL_42:
    *(_QWORD *)&v109[4] = 0;
  }
  else
  {
LABEL_38:
    v69 = 4 * (*(_DWORD *)&v109[4] - *(_DWORD *)&v109[8]);
    if ( v69 < 0x20 )
      v69 = 32;
    if ( v69 >= *(_DWORD *)v109 )
    {
      memset(v108, -1, 8LL * *(unsigned int *)v109);
      goto LABEL_42;
    }
    sub_C8C990((__int64)&v107, v64);
  }
  v32 = (unsigned int)v105;
  if ( (_DWORD)v105 )
    goto LABEL_15;
LABEL_44:
  v70 = *(__int64 **)(v13 + 448);
  if ( *(_BYTE *)(v13 + 468) )
    v71 = *(unsigned int *)(v13 + 460);
  else
    v71 = *(unsigned int *)(v13 + 456);
  v72 = &v70[v71];
  if ( v70 != v72 )
  {
    while ( 1 )
    {
      v73 = *v70;
      v74 = v70;
      if ( (unsigned __int64)*v70 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v72 == ++v70 )
        goto LABEL_49;
    }
    if ( v70 != v72 )
    {
      do
      {
        if ( *(_QWORD *)(*a9 + 8LL * *(int *)(v73 + 24)) )
          sub_37E2870((__int64)v100, v73, a7);
        v83 = v74 + 1;
        if ( v74 + 1 == v72 )
          break;
        v73 = *v83;
        for ( ++v74; (unsigned __int64)*v83 >= 0xFFFFFFFFFFFFFFFELL; v74 = v83 )
        {
          if ( v72 == ++v83 )
            goto LABEL_49;
          v73 = *v83;
        }
      }
      while ( v72 != v74 );
    }
  }
LABEL_49:
  LOBYTE(v75) = sub_37C82C0(v13);
  v24 = v75;
  if ( v104 != v106 )
    _libc_free((unsigned __int64)v104);
  if ( !v110 )
    _libc_free((unsigned __int64)v108);
  if ( v101 != s )
    _libc_free((unsigned __int64)v101);
  return v24;
}
