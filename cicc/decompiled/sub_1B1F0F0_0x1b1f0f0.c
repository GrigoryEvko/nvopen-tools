// Function: sub_1B1F0F0
// Address: 0x1b1f0f0
//
__int64 __fastcall sub_1B1F0F0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  unsigned __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // r14
  unsigned int v15; // r15d
  bool v16; // al
  unsigned __int64 v17; // rax
  const char *v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  const char *v21; // rax
  __int64 v22; // rdx
  __int64 v23; // r9
  __int64 v24; // rdx
  _QWORD *v25; // r14
  _QWORD *v26; // rax
  __int64 v27; // r13
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rcx
  __int64 v31; // rax
  int v32; // r8d
  unsigned int v33; // esi
  __int64 *v34; // rdx
  _QWORD *v35; // r9
  __int64 *v36; // rax
  __int64 v37; // rbx
  unsigned int v38; // esi
  __int64 *v39; // rdx
  __int64 v40; // r9
  __int64 v41; // r14
  __int64 v42; // r13
  char *v43; // rdx
  char *v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rcx
  char *v47; // rax
  _BYTE *v48; // rsi
  const char *v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 *v53; // r14
  __int64 *v54; // rbx
  __int64 v55; // r13
  _QWORD *v56; // rbx
  _QWORD *v57; // r12
  __int64 v58; // rax
  int v60; // edx
  int v61; // r10d
  int v62; // edx
  int v63; // r10d
  __int64 v64; // [rsp-10h] [rbp-470h]
  __int64 v65; // [rsp+8h] [rbp-458h]
  __int64 v66; // [rsp+10h] [rbp-450h]
  __int64 v67; // [rsp+18h] [rbp-448h]
  const char *v69; // [rsp+30h] [rbp-430h] BYREF
  __int64 v70; // [rsp+38h] [rbp-428h]
  _BYTE v71[64]; // [rsp+40h] [rbp-420h] BYREF
  _QWORD v72[4]; // [rsp+80h] [rbp-3E0h] BYREF
  _QWORD *v73; // [rsp+A0h] [rbp-3C0h]
  __int64 v74; // [rsp+A8h] [rbp-3B8h]
  unsigned int v75; // [rsp+B0h] [rbp-3B0h]
  __int64 v76; // [rsp+B8h] [rbp-3A8h]
  __int64 v77; // [rsp+C0h] [rbp-3A0h]
  __int64 v78; // [rsp+C8h] [rbp-398h]
  __int64 v79; // [rsp+D0h] [rbp-390h]
  __int64 v80; // [rsp+D8h] [rbp-388h]
  __int64 v81; // [rsp+E0h] [rbp-380h]
  __int64 v82; // [rsp+E8h] [rbp-378h]
  __int64 v83; // [rsp+F0h] [rbp-370h]
  __int64 v84; // [rsp+F8h] [rbp-368h]
  __int64 v85; // [rsp+100h] [rbp-360h]
  __int64 v86; // [rsp+108h] [rbp-358h]
  int v87; // [rsp+110h] [rbp-350h]
  __int64 v88; // [rsp+118h] [rbp-348h]
  _BYTE *v89; // [rsp+120h] [rbp-340h]
  _BYTE *v90; // [rsp+128h] [rbp-338h]
  __int64 v91; // [rsp+130h] [rbp-330h]
  int v92; // [rsp+138h] [rbp-328h]
  _BYTE v93[16]; // [rsp+140h] [rbp-320h] BYREF
  __int64 v94; // [rsp+150h] [rbp-310h]
  __int64 v95; // [rsp+158h] [rbp-308h]
  __int64 v96; // [rsp+160h] [rbp-300h]
  __int64 v97; // [rsp+168h] [rbp-2F8h]
  __int64 v98; // [rsp+170h] [rbp-2F0h]
  __int64 v99; // [rsp+178h] [rbp-2E8h]
  __int16 v100; // [rsp+180h] [rbp-2E0h]
  __int64 v101[5]; // [rsp+188h] [rbp-2D8h] BYREF
  int v102; // [rsp+1B0h] [rbp-2B0h]
  __int64 v103; // [rsp+1B8h] [rbp-2A8h]
  __int64 v104; // [rsp+1C0h] [rbp-2A0h]
  __int64 v105; // [rsp+1C8h] [rbp-298h]
  _BYTE *v106; // [rsp+1D0h] [rbp-290h]
  __int64 v107; // [rsp+1D8h] [rbp-288h]
  _BYTE v108[64]; // [rsp+1E0h] [rbp-280h] BYREF
  const char *v109; // [rsp+220h] [rbp-240h] BYREF
  char *v110; // [rsp+228h] [rbp-238h]
  _QWORD v111[70]; // [rsp+230h] [rbp-230h] BYREF

  v3 = (_QWORD *)sub_13FC520(*(_QWORD *)a1);
  v4 = sub_157EBA0((__int64)v3);
  sub_3860710(*(_QWORD *)(a1 + 488), v4, a1 + 96);
  v6 = v5;
  v7 = sub_1458800(**(_QWORD **)(a1 + 488));
  v8 = sub_157EB90((__int64)v3);
  v9 = sub_1632FA0(v8);
  v10 = *(_QWORD *)(a1 + 512);
  v72[2] = "scev.check";
  v89 = v93;
  v90 = v93;
  v72[0] = v10;
  v72[1] = v9;
  v72[3] = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v91 = 2;
  v92 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 1;
  v11 = sub_15E0530(*(_QWORD *)(v10 + 24));
  v105 = v9;
  v101[3] = v11;
  v106 = v108;
  memset(v101, 0, 24);
  v101[4] = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v107 = 0x800000000LL;
  v12 = sub_157EBA0((__int64)v3);
  v13 = sub_387DD00(v72, v7, v12);
  v14 = v13;
  if ( *(_BYTE *)(v13 + 16) != 13
    || ((v15 = *(_DWORD *)(v13 + 32), v15 <= 0x40)
      ? (v16 = *(_QWORD *)(v13 + 24) == 0)
      : (v16 = v15 == (unsigned int)sub_16A57B0(v13 + 24)),
        !v16) )
  {
    if ( v6 )
    {
      v109 = "lver.safe";
      LOWORD(v111[0]) = 259;
      v6 = sub_15FB440(27, (__int64 *)v6, v14, (__int64)&v109, 0);
      if ( *(_BYTE *)(v6 + 16) > 0x17u )
      {
        v17 = sub_157EBA0((__int64)v3);
        sub_15F2120(v6, v17);
      }
    }
    else
    {
      v6 = v14;
    }
  }
  v18 = sub_1649960(**(_QWORD **)(*(_QWORD *)a1 + 32LL));
  v109 = (const char *)&v69;
  v69 = v18;
  LOWORD(v111[0]) = 773;
  v70 = v19;
  v110 = ".lver.check";
  sub_164B780((__int64)v3, (__int64 *)&v109);
  v20 = sub_157EBA0((__int64)v3);
  v67 = sub_1AA8CA0(v3, v20, *(_QWORD *)(a1 + 504), *(_QWORD *)(a1 + 496));
  v21 = sub_1649960(**(_QWORD **)(*(_QWORD *)a1 + 32LL));
  v109 = (const char *)&v69;
  v69 = v21;
  v70 = v22;
  LOWORD(v111[0]) = 773;
  v110 = ".ph";
  sub_164B780(v67, (__int64 *)&v109);
  v64 = *(_QWORD *)(a1 + 504);
  v23 = *(_QWORD *)(a1 + 496);
  v24 = *(_QWORD *)a1;
  v69 = v71;
  v70 = 0x800000000LL;
  v109 = ".lver.orig";
  LOWORD(v111[0]) = 259;
  *(_QWORD *)(a1 + 8) = sub_1AB91F0(v67, (__int64)v3, v24, a1 + 16, (__int64 *)&v109, v23, v64, (__int64)&v69);
  sub_1AB3E10((__int64)&v69, a1 + 16);
  v25 = (_QWORD *)sub_157EBA0((__int64)v3);
  v65 = sub_13FC520(*(_QWORD *)a1);
  v66 = sub_13FC520(*(_QWORD *)(a1 + 8));
  v26 = sub_1648A60(56, 3u);
  if ( v26 )
    sub_15F83E0((__int64)v26, v66, v65, v6, (__int64)v25);
  sub_15F20C0(v25);
  v27 = *(_QWORD *)(a1 + 504);
  v28 = sub_13FA090(*(_QWORD *)a1);
  v29 = *(_QWORD *)(v27 + 32);
  v30 = v28;
  v31 = *(unsigned int *)(v27 + 48);
  if ( !(_DWORD)v31 )
    goto LABEL_91;
  v32 = v31 - 1;
  v33 = (v31 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
  v34 = (__int64 *)(v29 + 16LL * v33);
  v35 = (_QWORD *)*v34;
  if ( v3 == (_QWORD *)*v34 )
  {
LABEL_12:
    v36 = (__int64 *)(v29 + 16 * v31);
    if ( v34 != v36 )
    {
      v37 = v34[1];
      goto LABEL_14;
    }
  }
  else
  {
    v62 = 1;
    while ( v35 != (_QWORD *)-8LL )
    {
      v63 = v62 + 1;
      v33 = v32 & (v33 + v62);
      v34 = (__int64 *)(v29 + 16LL * v33);
      v35 = (_QWORD *)*v34;
      if ( v3 == (_QWORD *)*v34 )
        goto LABEL_12;
      v62 = v63;
    }
    v36 = (__int64 *)(v29 + 16 * v31);
  }
  v37 = 0;
LABEL_14:
  v38 = v32 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
  v39 = (__int64 *)(v29 + 16LL * v38);
  v40 = *v39;
  if ( v30 != *v39 )
  {
    v60 = 1;
    while ( v40 != -8 )
    {
      v61 = v60 + 1;
      v38 = v32 & (v38 + v60);
      v39 = (__int64 *)(v29 + 16LL * v38);
      v40 = *v39;
      if ( v30 == *v39 )
        goto LABEL_15;
      v60 = v61;
    }
LABEL_91:
    *(_BYTE *)(v27 + 72) = 0;
    BUG();
  }
LABEL_15:
  if ( v39 == v36 )
    goto LABEL_91;
  v41 = v39[1];
  *(_BYTE *)(v27 + 72) = 0;
  v42 = *(_QWORD *)(v41 + 8);
  if ( v42 == v37 )
    goto LABEL_42;
  v43 = *(char **)(v42 + 32);
  v44 = *(char **)(v42 + 24);
  v45 = (v43 - v44) >> 5;
  v46 = (v43 - v44) >> 3;
  if ( v45 > 0 )
  {
    v47 = &v44[32 * v45];
    while ( v41 != *(_QWORD *)v44 )
    {
      if ( v41 == *((_QWORD *)v44 + 1) )
      {
        v44 += 8;
        goto LABEL_24;
      }
      if ( v41 == *((_QWORD *)v44 + 2) )
      {
        v44 += 16;
        goto LABEL_24;
      }
      if ( v41 == *((_QWORD *)v44 + 3) )
      {
        v44 += 24;
        goto LABEL_24;
      }
      v44 += 32;
      if ( v44 == v47 )
      {
        v46 = (v43 - v44) >> 3;
        goto LABEL_66;
      }
    }
    goto LABEL_24;
  }
LABEL_66:
  switch ( v46 )
  {
    case 2LL:
LABEL_76:
      if ( v41 == *(_QWORD *)v44 )
        goto LABEL_24;
      v44 += 8;
LABEL_78:
      if ( v41 != *(_QWORD *)v44 )
        v44 = *(char **)(v42 + 32);
      goto LABEL_24;
    case 3LL:
      if ( v41 == *(_QWORD *)v44 )
        goto LABEL_24;
      v44 += 8;
      goto LABEL_76;
    case 1LL:
      goto LABEL_78;
  }
  v44 = *(char **)(v42 + 32);
LABEL_24:
  if ( v44 + 8 != v43 )
  {
    memmove(v44, v44 + 8, v43 - (v44 + 8));
    v43 = *(char **)(v42 + 32);
  }
  *(_QWORD *)(v42 + 32) = v43 - 8;
  *(_QWORD *)(v41 + 8) = v37;
  v109 = (const char *)v41;
  v48 = *(_BYTE **)(v37 + 32);
  if ( v48 == *(_BYTE **)(v37 + 40) )
  {
    sub_15CE310(v37 + 24, v48, &v109);
  }
  else
  {
    if ( v48 )
    {
      *(_QWORD *)v48 = v41;
      v48 = *(_BYTE **)(v37 + 32);
    }
    *(_QWORD *)(v37 + 32) = v48 + 8;
  }
  if ( *(_DWORD *)(v41 + 16) != *(_DWORD *)(*(_QWORD *)(v41 + 8) + 16LL) + 1 )
  {
    v111[0] = v41;
    v109 = (const char *)v111;
    v49 = (const char *)v111;
    v110 = (char *)0x4000000001LL;
    LODWORD(v50) = 1;
    do
    {
      v51 = (unsigned int)v50;
      v50 = (unsigned int)(v50 - 1);
      v52 = *(_QWORD *)&v49[8 * v51 - 8];
      LODWORD(v110) = v50;
      v53 = *(__int64 **)(v52 + 32);
      v54 = *(__int64 **)(v52 + 24);
      *(_DWORD *)(v52 + 16) = *(_DWORD *)(*(_QWORD *)(v52 + 8) + 16LL) + 1;
      if ( v54 != v53 )
      {
        do
        {
          v55 = *v54;
          if ( *(_DWORD *)(*v54 + 16) != *(_DWORD *)(*(_QWORD *)(*v54 + 8) + 16LL) + 1 )
          {
            if ( (unsigned int)v50 >= HIDWORD(v110) )
            {
              sub_16CD150((__int64)&v109, v111, 0, 8, v32, v40);
              v50 = (unsigned int)v110;
            }
            *(_QWORD *)&v109[8 * v50] = v55;
            v50 = (unsigned int)((_DWORD)v110 + 1);
            LODWORD(v110) = (_DWORD)v110 + 1;
          }
          ++v54;
        }
        while ( v53 != v54 );
        v49 = v109;
      }
    }
    while ( (_DWORD)v50 );
    if ( v49 != (const char *)v111 )
      _libc_free((unsigned __int64)v49);
  }
LABEL_42:
  sub_1B1E520(a1, a2);
  if ( v69 != v71 )
    _libc_free((unsigned __int64)v69);
  if ( v106 != v108 )
    _libc_free((unsigned __int64)v106);
  if ( v101[0] )
    sub_161E7C0((__int64)v101, v101[0]);
  j___libc_free_0(v97);
  if ( v90 != v89 )
    _libc_free((unsigned __int64)v90);
  j___libc_free_0(v85);
  j___libc_free_0(v81);
  j___libc_free_0(v77);
  if ( v75 )
  {
    v56 = v73;
    v57 = &v73[5 * v75];
    do
    {
      while ( *v56 == -8 )
      {
        if ( v56[1] != -8 )
          goto LABEL_53;
        v56 += 5;
        if ( v57 == v56 )
          return j___libc_free_0(v73);
      }
      if ( *v56 != -16 || v56[1] != -16 )
      {
LABEL_53:
        v58 = v56[4];
        if ( v58 != -8 && v58 != 0 && v58 != -16 )
          sub_1649B30(v56 + 2);
      }
      v56 += 5;
    }
    while ( v57 != v56 );
  }
  return j___libc_free_0(v73);
}
