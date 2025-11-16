// Function: sub_D640E0
// Address: 0xd640e0
//
__int64 __fastcall sub_D640E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5, __int64 a6)
{
  int v9; // edx
  __int64 v10; // r13
  __int64 v11; // rdi
  unsigned int v12; // r15d
  __int64 v13; // r15
  __int64 v14; // rdi
  unsigned int v15; // esi
  bool v16; // r8
  __int64 v17; // rax
  unsigned __int64 v18; // r8
  __int64 v19; // r15
  __int64 v20; // r8
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rsi
  __int64 v24; // r12
  int v25; // edx
  unsigned __int64 v26; // rax
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r8
  __int64 v31; // rsi
  __int128 v32; // rax
  __int64 v33; // r13
  __int64 *(__fastcall *v34)(__int64 **, __int64 *, __int64, __int64, __int64, __int64); // rdx
  __m128i v35; // xmm1
  __m128i v36; // xmm0
  void *v37; // r8
  __int64 v38; // r12
  _QWORD *v39; // rbx
  __int64 v40; // rax
  __int64 v41; // r12
  __int64 v42; // rax
  _QWORD *v43; // rsi
  __int64 v44; // rbx
  _QWORD *v45; // r15
  __int64 v46; // rax
  _QWORD *v47; // rbx
  _QWORD *v48; // r13
  __int64 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rdx
  int v53; // ecx
  int v54; // eax
  _QWORD *v55; // rdi
  __int64 *v56; // rax
  __int64 v57; // rsi
  __int64 v58; // rdx
  unsigned int *v59; // r15
  unsigned int *v60; // r12
  __int64 v61; // rdx
  unsigned int v62; // esi
  unsigned int *v63; // r13
  unsigned int *v64; // rbx
  __int64 v65; // rdx
  unsigned int v66; // esi
  _QWORD **v67; // rdx
  int v68; // ecx
  int v69; // eax
  __int64 *v70; // rax
  __int64 v71; // rsi
  unsigned int *v72; // r13
  unsigned int *v73; // rbx
  __int64 v74; // rdx
  unsigned int v75; // esi
  __int64 v76; // rax
  _QWORD *v77; // r12
  _QWORD *v78; // r13
  __int64 v79; // rax
  __int64 v80; // rax
  __int64 v81; // [rsp+0h] [rbp-380h]
  __int64 v82; // [rsp+8h] [rbp-378h]
  __int64 v83; // [rsp+8h] [rbp-378h]
  __int64 v84; // [rsp+10h] [rbp-370h]
  int v85; // [rsp+18h] [rbp-368h]
  __int64 v86; // [rsp+18h] [rbp-368h]
  __int64 v87; // [rsp+18h] [rbp-368h]
  bool v88; // [rsp+20h] [rbp-360h]
  _BYTE *v90; // [rsp+28h] [rbp-358h]
  __int64 v92; // [rsp+38h] [rbp-348h] BYREF
  __int64 v93; // [rsp+48h] [rbp-338h]
  void *v94; // [rsp+50h] [rbp-330h]
  __int64 v95; // [rsp+58h] [rbp-328h]
  __m128i v96; // [rsp+60h] [rbp-320h] BYREF
  void (__fastcall *v97)(__m128i *, __m128i *, __int64); // [rsp+70h] [rbp-310h]
  __int64 *(__fastcall *v98)(__int64 **, __int64 *, __int64, __int64, __int64, __int64); // [rsp+78h] [rbp-308h]
  __int16 v99; // [rsp+80h] [rbp-300h]
  void *v100; // [rsp+90h] [rbp-2F0h] BYREF
  __m128i v101; // [rsp+98h] [rbp-2E8h] BYREF
  __int64 (__fastcall *v102)(_QWORD *, _QWORD *, int); // [rsp+A8h] [rbp-2D8h]
  __int64 *(__fastcall *v103)(__int64 **, __int64 *, __int64, __int64, __int64, __int64); // [rsp+B0h] [rbp-2D0h]
  unsigned int *v104; // [rsp+C0h] [rbp-2C0h] BYREF
  __int64 v105; // [rsp+C8h] [rbp-2B8h]
  _BYTE v106[32]; // [rsp+D0h] [rbp-2B0h] BYREF
  __int64 v107; // [rsp+F0h] [rbp-290h]
  __int64 v108; // [rsp+F8h] [rbp-288h]
  __int64 v109; // [rsp+100h] [rbp-280h]
  __int64 v110; // [rsp+108h] [rbp-278h]
  _QWORD *v111; // [rsp+110h] [rbp-270h]
  void **v112; // [rsp+118h] [rbp-268h]
  __int64 v113; // [rsp+120h] [rbp-260h]
  int v114; // [rsp+128h] [rbp-258h]
  __int16 v115; // [rsp+12Ch] [rbp-254h]
  char v116; // [rsp+12Eh] [rbp-252h]
  __int64 v117; // [rsp+130h] [rbp-250h]
  __int64 v118; // [rsp+138h] [rbp-248h]
  _QWORD v119[2]; // [rsp+140h] [rbp-240h] BYREF
  void *v120; // [rsp+150h] [rbp-230h] BYREF
  __int64 v121; // [rsp+158h] [rbp-228h] BYREF
  __int64 (__fastcall *v122)(_QWORD *, _QWORD *, int); // [rsp+168h] [rbp-218h]
  __int64 *(__fastcall *v123)(__int64 **, __int64 *, __int64, __int64, __int64, __int64); // [rsp+170h] [rbp-210h]
  __int64 v124[3]; // [rsp+180h] [rbp-200h] BYREF
  _BYTE *v125; // [rsp+198h] [rbp-1E8h]
  _BYTE v126[112]; // [rsp+1A8h] [rbp-1D8h] BYREF
  void *v127; // [rsp+218h] [rbp-168h]
  _QWORD v128[8]; // [rsp+228h] [rbp-158h] BYREF
  _QWORD *v129; // [rsp+268h] [rbp-118h]
  unsigned int v130; // [rsp+278h] [rbp-108h]
  __int64 v131; // [rsp+288h] [rbp-F8h]
  char v132; // [rsp+29Ch] [rbp-E4h]
  __int64 v133; // [rsp+2F8h] [rbp-88h]
  char v134; // [rsp+30Ch] [rbp-74h]

  v9 = *(_DWORD *)(a1 + 4);
  v92 = a6;
  v10 = v9 & 0x7FFFFFF;
  v11 = *(_QWORD *)(a1 + 32 * (1 - v10));
  v12 = *(_DWORD *)(v11 + 32);
  if ( v12 <= 0x40 )
    v88 = *(_QWORD *)(v11 + 24) == 0;
  else
    v88 = v12 == (unsigned int)sub_C444A0(v11 + 24);
  v13 = 0;
  if ( a5 )
    LOBYTE(v13) = v88 + 2;
  v14 = *(_QWORD *)(a1 + 32 * (2 - v10));
  v15 = *(_DWORD *)(v14 + 32);
  if ( v15 <= 0x40 )
    v16 = *(_QWORD *)(v14 + 24) == 1;
  else
    v16 = v15 - 1 == (unsigned int)sub_C444A0(v14 + 24);
  v17 = v16;
  v18 = v13 & 0xFFFFFFFFFF00FFFFLL;
  v19 = *(_QWORD *)(a1 + 8);
  v20 = (v17 << 16) | v18;
  v21 = *(_QWORD *)(a1 + 32 * (3 - v10));
  if ( *(_DWORD *)(v21 + 32) <= 0x40u )
  {
    if ( !*(_QWORD *)(v21 + 24) )
      goto LABEL_9;
  }
  else
  {
    v84 = v20;
    v85 = *(_DWORD *)(v21 + 32);
    v22 = sub_C444A0(v21 + 24);
    v20 = v84;
    if ( v85 == v22 )
    {
LABEL_9:
      v23 = (__int64)v124;
      if ( (unsigned __int8)sub_D62CA0(*(_QWORD *)(a1 - 32 * v10), v124, a2, a3, v20, a4) )
      {
        v23 = v124[0];
        v25 = *(_DWORD *)(v19 + 8) >> 8;
        if ( *(_DWORD *)(v19 + 8) > 0x3FFFu )
          return sub_ACD640(v19, v124[0], 0);
        v26 = 0;
        if ( v25 )
          v26 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v25);
        if ( v124[0] <= v26 )
          return sub_ACD640(v19, v124[0], 0);
      }
      goto LABEL_10;
    }
  }
  v86 = v20;
  v28 = sub_B43CB0(a1);
  v29 = sub_B2BE50(v28);
  v30 = v86;
  v87 = v29;
  sub_D5EB90((__int64)v124, a2, a3, v29, v30, a4);
  v31 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  *(_QWORD *)&v32 = sub_D63C20((__int64)v124, v31);
  v90 = (_BYTE *)*((_QWORD *)&v32 + 1);
  v33 = v32;
  if ( v32 == 0 )
  {
    if ( !v134 )
      _libc_free(v133, v31);
    if ( !v132 )
      _libc_free(v131, v31);
    v76 = v130;
    if ( v130 )
    {
      v77 = v129;
      v78 = &v129[7 * v130];
      do
      {
        if ( *v77 != -8192 && *v77 != -4096 )
        {
          v79 = v77[6];
          if ( v79 != -4096 && v79 != 0 && v79 != -8192 )
            sub_BD60C0(v77 + 4);
          v80 = v77[3];
          if ( v80 != -4096 && v80 != 0 && v80 != -8192 )
            sub_BD60C0(v77 + 1);
        }
        v77 += 7;
      }
      while ( v78 != v77 );
      v76 = v130;
    }
    v23 = 56 * v76;
    sub_C7D6A0((__int64)v129, 56 * v76, 8);
    sub_B32BF0(v128);
    v127 = &unk_49D94D0;
    nullsub_63();
    if ( v125 != v126 )
      _libc_free(v125, v23);
LABEL_10:
    if ( !a5 )
      return 0;
    if ( v88 )
      return sub_AD62B0(v19);
    return sub_AD6530(v19, v23);
  }
  v34 = v103;
  v35 = _mm_loadu_si128(&v101);
  v103 = sub_D5BA50;
  v104 = (unsigned int *)v106;
  v105 = 0x200000000LL;
  v96.m128i_i64[0] = (__int64)&v92;
  v102 = sub_D5B9F0;
  v36 = _mm_loadu_si128(&v96);
  v110 = v87;
  v111 = v119;
  v94 = &unk_49D94D0;
  v112 = &v120;
  v115 = 512;
  v96 = v35;
  v101 = v36;
  v100 = &unk_49DA0D8;
  v98 = v34;
  v97 = 0;
  v95 = a2;
  v113 = 0;
  v114 = 0;
  v116 = 7;
  v117 = 0;
  v118 = 0;
  v107 = 0;
  v108 = 0;
  LOWORD(v109) = 0;
  v119[0] = &unk_49D94D0;
  v119[1] = a2;
  v120 = &unk_49DA0D8;
  v122 = 0;
  sub_D5B9F0(&v121, &v101, 2);
  v94 = v37;
  v123 = v103;
  v122 = v102;
  nullsub_63();
  sub_B32BF0(&v100);
  if ( v97 )
    v97(&v96, &v96, 3);
  sub_D5F1F0((__int64)&v104, a1);
  v99 = 257;
  v38 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, _BYTE *, _QWORD, _QWORD))(*v111 + 32LL))(
          v111,
          15,
          v33,
          v90,
          0,
          0);
  if ( !v38 )
  {
    LOWORD(v103) = 257;
    v38 = sub_B504D0(15, v33, (__int64)v90, (__int64)&v100, 0, 0);
    (*((void (__fastcall **)(void **, __int64, __m128i *, __int64, __int64))*v112 + 2))(v112, v38, &v96, v108, v109);
    if ( v104 != &v104[4 * (unsigned int)v105] )
    {
      v83 = v33;
      v63 = v104;
      v64 = &v104[4 * (unsigned int)v105];
      do
      {
        v65 = *((_QWORD *)v63 + 1);
        v66 = *v63;
        v63 += 4;
        sub_B99FD0(v38, v66, v65);
      }
      while ( v64 != v63 );
      v33 = v83;
    }
  }
  v99 = 257;
  v39 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, _BYTE *))(*v111 + 56LL))(v111, 36, v33, v90);
  if ( !v39 )
  {
    LOWORD(v103) = 257;
    v39 = sub_BD2C40(72, unk_3F10FD0);
    if ( v39 )
    {
      v52 = *(_QWORD *)(v33 + 8);
      v53 = *(unsigned __int8 *)(v52 + 8);
      if ( (unsigned int)(v53 - 17) > 1 )
      {
        v57 = sub_BCB2A0(*(_QWORD **)v52);
      }
      else
      {
        v54 = *(_DWORD *)(v52 + 32);
        v55 = *(_QWORD **)v52;
        BYTE4(v93) = (_BYTE)v53 == 18;
        LODWORD(v93) = v54;
        v56 = (__int64 *)sub_BCB2A0(v55);
        v57 = sub_BCE1B0(v56, v93);
      }
      sub_B523C0((__int64)v39, v57, 53, 36, v33, (__int64)v90, (__int64)&v100, 0, 0, 0);
    }
    (*((void (__fastcall **)(void **, _QWORD *, __m128i *, __int64, __int64))*v112 + 2))(v112, v39, &v96, v108, v109);
    v58 = 4LL * (unsigned int)v105;
    if ( v104 != &v104[v58] )
    {
      v82 = v19;
      v59 = &v104[v58];
      v81 = v38;
      v60 = v104;
      do
      {
        v61 = *((_QWORD *)v60 + 1);
        v62 = *v60;
        v60 += 4;
        sub_B99FD0((__int64)v39, v62, v61);
      }
      while ( v59 != v60 );
      v19 = v82;
      v38 = v81;
    }
  }
  LOWORD(v103) = 257;
  v40 = sub_A830B0(&v104, v38, v19, (__int64)&v100);
  LOWORD(v103) = 257;
  v41 = v40;
  v42 = sub_ACD640(v19, 0, 0);
  v43 = v39;
  v24 = sub_B36550(&v104, (__int64)v39, v42, v41, (__int64)&v100, 0);
  if ( *(_BYTE *)v33 > 0x15u || *v90 > 0x15u )
  {
    v99 = 257;
    v44 = sub_ACD640(v19, -1, 0);
    v45 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD *, __int64, __int64, __int64))(*v111 + 56LL))(v111, 33, v24, v44);
    if ( !v45 )
    {
      LOWORD(v103) = 257;
      v45 = sub_BD2C40(72, unk_3F10FD0);
      if ( v45 )
      {
        v67 = *(_QWORD ***)(v24 + 8);
        v68 = *((unsigned __int8 *)v67 + 8);
        if ( (unsigned int)(v68 - 17) > 1 )
        {
          v71 = sub_BCB2A0(*v67);
        }
        else
        {
          v69 = *((_DWORD *)v67 + 8);
          BYTE4(v94) = (_BYTE)v68 == 18;
          LODWORD(v94) = v69;
          v70 = (__int64 *)sub_BCB2A0(*v67);
          v71 = sub_BCE1B0(v70, (__int64)v94);
        }
        sub_B523C0((__int64)v45, v71, 53, 33, v24, v44, (__int64)&v100, 0, 0, 0);
      }
      (*((void (__fastcall **)(void **, _QWORD *, __m128i *, __int64, __int64))*v112 + 2))(v112, v45, &v96, v108, v109);
      v72 = v104;
      v73 = &v104[4 * (unsigned int)v105];
      if ( v104 != v73 )
      {
        do
        {
          v74 = *((_QWORD *)v72 + 1);
          v75 = *v72;
          v72 += 4;
          sub_B99FD0((__int64)v45, v75, v74);
        }
        while ( v73 != v72 );
      }
    }
    v43 = v45;
    sub_B33B40((__int64)&v104, (__int64)v45, 0, 0);
  }
  sub_B32BF0(&v120);
  v119[0] = &unk_49D94D0;
  nullsub_63();
  if ( v104 != (unsigned int *)v106 )
    _libc_free(v104, v43);
  if ( !v134 )
    _libc_free(v133, v43);
  if ( !v132 )
    _libc_free(v131, v43);
  v46 = v130;
  if ( v130 )
  {
    v47 = v129;
    v48 = &v129[7 * v130];
    do
    {
      if ( *v47 != -8192 && *v47 != -4096 )
      {
        v49 = v47[6];
        if ( v49 != -4096 && v49 != 0 && v49 != -8192 )
          sub_BD60C0(v47 + 4);
        v50 = v47[3];
        if ( v50 != -4096 && v50 != 0 && v50 != -8192 )
          sub_BD60C0(v47 + 1);
      }
      v47 += 7;
    }
    while ( v48 != v47 );
    v46 = v130;
  }
  v51 = 56 * v46;
  sub_C7D6A0((__int64)v129, 56 * v46, 8);
  sub_B32BF0(v128);
  v127 = &unk_49D94D0;
  nullsub_63();
  if ( v125 != v126 )
    _libc_free(v125, v51);
  return v24;
}
