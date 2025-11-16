// Function: sub_188F8B0
// Address: 0x188f8b0
//
__int64 __fastcall sub_188F8B0(
        _QWORD **a1,
        __m128 a2,
        __m128 si128,
        __m128i a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  __int64 *v9; // rcx
  __int64 *v10; // rdx
  double v11; // xmm4_8
  double v12; // xmm5_8
  unsigned __int8 v13; // al
  __int64 v14; // rbx
  __int64 v15; // r12
  __int64 v16; // r14
  __int64 v17; // rdi
  _QWORD *v18; // rbx
  _QWORD *v19; // r12
  __int64 v20; // rdi
  unsigned __int64 *v21; // rbx
  unsigned __int64 *v22; // r12
  unsigned __int64 v23; // rdi
  unsigned __int64 *v24; // rbx
  unsigned __int64 *v25; // r12
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // r8
  __int64 v28; // r12
  __int64 v29; // rbx
  unsigned __int64 v30; // rdi
  __int64 v32; // rax
  __int64 v33; // rax
  __m128i **v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  char v38; // al
  bool v39; // zf
  __int64 *v40; // r10
  __int64 v41; // rsi
  int v42; // eax
  __int64 v43; // rdx
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rsi
  __int64 v47; // rax
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // rax
  _BYTE *v52; // rsi
  __int64 v53; // rdx
  __int64 v54; // rcx
  int v55; // r8d
  int v56; // r9d
  unsigned __int8 v57; // [rsp+10h] [rbp-3A0h]
  __int64 *v58; // [rsp+10h] [rbp-3A0h]
  __int64 *v59; // [rsp+10h] [rbp-3A0h]
  __int64 v60; // [rsp+28h] [rbp-388h] BYREF
  __int64 v61; // [rsp+30h] [rbp-380h] BYREF
  __int64 v62; // [rsp+38h] [rbp-378h]
  __m128i *v63; // [rsp+40h] [rbp-370h] BYREF
  __int64 v64; // [rsp+48h] [rbp-368h]
  __m128i v65; // [rsp+50h] [rbp-360h] BYREF
  _DWORD v66[4]; // [rsp+60h] [rbp-350h] BYREF
  __int64 (__fastcall *v67)(_QWORD *, _DWORD *, int); // [rsp+70h] [rbp-340h]
  __int64 (__fastcall *v68)(unsigned int *); // [rsp+78h] [rbp-338h]
  __m128 *v69; // [rsp+80h] [rbp-330h] BYREF
  __int64 v70; // [rsp+88h] [rbp-328h]
  __m128 v71; // [rsp+90h] [rbp-320h] BYREF
  _DWORD v72[4]; // [rsp+A0h] [rbp-310h] BYREF
  __int64 (__fastcall *v73)(_QWORD *, _DWORD *, int); // [rsp+B0h] [rbp-300h]
  __int64 (__fastcall *v74)(unsigned int *); // [rsp+B8h] [rbp-2F8h]
  __m128i *v75; // [rsp+D0h] [rbp-2E0h] BYREF
  __int64 v76; // [rsp+D8h] [rbp-2D8h]
  __m128i v77[6]; // [rsp+E0h] [rbp-2D0h] BYREF
  _QWORD *v78; // [rsp+148h] [rbp-268h]
  unsigned int v79; // [rsp+158h] [rbp-258h]
  __int64 v80; // [rsp+160h] [rbp-250h]
  __int64 v81; // [rsp+168h] [rbp-248h]
  __int64 v82; // [rsp+170h] [rbp-240h]
  _BYTE v83[8]; // [rsp+1F0h] [rbp-1C0h] BYREF
  int v84; // [rsp+1F8h] [rbp-1B8h] BYREF
  _QWORD *v85; // [rsp+200h] [rbp-1B0h]
  int *v86; // [rsp+208h] [rbp-1A8h]
  int *v87; // [rsp+210h] [rbp-1A0h]
  __int64 v88; // [rsp+218h] [rbp-198h]
  unsigned __int64 v89; // [rsp+220h] [rbp-190h]
  __int64 v90; // [rsp+228h] [rbp-188h]
  __int64 v91; // [rsp+230h] [rbp-180h]
  int v92; // [rsp+248h] [rbp-168h] BYREF
  _QWORD *v93; // [rsp+250h] [rbp-160h]
  int *v94; // [rsp+258h] [rbp-158h]
  int *v95; // [rsp+260h] [rbp-150h]
  __int64 v96; // [rsp+268h] [rbp-148h]
  int v97; // [rsp+278h] [rbp-138h] BYREF
  __int64 v98; // [rsp+280h] [rbp-130h]
  int *v99; // [rsp+288h] [rbp-128h]
  int *v100; // [rsp+290h] [rbp-120h]
  __int64 v101; // [rsp+298h] [rbp-118h]
  __int16 v102; // [rsp+2A0h] [rbp-110h]
  char v103; // [rsp+2A2h] [rbp-10Eh]
  int v104; // [rsp+2B0h] [rbp-100h] BYREF
  _QWORD *v105; // [rsp+2B8h] [rbp-F8h]
  int *v106; // [rsp+2C0h] [rbp-F0h]
  int *v107; // [rsp+2C8h] [rbp-E8h]
  __int64 v108; // [rsp+2D0h] [rbp-E0h]
  int v109; // [rsp+2E0h] [rbp-D0h] BYREF
  _QWORD *v110; // [rsp+2E8h] [rbp-C8h]
  int *v111; // [rsp+2F0h] [rbp-C0h]
  int *v112; // [rsp+2F8h] [rbp-B8h]
  __int64 v113; // [rsp+300h] [rbp-B0h]
  _QWORD v114[2]; // [rsp+308h] [rbp-A8h] BYREF
  unsigned __int64 *v115; // [rsp+318h] [rbp-98h]
  __int64 v116; // [rsp+320h] [rbp-90h]
  _BYTE v117[32]; // [rsp+328h] [rbp-88h] BYREF
  unsigned __int64 *v118; // [rsp+348h] [rbp-68h]
  __int64 v119; // [rsp+350h] [rbp-60h]
  _QWORD v120[11]; // [rsp+358h] [rbp-58h] BYREF

  v86 = &v84;
  v87 = &v84;
  v91 = 0x2800000000LL;
  v94 = &v92;
  v95 = &v92;
  v99 = &v97;
  v100 = &v97;
  v106 = &v104;
  v107 = &v104;
  v84 = 0;
  v85 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v92 = 0;
  v93 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v108 = 0;
  v109 = 0;
  v111 = &v109;
  v112 = &v109;
  v115 = (unsigned __int64 *)v117;
  v116 = 0x400000000LL;
  v110 = 0;
  v113 = 0;
  v114[0] = 0;
  v114[1] = 0;
  v118 = v120;
  v119 = 0;
  v120[0] = 0;
  v120[1] = 1;
  v120[3] = v114;
  if ( qword_4FAC228 )
  {
    sub_8FD6D0((__int64)&v63, "-lowertypetests-read-summary: ", &qword_4FAC220);
    if ( v64 == 0x3FFFFFFFFFFFFFFFLL || v64 == 4611686018427387902LL )
      goto LABEL_75;
    v32 = sub_2241490(&v63, ": ", 2);
    v75 = v77;
    if ( *(_QWORD *)v32 == v32 + 16 )
    {
      a2 = (__m128)_mm_loadu_si128((const __m128i *)(v32 + 16));
      v77[0] = (__m128i)a2;
    }
    else
    {
      v75 = *(__m128i **)v32;
      v77[0].m128i_i64[0] = *(_QWORD *)(v32 + 16);
    }
    v76 = *(_QWORD *)(v32 + 8);
    *(_QWORD *)v32 = v32 + 16;
    *(_QWORD *)(v32 + 8) = 0;
    *(_BYTE *)(v32 + 16) = 0;
    v69 = &v71;
    if ( v75 == v77 )
    {
      si128 = (__m128)_mm_load_si128(v77);
      v71 = si128;
    }
    else
    {
      v69 = (__m128 *)v75;
      v71.m128_u64[0] = v77[0].m128i_i64[0];
    }
    v33 = v76;
    v75 = v77;
    v76 = 0;
    v70 = v33;
    v77[0].m128i_i8[0] = 0;
    v74 = sub_1872040;
    v72[0] = 1;
    v73 = sub_1872500;
    sub_2240A30(&v75);
    sub_2240A30(&v63);
    v34 = &v63;
    v65.m128i_i16[0] = 260;
    v63 = (__m128i *)&qword_4FAC220;
    sub_16C2DE0((__int64)&v75, (__int64)&v63, 0xFFFFFFFFFFFFFFFFLL, 1, 0);
    v38 = v77[0].m128i_i8[0] & 1;
    if ( (v77[0].m128i_i8[0] & 1) != 0 && (v34 = (__m128i **)(unsigned int)v75, v37 = v76, (_DWORD)v75) )
    {
      sub_16BCB40(&v61, (int)v75, v76);
      v35 = v61 | 1;
      v39 = (v61 & 0xFFFFFFFFFFFFFFFELL) == 0;
      v61 |= 1uLL;
      if ( !v39 )
        sub_1873DF0((__int64)&v69, &v61, v35);
      v40 = 0;
      v38 = v77[0].m128i_i8[0] & 1;
    }
    else
    {
      v40 = (__int64 *)v75;
      v75 = 0;
    }
    if ( !v38 && v75 )
    {
      v59 = v40;
      (*(void (__fastcall **)(__m128i *, __m128i **, __int64, __int64, __int64))(v75->m128i_i64[0] + 8))(
        v75,
        v34,
        v35,
        v36,
        v37);
      v40 = v59;
    }
    v41 = v40[1];
    v58 = v40;
    sub_16E40A0((__int64)&v75, v41, v40[2] - v41, 0, 0, 0);
    sub_16E7420((__int64)&v75, v41);
    sub_16E3830((__int64)&v75);
    sub_1885EF0((char *)&v75, (__int64)v83);
    sub_16E46D0((__int64)&v75);
    v42 = sub_16E4240((__int64)&v75);
    sub_16BCB40(&v61, v42, v43);
    v44 = v61;
    v61 = 0;
    v63 = (__m128i *)(v44 | 1);
    if ( (v44 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_1873DF0((__int64)&v69, (__int64 *)&v63, v44 | 1);
    sub_16E3EB0((__int64)&v75);
    (*(void (__fastcall **)(__int64 *))(*v58 + 8))(v58);
    if ( v73 )
      v73(v72, v72, 3);
    sub_2240A30(&v69);
  }
  if ( dword_4FAC340 == 1 )
  {
    v9 = (__int64 *)v83;
  }
  else
  {
    v9 = 0;
    v10 = (__int64 *)v83;
    if ( dword_4FAC340 == 2 )
      goto LABEL_4;
  }
  v10 = 0;
LABEL_4:
  sub_1875960((__int64 **)&v75, a1, v10, v9);
  v13 = sub_188C730((__int64 *)&v75, a2, si128, a4, a5, v11, v12, a8, a9);
  v14 = v81;
  v15 = v80;
  v57 = v13;
  if ( v81 != v80 )
  {
    do
    {
      v16 = *(_QWORD *)(v15 + 16);
      while ( v16 )
      {
        sub_1876060(*(_QWORD *)(v16 + 24));
        v17 = v16;
        v16 = *(_QWORD *)(v16 + 16);
        j_j___libc_free_0(v17, 40);
      }
      v15 += 80;
    }
    while ( v14 != v15 );
    v15 = v80;
  }
  if ( v15 )
    j_j___libc_free_0(v15, v82 - v15);
  if ( v79 )
  {
    v18 = v78;
    v19 = &v78[5 * v79];
    do
    {
      if ( *v18 != -8 && *v18 != -4 )
      {
        v20 = v18[1];
        if ( v20 )
          j_j___libc_free_0(v20, v18[3] - v20);
      }
      v18 += 5;
    }
    while ( v19 != v18 );
  }
  j___libc_free_0(v78);
  if ( qword_4FAC108 )
  {
    sub_8FD6D0((__int64)&v69, "-lowertypetests-write-summary: ", &qword_4FAC100);
    if ( v70 != 0x3FFFFFFFFFFFFFFFLL && v70 != 4611686018427387902LL )
    {
      v45 = sub_2241490(&v69, ": ", 2);
      v75 = v77;
      if ( *(_QWORD *)v45 == v45 + 16 )
      {
        v77[0] = _mm_loadu_si128((const __m128i *)(v45 + 16));
      }
      else
      {
        v75 = *(__m128i **)v45;
        v77[0].m128i_i64[0] = *(_QWORD *)(v45 + 16);
      }
      v46 = *(_QWORD *)(v45 + 8);
      v76 = v46;
      *(_QWORD *)v45 = v45 + 16;
      *(_QWORD *)(v45 + 8) = 0;
      *(_BYTE *)(v45 + 16) = 0;
      v63 = &v65;
      if ( v75 == v77 )
      {
        v65 = _mm_load_si128(v77);
      }
      else
      {
        v63 = v75;
        v65.m128i_i64[0] = v77[0].m128i_i64[0];
      }
      v47 = v76;
      v75 = v77;
      v76 = 0;
      v64 = v47;
      v77[0].m128i_i8[0] = 0;
      v68 = sub_1872040;
      v66[0] = 1;
      v67 = sub_1872500;
      sub_2240A30(&v75);
      sub_2240A30(&v69);
      LODWORD(v61) = 0;
      v62 = sub_2241E40(&v69, v46, v48, v49, v50);
      sub_16E8AF0((__int64)&v69, (_BYTE *)qword_4FAC100, qword_4FAC108, (__int64)&v61, 1u);
      sub_16BCB40(&v60, v61, v62);
      v51 = v60;
      v60 = 0;
      v75 = (__m128i *)(v51 | 1);
      if ( (v51 & 0xFFFFFFFFFFFFFFFELL) != 0 )
        sub_1873DF0((__int64)&v63, (__int64 *)&v75, v51 | 1);
      sub_16E4AB0((__int64)&v75, (__int64)&v69, 0, 70);
      nullsub_622();
      v52 = 0;
      if ( (unsigned __int8)sub_16E4B20() )
      {
        sub_16E3D10((__int64)&v75, 0, v53, v54, v55, v56);
        v52 = v83;
        sub_1885EF0((char *)&v75, (__int64)v83);
        sub_16E3410((__int64)&v75);
        nullsub_628();
      }
      sub_16E4BA0((__int64)&v75);
      sub_16E3E40(&v75);
      sub_16E7C30((int *)&v69, (__int64)v52);
      if ( v67 )
        v67(v66, v66, 3);
      sub_2240A30(&v63);
      goto LABEL_19;
    }
LABEL_75:
    sub_4262D8((__int64)"basic_string::append");
  }
LABEL_19:
  v21 = v115;
  v22 = &v115[(unsigned int)v116];
  if ( v115 != v22 )
  {
    do
    {
      v23 = *v21++;
      _libc_free(v23);
    }
    while ( v22 != v21 );
  }
  v24 = v118;
  v25 = &v118[2 * (unsigned int)v119];
  if ( v118 != v25 )
  {
    do
    {
      v26 = *v24;
      v24 += 2;
      _libc_free(v26);
    }
    while ( v25 != v24 );
    v25 = v118;
  }
  if ( v25 != v120 )
    _libc_free((unsigned __int64)v25);
  if ( v115 != (unsigned __int64 *)v117 )
    _libc_free((unsigned __int64)v115);
  sub_1875D60(v110);
  sub_1875D60(v105);
  sub_1873C20(v98);
  sub_1874200(v93);
  if ( HIDWORD(v90) )
  {
    v27 = v89;
    if ( (_DWORD)v90 )
    {
      v28 = 8LL * (unsigned int)v90;
      v29 = 0;
      do
      {
        v30 = *(_QWORD *)(v27 + v29);
        if ( v30 && v30 != -8 )
        {
          _libc_free(v30);
          v27 = v89;
        }
        v29 += 8;
      }
      while ( v29 != v28 );
    }
  }
  else
  {
    v27 = v89;
  }
  _libc_free(v27);
  sub_1873440(v85);
  return v57;
}
