// Function: sub_F2DD30
// Address: 0xf2dd30
//
__int64 __fastcall sub_F2DD30(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9,
        __int64 a10,
        __int64 a11,
        char *a12)
{
  __m128i v12; // xmm1
  __m128i v13; // xmm0
  __int64 v14; // r15
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  __int64 v17; // r14
  int v18; // eax
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 (__fastcall ***v28)(); // rsi
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  char v32; // bl
  char v33; // al
  __int64 v34; // r8
  __int64 v35; // r9
  char *v36; // rax
  char *v37; // rax
  char *v38; // rax
  unsigned int v39; // ebx
  const char *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rcx
  __int64 v49; // r8
  __int64 v50; // r9
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v59; // [rsp+A8h] [rbp-EF8h]
  char v60; // [rsp+BAh] [rbp-EE6h]
  unsigned __int8 v61; // [rsp+BBh] [rbp-EE5h]
  unsigned int v62; // [rsp+BCh] [rbp-EE4h]
  __m128i v63; // [rsp+C0h] [rbp-EE0h] BYREF
  const char *v64; // [rsp+D0h] [rbp-ED0h]
  __int64 v65; // [rsp+D8h] [rbp-EC8h]
  __int16 v66; // [rsp+E0h] [rbp-EC0h]
  __m128i v67[2]; // [rsp+F0h] [rbp-EB0h] BYREF
  char v68; // [rsp+110h] [rbp-E90h]
  char v69; // [rsp+111h] [rbp-E8Fh]
  __m128i v70[3]; // [rsp+120h] [rbp-E80h] BYREF
  __m128i v71[2]; // [rsp+150h] [rbp-E50h] BYREF
  __int16 v72; // [rsp+170h] [rbp-E30h]
  _QWORD v73[2]; // [rsp+180h] [rbp-E20h] BYREF
  _BYTE v74[64]; // [rsp+190h] [rbp-E10h] BYREF
  _QWORD v75[2]; // [rsp+1D0h] [rbp-DD0h] BYREF
  _BYTE v76[32]; // [rsp+1E0h] [rbp-DC0h] BYREF
  __int64 v77; // [rsp+200h] [rbp-DA0h]
  __int64 v78; // [rsp+208h] [rbp-D98h]
  __int16 v79; // [rsp+210h] [rbp-D90h]
  __int64 v80; // [rsp+218h] [rbp-D88h]
  _QWORD *v81; // [rsp+220h] [rbp-D80h]
  void **v82; // [rsp+228h] [rbp-D78h]
  __int64 v83; // [rsp+230h] [rbp-D70h]
  int v84; // [rsp+238h] [rbp-D68h]
  __int16 v85; // [rsp+23Ch] [rbp-D64h]
  char v86; // [rsp+23Eh] [rbp-D62h]
  __int64 v87; // [rsp+240h] [rbp-D60h]
  __int64 v88; // [rsp+248h] [rbp-D58h]
  _QWORD v89[2]; // [rsp+250h] [rbp-D50h] BYREF
  void *v90; // [rsp+260h] [rbp-D40h] BYREF
  char v91; // [rsp+268h] [rbp-D38h] BYREF
  __int64 (__fastcall *v92)(const __m128i **, const __m128i *, int); // [rsp+278h] [rbp-D28h]
  void (__fastcall *v93)(_QWORD *, __int64 *); // [rsp+280h] [rbp-D20h]
  __m128i v94[27]; // [rsp+290h] [rbp-D10h] BYREF
  __m128i v95; // [rsp+440h] [rbp-B60h] BYREF
  int v96; // [rsp+450h] [rbp-B50h]
  int v97; // [rsp+454h] [rbp-B4Ch]
  int v98; // [rsp+458h] [rbp-B48h]
  char v99; // [rsp+45Ch] [rbp-B44h]
  __int64 v100; // [rsp+460h] [rbp-B40h] BYREF
  __int64 *v101; // [rsp+4A0h] [rbp-B00h]
  __int64 v102; // [rsp+4A8h] [rbp-AF8h]
  __int64 v103; // [rsp+4B0h] [rbp-AF0h] BYREF
  int v104; // [rsp+4B8h] [rbp-AE8h]
  __int64 v105; // [rsp+4C0h] [rbp-AE0h]
  int v106; // [rsp+4C8h] [rbp-AD8h]
  __int64 v107; // [rsp+4D0h] [rbp-AD0h]
  __m128i v108; // [rsp+5F0h] [rbp-9B0h] BYREF
  char v109; // [rsp+60Ch] [rbp-994h]
  char *v110; // [rsp+650h] [rbp-950h]
  char v111; // [rsp+660h] [rbp-940h] BYREF
  __m128i v112; // [rsp+7A0h] [rbp-800h] BYREF
  char v113; // [rsp+7BCh] [rbp-7E4h]
  char v114; // [rsp+7C0h] [rbp-7E0h]
  char v115; // [rsp+7C1h] [rbp-7DFh]
  char *v116; // [rsp+800h] [rbp-7A0h]
  char v117; // [rsp+810h] [rbp-790h] BYREF
  __m128i v118; // [rsp+950h] [rbp-650h] BYREF
  void (__fastcall *v119)(__m128i *, __m128i *, __int64); // [rsp+960h] [rbp-640h]
  void (__fastcall *v120)(_QWORD *, __int64 *); // [rsp+968h] [rbp-638h]
  char *v121; // [rsp+9B0h] [rbp-5F0h]
  char v122; // [rsp+9C0h] [rbp-5E0h] BYREF
  __int64 (__fastcall **v123)(); // [rsp+B00h] [rbp-4A0h] BYREF
  __m128i v124; // [rsp+B08h] [rbp-498h] BYREF
  __int64 (__fastcall *v125)(const __m128i **, const __m128i *, int); // [rsp+B18h] [rbp-488h]
  void (__fastcall *v126)(_QWORD *, __int64 *); // [rsp+B20h] [rbp-480h]
  __int64 v127; // [rsp+B28h] [rbp-478h]
  char v128; // [rsp+B30h] [rbp-470h]
  __int64 v129; // [rsp+B38h] [rbp-468h]
  __int64 v130; // [rsp+B40h] [rbp-460h]
  __int64 v131; // [rsp+B48h] [rbp-458h]
  __int64 v132; // [rsp+B50h] [rbp-450h]
  __int64 v133; // [rsp+B58h] [rbp-448h]
  _QWORD *v134; // [rsp+B60h] [rbp-440h]
  __int64 v135; // [rsp+B68h] [rbp-438h]
  _QWORD v136[6]; // [rsp+B70h] [rbp-430h] BYREF
  __int16 v137; // [rsp+BA0h] [rbp-400h]
  __int64 v138; // [rsp+BA8h] [rbp-3F8h]
  __int64 v139; // [rsp+BB0h] [rbp-3F0h]
  __int64 v140; // [rsp+BB8h] [rbp-3E8h]
  __int64 v141; // [rsp+BC0h] [rbp-3E0h]
  _QWORD v142[3]; // [rsp+BC8h] [rbp-3D8h] BYREF
  int v143; // [rsp+BE0h] [rbp-3C0h]
  _QWORD *v144; // [rsp+BE8h] [rbp-3B8h]
  char v145; // [rsp+BF0h] [rbp-3B0h]
  __int64 v146; // [rsp+BF8h] [rbp-3A8h]
  __int64 v147; // [rsp+C00h] [rbp-3A0h]
  char v148; // [rsp+C08h] [rbp-398h] BYREF
  _QWORD v149[2]; // [rsp+C88h] [rbp-318h] BYREF
  char v150; // [rsp+C98h] [rbp-308h] BYREF
  _QWORD v151[2]; // [rsp+ED8h] [rbp-C8h] BYREF
  char v152; // [rsp+EE8h] [rbp-B8h] BYREF
  char v153; // [rsp+F68h] [rbp-38h] BYREF

  v59 = sub_B2BEC0(a1);
  v60 = *a12;
  if ( *a12 )
    v60 = sub_B2D620(a1, "instcombine-no-verify-fixpoint", 0x1Eu) ^ 1;
  v12 = _mm_loadu_si128(&v124);
  v119 = 0;
  v118.m128i_i64[0] = a2;
  v118.m128i_i64[1] = a4;
  v13 = _mm_loadu_si128(&v118);
  v118 = v12;
  v125 = sub_F066C0;
  v123 = (__int64 (__fastcall **)())&unk_49DA0D8;
  v120 = v126;
  v126 = sub_F205E0;
  v124 = v13;
  v112.m128i_i64[0] = (__int64)&unk_49D94D0;
  v112.m128i_i64[1] = v59;
  v80 = sub_B2BE50(a1);
  v81 = v89;
  v82 = &v90;
  v89[1] = v59;
  v75[0] = v76;
  v75[1] = 0x200000000LL;
  v83 = 0;
  v84 = 0;
  v85 = 512;
  v86 = 7;
  v87 = 0;
  v88 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v89[0] = &unk_49D94D0;
  v90 = &unk_49DA0D8;
  v92 = 0;
  if ( v125 )
  {
    v125((const __m128i **)&v91, &v124, 2);
    v93 = v126;
    v92 = v125;
  }
  v112.m128i_i64[0] = (__int64)&unk_49D94D0;
  nullsub_63();
  sub_B32BF0(&v123);
  if ( v119 )
    v119(&v118, &v118, 3);
  v14 = *(_QWORD *)(a1 + 80);
  v102 = 0x800000000LL;
  v96 = 8;
  v98 = 0;
  if ( v14 )
    v14 -= 24;
  v99 = 1;
  v73[0] = v74;
  memset(v94, 0, sizeof(v94));
  v94[1].m128i_i32[0] = 8;
  v94[0].m128i_i64[1] = (__int64)v94[2].m128i_i64;
  v94[1].m128i_i8[12] = 1;
  v94[6].m128i_i64[0] = (__int64)v94[7].m128i_i64;
  v94[6].m128i_i32[3] = 8;
  v95.m128i_i64[1] = (__int64)&v100;
  v101 = &v103;
  v97 = 1;
  v100 = v14;
  v95.m128i_i64[0] = 1;
  v15 = *(_QWORD *)(v14 + 48);
  v73[1] = 0x800000000LL;
  v16 = v15 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v16 == v14 + 48 )
    goto LABEL_57;
  if ( !v16 )
    BUG();
  v17 = v16 - 24;
  if ( (unsigned int)*(unsigned __int8 *)(v16 - 24) - 30 > 0xA )
  {
LABEL_57:
    v18 = 0;
    v19 = 0;
    v17 = 0;
  }
  else
  {
    v18 = sub_B46E30(v17);
    v19 = v17;
  }
  v105 = v17;
  v103 = v19;
  v104 = v18;
  v107 = v14;
  v106 = 0;
  LODWORD(v102) = 1;
  sub_D4D230((__int64)&v95);
  sub_F1FCA0((__int64)&v118, (__int64)v94, v20, v21, v22, v23);
  sub_F1FB80((__int64)&v123, (__int64)&v118);
  sub_F1FCA0((__int64)&v108, (__int64)&v95, v24, v25, v26, v27);
  sub_F1FB80((__int64)&v112, (__int64)&v108);
  v28 = &v123;
  sub_F1FD70((__int64)&v112, (__int64)&v123, (__int64)v73, v29, v30, v31);
  if ( v116 != &v117 )
    _libc_free(v116, &v123);
  if ( !v113 )
    _libc_free(v112.m128i_i64[1], &v123);
  if ( v110 != &v111 )
    _libc_free(v110, &v123);
  if ( !v109 )
    _libc_free(v108.m128i_i64[1], &v123);
  if ( v134 != v136 )
    _libc_free(v134, &v123);
  if ( !BYTE4(v125) )
    _libc_free(v124.m128i_i64[0], &v123);
  if ( v121 != &v122 )
    _libc_free(v121, &v123);
  if ( !BYTE4(v120) )
    _libc_free(v118.m128i_i64[1], &v123);
  if ( v101 != &v103 )
    _libc_free(v101, &v123);
  if ( !v99 )
    _libc_free(v95.m128i_i64[1], &v123);
  if ( (__m128i *)v94[6].m128i_i64[0] != &v94[7] )
    _libc_free(v94[6].m128i_i64[0], &v123);
  if ( !v94[1].m128i_i8[12] )
    _libc_free(v94[0].m128i_i64[1], &v123);
  v61 = 0;
  if ( (_DWORD)qword_4F8AF68 )
    v61 = sub_F521F0(a1);
  if ( *((_DWORD *)a12 + 1) || v60 )
  {
    v62 = 1;
    while ( 1 )
    {
      v32 = a12[8];
      v33 = sub_B2D610(a1, 18);
      v136[3] = 0;
      v128 = v33;
      LOBYTE(v125) = v32;
      v129 = a3;
      v126 = (void (__fastcall *)(_QWORD *, __int64 *))v75;
      v130 = a4;
      v136[2] = a4;
      v133 = v59;
      v136[4] = v142;
      v134 = (_QWORD *)v59;
      v138 = a8;
      v124.m128i_i64[0] = a6;
      v127 = a2;
      v131 = a5;
      v132 = a7;
      v135 = a5;
      v136[0] = a6;
      v136[1] = a7;
      v136[5] = 0;
      v137 = 257;
      v139 = a9;
      memset(v142, 0, sizeof(v142));
      v140 = a10;
      v141 = a11;
      v143 = 0;
      v145 = 0;
      v146 = 0;
      v147 = 1;
      v144 = v73;
      v36 = &v148;
      do
      {
        *(_QWORD *)v36 = -4096;
        v36 += 16;
        *((_QWORD *)v36 - 1) = -4096;
      }
      while ( v36 != (char *)v149 );
      v149[0] = 0;
      v37 = &v150;
      v149[1] = 1;
      do
      {
        *(_QWORD *)v37 = -4096;
        v37 += 72;
      }
      while ( v37 != (char *)v151 );
      v151[0] = 0;
      v38 = &v152;
      v151[1] = 1;
      do
      {
        *(_QWORD *)v38 = -4096;
        v38 += 16;
        *((_QWORD *)v38 - 1) = -4096;
      }
      while ( v38 != &v153 );
      v28 = (__int64 (__fastcall ***)())a1;
      v153 = 0;
      v123 = off_497C110;
      v124.m128i_i64[1] = (unsigned int)qword_4F8B048;
      v39 = sub_F1A180((__int64)&v123, a1, a5, a6, v34, v35);
      LOBYTE(v39) = sub_F2D1B0((__int64)&v123) | v39;
      if ( !(_BYTE)v39 )
        break;
      if ( *((_DWORD *)a12 + 1) < v62 )
      {
        v71[0].m128i_i32[0] = *((_DWORD *)a12 + 1);
        v112.m128i_i64[0] = (__int64)"Use 'instcombine<no-verify-fixpoint>' or function attribute 'instcombine-no-verify-"
                                     "fixpoint' to suppress this error.";
        v95.m128i_i64[0] = (__int64)" iterations. ";
        v115 = 1;
        v114 = 3;
        LOWORD(v100) = 259;
        v72 = 265;
        v69 = 1;
        v67[0].m128i_i64[0] = (__int64)" did not reach a fixpoint after ";
        v68 = 3;
        v41 = sub_BD5D20(a1);
        v65 = v42;
        v64 = v41;
        v66 = 1283;
        v63.m128i_i64[0] = (__int64)"Instruction Combining on ";
        sub_9C6370(v70, &v63, v67, 1283, v43, v44);
        sub_9C6370(v94, v70, v71, v45, v46, v47);
        sub_9C6370(&v108, v94, &v95, v48, v49, v50);
        sub_9C6370(&v118, &v108, &v112, v51, v52, v53);
        sub_C64D30((__int64)&v118, 0);
      }
      v123 = off_497C110;
      sub_F0BE20((__int64)&v123, a1);
      v61 = v60 | (*((_DWORD *)a12 + 1) >= ++v62);
      if ( !v61 )
        goto LABEL_51;
    }
    v123 = off_497C110;
    sub_F0BE20((__int64)&v123, a1);
    v39 = v61;
  }
  else
  {
    v39 = v61;
  }
LABEL_51:
  if ( (_BYTE *)v73[0] != v74 )
    _libc_free(v73[0], v28);
  sub_B32BF0(&v90);
  v89[0] = &unk_49D94D0;
  nullsub_63();
  if ( (_BYTE *)v75[0] != v76 )
    _libc_free(v75[0], v28);
  return v39;
}
