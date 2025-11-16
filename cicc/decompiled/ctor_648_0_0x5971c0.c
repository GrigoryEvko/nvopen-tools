// Function: ctor_648_0
// Address: 0x5971c0
//
int __fastcall ctor_648_0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v4; // edx
  __int64 v5; // rbx
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rcx
  int v10; // edx
  __int64 v11; // r13
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  int v14; // edx
  int v15; // r8d
  int v16; // r9d
  int v17; // edx
  int v18; // r8d
  int v19; // r9d
  int v20; // edx
  int v21; // r8d
  int v22; // r9d
  int v23; // edx
  int v24; // r8d
  int v25; // r9d
  int v26; // ecx
  int v27; // r8d
  int v28; // r9d
  __m128i *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rcx
  int v32; // edx
  __int64 v33; // rbx
  __int64 v34; // rax
  __int128 v36; // [rsp-80h] [rbp-530h]
  __int128 v37; // [rsp-80h] [rbp-530h]
  __int128 v38; // [rsp-80h] [rbp-530h]
  __int128 v39; // [rsp-80h] [rbp-530h]
  __int128 v40; // [rsp-80h] [rbp-530h]
  __int128 v41; // [rsp-70h] [rbp-520h]
  __int128 v42; // [rsp-70h] [rbp-520h]
  __int128 v43; // [rsp-70h] [rbp-520h]
  __int128 v44; // [rsp-70h] [rbp-520h]
  __int128 v45; // [rsp-70h] [rbp-520h]
  __int128 v46; // [rsp-58h] [rbp-508h]
  __int128 v47; // [rsp-58h] [rbp-508h]
  __int128 v48; // [rsp-58h] [rbp-508h]
  __int128 v49; // [rsp-58h] [rbp-508h]
  __int128 v50; // [rsp-58h] [rbp-508h]
  __int128 v51; // [rsp-48h] [rbp-4F8h]
  __int128 v52; // [rsp-48h] [rbp-4F8h]
  __int128 v53; // [rsp-48h] [rbp-4F8h]
  __int128 v54; // [rsp-48h] [rbp-4F8h]
  __int128 v55; // [rsp-48h] [rbp-4F8h]
  __int128 v56; // [rsp-30h] [rbp-4E0h]
  __int128 v57; // [rsp-30h] [rbp-4E0h]
  __int128 v58; // [rsp-30h] [rbp-4E0h]
  __int128 v59; // [rsp-30h] [rbp-4E0h]
  __int128 v60; // [rsp-30h] [rbp-4E0h]
  __int128 v61; // [rsp-20h] [rbp-4D0h]
  __int128 v62; // [rsp-20h] [rbp-4D0h]
  __int128 v63; // [rsp-20h] [rbp-4D0h]
  __int128 v64; // [rsp-20h] [rbp-4D0h]
  __int128 v65; // [rsp-20h] [rbp-4D0h]
  int v66; // [rsp+20h] [rbp-490h] BYREF
  int v67; // [rsp+24h] [rbp-48Ch] BYREF
  int *v68; // [rsp+28h] [rbp-488h] BYREF
  _QWORD v69[2]; // [rsp+30h] [rbp-480h] BYREF
  __int64 v70; // [rsp+40h] [rbp-470h]
  const char *v71; // [rsp+48h] [rbp-468h]
  __int64 v72; // [rsp+50h] [rbp-460h]
  _QWORD v73[2]; // [rsp+60h] [rbp-450h] BYREF
  __int64 v74; // [rsp+70h] [rbp-440h]
  char *v75; // [rsp+78h] [rbp-438h]
  __int64 v76; // [rsp+80h] [rbp-430h]
  _QWORD v77[2]; // [rsp+90h] [rbp-420h] BYREF
  __int64 v78; // [rsp+A0h] [rbp-410h]
  const char *v79; // [rsp+A8h] [rbp-408h]
  __int64 v80; // [rsp+B0h] [rbp-400h]
  _QWORD v81[2]; // [rsp+C0h] [rbp-3F0h] BYREF
  __int64 v82; // [rsp+D0h] [rbp-3E0h]
  const char *v83; // [rsp+D8h] [rbp-3D8h]
  __int64 v84; // [rsp+E0h] [rbp-3D0h]
  _QWORD v85[2]; // [rsp+F0h] [rbp-3C0h] BYREF
  __int64 v86; // [rsp+100h] [rbp-3B0h]
  char *v87; // [rsp+108h] [rbp-3A8h]
  __int64 v88; // [rsp+110h] [rbp-3A0h]
  _QWORD v89[2]; // [rsp+120h] [rbp-390h] BYREF
  __int64 v90; // [rsp+130h] [rbp-380h]
  char *v91; // [rsp+138h] [rbp-378h]
  __int64 v92; // [rsp+140h] [rbp-370h]
  _QWORD v93[2]; // [rsp+150h] [rbp-360h] BYREF
  __int64 v94; // [rsp+160h] [rbp-350h]
  const char *v95; // [rsp+168h] [rbp-348h]
  __int64 v96; // [rsp+170h] [rbp-340h]
  _QWORD v97[2]; // [rsp+180h] [rbp-330h] BYREF
  __int64 v98; // [rsp+190h] [rbp-320h]
  char *v99; // [rsp+198h] [rbp-318h]
  __int64 v100; // [rsp+1A0h] [rbp-310h]
  _QWORD v101[2]; // [rsp+1B0h] [rbp-300h] BYREF
  __int64 v102; // [rsp+1C0h] [rbp-2F0h]
  char *v103; // [rsp+1C8h] [rbp-2E8h]
  __int64 v104; // [rsp+1D0h] [rbp-2E0h]
  _QWORD v105[2]; // [rsp+1E0h] [rbp-2D0h] BYREF
  __int64 v106; // [rsp+1F0h] [rbp-2C0h]
  const char *v107; // [rsp+1F8h] [rbp-2B8h]
  __int64 v108; // [rsp+200h] [rbp-2B0h]
  _QWORD v109[2]; // [rsp+210h] [rbp-2A0h] BYREF
  __int64 v110; // [rsp+220h] [rbp-290h]
  char *v111; // [rsp+228h] [rbp-288h]
  __int64 v112; // [rsp+230h] [rbp-280h]
  _QWORD v113[2]; // [rsp+240h] [rbp-270h] BYREF
  __int64 v114; // [rsp+250h] [rbp-260h]
  char *v115; // [rsp+258h] [rbp-258h]
  __int64 v116; // [rsp+260h] [rbp-250h]
  char *v117; // [rsp+270h] [rbp-240h]
  __int64 v118; // [rsp+278h] [rbp-238h]
  __int64 v119; // [rsp+280h] [rbp-230h]
  const char *v120; // [rsp+288h] [rbp-228h]
  __int64 v121; // [rsp+290h] [rbp-220h]
  _QWORD v122[2]; // [rsp+2A0h] [rbp-210h] BYREF
  __int64 v123; // [rsp+2B0h] [rbp-200h]
  const char *v124; // [rsp+2B8h] [rbp-1F8h]
  __int64 v125; // [rsp+2C0h] [rbp-1F0h]
  _QWORD v126[2]; // [rsp+2D0h] [rbp-1E0h] BYREF
  __int64 v127; // [rsp+2E0h] [rbp-1D0h]
  const char *v128; // [rsp+2E8h] [rbp-1C8h]
  __int64 v129; // [rsp+2F0h] [rbp-1C0h]
  const char *v130; // [rsp+300h] [rbp-1B0h] BYREF
  __int64 v131; // [rsp+308h] [rbp-1A8h]
  _BYTE v132[160]; // [rsp+310h] [rbp-1A0h] BYREF
  __m128i v133; // [rsp+3B0h] [rbp-100h] BYREF
  __m128i v134; // [rsp+3C0h] [rbp-F0h] BYREF
  __m128i v135; // [rsp+3D0h] [rbp-E0h] BYREF
  __m128i v136; // [rsp+3E0h] [rbp-D0h] BYREF
  __m128i v137; // [rsp+3F0h] [rbp-C0h] BYREF
  __m128i v138; // [rsp+400h] [rbp-B0h] BYREF
  __m128i v139; // [rsp+410h] [rbp-A0h] BYREF
  __m128i v140; // [rsp+420h] [rbp-90h] BYREF
  __m128i v141; // [rsp+430h] [rbp-80h] BYREF
  __m128i v142; // [rsp+440h] [rbp-70h] BYREF
  __m128i v143; // [rsp+450h] [rbp-60h] BYREF
  __m128i v144; // [rsp+460h] [rbp-50h] BYREF
  __int64 v145; // [rsp+470h] [rbp-40h]

  qword_50379A0 = (__int64)&unk_49DC150;
  v4 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(a1, a2, a3, a4), 1u);
  qword_50379F0 = 0x100000000LL;
  dword_50379AC &= 0x8000u;
  word_50379B0 = 0;
  qword_50379B8 = 0;
  qword_50379C0 = 0;
  dword_50379A8 = v4;
  qword_50379C8 = 0;
  qword_50379D0 = 0;
  qword_50379D8 = 0;
  qword_50379E0 = 0;
  qword_50379E8 = (__int64)&unk_50379F8;
  qword_5037A00 = 0;
  qword_5037A08 = (__int64)&unk_5037A20;
  qword_5037A10 = 1;
  dword_5037A18 = 0;
  byte_5037A1C = 1;
  v5 = sub_C57470();
  v6 = (unsigned int)qword_50379F0;
  v7 = (unsigned int)qword_50379F0 + 1LL;
  if ( v7 > HIDWORD(qword_50379F0) )
  {
    sub_C8D5F0((char *)&unk_50379F8 - 16, &unk_50379F8, v7, 8);
    v6 = (unsigned int)qword_50379F0;
  }
  *(_QWORD *)(qword_50379E8 + 8 * v6) = v5;
  LODWORD(qword_50379F0) = qword_50379F0 + 1;
  qword_5037A28 = 0;
  qword_5037A30 = (__int64)&unk_49D9748;
  qword_5037A38 = 0;
  qword_50379A0 = (__int64)&unk_49DC090;
  qword_5037A40 = (__int64)&unk_49DC1D0;
  qword_5037A60 = (__int64)nullsub_23;
  qword_5037A58 = (__int64)sub_984030;
  sub_C53080(&qword_50379A0, "use-dwarf-ranges-base-address-specifier", 39);
  LOWORD(qword_5037A38) = 256;
  LOBYTE(qword_5037A28) = 0;
  qword_50379D0 = 43;
  LOBYTE(dword_50379AC) = dword_50379AC & 0x9F | 0x20;
  qword_50379C8 = (__int64)"Use base address specifiers in debug_ranges";
  sub_C53130(&qword_50379A0);
  __cxa_atexit(sub_984900, &qword_50379A0, &qword_4A427C0);
  LOBYTE(v122[0]) = 0;
  v133.m128i_i64[0] = (__int64)"Generate dwarf aranges";
  v130 = (const char *)v122;
  v133.m128i_i64[1] = 22;
  LODWORD(v126[0]) = 1;
  sub_3223230(&unk_50378C0, "generate-arange-section", v126, &v133, &v130);
  __cxa_atexit(sub_984900, &unk_50378C0, &qword_4A427C0);
  LOBYTE(v122[0]) = 0;
  v133.m128i_i64[0] = (__int64)"Generate DWARF4 type units.";
  v130 = (const char *)v122;
  v133.m128i_i64[1] = 27;
  LODWORD(v126[0]) = 1;
  sub_3223440(&unk_50377E0, "generate-type-units", v126, &v133, &v130);
  __cxa_atexit(sub_984900, &unk_50377E0, &qword_4A427C0);
  qword_5037700 = (__int64)&unk_49DC150;
  v10 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_984900, &unk_50377E0, v8, v9), 1u);
  qword_5037750 = 0x100000000LL;
  dword_503770C &= 0x8000u;
  word_5037710 = 0;
  qword_5037718 = 0;
  qword_5037720 = 0;
  dword_5037708 = v10;
  qword_5037728 = 0;
  qword_5037730 = 0;
  qword_5037738 = 0;
  qword_5037740 = 0;
  qword_5037748 = (__int64)&unk_5037758;
  qword_5037760 = 0;
  qword_5037768 = (__int64)&unk_5037780;
  qword_5037770 = 1;
  dword_5037778 = 0;
  byte_503777C = 1;
  v11 = sub_C57470();
  v12 = (unsigned int)qword_5037750;
  v13 = (unsigned int)qword_5037750 + 1LL;
  if ( v13 > HIDWORD(qword_5037750) )
  {
    sub_C8D5F0((char *)&unk_5037758 - 16, &unk_5037758, v13, 8);
    v12 = (unsigned int)qword_5037750;
  }
  *(_QWORD *)(qword_5037748 + 8 * v12) = v11;
  LODWORD(qword_5037750) = qword_5037750 + 1;
  qword_5037788 = 0;
  qword_5037790 = (__int64)&unk_49D9748;
  qword_5037798 = 0;
  qword_5037700 = (__int64)&unk_49DC090;
  qword_50377A0 = (__int64)&unk_49DC1D0;
  qword_50377C0 = (__int64)nullsub_23;
  qword_50377B8 = (__int64)sub_984030;
  sub_C53080(&qword_5037700, "split-dwarf-cross-cu-references", 31);
  LOWORD(qword_5037798) = 256;
  LOBYTE(qword_5037788) = 0;
  qword_5037730 = 39;
  LOBYTE(dword_503770C) = dword_503770C & 0x9F | 0x20;
  qword_5037728 = (__int64)"Enable cross-cu references in DWO files";
  sub_C53130(&qword_5037700);
  __cxa_atexit(sub_984900, &qword_5037700, &qword_4A427C0);
  v113[0] = v109;
  v128 = "Never";
  v124 = "In all cases";
  v122[0] = "Enable";
  v120 = "At top of block or after label";
  v126[1] = 7;
  LODWORD(v127) = 2;
  v129 = 5;
  v125 = 12;
  LODWORD(v109[0]) = 0;
  *((_QWORD *)&v61 + 1) = "Never";
  v126[0] = "Disable";
  *(_QWORD *)&v61 = v127;
  v122[1] = 6;
  *((_QWORD *)&v56 + 1) = 7;
  LODWORD(v123) = 1;
  *(_QWORD *)&v56 = "Disable";
  v117 = "Default";
  v118 = 7;
  *((_QWORD *)&v51 + 1) = "In all cases";
  LODWORD(v119) = 0;
  v121 = 30;
  *(_QWORD *)&v51 = v123;
  *((_QWORD *)&v46 + 1) = 6;
  *(_QWORD *)&v46 = "Enable";
  *((_QWORD *)&v41 + 1) = "At top of block or after label";
  *(_QWORD *)&v41 = v119;
  *((_QWORD *)&v36 + 1) = 7;
  *(_QWORD *)&v36 = "Default";
  sub_22735E0(
    (unsigned int)&v133,
    (unsigned int)&qword_5037700,
    v14,
    (unsigned int)"At top of block or after label",
    v15,
    v16,
    v36,
    v41,
    30,
    v46,
    v51,
    12,
    v56,
    v61,
    5);
  v131 = 55;
  v130 = "Make an absence of debug location information explicit.";
  LODWORD(v105[0]) = 1;
  sub_323E5C0(&unk_50374A0, "use-unknown-locations", v105, &v130, &v133);
  if ( (__m128i *)v133.m128i_i64[0] != &v134 )
    _libc_free(v133.m128i_i64[0], "use-unknown-locations");
  __cxa_atexit(sub_3219C00, &unk_50374A0, &qword_4A427C0);
  v135.m128i_i64[1] = (__int64)"Default for platform";
  v138.m128i_i64[0] = (__int64)"Disabled.";
  v113[0] = v109;
  v141.m128i_i64[1] = (__int64)"Dwarf";
  v143.m128i_i64[0] = (__int64)"DWARF";
  v133.m128i_i64[0] = (__int64)&v134;
  v133.m128i_i64[1] = 0x400000004LL;
  v130 = "Output dwarf accelerator tables.";
  LODWORD(v109[0]) = 0;
  v134.m128i_i64[0] = (__int64)"Default";
  v134.m128i_i64[1] = 7;
  v135.m128i_i32[0] = 0;
  v136.m128i_i64[0] = 20;
  v136.m128i_i64[1] = (__int64)"Disable";
  v137.m128i_i64[0] = 7;
  v137.m128i_i32[2] = 1;
  v138.m128i_i64[1] = 9;
  v139.m128i_i64[0] = (__int64)"Apple";
  v139.m128i_i64[1] = 5;
  v140.m128i_i32[0] = 2;
  v140.m128i_i64[1] = (__int64)"Apple";
  v141.m128i_i64[0] = 5;
  v142.m128i_i64[0] = 5;
  v142.m128i_i32[2] = 3;
  v143.m128i_i64[1] = 5;
  v131 = 32;
  LODWORD(v105[0]) = 1;
  sub_323F220(&unk_5037240, "accel-tables", v105, &v130, &v133, v113);
  if ( (__m128i *)v133.m128i_i64[0] != &v134 )
    _libc_free(v133.m128i_i64[0], "accel-tables");
  __cxa_atexit(sub_3219AE0, &unk_5037240, &qword_4A427C0);
  v101[0] = v97;
  v115 = "Disabled";
  v111 = "Enabled";
  v109[0] = "Enable";
  v107 = "Default for platform";
  v113[1] = 7;
  LODWORD(v114) = 2;
  v116 = 8;
  v112 = 7;
  LODWORD(v97[0]) = 0;
  *((_QWORD *)&v62 + 1) = "Disabled";
  v113[0] = "Disable";
  *(_QWORD *)&v62 = v114;
  v109[1] = 6;
  *((_QWORD *)&v57 + 1) = 7;
  LODWORD(v110) = 1;
  *(_QWORD *)&v57 = "Disable";
  v105[0] = "Default";
  v105[1] = 7;
  *((_QWORD *)&v52 + 1) = "Enabled";
  LODWORD(v106) = 0;
  v108 = 20;
  *(_QWORD *)&v52 = v110;
  *((_QWORD *)&v47 + 1) = 6;
  *(_QWORD *)&v47 = "Enable";
  *((_QWORD *)&v42 + 1) = "Default for platform";
  *(_QWORD *)&v42 = v106;
  *((_QWORD *)&v37 + 1) = 7;
  *(_QWORD *)&v37 = "Default";
  sub_22735E0(
    (unsigned int)&v133,
    (unsigned int)"Enabled",
    v17,
    (unsigned int)"Default for platform",
    v18,
    v19,
    v37,
    v42,
    20,
    v47,
    v52,
    7,
    v57,
    v62,
    8);
  v131 = 47;
  v130 = "Use inlined strings rather than string section.";
  LODWORD(v93[0]) = 1;
  sub_323E5C0(&unk_5036FE0, "dwarf-inlined-strings", v93, &v130, &v133);
  if ( (__m128i *)v133.m128i_i64[0] != &v134 )
    _libc_free(v133.m128i_i64[0], "dwarf-inlined-strings");
  __cxa_atexit(sub_3219C00, &unk_5036FE0, &qword_4A427C0);
  LOBYTE(v97[0]) = 0;
  v133.m128i_i64[1] = 39;
  v130 = (const char *)v97;
  v133.m128i_i64[0] = (__int64)"Disable emission .debug_ranges section.";
  LODWORD(v101[0]) = 1;
  sub_3223230(&unk_5036F00, "no-dwarf-ranges-section", v101, &v133, &v130);
  __cxa_atexit(sub_984900, &unk_5036F00, &qword_4A427C0);
  v89[0] = v85;
  v103 = "Disabled";
  v99 = "Enabled";
  v97[0] = "Enable";
  v95 = "Default for platform";
  v101[1] = 7;
  LODWORD(v102) = 2;
  v104 = 8;
  v100 = 7;
  LODWORD(v85[0]) = 0;
  *((_QWORD *)&v63 + 1) = "Disabled";
  v101[0] = "Disable";
  *(_QWORD *)&v63 = v102;
  v97[1] = 6;
  *((_QWORD *)&v58 + 1) = 7;
  LODWORD(v98) = 1;
  *(_QWORD *)&v58 = "Disable";
  v93[0] = "Default";
  v93[1] = 7;
  *((_QWORD *)&v53 + 1) = "Enabled";
  LODWORD(v94) = 0;
  v96 = 20;
  *(_QWORD *)&v53 = v98;
  *((_QWORD *)&v48 + 1) = 6;
  *(_QWORD *)&v48 = "Enable";
  *((_QWORD *)&v43 + 1) = "Default for platform";
  *(_QWORD *)&v43 = v94;
  *((_QWORD *)&v38 + 1) = 7;
  *(_QWORD *)&v38 = "Default";
  sub_22735E0(
    (unsigned int)&v133,
    (unsigned int)"Enabled",
    v20,
    (unsigned int)"Default for platform",
    v21,
    v22,
    v38,
    v43,
    20,
    v48,
    v53,
    7,
    v58,
    v63,
    8);
  v131 = 53;
  v130 = "Use sections+offset as references rather than labels.";
  LODWORD(v81[0]) = 1;
  sub_323ED80(&unk_5036CA0, "dwarf-sections-as-references", v81, &v130, &v133, v89);
  if ( (__m128i *)v133.m128i_i64[0] != &v134 )
    _libc_free(v133.m128i_i64[0], "dwarf-sections-as-references");
  __cxa_atexit(sub_3219C00, &unk_5036CA0, &qword_4A427C0);
  LOBYTE(v85[0]) = 0;
  v133.m128i_i64[1] = 46;
  v130 = (const char *)v85;
  v133.m128i_i64[0] = (__int64)"Emit the GNU .debug_macro format with DWARF <5";
  LODWORD(v89[0]) = 1;
  sub_3223440(&unk_5036BC0, "use-gnu-debug-macro", v89, &v133, &v130);
  __cxa_atexit(sub_984900, &unk_5036BC0, &qword_4A427C0);
  v77[0] = v73;
  v91 = "Disabled";
  v87 = "Enabled";
  v85[0] = "Enable";
  v83 = "Default for platform";
  v89[1] = 7;
  LODWORD(v90) = 2;
  v92 = 8;
  LODWORD(v86) = 1;
  v88 = 7;
  *((_QWORD *)&v64 + 1) = "Disabled";
  LODWORD(v73[0]) = 0;
  *(_QWORD *)&v64 = v90;
  v89[0] = "Disable";
  *((_QWORD *)&v59 + 1) = 7;
  v85[1] = 6;
  *(_QWORD *)&v59 = "Disable";
  v81[0] = "Default";
  v81[1] = 7;
  *((_QWORD *)&v54 + 1) = "Enabled";
  LODWORD(v82) = 0;
  *(_QWORD *)&v54 = v86;
  v84 = 20;
  *((_QWORD *)&v49 + 1) = 6;
  *(_QWORD *)&v49 = "Enable";
  *((_QWORD *)&v44 + 1) = "Default for platform";
  *(_QWORD *)&v44 = v82;
  *((_QWORD *)&v39 + 1) = 7;
  *(_QWORD *)&v39 = "Default";
  sub_22735E0(
    (unsigned int)&v133,
    (unsigned int)"Enabled",
    v23,
    (unsigned int)"Default for platform",
    v24,
    v25,
    v39,
    v44,
    20,
    v49,
    v54,
    7,
    v59,
    v64,
    8);
  v130 = "Enable use of the DWARFv5 DW_OP_convert operator";
  v131 = 48;
  LODWORD(v69[0]) = 1;
  sub_323E9A0(&unk_5036960, "dwarf-op-convert", v69, &v130, &v133, v77);
  if ( (__m128i *)v133.m128i_i64[0] != &v134 )
    _libc_free(v133.m128i_i64[0], "dwarf-op-convert");
  __cxa_atexit(sub_3219C00, &unk_5036960, &qword_4A427C0);
  v68 = &v67;
  v79 = "Abstract subprograms";
  v75 = "All";
  v77[0] = "Abstract";
  v73[0] = "All";
  v71 = "Default for platform";
  v77[1] = 8;
  LODWORD(v78) = 2;
  v80 = 20;
  v76 = 3;
  v67 = 0;
  *((_QWORD *)&v65 + 1) = "Abstract subprograms";
  v73[1] = 3;
  *(_QWORD *)&v65 = v78;
  LODWORD(v74) = 1;
  *((_QWORD *)&v60 + 1) = 8;
  v69[0] = "Default";
  *(_QWORD *)&v60 = "Abstract";
  v69[1] = 7;
  LODWORD(v70) = 0;
  *((_QWORD *)&v55 + 1) = "All";
  v72 = 20;
  *(_QWORD *)&v55 = v74;
  *((_QWORD *)&v50 + 1) = 3;
  *(_QWORD *)&v50 = "All";
  *((_QWORD *)&v45 + 1) = "Default for platform";
  *(_QWORD *)&v45 = v70;
  *((_QWORD *)&v40 + 1) = 7;
  *(_QWORD *)&v40 = "Default";
  sub_22735E0(
    (unsigned int)&v133,
    (unsigned int)"Default for platform",
    (unsigned int)"Abstract",
    v26,
    v27,
    v28,
    v40,
    v45,
    20,
    v50,
    v55,
    3,
    v60,
    v65,
    20);
  v130 = "Which DWARF linkage-name attributes to emit.";
  v131 = 44;
  v66 = 1;
  sub_323F6C0(&unk_5036700, "dwarf-linkage-names", &v66, &v130, &v133, &v68);
  if ( (__m128i *)v133.m128i_i64[0] != &v134 )
    _libc_free(v133.m128i_i64[0], "dwarf-linkage-names");
  __cxa_atexit(sub_3219B70, &unk_5036700, &qword_4A427C0);
  v133.m128i_i64[0] = (__int64)"Default";
  v68 = &v67;
  v134.m128i_i64[1] = (__int64)"Default address minimization strategy";
  v135.m128i_i64[1] = (__int64)"Ranges";
  v137.m128i_i64[0] = (__int64)"Use rnglists for contiguous ranges if that allows using a pre-existing base address";
  v138.m128i_i64[0] = (__int64)"Expressions";
  v139.m128i_i64[1] = (__int64)"Use exprloc addrx+offset expressions for any address with a prior base address";
  v140.m128i_i64[1] = (__int64)"Form";
  v142.m128i_i64[0] = (__int64)"Use addrx+offset extension form for any address with a prior base address";
  v143.m128i_i64[0] = (__int64)"Disabled";
  v144.m128i_i64[1] = (__int64)"Stuff";
  v131 = 0x400000000LL;
  v67 = 0;
  v133.m128i_i64[1] = 7;
  v134.m128i_i32[0] = 0;
  v135.m128i_i64[0] = 37;
  v136.m128i_i64[0] = 6;
  v136.m128i_i32[2] = 2;
  v137.m128i_i64[1] = 83;
  v138.m128i_i64[1] = 11;
  v139.m128i_i32[0] = 3;
  v140.m128i_i64[0] = 78;
  v141.m128i_i64[0] = 4;
  v141.m128i_i32[2] = 4;
  v142.m128i_i64[1] = 73;
  v143.m128i_i64[1] = 8;
  v144.m128i_i32[0] = 1;
  v145 = 5;
  v130 = v132;
  sub_C8D5F0(&v130, v132, 5, 40);
  v29 = (__m128i *)&v130[40 * (unsigned int)v131];
  *v29 = _mm_loadu_si128(&v133);
  v29[1] = _mm_loadu_si128(&v134);
  v29[2] = _mm_loadu_si128(&v135);
  v29[3] = _mm_loadu_si128(&v136);
  v29[4] = _mm_loadu_si128(&v137);
  v29[5] = _mm_loadu_si128(&v138);
  v29[6] = _mm_loadu_si128(&v139);
  v29[7] = _mm_loadu_si128(&v140);
  v29[8] = _mm_loadu_si128(&v141);
  v29[9] = _mm_loadu_si128(&v142);
  v29[10] = _mm_loadu_si128(&v143);
  v29[11] = _mm_loadu_si128(&v144);
  v29[12].m128i_i64[0] = v145;
  v133.m128i_i64[0] = (__int64)"Always use DW_AT_ranges in DWARFv5 whenever it could allow more address pool entry sharin"
                               "g to reduce relocations/object size";
  LODWORD(v131) = v131 + 5;
  v133.m128i_i64[1] = 124;
  v66 = 1;
  sub_323FB60(&unk_50364A0, "minimize-addr-in-v5", &v66, &v133, &v130, &v68);
  if ( v130 != v132 )
    _libc_free(v130, "minimize-addr-in-v5");
  __cxa_atexit(sub_3219C90, &unk_50364A0, &qword_4A427C0);
  qword_50363C0 = &unk_49DC150;
  v32 = _InterlockedExchangeAdd64((volatile signed __int64 *)sub_C523C0(sub_3219C90, &unk_50364A0, v30, v31), 1u);
  *(_DWORD *)&word_50363CC = word_50363CC & 0x8000;
  unk_50363D0 = 0;
  qword_5036408[1] = 0x100000000LL;
  unk_50363C8 = v32;
  unk_50363D8 = 0;
  unk_50363E0 = 0;
  unk_50363E8 = 0;
  unk_50363F0 = 0;
  unk_50363F8 = 0;
  unk_5036400 = 0;
  qword_5036408[0] = &qword_5036408[2];
  qword_5036408[3] = 0;
  qword_5036408[4] = &qword_5036408[7];
  qword_5036408[5] = 1;
  LODWORD(qword_5036408[6]) = 0;
  BYTE4(qword_5036408[6]) = 1;
  v33 = sub_C57470();
  v34 = LODWORD(qword_5036408[1]);
  if ( (unsigned __int64)LODWORD(qword_5036408[1]) + 1 > HIDWORD(qword_5036408[1]) )
  {
    sub_C8D5F0(qword_5036408, &qword_5036408[2], LODWORD(qword_5036408[1]) + 1LL, 8);
    v34 = LODWORD(qword_5036408[1]);
  }
  *(_QWORD *)(qword_5036408[0] + 8 * v34) = v33;
  ++LODWORD(qword_5036408[1]);
  qword_5036408[8] = 0;
  qword_5036408[9] = &unk_49D9748;
  qword_5036408[10] = 0;
  qword_50363C0 = &unk_49DC090;
  qword_5036408[11] = &unk_49DC1D0;
  qword_5036408[15] = nullsub_23;
  qword_5036408[14] = sub_984030;
  sub_C53080(&qword_50363C0, "nvptx-emit-src", 14);
  unk_50363F0 = 44;
  LOBYTE(word_50363CC) = word_50363CC & 0xF8 | 1;
  unk_50363E8 = "NVPTX Specific: Emit source line in ptx file";
  sub_C53130(&qword_50363C0);
  return __cxa_atexit(sub_984900, &qword_50363C0, &qword_4A427C0);
}
