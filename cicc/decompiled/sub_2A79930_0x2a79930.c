// Function: sub_2A79930
// Address: 0x2a79930
//
__int64 __fastcall sub_2A79930(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // rax
  __m128i v10; // xmm1
  __m128i v11; // xmm0
  __int64 v12; // rdi
  __m128i v13; // xmm2
  __m128i v14; // xmm3
  __m128i v15; // xmm4
  __m128i v16; // xmm5
  unsigned int v17; // r12d
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // r15
  __int64 v21; // r14
  int v22; // eax
  __m128i v26; // [rsp+30h] [rbp-450h] BYREF
  void (__fastcall *v27)(__m128i *, __m128i *, __int64); // [rsp+40h] [rbp-440h]
  char (__fastcall *v28)(__int64 *, __int64 *); // [rsp+48h] [rbp-438h]
  void *v29; // [rsp+50h] [rbp-430h] BYREF
  __m128i v30; // [rsp+58h] [rbp-428h] BYREF
  __int64 (__fastcall *v31)(char *, __m128i *, int); // [rsp+68h] [rbp-418h]
  char (__fastcall *v32)(__int64 *, __int64 *); // [rsp+70h] [rbp-410h]
  void *v33; // [rsp+80h] [rbp-400h]
  void *v34; // [rsp+88h] [rbp-3F8h]
  __int64 v35; // [rsp+90h] [rbp-3F0h]
  __m128i v36; // [rsp+98h] [rbp-3E8h] BYREF
  __m128i v37; // [rsp+A8h] [rbp-3D8h] BYREF
  __m128i v38; // [rsp+B8h] [rbp-3C8h] BYREF
  __m128i v39; // [rsp+C8h] [rbp-3B8h] BYREF
  __int64 v40; // [rsp+D8h] [rbp-3A8h]
  _QWORD v41[3]; // [rsp+E0h] [rbp-3A0h] BYREF
  char v42; // [rsp+F8h] [rbp-388h]
  __int64 v43; // [rsp+100h] [rbp-380h]
  __int64 v44; // [rsp+108h] [rbp-378h]
  __int64 v45; // [rsp+110h] [rbp-370h]
  int v46; // [rsp+118h] [rbp-368h]
  __int64 v47; // [rsp+120h] [rbp-360h]
  __int64 v48; // [rsp+128h] [rbp-358h]
  __int64 v49; // [rsp+130h] [rbp-350h]
  __int64 v50; // [rsp+138h] [rbp-348h]
  __int64 v51; // [rsp+140h] [rbp-340h]
  __int64 v52; // [rsp+148h] [rbp-338h]
  __int64 v53; // [rsp+150h] [rbp-330h]
  __int64 v54; // [rsp+158h] [rbp-328h]
  __int64 v55; // [rsp+160h] [rbp-320h]
  char *v56; // [rsp+168h] [rbp-318h]
  __int64 v57; // [rsp+170h] [rbp-310h]
  int v58; // [rsp+178h] [rbp-308h]
  char v59; // [rsp+17Ch] [rbp-304h]
  char v60; // [rsp+180h] [rbp-300h] BYREF
  __int64 v61; // [rsp+200h] [rbp-280h]
  __int64 v62; // [rsp+208h] [rbp-278h]
  __int64 v63; // [rsp+210h] [rbp-270h]
  int v64; // [rsp+218h] [rbp-268h]
  char *v65; // [rsp+220h] [rbp-260h]
  __int64 v66; // [rsp+228h] [rbp-258h]
  char v67; // [rsp+230h] [rbp-250h] BYREF
  __int64 v68; // [rsp+260h] [rbp-220h]
  __int64 v69; // [rsp+268h] [rbp-218h]
  __int64 v70; // [rsp+270h] [rbp-210h]
  int v71; // [rsp+278h] [rbp-208h]
  __int64 v72; // [rsp+280h] [rbp-200h]
  char *v73; // [rsp+288h] [rbp-1F8h]
  __int64 v74; // [rsp+290h] [rbp-1F0h]
  int v75; // [rsp+298h] [rbp-1E8h]
  char v76; // [rsp+29Ch] [rbp-1E4h]
  char v77; // [rsp+2A0h] [rbp-1E0h] BYREF
  __int64 v78; // [rsp+2B0h] [rbp-1D0h]
  __int64 v79; // [rsp+2B8h] [rbp-1C8h]
  __int64 v80; // [rsp+2C0h] [rbp-1C0h]
  __int64 v81; // [rsp+2C8h] [rbp-1B8h]
  __int64 v82; // [rsp+2D0h] [rbp-1B0h]
  __int64 v83; // [rsp+2D8h] [rbp-1A8h]
  __int16 v84; // [rsp+2E0h] [rbp-1A0h]
  char v85; // [rsp+2E2h] [rbp-19Eh]
  char *v86; // [rsp+2E8h] [rbp-198h]
  __int64 v87; // [rsp+2F0h] [rbp-190h]
  char v88; // [rsp+2F8h] [rbp-188h] BYREF
  __int64 v89; // [rsp+318h] [rbp-168h]
  __int64 v90; // [rsp+320h] [rbp-160h]
  __int16 v91; // [rsp+328h] [rbp-158h]
  __int64 v92; // [rsp+330h] [rbp-150h]
  _QWORD *v93; // [rsp+338h] [rbp-148h]
  void **v94; // [rsp+340h] [rbp-140h]
  __int64 v95; // [rsp+348h] [rbp-138h]
  int v96; // [rsp+350h] [rbp-130h]
  __int16 v97; // [rsp+354h] [rbp-12Ch]
  char v98; // [rsp+356h] [rbp-12Ah]
  __int64 v99; // [rsp+358h] [rbp-128h]
  __int64 v100; // [rsp+360h] [rbp-120h]
  _QWORD v101[3]; // [rsp+368h] [rbp-118h] BYREF
  __m128i v102; // [rsp+380h] [rbp-100h]
  __m128i v103; // [rsp+390h] [rbp-F0h]
  __m128i v104; // [rsp+3A0h] [rbp-E0h]
  __m128i v105; // [rsp+3B0h] [rbp-D0h]
  __int64 v106; // [rsp+3C0h] [rbp-C0h]
  void *v107; // [rsp+3C8h] [rbp-B8h] BYREF
  char v108[16]; // [rsp+3D0h] [rbp-B0h] BYREF
  __int64 (__fastcall *v109)(char *, __m128i *, int); // [rsp+3E0h] [rbp-A0h]
  char (__fastcall *v110)(__int64 *, __int64 *); // [rsp+3E8h] [rbp-98h]
  char *v111; // [rsp+3F0h] [rbp-90h]
  __int64 v112; // [rsp+3F8h] [rbp-88h]
  char v113; // [rsp+400h] [rbp-80h] BYREF
  char *v114; // [rsp+440h] [rbp-40h]

  v9 = a2[1];
  v41[0] = a2;
  v56 = &v60;
  v41[1] = v9;
  v41[2] = "indvars";
  v42 = 1;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v57 = 16;
  v58 = 0;
  v59 = 1;
  v61 = 0;
  v10 = _mm_loadu_si128(&v30);
  v65 = &v67;
  v73 = &v77;
  v26.m128i_i64[0] = (__int64)v41;
  v11 = _mm_loadu_si128(&v26);
  v84 = 1;
  v66 = 0x200000000LL;
  v29 = &unk_49DA0D8;
  v26 = v10;
  v30 = v11;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v74 = 2;
  v75 = 0;
  v76 = 1;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v85 = 0;
  v27 = 0;
  v31 = (__int64 (__fastcall *)(char *, __m128i *, int))sub_27BFDD0;
  v12 = *a2;
  v35 = v9;
  v28 = v32;
  v36 = (__m128i)(unsigned __int64)v9;
  v32 = sub_27BFD20;
  v37 = 0u;
  v33 = &unk_49E5698;
  v34 = &unk_49D94D0;
  LOWORD(v40) = 257;
  v38 = 0u;
  v39 = 0u;
  v92 = sub_B2BE50(v12);
  v13 = _mm_loadu_si128(&v36);
  v93 = v101;
  v14 = _mm_loadu_si128(&v37);
  v94 = &v107;
  v15 = _mm_loadu_si128(&v38);
  v16 = _mm_loadu_si128(&v39);
  v86 = &v88;
  v101[2] = v35;
  v87 = 0x200000000LL;
  v95 = 0;
  v106 = v40;
  v96 = 0;
  v97 = 512;
  v98 = 7;
  v99 = 0;
  v100 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v101[0] = &unk_49E5698;
  v101[1] = &unk_49D94D0;
  v107 = &unk_49DA0D8;
  v102 = v13;
  v103 = v14;
  v104 = v15;
  v105 = v16;
  v109 = 0;
  if ( v31 )
  {
    v31(v108, &v30, 2);
    v110 = v32;
    v109 = v31;
  }
  v33 = &unk_49E5698;
  v34 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  sub_B32BF0(&v29);
  if ( v27 )
    v27(&v26, &v26, 3);
  v114 = "indvars";
  v17 = 0;
  v111 = &v113;
  v112 = 0x800000000LL;
  v18 = *(_QWORD *)(**(_QWORD **)(a1 + 32) + 56LL);
  v19 = a6;
  v20 = a5;
  v21 = v19;
  while ( 1 )
  {
    if ( !v18 )
      BUG();
    if ( *(_BYTE *)(v18 - 24) != 84 )
      break;
    LOWORD(v22) = sub_2A76A40(v18 - 24, a2, a3, a4, v20, v21, (__int64)v41, 0);
    v18 = *(_QWORD *)(v18 + 8);
    v17 |= v22;
  }
  sub_27C20B0((__int64)v41);
  return v17;
}
