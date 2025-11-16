// Function: sub_3880360
// Address: 0x3880360
//
__int64 __fastcall sub_3880360(
        __int64 **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        __int64 a18,
        __int8 *a19,
        size_t a20)
{
  __int64 v22; // rax
  _QWORD *v23; // rdi
  __int64 *v24; // r9
  __int64 v25; // rsi
  unsigned int v26; // eax
  unsigned int v27; // r15d
  double v28; // xmm4_8
  double v29; // xmm5_8
  unsigned __int64 *v30; // rbx
  unsigned __int64 *v31; // r12
  _QWORD *v32; // rbx
  unsigned __int64 v33; // r12
  __int64 *v34; // rdi
  char v36; // [rsp+14h] [rbp-67Ch]
  __int64 v39; // [rsp+30h] [rbp-660h] BYREF
  __int64 v40; // [rsp+38h] [rbp-658h] BYREF
  _QWORD *v41; // [rsp+40h] [rbp-650h] BYREF
  _QWORD *v42; // [rsp+48h] [rbp-648h]
  _QWORD *v43; // [rsp+50h] [rbp-640h]
  unsigned __int64 *v44; // [rsp+58h] [rbp-638h]
  unsigned __int64 *v45; // [rsp+60h] [rbp-630h]
  __int64 v46; // [rsp+68h] [rbp-628h]
  __int64 v47; // [rsp+70h] [rbp-620h]
  __int64 v48; // [rsp+78h] [rbp-618h]
  __int64 v49; // [rsp+80h] [rbp-610h] BYREF
  _QWORD v50[26]; // [rsp+88h] [rbp-608h] BYREF
  char v51; // [rsp+158h] [rbp-538h] BYREF
  __int64 v52; // [rsp+358h] [rbp-338h]
  __int64 v53; // [rsp+360h] [rbp-330h]
  __int64 v54; // [rsp+368h] [rbp-328h]
  int v55; // [rsp+380h] [rbp-310h] BYREF
  __int64 v56; // [rsp+388h] [rbp-308h]
  int *v57; // [rsp+390h] [rbp-300h]
  int *v58; // [rsp+398h] [rbp-2F8h]
  __int64 v59; // [rsp+3A0h] [rbp-2F0h]
  int v60; // [rsp+3B0h] [rbp-2E0h] BYREF
  __int64 v61; // [rsp+3B8h] [rbp-2D8h]
  int *v62; // [rsp+3C0h] [rbp-2D0h]
  int *v63; // [rsp+3C8h] [rbp-2C8h]
  __int64 v64; // [rsp+3D0h] [rbp-2C0h]
  int v65; // [rsp+3E0h] [rbp-2B0h] BYREF
  __int64 v66; // [rsp+3E8h] [rbp-2A8h]
  int *v67; // [rsp+3F0h] [rbp-2A0h]
  int *v68; // [rsp+3F8h] [rbp-298h]
  __int64 v69; // [rsp+400h] [rbp-290h]
  int v70; // [rsp+410h] [rbp-280h] BYREF
  __int64 v71; // [rsp+418h] [rbp-278h]
  int *v72; // [rsp+420h] [rbp-270h]
  int *v73; // [rsp+428h] [rbp-268h]
  __int64 v74; // [rsp+430h] [rbp-260h]
  int v75; // [rsp+440h] [rbp-250h] BYREF
  __int64 v76; // [rsp+448h] [rbp-248h]
  int *v77; // [rsp+450h] [rbp-240h]
  int *v78; // [rsp+458h] [rbp-238h]
  __int64 v79; // [rsp+460h] [rbp-230h]
  __int64 v80; // [rsp+468h] [rbp-228h]
  __int64 v81; // [rsp+470h] [rbp-220h]
  __int64 v82; // [rsp+478h] [rbp-218h]
  int v83; // [rsp+488h] [rbp-208h] BYREF
  __int64 v84; // [rsp+490h] [rbp-200h]
  int *v85; // [rsp+498h] [rbp-1F8h]
  int *v86; // [rsp+4A0h] [rbp-1F0h]
  __int64 v87; // [rsp+4A8h] [rbp-1E8h]
  int v88; // [rsp+4B8h] [rbp-1D8h] BYREF
  __int64 v89; // [rsp+4C0h] [rbp-1D0h]
  int *v90; // [rsp+4C8h] [rbp-1C8h]
  int *v91; // [rsp+4D0h] [rbp-1C0h]
  __int64 v92; // [rsp+4D8h] [rbp-1B8h]
  __int64 v93; // [rsp+4E0h] [rbp-1B0h]
  int v94; // [rsp+4F0h] [rbp-1A0h] BYREF
  __int64 v95; // [rsp+4F8h] [rbp-198h]
  int *v96; // [rsp+500h] [rbp-190h]
  int *v97; // [rsp+508h] [rbp-188h]
  __int64 v98; // [rsp+510h] [rbp-180h]
  int v99; // [rsp+520h] [rbp-170h] BYREF
  __int64 v100; // [rsp+528h] [rbp-168h]
  int *v101; // [rsp+530h] [rbp-160h]
  int *v102; // [rsp+538h] [rbp-158h]
  __int64 v103; // [rsp+540h] [rbp-150h]
  int v104; // [rsp+550h] [rbp-140h] BYREF
  __int64 v105; // [rsp+558h] [rbp-138h]
  int *v106; // [rsp+560h] [rbp-130h]
  int *v107; // [rsp+568h] [rbp-128h]
  __int64 v108; // [rsp+570h] [rbp-120h]
  int v109; // [rsp+580h] [rbp-110h] BYREF
  __int64 v110; // [rsp+588h] [rbp-108h]
  int *v111; // [rsp+590h] [rbp-100h]
  int *v112; // [rsp+598h] [rbp-F8h]
  __int64 v113; // [rsp+5A0h] [rbp-F0h]
  __int64 v114; // [rsp+5A8h] [rbp-E8h]
  __int64 v115; // [rsp+5B0h] [rbp-E0h]
  __int64 v116; // [rsp+5B8h] [rbp-D8h]
  int v117; // [rsp+5C8h] [rbp-C8h] BYREF
  __int64 v118; // [rsp+5D0h] [rbp-C0h]
  int *v119; // [rsp+5D8h] [rbp-B8h]
  int *v120; // [rsp+5E0h] [rbp-B0h]
  __int64 v121; // [rsp+5E8h] [rbp-A8h]
  int v122; // [rsp+5F8h] [rbp-98h] BYREF
  __int64 v123; // [rsp+600h] [rbp-90h]
  int *v124; // [rsp+608h] [rbp-88h]
  int *v125; // [rsp+610h] [rbp-80h]
  __int64 v126; // [rsp+618h] [rbp-78h]
  char v127; // [rsp+620h] [rbp-70h]
  __int8 *v128; // [rsp+628h] [rbp-68h]
  size_t v129; // [rsp+630h] [rbp-60h]
  char *v130; // [rsp+638h] [rbp-58h]
  __int64 v131; // [rsp+640h] [rbp-50h]
  char v132; // [rsp+648h] [rbp-48h] BYREF

  v36 = a5;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  sub_16C24D0(&v39, 1, a3, a4, a5, a6, a15, a16, a17, a18);
  v22 = v39;
  v23 = v42;
  v39 = 0;
  v50[0] = 0;
  v49 = v22;
  v50[1] = 0;
  if ( v42 == v43 )
  {
    sub_168C7C0((__int64 *)&v41, (__int64)v42, (__int64)&v49);
  }
  else
  {
    if ( v42 )
    {
      sub_16CE2D0(v42, &v49);
      v23 = v42;
    }
    v42 = v23 + 3;
  }
  sub_16CE300(&v49);
  sub_1602D10(&v40);
  v24 = &v40;
  if ( a1 )
    v24 = *a1;
  v25 = a15;
  v49 = (__int64)v24;
  sub_3880EA0(v50, a15, a16, &v41, a3);
  v50[21] = a1;
  v52 = 0;
  v50[22] = a2;
  v53 = 0;
  v50[23] = a4;
  v50[24] = &v51;
  v50[25] = 0x4000000000LL;
  v54 = 0x1800000000LL;
  v57 = &v55;
  v58 = &v55;
  v62 = &v60;
  v63 = &v60;
  v67 = &v65;
  v68 = &v65;
  v72 = &v70;
  v73 = &v70;
  v55 = 0;
  v56 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = &v75;
  v78 = &v75;
  v85 = &v83;
  v86 = &v83;
  v90 = &v88;
  v91 = &v88;
  v96 = &v94;
  v97 = &v94;
  v101 = &v99;
  v102 = &v99;
  v106 = &v104;
  v107 = &v104;
  v79 = 0;
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = &v109;
  v112 = &v109;
  v119 = &v117;
  v120 = &v117;
  v124 = &v122;
  v125 = &v122;
  v113 = 0;
  v127 = v36;
  v114 = 0;
  v128 = a19;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v126 = 0;
  v129 = a20;
  v130 = &v132;
  v131 = 0;
  v132 = 0;
  if ( a1 && a20 )
  {
    v25 = (__int64)a19;
    sub_1632B30((__int64)a1, a19, a20);
  }
  *(double *)a7.m128_u64 = sub_38B8480(&v49);
  v27 = v26;
  sub_38800E0((__int64)&v49, v25, a7, a8, a9, a10, v28, v29, a13, a14);
  sub_16025D0(&v40);
  if ( v39 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v39 + 8LL))(v39);
  v30 = v45;
  v31 = v44;
  if ( v45 != v44 )
  {
    do
    {
      if ( (unsigned __int64 *)*v31 != v31 + 2 )
        j_j___libc_free_0(*v31);
      v31 += 4;
    }
    while ( v30 != v31 );
    v31 = v44;
  }
  if ( v31 )
    j_j___libc_free_0((unsigned __int64)v31);
  v32 = v42;
  v33 = (unsigned __int64)v41;
  if ( v42 != v41 )
  {
    do
    {
      v34 = (__int64 *)v33;
      v33 += 24LL;
      sub_16CE300(v34);
    }
    while ( v32 != (_QWORD *)v33 );
    v33 = (unsigned __int64)v41;
  }
  if ( v33 )
    j_j___libc_free_0(v33);
  return v27;
}
