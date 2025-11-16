// Function: sub_105F9E0
// Address: 0x105f9e0
//
__int64 __fastcall sub_105F9E0(
        __int128 **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        const char *a9,
        const char *a10,
        __int64 a11,
        __int64 a12)
{
  __int64 v14; // rax
  char *v15; // rdi
  __int128 *v16; // r9
  unsigned int v17; // r14d
  _QWORD *v18; // rbx
  _QWORD *v19; // r12
  __int64 *v20; // rbx
  __int64 *v21; // r12
  __int64 *v22; // rdi
  unsigned __int8 v24; // [rsp+24h] [rbp-79Ch]
  __int64 v26; // [rsp+38h] [rbp-788h] BYREF
  __int128 v27; // [rsp+40h] [rbp-780h] BYREF
  char *v28; // [rsp+50h] [rbp-770h] BYREF
  char *v29; // [rsp+58h] [rbp-768h]
  char *v30; // [rsp+60h] [rbp-760h]
  _QWORD *v31; // [rsp+68h] [rbp-758h]
  _QWORD *v32; // [rsp+70h] [rbp-750h]
  __int64 v33; // [rsp+78h] [rbp-748h]
  __int64 v34; // [rsp+80h] [rbp-740h]
  __int64 v35; // [rsp+88h] [rbp-738h]
  _OWORD *v36; // [rsp+90h] [rbp-730h] BYREF
  _QWORD v37[21]; // [rsp+98h] [rbp-728h] BYREF
  char v38[168]; // [rsp+140h] [rbp-680h] BYREF
  __int128 **v39; // [rsp+1E8h] [rbp-5D8h]
  __int64 v40; // [rsp+1F0h] [rbp-5D0h]
  __int64 v41; // [rsp+1F8h] [rbp-5C8h]
  char *v42; // [rsp+200h] [rbp-5C0h]
  __int64 v43; // [rsp+208h] [rbp-5B8h]
  char v44; // [rsp+210h] [rbp-5B0h] BYREF
  __int64 v45; // [rsp+410h] [rbp-3B0h]
  __int64 v46; // [rsp+418h] [rbp-3A8h]
  __int64 v47; // [rsp+420h] [rbp-3A0h]
  int v48; // [rsp+428h] [rbp-398h]
  __int64 v49; // [rsp+430h] [rbp-390h]
  __int64 v50; // [rsp+438h] [rbp-388h]
  __int64 v51; // [rsp+440h] [rbp-380h]
  _QWORD v52[6]; // [rsp+450h] [rbp-370h] BYREF
  int v53; // [rsp+480h] [rbp-340h] BYREF
  __int64 v54; // [rsp+488h] [rbp-338h]
  int *v55; // [rsp+490h] [rbp-330h]
  int *v56; // [rsp+498h] [rbp-328h]
  __int64 v57; // [rsp+4A0h] [rbp-320h]
  int v58; // [rsp+4B0h] [rbp-310h] BYREF
  __int64 v59; // [rsp+4B8h] [rbp-308h]
  int *v60; // [rsp+4C0h] [rbp-300h]
  int *v61; // [rsp+4C8h] [rbp-2F8h]
  __int64 v62; // [rsp+4D0h] [rbp-2F0h]
  int v63; // [rsp+4E0h] [rbp-2E0h] BYREF
  __int64 v64; // [rsp+4E8h] [rbp-2D8h]
  int *v65; // [rsp+4F0h] [rbp-2D0h]
  int *v66; // [rsp+4F8h] [rbp-2C8h]
  __int64 v67; // [rsp+500h] [rbp-2C0h]
  int v68; // [rsp+510h] [rbp-2B0h] BYREF
  __int64 v69; // [rsp+518h] [rbp-2A8h]
  int *v70; // [rsp+520h] [rbp-2A0h]
  int *v71; // [rsp+528h] [rbp-298h]
  __int64 v72; // [rsp+530h] [rbp-290h]
  __int64 v73; // [rsp+538h] [rbp-288h]
  __int64 v74; // [rsp+540h] [rbp-280h]
  __int64 v75; // [rsp+548h] [rbp-278h]
  int v76; // [rsp+550h] [rbp-270h]
  int v77; // [rsp+558h] [rbp-268h]
  int v78; // [rsp+568h] [rbp-258h] BYREF
  __int64 v79; // [rsp+570h] [rbp-250h]
  int *v80; // [rsp+578h] [rbp-248h]
  int *v81; // [rsp+580h] [rbp-240h]
  __int64 v82; // [rsp+588h] [rbp-238h]
  int v83; // [rsp+598h] [rbp-228h] BYREF
  __int64 v84; // [rsp+5A0h] [rbp-220h]
  int *v85; // [rsp+5A8h] [rbp-218h]
  int *v86; // [rsp+5B0h] [rbp-210h]
  __int64 v87; // [rsp+5B8h] [rbp-208h]
  __int64 v88; // [rsp+5C0h] [rbp-200h]
  int v89; // [rsp+5D0h] [rbp-1F0h] BYREF
  __int64 v90; // [rsp+5D8h] [rbp-1E8h]
  int *v91; // [rsp+5E0h] [rbp-1E0h]
  int *v92; // [rsp+5E8h] [rbp-1D8h]
  __int64 v93; // [rsp+5F0h] [rbp-1D0h]
  int v94; // [rsp+600h] [rbp-1C0h] BYREF
  __int64 v95; // [rsp+608h] [rbp-1B8h]
  int *v96; // [rsp+610h] [rbp-1B0h]
  int *v97; // [rsp+618h] [rbp-1A8h]
  __int64 v98; // [rsp+620h] [rbp-1A0h]
  int v99; // [rsp+630h] [rbp-190h] BYREF
  __int64 v100; // [rsp+638h] [rbp-188h]
  int *v101; // [rsp+640h] [rbp-180h]
  int *v102; // [rsp+648h] [rbp-178h]
  __int64 v103; // [rsp+650h] [rbp-170h]
  int v104; // [rsp+660h] [rbp-160h] BYREF
  __int64 v105; // [rsp+668h] [rbp-158h]
  int *v106; // [rsp+670h] [rbp-150h]
  int *v107; // [rsp+678h] [rbp-148h]
  __int64 v108; // [rsp+680h] [rbp-140h]
  int v109; // [rsp+690h] [rbp-130h] BYREF
  __int64 v110; // [rsp+698h] [rbp-128h]
  int *v111; // [rsp+6A0h] [rbp-120h]
  int *v112; // [rsp+6A8h] [rbp-118h]
  __int64 v113; // [rsp+6B0h] [rbp-110h]
  int v114; // [rsp+6C0h] [rbp-100h] BYREF
  __int64 v115; // [rsp+6C8h] [rbp-F8h]
  int *v116; // [rsp+6D0h] [rbp-F0h]
  int *v117; // [rsp+6D8h] [rbp-E8h]
  __int64 v118; // [rsp+6E0h] [rbp-E0h]
  __int64 v119; // [rsp+6E8h] [rbp-D8h]
  __int64 v120; // [rsp+6F0h] [rbp-D0h]
  __int64 v121; // [rsp+6F8h] [rbp-C8h]
  int v122; // [rsp+708h] [rbp-B8h] BYREF
  __int64 v123; // [rsp+710h] [rbp-B0h]
  int *v124; // [rsp+718h] [rbp-A8h]
  int *v125; // [rsp+720h] [rbp-A0h]
  __int64 v126; // [rsp+728h] [rbp-98h]
  int v127; // [rsp+738h] [rbp-88h] BYREF
  __int64 v128; // [rsp+740h] [rbp-80h]
  int *v129; // [rsp+748h] [rbp-78h]
  int *v130; // [rsp+750h] [rbp-70h]
  __int64 v131; // [rsp+758h] [rbp-68h]
  __int16 v132; // [rsp+761h] [rbp-5Fh]
  char *v133; // [rsp+768h] [rbp-58h]
  __int64 v134; // [rsp+770h] [rbp-50h]
  char v135; // [rsp+778h] [rbp-48h] BYREF

  v24 = a5;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  sub_C7DB10(&v26, 1u, a3, a4, a5, a6, a7, a8, a9, a10);
  v14 = v26;
  v15 = v29;
  v26 = 0;
  v37[0] = 0;
  v36 = (_OWORD *)v14;
  v37[1] = 0;
  if ( v29 == v30 )
  {
    sub_C12520((__int64 *)&v28, (__int64)v29, (__int64)&v36);
  }
  else
  {
    if ( v29 )
    {
      sub_C8EDF0(v29, &v36);
      v15 = v29;
    }
    v29 = v15 + 24;
  }
  sub_C8EE20((__int64 *)&v36);
  v27 = 0;
  if ( a1 )
  {
    v16 = *a1;
  }
  else
  {
    sub_B6EEA0((__int64 *)&v27);
    BYTE8(v27) = 1;
    v16 = &v27;
  }
  v36 = v16;
  sub_11FD2F0(v37, a7, a8, &v28, a3);
  sub_11FD2F0(v38, a7, a8, &v28, a3);
  v39 = a1;
  v40 = a2;
  v41 = a4;
  v42 = &v44;
  v43 = 0x4000000000LL;
  v51 = 0x1800000000LL;
  v52[2] = v52;
  v52[3] = v52;
  v55 = &v53;
  v56 = &v53;
  v60 = &v58;
  v61 = &v58;
  v65 = &v63;
  v66 = &v63;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v52[0] = 0;
  v52[1] = 0;
  v52[4] = 0;
  v53 = 0;
  v54 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v67 = 0;
  v70 = &v68;
  v71 = &v68;
  v80 = &v78;
  v81 = &v78;
  v85 = &v83;
  v86 = &v83;
  v91 = &v89;
  v92 = &v89;
  v96 = &v94;
  v97 = &v94;
  v68 = 0;
  v69 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v75 = 0;
  v76 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = &v99;
  v102 = &v99;
  v106 = &v104;
  v107 = &v104;
  v111 = &v109;
  v112 = &v109;
  v116 = &v114;
  v117 = &v114;
  v124 = &v122;
  v125 = &v122;
  v129 = &v127;
  v130 = &v127;
  v132 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v118 = 0;
  v119 = 0;
  v120 = 0;
  v121 = 0;
  v122 = 0;
  v123 = 0;
  v126 = 0;
  v127 = 0;
  v128 = 0;
  v131 = 0;
  v133 = &v135;
  v134 = 0;
  v135 = 0;
  v17 = sub_1249310(&v36, v24, a11, a12);
  sub_105F640((__int64)&v36, v24);
  if ( BYTE8(v27) )
  {
    BYTE8(v27) = 0;
    sub_B6E710(&v27);
  }
  if ( v26 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
  v18 = v32;
  v19 = v31;
  if ( v32 != v31 )
  {
    do
    {
      if ( (_QWORD *)*v19 != v19 + 2 )
        j_j___libc_free_0(*v19, v19[2] + 1LL);
      v19 += 4;
    }
    while ( v18 != v19 );
    v19 = v31;
  }
  if ( v19 )
    j_j___libc_free_0(v19, v33 - (_QWORD)v19);
  v20 = (__int64 *)v29;
  v21 = (__int64 *)v28;
  if ( v29 != v28 )
  {
    do
    {
      v22 = v21;
      v21 += 3;
      sub_C8EE20(v22);
    }
    while ( v20 != v21 );
    v21 = (__int64 *)v28;
  }
  if ( v21 )
    j_j___libc_free_0(v21, v30 - (char *)v21);
  return v17;
}
