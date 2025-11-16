// Function: sub_211E5A0
// Address: 0x211e5a0
//
__int64 __fastcall sub_211E5A0(
        __int64 **a1,
        unsigned __int64 *a2,
        __int64 a3,
        unsigned int *a4,
        __int64 a5,
        __m128 a6,
        double a7,
        __m128i a8)
{
  __int64 v10; // rdx
  unsigned __int64 v11; // rsi
  __int64 *v12; // r12
  unsigned __int8 *v13; // rax
  __int64 v14; // r15
  __int64 v15; // r14
  __int64 (__fastcall *v16)(__int64 *, __int64, __int64, __int64, __int64); // r13
  __int64 v17; // rax
  unsigned __int8 v18; // al
  unsigned __int64 v19; // r12
  __int16 *v20; // r13
  __int64 v21; // r14
  __int64 v22; // r15
  __int64 v23; // rdx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // rax
  __int64 *v29; // r10
  __int64 *v30; // r12
  unsigned int v31; // edx
  __int64 v32; // r14
  unsigned __int8 *v33; // rax
  __int64 v34; // r15
  __int64 (__fastcall *v35)(__int64 *, __int64, __int64, __int64, __int64); // r13
  __int64 v36; // rax
  unsigned __int8 v37; // al
  unsigned __int64 v38; // r12
  __int16 *v39; // r13
  __int64 v40; // rdx
  __int64 v41; // r14
  __int64 v42; // r15
  __int64 v43; // r8
  __int64 v44; // rax
  __int64 v45; // rdx
  unsigned int v46; // edx
  __int64 *v47; // rax
  __int64 *v48; // r10
  __int64 *v49; // r12
  __int64 v50; // r14
  unsigned int v51; // edx
  unsigned __int8 *v52; // rax
  __int64 v53; // r15
  __int64 (__fastcall *v54)(__int64 *, __int64, __int64, __int64, __int64); // r13
  __int64 v55; // rax
  unsigned __int8 v56; // al
  unsigned __int64 v57; // r12
  __int16 *v58; // r13
  __int64 v59; // r14
  __int64 v60; // r15
  __int64 v61; // rdx
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 *v66; // rax
  __int64 *v67; // r10
  __int64 *v68; // r12
  unsigned int *v69; // r9
  unsigned int v70; // edx
  __int64 v71; // r14
  unsigned __int8 *v72; // rax
  __int64 v73; // r15
  __int64 (__fastcall *v74)(__int64 *, __int64, __int64, __int64, __int64); // r13
  __int64 v75; // rax
  unsigned __int8 v76; // al
  unsigned __int64 v77; // r12
  __int16 *v78; // r13
  __int64 v79; // rdx
  __int64 v80; // r14
  __int64 v81; // r15
  __int64 v82; // r8
  __int64 v83; // rax
  __int64 v84; // rdx
  unsigned int v85; // edx
  unsigned int v86; // edx
  unsigned __int64 v87; // rcx
  const void ***v88; // rdx
  unsigned int v90; // edx
  __int128 v91; // [rsp-20h] [rbp-150h]
  __int128 v92; // [rsp-20h] [rbp-150h]
  __int128 v93; // [rsp-20h] [rbp-150h]
  __int128 v94; // [rsp-20h] [rbp-150h]
  __int128 v95; // [rsp-10h] [rbp-140h]
  __int128 v96; // [rsp-10h] [rbp-140h]
  __int64 v97; // [rsp+8h] [rbp-128h]
  __int64 v98; // [rsp+10h] [rbp-120h]
  unsigned int v99; // [rsp+18h] [rbp-118h]
  const void **v100; // [rsp+18h] [rbp-118h]
  const void **v101; // [rsp+18h] [rbp-118h]
  const void **v102; // [rsp+20h] [rbp-110h]
  __int64 v103; // [rsp+20h] [rbp-110h]
  unsigned __int64 v104; // [rsp+20h] [rbp-110h]
  __int64 v105; // [rsp+20h] [rbp-110h]
  unsigned __int64 v106; // [rsp+20h] [rbp-110h]
  __int64 v107; // [rsp+28h] [rbp-108h]
  unsigned __int64 v108; // [rsp+28h] [rbp-108h]
  __int64 *v109; // [rsp+28h] [rbp-108h]
  __int64 *v110; // [rsp+28h] [rbp-108h]
  __int64 v111; // [rsp+30h] [rbp-100h]
  unsigned __int64 *v112; // [rsp+38h] [rbp-F8h]
  const void **v113; // [rsp+40h] [rbp-F0h]
  __int64 *v114; // [rsp+40h] [rbp-F0h]
  __int128 v115; // [rsp+40h] [rbp-F0h]
  __int64 *v117; // [rsp+50h] [rbp-E0h]
  __int64 v119; // [rsp+60h] [rbp-D0h]
  unsigned __int64 v120; // [rsp+60h] [rbp-D0h]
  __int64 *v121; // [rsp+60h] [rbp-D0h]
  __int64 *v122; // [rsp+60h] [rbp-D0h]
  __int64 v123; // [rsp+68h] [rbp-C8h]
  __int64 *v124; // [rsp+70h] [rbp-C0h]
  __int64 v125; // [rsp+70h] [rbp-C0h]
  __int64 v126; // [rsp+70h] [rbp-C0h]
  __int64 *v127; // [rsp+70h] [rbp-C0h]
  unsigned __int64 v128; // [rsp+78h] [rbp-B8h]
  unsigned __int64 v129; // [rsp+78h] [rbp-B8h]
  __int64 *v130; // [rsp+B0h] [rbp-80h]
  unsigned __int64 v131; // [rsp+C0h] [rbp-70h] BYREF
  __int16 *v132; // [rsp+C8h] [rbp-68h]
  unsigned __int64 v133; // [rsp+D0h] [rbp-60h] BYREF
  __int16 *v134; // [rsp+D8h] [rbp-58h]
  __int64 v135; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v136; // [rsp+E8h] [rbp-48h]
  __int64 v137; // [rsp+F0h] [rbp-40h] BYREF
  __int64 v138; // [rsp+F8h] [rbp-38h]

  v10 = a2[1];
  v11 = *a2;
  v112 = a2;
  v131 = 0;
  LODWORD(v132) = 0;
  v133 = 0;
  LODWORD(v134) = 0;
  v135 = 0;
  LODWORD(v136) = 0;
  v137 = 0;
  LODWORD(v138) = 0;
  sub_2016B80((__int64)a1, v11, v10, &v131, &v133);
  v111 = a3;
  sub_2016B80((__int64)a1, *(_QWORD *)a3, *(_QWORD *)(a3 + 8), &v135, &v137);
  v12 = *a1;
  v13 = (unsigned __int8 *)(*(_QWORD *)(v133 + 40) + 16LL * (unsigned int)v134);
  v14 = *((_QWORD *)v13 + 1);
  v124 = a1[1];
  v15 = v124[6];
  v119 = *v13;
  v16 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(**a1 + 264);
  v17 = sub_1E0A0C0(v124[4]);
  v18 = v16(v12, v17, v15, v119, v14);
  v19 = v133;
  v20 = v134;
  v21 = v137;
  v22 = v138;
  v120 = v18;
  v113 = (const void **)v23;
  v26 = sub_1D28D50(v124, 1u, v23, v18, v24, v25);
  *((_QWORD *)&v91 + 1) = v22;
  *(_QWORD *)&v91 = v21;
  v28 = sub_1D3A900(v124, 0x89u, a5, v120, v113, 0, a6, a7, a8, v19, v20, v91, v26, v27);
  v29 = a1[1];
  v30 = *a1;
  v114 = v28;
  v125 = (__int64)v28;
  v98 = v31;
  v32 = v29[6];
  v121 = v29;
  v128 = v31;
  v33 = (unsigned __int8 *)(*(_QWORD *)(v131 + 40) + 16LL * (unsigned int)v132);
  v34 = *((_QWORD *)v33 + 1);
  v99 = *a4;
  v107 = *v33;
  v35 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(**a1 + 264);
  v36 = sub_1E0A0C0(v29[4]);
  v37 = v35(v30, v36, v32, v107, v34);
  v38 = v131;
  v39 = v132;
  v102 = (const void **)v40;
  v41 = v135;
  v42 = v136;
  v108 = v37;
  v44 = sub_1D28D50(v121, v99, v40, v37, v43, v99);
  *((_QWORD *)&v92 + 1) = v42;
  *(_QWORD *)&v92 = v41;
  v130 = sub_1D3A900(v121, 0x89u, a5, v108, v102, 0, a6, a7, a8, v38, v39, v92, v44, v45);
  v123 = v46;
  *((_QWORD *)&v95 + 1) = v46;
  *(_QWORD *)&v95 = v130;
  v47 = sub_1D332F0(
          a1[1],
          118,
          a5,
          *(unsigned __int8 *)(v114[5] + 16 * v98),
          *(const void ***)(v114[5] + 16 * v98 + 8),
          0,
          *(double *)a6.m128_u64,
          a7,
          a8,
          v125,
          v128,
          v95);
  v48 = a1[1];
  v49 = *a1;
  *(_QWORD *)&v115 = v47;
  v50 = v48[6];
  v109 = v48;
  *((_QWORD *)&v115 + 1) = v51;
  v52 = (unsigned __int8 *)(*(_QWORD *)(v133 + 40) + 16LL * (unsigned int)v134);
  v53 = *((_QWORD *)v52 + 1);
  v103 = *v52;
  v54 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(**a1 + 264);
  v55 = sub_1E0A0C0(v48[4]);
  v56 = v54(v49, v55, v50, v103, v53);
  v57 = v133;
  v58 = v134;
  v59 = v137;
  v60 = v138;
  v104 = v56;
  v100 = (const void **)v61;
  v64 = sub_1D28D50(v109, 0xEu, v61, v56, v62, v63);
  *((_QWORD *)&v93 + 1) = v60;
  *(_QWORD *)&v93 = v59;
  v66 = sub_1D3A900(v109, 0x89u, a5, v104, v100, 0, a6, a7, a8, v57, v58, v93, v64, v65);
  v67 = a1[1];
  v68 = *a1;
  v110 = v66;
  v126 = (__int64)v66;
  v69 = a4;
  v97 = v70;
  v71 = v67[6];
  v117 = v67;
  v72 = (unsigned __int8 *)(*(_QWORD *)(v133 + 40) + 16LL * (unsigned int)v134);
  v129 = v70 | v128 & 0xFFFFFFFF00000000LL;
  v73 = *((_QWORD *)v72 + 1);
  LODWORD(v98) = *v69;
  v105 = *v72;
  v74 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(**a1 + 264);
  v75 = sub_1E0A0C0(v67[4]);
  v76 = v74(v68, v75, v71, v105, v73);
  v77 = v133;
  v78 = v134;
  v101 = (const void **)v79;
  v80 = v137;
  v81 = v138;
  v106 = v76;
  v83 = sub_1D28D50(v117, v98, v79, v76, v82, (unsigned int)v98);
  *((_QWORD *)&v94 + 1) = v81;
  *(_QWORD *)&v94 = v80;
  v122 = sub_1D3A900(v117, 0x89u, a5, v106, v101, 0, a6, a7, a8, v77, v78, v94, v83, v84);
  *((_QWORD *)&v96 + 1) = v85 | v123 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v96 = v122;
  v127 = sub_1D332F0(
           a1[1],
           118,
           a5,
           *(unsigned __int8 *)(v110[5] + 16 * v97),
           *(const void ***)(v110[5] + 16 * v97 + 8),
           0,
           *(double *)a6.m128_u64,
           a7,
           a8,
           v126,
           v129,
           v96);
  v87 = v86 | v129 & 0xFFFFFFFF00000000LL;
  v88 = (const void ***)(v127[5] + 16LL * v86);
  *v112 = (unsigned __int64)sub_1D332F0(
                              a1[1],
                              119,
                              a5,
                              *(unsigned __int8 *)v88,
                              v88[1],
                              0,
                              *(double *)a6.m128_u64,
                              a7,
                              a8,
                              (__int64)v127,
                              v87,
                              v115);
  *((_DWORD *)v112 + 2) = v90;
  *(_QWORD *)v111 = 0;
  *(_DWORD *)(v111 + 8) = 0;
  return v90;
}
