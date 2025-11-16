// Function: sub_3443060
// Address: 0x3443060
//
__int64 __fastcall sub_3443060(_QWORD **a1, unsigned int a2, int a3, unsigned int a4, int a5)
{
  __m128i *v7; // rax
  unsigned int v8; // r10d
  unsigned __int64 v9; // r13
  int v10; // edx
  int v11; // r14d
  __m128i *i; // r12
  _QWORD *v13; // r15
  const __m128i *v14; // rax
  __int64 v15; // rax
  __int64 v16; // rdx
  char v17; // al
  unsigned int v18; // r10d
  unsigned int v19; // r12d
  __m128i *v20; // r15
  const __m128i *v21; // rax
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 *v24; // rax
  __int64 *v25; // rcx
  __int64 *v26; // rdx
  __int64 v27; // r12
  __int64 v28; // r13
  __int64 v29; // r14
  __int64 v30; // r15
  __int128 v31; // rax
  __int64 v32; // rax
  __int64 *v33; // r9
  _QWORD *v34; // r14
  unsigned __int64 *v35; // rcx
  __int64 *v36; // r8
  __int64 v37; // r12
  __int64 v38; // rsi
  unsigned int v39; // edx
  __int64 v40; // r15
  __int64 v41; // rbx
  __int64 v42; // r10
  __int64 v43; // r11
  __int64 v44; // rax
  __int16 v45; // di
  __int64 v46; // rax
  __int64 v47; // r13
  unsigned __int64 v48; // rdx
  unsigned __int64 v49; // rcx
  bool v50; // al
  unsigned int v51; // eax
  __int64 *v53; // rax
  __int64 *v54; // rdx
  __int64 v55; // r14
  __int64 v56; // r12
  __int64 v57; // r13
  __int128 v58; // rax
  __int64 v59; // r9
  __int64 v60; // rax
  unsigned int v61; // edx
  _QWORD *v62; // r15
  unsigned __int64 *v63; // rcx
  __int64 *v64; // r8
  __int64 v65; // rdi
  __int64 *v66; // r9
  __int64 v67; // r14
  __int64 v68; // r12
  __int64 v69; // r10
  __int64 v70; // r11
  __int64 v71; // rsi
  __int64 v72; // rbx
  __int64 v73; // rax
  __int16 v74; // di
  __int64 v75; // rax
  __int64 v76; // r13
  unsigned __int64 v77; // rdx
  unsigned __int64 v78; // rcx
  unsigned int v79; // eax
  bool v80; // al
  __int64 *v81; // rax
  unsigned int v82; // esi
  _QWORD *v83; // r12
  __int64 *v84; // rdx
  __int64 v85; // r13
  __int64 v86; // r15
  __int64 v87; // r14
  __int128 v88; // rax
  __int64 v89; // rax
  __int128 v90; // [rsp-30h] [rbp-F0h]
  __int128 v91; // [rsp-20h] [rbp-E0h]
  __int128 v92; // [rsp-20h] [rbp-E0h]
  __int128 v93; // [rsp-20h] [rbp-E0h]
  __int128 v94; // [rsp-20h] [rbp-E0h]
  __int128 v95; // [rsp-20h] [rbp-E0h]
  __int64 v96; // [rsp+10h] [rbp-B0h]
  __int128 v97; // [rsp+10h] [rbp-B0h]
  __int64 v99; // [rsp+20h] [rbp-A0h]
  __int64 v100; // [rsp+20h] [rbp-A0h]
  __int64 v101; // [rsp+20h] [rbp-A0h]
  __int64 v102; // [rsp+20h] [rbp-A0h]
  __int128 v103; // [rsp+20h] [rbp-A0h]
  __int64 v104; // [rsp+28h] [rbp-98h]
  __int64 v105; // [rsp+28h] [rbp-98h]
  unsigned int v106; // [rsp+30h] [rbp-90h]
  _QWORD *v107; // [rsp+30h] [rbp-90h]
  __int64 v108; // [rsp+30h] [rbp-90h]
  unsigned __int64 v109; // [rsp+30h] [rbp-90h]
  __int64 v110; // [rsp+30h] [rbp-90h]
  unsigned __int64 v111; // [rsp+30h] [rbp-90h]
  __int64 v112; // [rsp+30h] [rbp-90h]
  unsigned __int64 v113; // [rsp+38h] [rbp-88h]
  unsigned __int64 v114; // [rsp+38h] [rbp-88h]
  _QWORD *v116; // [rsp+40h] [rbp-80h]
  _QWORD *v117; // [rsp+40h] [rbp-80h]
  __int64 v118; // [rsp+40h] [rbp-80h]
  unsigned int v119; // [rsp+58h] [rbp-68h] BYREF
  int v120; // [rsp+5Ch] [rbp-64h]
  __m128i v121; // [rsp+60h] [rbp-60h] BYREF
  __m128i v122; // [rsp+70h] [rbp-50h]
  __int64 v123; // [rsp+80h] [rbp-40h]
  __int64 v124; // [rsp+88h] [rbp-38h]

  v7 = sub_33ED250((__int64)*a1, *(unsigned int *)a1[1], a1[1][1]);
  v120 = a3;
  v8 = a2;
  v9 = (unsigned __int64)v7;
  v119 = a2;
  v11 = v10;
  for ( i = (__m128i *)&v119; ; v8 = i->m128i_i32[0] )
  {
    v13 = *a1;
    v106 = v8;
    v14 = (const __m128i *)a1[3];
    v121 = _mm_loadu_si128((const __m128i *)a1[2]);
    v122 = _mm_loadu_si128(v14);
    v15 = sub_33ED040(v13, v8);
    v124 = v16;
    v123 = v15;
    v17 = sub_33CEDC0((__int64)v13, 208, v9, v11, (unsigned __int64 *)&v121, 3);
    v18 = v106;
    if ( v17 )
    {
      v53 = a1[1];
      v54 = a1[3];
      v117 = *a1;
      v55 = (__int64)a1[4];
      v56 = *v54;
      v110 = *v53;
      v97 = *(_OWORD *)a1[2];
      v57 = v54[1];
      v101 = v53[1];
      *(_QWORD *)&v58 = sub_33ED040(*a1, v18);
      *((_QWORD *)&v93 + 1) = v57;
      *(_QWORD *)&v93 = v56;
      v60 = sub_340F900(v117, 0xD0u, v55, v110, v101, v59, v97, v93, v58);
      v62 = *a1;
      v63 = a1[3];
      v64 = a1[2];
      v65 = v60;
LABEL_14:
      v66 = a1[5];
      v67 = (__int64)a1[4];
      v68 = v65;
      v69 = *v64;
      v70 = v64[1];
      v71 = *v66;
      v72 = v66[1];
      v73 = *(_QWORD *)(v65 + 48) + 16LL * v61;
      v74 = *(_WORD *)v73;
      v75 = *(_QWORD *)(v73 + 8);
      v76 = v61;
      v77 = *v63;
      v78 = v63[1];
      v121.m128i_i16[0] = v74;
      v121.m128i_i64[1] = v75;
      if ( v74 )
      {
        v79 = ((unsigned __int16)(v74 - 17) < 0xD4u) + 205;
      }
      else
      {
        v102 = v69;
        v105 = v70;
        v111 = v77;
        v114 = v78;
        v80 = sub_30070B0((__int64)&v121);
        v69 = v102;
        v70 = v105;
        v77 = v111;
        v78 = v114;
        v79 = 205 - (!v80 - 1);
      }
      *((_QWORD *)&v94 + 1) = v70;
      *(_QWORD *)&v94 = v69;
      return sub_340EC60(v62, v79, v67, v71, v72, 0, v68, v76, v94, __PAIR128__(v78, v77));
    }
    i = (__m128i *)((char *)i + 4);
    if ( &v121 == i )
      break;
  }
  v19 = a4;
  v20 = (__m128i *)&v119;
  v119 = a4;
  v120 = a5;
  while ( 1 )
  {
    v21 = (const __m128i *)a1[3];
    v107 = *a1;
    v121 = _mm_loadu_si128((const __m128i *)a1[2]);
    v122 = _mm_loadu_si128(v21);
    v22 = sub_33ED040(v107, v19);
    v124 = v23;
    v123 = v22;
    if ( (unsigned __int8)sub_33CEDC0((__int64)v107, 208, v9, v11, (unsigned __int64 *)&v121, 3) )
    {
      v81 = a1[1];
      v82 = v19;
      v83 = *a1;
      v84 = a1[3];
      v85 = *v81;
      v86 = v84[1];
      v103 = *(_OWORD *)a1[2];
      v118 = (__int64)a1[4];
      v87 = *v84;
      v112 = v81[1];
      *(_QWORD *)&v88 = sub_33ED040(*a1, v82);
      *((_QWORD *)&v95 + 1) = v86;
      *(_QWORD *)&v95 = v87;
      v89 = sub_340F900(v83, 0xD0u, v118, v85, v112, v118, v103, v95, v88);
      v62 = *a1;
      v63 = a1[2];
      v64 = a1[3];
      v65 = v89;
      goto LABEL_14;
    }
    v20 = (__m128i *)((char *)v20 + 4);
    if ( &v121 == v20 )
      break;
    v19 = v20->m128i_i32[0];
  }
  v24 = a1[1];
  v25 = a1[2];
  v26 = a1[3];
  v27 = *v25;
  v116 = *a1;
  v28 = v25[1];
  v29 = *v26;
  v108 = *v24;
  v30 = v26[1];
  v96 = (__int64)a1[4];
  v99 = v24[1];
  *(_QWORD *)&v31 = sub_33ED040(*a1, a2);
  *((_QWORD *)&v91 + 1) = v30;
  *(_QWORD *)&v91 = v29;
  *((_QWORD *)&v90 + 1) = v28;
  *(_QWORD *)&v90 = v27;
  v32 = sub_340F900(v116, 0xD0u, v96, v108, v99, v96, v90, v91, v31);
  v33 = a1[5];
  v34 = *a1;
  v35 = a1[3];
  v36 = a1[2];
  v37 = v32;
  v38 = *v33;
  v40 = (__int64)a1[4];
  v41 = v33[1];
  v42 = *v36;
  v43 = v36[1];
  v44 = *(_QWORD *)(v32 + 48) + 16LL * v39;
  v45 = *(_WORD *)v44;
  v46 = *(_QWORD *)(v44 + 8);
  v47 = v39;
  v48 = *v35;
  v49 = v35[1];
  v121.m128i_i16[0] = v45;
  v121.m128i_i64[1] = v46;
  if ( v45 )
  {
    v51 = ((unsigned __int16)(v45 - 17) < 0xD4u) + 205;
  }
  else
  {
    v100 = v42;
    v104 = v43;
    v109 = v48;
    v113 = v49;
    v50 = sub_30070B0((__int64)&v121);
    v42 = v100;
    v43 = v104;
    v48 = v109;
    v49 = v113;
    v51 = 205 - (!v50 - 1);
  }
  *((_QWORD *)&v92 + 1) = v43;
  *(_QWORD *)&v92 = v42;
  return sub_340EC60(v34, v51, v40, v38, v41, 0, v37, v47, v92, __PAIR128__(v49, v48));
}
