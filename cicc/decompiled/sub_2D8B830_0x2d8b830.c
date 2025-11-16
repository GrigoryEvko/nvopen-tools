// Function: sub_2D8B830
// Address: 0x2d8b830
//
__int64 __fastcall sub_2D8B830(__int64 a1, const char *a2)
{
  __int64 *v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 (*v8)(); // rax
  __int64 v9; // rdi
  __int64 (*v10)(); // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 *v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  __m128i v19; // xmm3
  __m128i v20; // xmm4
  unsigned int v21; // eax
  _QWORD **v22; // rbx
  __int64 v23; // rax
  _QWORD *v24; // r15
  unsigned __int64 v25; // r12
  __int64 v26; // rdi
  unsigned int v27; // eax
  _QWORD *v28; // rbx
  _QWORD *v29; // r12
  __int64 v30; // rdi
  __int64 *v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 *v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rdx
  _QWORD *v39; // rax
  _QWORD *v40; // r12
  __int64 v41; // r10
  _QWORD *v42; // rdx
  _QWORD *v43; // rax
  _QWORD *v44; // rax
  unsigned __int64 v45; // rdi
  __int64 *v46; // rax
  __int64 *v47; // rbx
  __int64 *v48; // r12
  __int64 *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // rax
  unsigned int v57; // r12d
  __m128i v59; // xmm5
  __m128i v60; // xmm6
  __m128i v61; // xmm7
  __m128i v62; // xmm0
  __m128i v63; // xmm1
  __int64 v64; // [rsp+0h] [rbp-560h]
  _QWORD **v65; // [rsp+8h] [rbp-558h]
  __m128i v66; // [rsp+10h] [rbp-550h] BYREF
  __m128i v67; // [rsp+20h] [rbp-540h] BYREF
  __m128i v68; // [rsp+30h] [rbp-530h] BYREF
  __m128i v69; // [rsp+40h] [rbp-520h] BYREF
  __m128i v70; // [rsp+50h] [rbp-510h] BYREF
  char v71[8]; // [rsp+60h] [rbp-500h] BYREF
  _QWORD *v72; // [rsp+68h] [rbp-4F8h]
  unsigned int v73; // [rsp+78h] [rbp-4E8h]
  __int64 v74; // [rsp+88h] [rbp-4D8h]
  unsigned int v75; // [rsp+98h] [rbp-4C8h]
  __int64 v76; // [rsp+A8h] [rbp-4B8h]
  unsigned int v77; // [rsp+B8h] [rbp-4A8h]
  __int64 v78; // [rsp+C0h] [rbp-4A0h] BYREF
  __int64 v79; // [rsp+C8h] [rbp-498h]
  __int64 v80; // [rsp+D0h] [rbp-490h]
  __int64 v81; // [rsp+D8h] [rbp-488h]
  __int64 v82; // [rsp+E0h] [rbp-480h]
  __int64 v83; // [rsp+E8h] [rbp-478h]
  __int64 v84; // [rsp+F0h] [rbp-470h]
  __int64 v85; // [rsp+F8h] [rbp-468h]
  __int64 *v86; // [rsp+100h] [rbp-460h]
  _QWORD *v87; // [rsp+108h] [rbp-458h]
  __int64 v88; // [rsp+110h] [rbp-450h]
  __int64 v89; // [rsp+118h] [rbp-448h]
  __int16 v90; // [rsp+120h] [rbp-440h]
  _QWORD v91[3]; // [rsp+128h] [rbp-438h] BYREF
  int v92; // [rsp+140h] [rbp-420h]
  char v93; // [rsp+168h] [rbp-3F8h]
  __int64 v94; // [rsp+178h] [rbp-3E8h]
  char *v95; // [rsp+180h] [rbp-3E0h]
  __int64 v96; // [rsp+188h] [rbp-3D8h]
  int v97; // [rsp+190h] [rbp-3D0h]
  char v98; // [rsp+194h] [rbp-3CCh]
  char v99; // [rsp+198h] [rbp-3C8h] BYREF
  __int64 v100; // [rsp+218h] [rbp-348h]
  __int64 v101; // [rsp+220h] [rbp-340h]
  __int64 v102; // [rsp+228h] [rbp-338h]
  int v103; // [rsp+230h] [rbp-330h]
  __int64 v104; // [rsp+238h] [rbp-328h]
  char *v105; // [rsp+240h] [rbp-320h]
  __int64 v106; // [rsp+248h] [rbp-318h]
  int v107; // [rsp+250h] [rbp-310h]
  char v108; // [rsp+254h] [rbp-30Ch]
  char v109; // [rsp+258h] [rbp-308h] BYREF
  __int64 v110; // [rsp+2D8h] [rbp-288h]
  __int64 v111; // [rsp+2E0h] [rbp-280h]
  __int64 v112; // [rsp+2E8h] [rbp-278h]
  int v113; // [rsp+2F0h] [rbp-270h]
  __int64 v114; // [rsp+2F8h] [rbp-268h]
  __int64 v115; // [rsp+300h] [rbp-260h]
  __int64 v116; // [rsp+308h] [rbp-258h]
  int v117; // [rsp+310h] [rbp-250h]
  _QWORD *v118; // [rsp+318h] [rbp-248h]
  __int64 v119; // [rsp+320h] [rbp-240h]
  _QWORD v120[2]; // [rsp+328h] [rbp-238h] BYREF
  char v121; // [rsp+338h] [rbp-228h] BYREF
  int v122; // [rsp+370h] [rbp-1F0h] BYREF
  __int64 v123; // [rsp+378h] [rbp-1E8h]
  int *v124; // [rsp+380h] [rbp-1E0h]
  int *v125; // [rsp+388h] [rbp-1D8h]
  __int64 v126; // [rsp+390h] [rbp-1D0h]
  __int64 v127; // [rsp+398h] [rbp-1C8h]
  __int64 v128; // [rsp+3A0h] [rbp-1C0h]
  __int64 v129; // [rsp+3A8h] [rbp-1B8h]
  int v130; // [rsp+3B0h] [rbp-1B0h]
  __int64 v131; // [rsp+3B8h] [rbp-1A8h]
  __int64 v132; // [rsp+3C0h] [rbp-1A0h]
  __int64 v133; // [rsp+3C8h] [rbp-198h]
  int v134; // [rsp+3D0h] [rbp-190h]
  char *v135; // [rsp+3D8h] [rbp-188h]
  __int64 v136; // [rsp+3E0h] [rbp-180h]
  char v137; // [rsp+3E8h] [rbp-178h] BYREF
  __int64 v138; // [rsp+3F0h] [rbp-170h]
  __int64 v139; // [rsp+3F8h] [rbp-168h]
  char v140; // [rsp+400h] [rbp-160h]
  __int64 v141; // [rsp+408h] [rbp-158h]
  char *v142; // [rsp+410h] [rbp-150h]
  __int64 v143; // [rsp+418h] [rbp-148h]
  int v144; // [rsp+420h] [rbp-140h]
  char v145; // [rsp+424h] [rbp-13Ch]
  char v146; // [rsp+428h] [rbp-138h] BYREF

  v2 = *(__int64 **)(a1 + 8);
  v3 = *v2;
  v4 = v2[1];
  if ( v3 == v4 )
LABEL_78:
    BUG();
  while ( *(_UNKNOWN **)v3 != &unk_5027190 )
  {
    v3 += 16;
    if ( v4 == v3 )
      goto LABEL_78;
  }
  v6 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v3 + 8) + 104LL))(*(_QWORD *)(v3 + 8), &unk_5027190);
  v79 = 0;
  v7 = *(_QWORD *)(v6 + 256);
  v80 = 0;
  v81 = 0;
  v78 = v7;
  v82 = 0;
  v83 = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91[0] = 0;
  v92 = 128;
  v91[1] = sub_C7D670(0x2000, 8);
  sub_2D69A40((__int64)v91);
  v93 = 0;
  v95 = &v99;
  v105 = &v109;
  v118 = v120;
  v120[0] = &v121;
  v120[1] = 0x200000000LL;
  v124 = &v122;
  v125 = &v122;
  v94 = 0;
  v96 = 16;
  v97 = 0;
  v98 = 1;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v106 = 16;
  v107 = 0;
  v108 = 1;
  v110 = 0;
  v111 = 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v119 = 0;
  v122 = 0;
  v123 = 0;
  v126 = 0;
  v127 = 0;
  v135 = &v137;
  v128 = 0;
  v129 = 0;
  v130 = 0;
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v136 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = &v146;
  v143 = 32;
  v144 = 0;
  v145 = 1;
  v138 = sub_B2BEC0((__int64)a2);
  v8 = *(__int64 (**)())(*(_QWORD *)v7 + 16LL);
  if ( v8 == sub_23CE270 )
  {
    v79 = 0;
    BUG();
  }
  v79 = ((__int64 (__fastcall *)(__int64, const char *))v8)(v7, a2);
  v9 = v79;
  v10 = *(__int64 (**)())(*(_QWORD *)v79 + 144LL);
  v11 = 0;
  if ( v10 != sub_2C8F680 )
  {
    v11 = ((__int64 (__fastcall *)(__int64))v10)(v79);
    v9 = v79;
  }
  v80 = v11;
  v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v9 + 200LL))(v9);
  v13 = *(__int64 **)(a1 + 8);
  v81 = v12;
  v14 = *v13;
  v15 = v13[1];
  if ( v14 == v15 )
LABEL_74:
    BUG();
  while ( *(_UNKNOWN **)v14 != &unk_4F6D3F0 )
  {
    v14 += 16;
    if ( v15 == v14 )
      goto LABEL_74;
  }
  v16 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v14 + 8) + 104LL))(*(_QWORD *)(v14 + 8), &unk_4F6D3F0);
  sub_BBB200((__int64)v71);
  sub_983BD0((__int64)&v66, v16 + 176, (__int64)a2);
  v64 = v16 + 408;
  if ( *(_BYTE *)(v16 + 488) )
  {
    v17 = _mm_loadu_si128(&v67);
    v18 = _mm_loadu_si128(&v68);
    v19 = _mm_loadu_si128(&v69);
    v20 = _mm_loadu_si128(&v70);
    *(__m128i *)(v16 + 408) = _mm_loadu_si128(&v66);
    *(__m128i *)(v16 + 424) = v17;
    *(__m128i *)(v16 + 440) = v18;
    *(__m128i *)(v16 + 456) = v19;
    *(__m128i *)(v16 + 472) = v20;
  }
  else
  {
    v59 = _mm_loadu_si128(&v66);
    v60 = _mm_loadu_si128(&v67);
    *(_BYTE *)(v16 + 488) = 1;
    v61 = _mm_loadu_si128(&v68);
    v62 = _mm_loadu_si128(&v69);
    v63 = _mm_loadu_si128(&v70);
    *(__m128i *)(v16 + 408) = v59;
    *(__m128i *)(v16 + 424) = v60;
    *(__m128i *)(v16 + 440) = v61;
    *(__m128i *)(v16 + 456) = v62;
    *(__m128i *)(v16 + 472) = v63;
  }
  sub_C7D6A0(v76, 24LL * v77, 8);
  v21 = v75;
  if ( v75 )
  {
    v22 = (_QWORD **)(v74 + 8);
    v65 = (_QWORD **)(v74 + 32LL * v75);
    while ( 1 )
    {
      v23 = (__int64)*(v22 - 1);
      if ( v23 != -4096 && v23 != -8192 )
      {
        v24 = *v22;
        while ( v24 != v22 )
        {
          v25 = (unsigned __int64)v24;
          v24 = (_QWORD *)*v24;
          v26 = *(_QWORD *)(v25 + 24);
          if ( v26 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v26 + 8LL))(v26);
          j_j___libc_free_0(v25);
        }
      }
      if ( v65 == v22 + 3 )
        break;
      v22 += 4;
    }
    v21 = v75;
  }
  sub_C7D6A0(v74, 32LL * v21, 8);
  v27 = v73;
  if ( v73 )
  {
    v28 = v72;
    v29 = &v72[2 * v73];
    do
    {
      if ( *v28 != -4096 && *v28 != -8192 )
      {
        v30 = v28[1];
        if ( v30 )
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v30 + 8LL))(v30);
      }
      v28 += 2;
    }
    while ( v29 != v28 );
    v27 = v73;
  }
  sub_C7D6A0((__int64)v72, 16LL * v27, 8);
  v31 = *(__int64 **)(a1 + 8);
  v84 = v64;
  v32 = *v31;
  v33 = v31[1];
  if ( v32 == v33 )
LABEL_75:
    BUG();
  while ( *(_UNKNOWN **)v32 != &unk_4F89C28 )
  {
    v32 += 16;
    if ( v33 == v32 )
      goto LABEL_75;
  }
  v34 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v32 + 8) + 104LL))(*(_QWORD *)(v32 + 8), &unk_4F89C28);
  v35 = sub_DFED00(v34, (__int64)a2);
  v36 = *(__int64 **)(a1 + 8);
  v82 = v35;
  v37 = *v36;
  v38 = v36[1];
  if ( v37 == v38 )
LABEL_76:
    BUG();
  while ( *(_UNKNOWN **)v37 != &unk_4F875EC )
  {
    v37 += 16;
    if ( v38 == v37 )
      goto LABEL_76;
  }
  v85 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v37 + 8) + 104LL))(*(_QWORD *)(v37 + 8), &unk_4F875EC)
      + 176;
  v39 = (_QWORD *)sub_22077B0(0x118u);
  v40 = v39;
  if ( v39 )
  {
    v41 = v85;
    *v39 = 0;
    v42 = v39 + 21;
    v43 = v39 + 13;
    *(v43 - 12) = 0;
    *(v43 - 11) = 0;
    *((_DWORD *)v43 - 20) = 0;
    *(v43 - 9) = 0;
    *(v43 - 8) = 0;
    *(v43 - 7) = 0;
    *((_DWORD *)v43 - 12) = 0;
    *(v43 - 5) = 0;
    *(v43 - 4) = 0;
    *(v43 - 3) = 0;
    *(v43 - 2) = 0;
    *(v43 - 1) = 1;
    do
    {
      if ( v43 )
        *v43 = -4096;
      v43 += 2;
    }
    while ( v43 != v42 );
    v44 = v40 + 23;
    v40[21] = 0;
    v40[22] = 1;
    do
    {
      if ( v44 )
      {
        *v44 = -4096;
        *((_DWORD *)v44 + 2) = 0x7FFFFFFF;
      }
      v44 += 3;
    }
    while ( v44 != v40 + 35 );
    sub_FF9360(v40, (__int64)a2, v41, 0, 0, 0);
  }
  v45 = (unsigned __int64)v87;
  v87 = v40;
  if ( v45 )
  {
    sub_2D59CD0(v45);
    v40 = v87;
  }
  v46 = (__int64 *)sub_22077B0(8u);
  v47 = v46;
  if ( v46 )
    sub_FE7FB0(v46, a2, (__int64)v40, v85);
  v48 = v86;
  v86 = v47;
  if ( v48 )
  {
    sub_FDC110(v48);
    j_j___libc_free_0((unsigned __int64)v48);
  }
  v49 = *(__int64 **)(a1 + 8);
  v50 = *v49;
  v51 = v49[1];
  if ( v50 == v51 )
LABEL_77:
    BUG();
  while ( *(_UNKNOWN **)v50 != &unk_4F87C64 )
  {
    v50 += 16;
    if ( v51 == v50 )
      goto LABEL_77;
  }
  v52 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v50 + 8) + 104LL))(*(_QWORD *)(v50 + 8), &unk_4F87C64);
  v53 = *(_QWORD *)(a1 + 8);
  v88 = *(_QWORD *)(v52 + 176);
  v54 = sub_B82360(v53, (__int64)&unk_501695C);
  if ( v54 && (v55 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v54 + 104LL))(v54, &unk_501695C)) != 0 )
    v56 = sub_2D514A0(v55);
  else
    v56 = 0;
  v83 = v56;
  v57 = sub_2D88660((__int64)&v78, (__int64)a2);
  sub_2D5C240((__int64)&v78);
  return v57;
}
