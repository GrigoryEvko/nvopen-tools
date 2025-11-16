// Function: sub_E7A190
// Address: 0xe7a190
//
__int64 __fastcall sub_E7A190(__int64 *a1)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 result; // rax
  void (__fastcall *v7)(__int64 *, __int64, _QWORD); // r8
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r15
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // rbx
  char v18; // dl
  int v19; // r10d
  int v20; // edi
  unsigned int v21; // r8d
  int v22; // ecx
  unsigned int v23; // r12d
  int v24; // esi
  int v25; // r13d
  int v26; // eax
  int v27; // r15d
  __int64 *v28; // r15
  __int64 v29; // rdi
  __int64 v30; // r13
  __int64 v31; // rax
  __int64 v32; // r13
  __int64 v33; // rax
  __int64 v34; // r13
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // rbx
  __int64 v38; // r12
  __int64 v39; // rbx
  __int64 v40; // rsi
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r12
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  char v48; // al
  __int64 v49; // r15
  unsigned int v50; // r13d
  __int64 v51; // rax
  __int64 v52; // r12
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r15
  unsigned int v58; // r12d
  __int64 v59; // rsi
  __int64 v60; // rax
  __int64 v61; // r13
  __int64 v62; // rsi
  void (__fastcall *v63)(__int64 *, char *); // r15
  char *v64; // rax
  __int64 v65; // rax
  __int64 v66; // rsi
  _QWORD *v67; // rsi
  __int64 v68; // rdx
  const char *v69; // rsi
  __int64 v70; // rax
  __int64 v71; // r13
  __int64 v72; // r15
  __int64 v73; // rax
  _QWORD *v74; // rax
  __int64 v75; // r15
  __int64 v76; // r13
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rbx
  __int64 v80; // rdx
  unsigned int v81; // r12d
  void (__fastcall *v82)(__int64 *, __int64, _QWORD); // rax
  __int64 v83; // rsi
  __int64 v84; // rdx
  __int64 v85; // rcx
  __int64 v86; // r8
  __int64 v87; // rax
  void (*v88)(); // rcx
  __int64 v89; // rcx
  __int64 v90; // r8
  __int64 *v91; // r13
  __int64 v92; // rdi
  __int64 v93; // r12
  __int64 v94; // r15
  __int64 v95; // r15
  __int64 v96; // rax
  __int64 v97; // r15
  __int64 v98; // rax
  __int64 v99; // r15
  __int64 v100; // rax
  __int64 v101; // rsi
  __int64 v102; // rcx
  __int64 v103; // r8
  __int64 *v104; // r15
  __int64 *v105; // r13
  __int64 v106; // rdi
  __int64 v107; // r15
  __int64 v108; // rax
  __int64 v109; // r15
  __int64 v110; // rax
  __int64 v111; // r15
  __int64 v112; // rcx
  __int64 v113; // r8
  int v114; // [rsp+8h] [rbp-A8h]
  __int64 v115; // [rsp+8h] [rbp-A8h]
  __int64 v116; // [rsp+10h] [rbp-A0h]
  __int64 v117; // [rsp+18h] [rbp-98h]
  char v118; // [rsp+20h] [rbp-90h]
  unsigned int v119; // [rsp+20h] [rbp-90h]
  __int64 *v120; // [rsp+20h] [rbp-90h]
  unsigned int v121; // [rsp+28h] [rbp-88h]
  __int64 *v122; // [rsp+28h] [rbp-88h]
  __int64 v123; // [rsp+28h] [rbp-88h]
  __int64 v124; // [rsp+30h] [rbp-80h]
  __int64 v125; // [rsp+30h] [rbp-80h]
  __int64 v126; // [rsp+30h] [rbp-80h]
  __int64 *v127; // [rsp+30h] [rbp-80h]
  __int64 v128; // [rsp+30h] [rbp-80h]
  int v129; // [rsp+38h] [rbp-78h]
  unsigned int v130; // [rsp+38h] [rbp-78h]
  __int64 v131; // [rsp+38h] [rbp-78h]
  __int64 v132; // [rsp+38h] [rbp-78h]
  __int64 v133; // [rsp+38h] [rbp-78h]
  __int64 v134; // [rsp+38h] [rbp-78h]
  __int64 v135; // [rsp+38h] [rbp-78h]
  __int64 v136; // [rsp+38h] [rbp-78h]
  int v137; // [rsp+4Ch] [rbp-64h] BYREF
  unsigned int *v138[4]; // [rsp+50h] [rbp-60h] BYREF
  char v139; // [rsp+70h] [rbp-40h]
  char v140; // [rsp+71h] [rbp-3Fh]

  v2 = a1[1];
  v118 = *(_BYTE *)(*(_QWORD *)(v2 + 152) + 348LL);
  if ( v118 )
  {
    v3 = (*(__int64 (__fastcall **)(__int64 *, _QWORD))(*a1 + 848))(a1, 0);
    v4 = a1[1];
    v117 = v3;
  }
  else
  {
    v117 = 0;
    v4 = a1[1];
  }
  sub_E65B10(v4, (__int64)a1);
  v5 = a1[1];
  result = *(unsigned int *)(v5 + 1840);
  if ( !(_DWORD)result )
    return result;
  v7 = *(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176);
  v8 = *(_QWORD *)(*(_QWORD *)(v2 + 168) + 88LL);
  if ( (_DWORD)result != 1 && *(_WORD *)(v5 + 1904) > 2u )
  {
    v7(a1, v8, 0);
    v118 = 1;
LABEL_7:
    v12 = sub_E6C430(v2, v8, v9, v10, v11);
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 208))(a1, v12, 0);
    v13 = *(_QWORD *)(*(_QWORD *)(v2 + 168) + 80LL);
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, v13, 0);
    v116 = sub_E6C430(v2, v13, v14, v15, v16);
    (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 208))(a1, v116, 0);
    goto LABEL_8;
  }
  v7(a1, v8, 0);
  if ( v118 )
  {
    v118 = 0;
    goto LABEL_7;
  }
  v12 = 0;
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 176))(a1, *(_QWORD *)(*(_QWORD *)(v2 + 168) + 80LL), 0);
  v116 = 0;
LABEL_8:
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 176))(a1, *(_QWORD *)(*(_QWORD *)(v2 + 168) + 152LL), 0);
  v17 = a1[1];
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 176))(a1, *(_QWORD *)(*(_QWORD *)(v17 + 168) + 152LL), 0);
  v18 = *(_BYTE *)(v17 + 1906);
  if ( v18 )
  {
    if ( v18 != 1 )
      goto LABEL_99;
    v19 = 24;
    v20 = 24;
    v21 = 8;
    v22 = 12;
  }
  else
  {
    v19 = 12;
    v20 = 12;
    v21 = 4;
    v22 = 4;
  }
  v23 = *(_DWORD *)(*(_QWORD *)(v17 + 152) + 8LL);
  v124 = *(_QWORD *)(v17 + 152);
  v24 = 2 * v23;
  v25 = 2 * v23 - (v19 & (2 * v23 - 1));
  if ( 2 * v23 == v25 )
    v25 = 0;
  else
    v20 = v25 + v19;
  v26 = v24 + v20 + v24 * *(_DWORD *)(v17 + 1840);
  if ( v18 == 1 )
  {
    v129 = v24 + v20 + v24 * *(_DWORD *)(v17 + 1840);
    v114 = v22;
    v121 = v21;
    (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 536))(a1, 0xFFFFFFFFLL, 4);
    v22 = v114;
    v26 = v129;
    v21 = v121;
  }
  v130 = v21;
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 536))(a1, (unsigned int)(v26 - v22), v21);
  (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 536))(a1, 2, 2);
  if ( v12 )
    sub_E9A500(a1, v12, v130, *(unsigned __int8 *)(v124 + 259));
  else
    (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 536))(a1, 0, v130);
  v27 = 0;
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, (int)v23, 1);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
  if ( v25 > 0 )
  {
    do
    {
      ++v27;
      (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
    }
    while ( v25 != v27 );
  }
  v28 = *(__int64 **)(v17 + 1832);
  v122 = &v28[*(unsigned int *)(v17 + 1840)];
  while ( v122 != v28 )
  {
    v29 = *v28++;
    v30 = *(_QWORD *)(v29 + 16);
    v131 = sub_E92830(v29, v17);
    v125 = sub_E808D0(v30, 0, v17, 0);
    v132 = sub_E808D0(v131, 0, v17, 0);
    v31 = sub_E808D0(v30, 0, v17, 0);
    v32 = sub_E81A00(18, v132, v31, v17, 0);
    v33 = sub_E81A90(0, v17, 0, 0);
    v34 = sub_E81A00(18, v32, v33, v17, 0);
    sub_E9A5B0(a1, v125, v23, 0);
    sub_E71DA0(a1, v34, v23, v35, v36);
  }
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 536))(a1, 0, v23);
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 536))(a1, 0, v23);
  if ( v118 )
  {
    v79 = a1[1];
    v80 = *(_QWORD *)(v79 + 168);
    v81 = *(_DWORD *)(*(_QWORD *)(v79 + 152) + 8LL);
    v82 = *(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176);
    if ( *(_WORD *)(v79 + 1904) <= 4u )
    {
      v82(a1, *(_QWORD *)(v80 + 160), 0);
      v140 = 1;
      v138[0] = (unsigned int *)"debug_ranges_start";
      v139 = 3;
      v123 = sub_E6C380(v79, (__int64 *)v138, 1, v102, v103);
      (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 208))(a1, v123, 0);
      v104 = *(__int64 **)(v79 + 1832);
      v105 = v104;
      v120 = &v104[*(unsigned int *)(v79 + 1840)];
      if ( v104 != v120 )
      {
        do
        {
          v106 = *v105++;
          v107 = *(_QWORD *)(v106 + 16);
          v128 = sub_E92830(v106, v79);
          v135 = sub_E808D0(v107, 0, v79, 0);
          sub_E99280(a1, (int)v81, 255);
          sub_E9A5B0(a1, v135, v81, 0);
          v136 = sub_E808D0(v128, 0, v79, 0);
          v108 = sub_E808D0(v107, 0, v79, 0);
          v109 = sub_E81A00(18, v136, v108, v79, 0);
          v110 = sub_E81A90(0, v79, 0, 0);
          v111 = sub_E81A00(18, v109, v110, v79, 0);
          (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 536))(a1, 0, v81);
          sub_E71DA0(a1, v111, v81, v112, v113);
        }
        while ( v120 != v105 );
      }
      (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 536))(a1, 0, v81);
      (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 536))(a1, 0, v81);
    }
    else
    {
      v83 = *(_QWORD *)(v80 + 320);
      v82(a1, v83, 0);
      v115 = sub_E75E60(a1, v83, v84, v85, v86);
      v87 = *a1;
      v88 = *(void (**)())(*a1 + 120);
      v140 = 1;
      v138[0] = (unsigned int *)"Offset entry count";
      v139 = 3;
      if ( v88 != nullsub_98 )
      {
        ((void (__fastcall *)(__int64 *, unsigned int **, __int64))v88)(a1, v138, 1);
        v87 = *a1;
      }
      (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(v87 + 536))(a1, 0, 4);
      v140 = 1;
      v139 = 3;
      v138[0] = (unsigned int *)"debug_rnglist0_start";
      v123 = sub_E6C380(v79, (__int64 *)v138, 1, v89, v90);
      (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 208))(a1, v123, 0);
      v91 = *(__int64 **)(v79 + 1832);
      v127 = &v91[*(unsigned int *)(v79 + 1840)];
      if ( v91 != v127 )
      {
        v119 = v81;
        do
        {
          v92 = *v91++;
          v93 = *(_QWORD *)(v92 + 16);
          v94 = sub_E92830(v92, v79);
          v134 = sub_E808D0(v93, 0, v79, 0);
          v95 = sub_E808D0(v94, 0, v79, 0);
          v96 = sub_E808D0(v93, 0, v79, 0);
          v97 = sub_E81A00(18, v95, v96, v79, 0);
          v98 = sub_E81A90(0, v79, 0, 0);
          v99 = sub_E81A00(18, v97, v98, v79, 0);
          (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 536))(a1, 7, 1);
          sub_E9A5B0(a1, v134, v119, 0);
          (*(void (__fastcall **)(__int64 *, __int64))(*a1 + 568))(a1, v99);
        }
        while ( v127 != v91 );
      }
      (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
      (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 208))(a1, v115, 0);
    }
  }
  else
  {
    v123 = 0;
  }
  v37 = a1[1];
  v38 = 23;
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 176))(a1, *(_QWORD *)(*(_QWORD *)(v37 + 168) + 80LL), 0);
  sub_E98EB0(a1, 1, 0);
  sub_E98EB0(a1, 17, 0);
  (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 536))(a1, 1, 1);
  if ( *(_WORD *)(v37 + 1904) <= 3u )
    v38 = (*(_BYTE *)(v37 + 1906) == 1) + 6LL;
  sub_E98EB0(a1, 16, 0);
  sub_E98EB0(a1, v38, 0);
  if ( *(_DWORD *)(v37 + 1840) > 1u && *(_WORD *)(v37 + 1904) > 2u )
  {
    sub_E98EB0(a1, 85, 0);
    sub_E98EB0(a1, v38, 0);
  }
  else
  {
    sub_E98EB0(a1, 17, 0);
    sub_E98EB0(a1, 1, 0);
    sub_E98EB0(a1, 18, 0);
    sub_E98EB0(a1, 1, 0);
  }
  sub_E98EB0(a1, 3, 0);
  sub_E98EB0(a1, 8, 0);
  if ( *(_QWORD *)(v37 + 1536) )
  {
    sub_E98EB0(a1, 27, 0);
    sub_E98EB0(a1, 8, 0);
  }
  if ( *(_QWORD *)(v37 + 1880) )
  {
    sub_E98EB0(a1, 16354, 0);
    sub_E98EB0(a1, 8, 0);
  }
  sub_E98EB0(a1, 37, 0);
  sub_E98EB0(a1, 8, 0);
  sub_E98EB0(a1, 19, 0);
  sub_E98EB0(a1, 5, 0);
  sub_E98EB0(a1, 0, 0);
  sub_E98EB0(a1, 0, 0);
  sub_E98EB0(a1, 2, 0);
  sub_E98EB0(a1, 10, 0);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
  sub_E98EB0(a1, 3, 0);
  sub_E98EB0(a1, 8, 0);
  sub_E98EB0(a1, 58, 0);
  sub_E98EB0(a1, 6, 0);
  sub_E98EB0(a1, 59, 0);
  sub_E98EB0(a1, 6, 0);
  sub_E98EB0(a1, 17, 0);
  sub_E98EB0(a1, 1, 0);
  sub_E98EB0(a1, 0, 0);
  sub_E98EB0(a1, 0, 0);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
  v39 = a1[1];
  v40 = *(_QWORD *)(*(_QWORD *)(v39 + 168) + 88LL);
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 176))(a1, v40, 0);
  v44 = sub_E6C430(v39, v40, v41, v42, v43);
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 208))(a1, v44, 0);
  v133 = sub_E6C430(v39, v44, v45, v46, v47);
  v48 = *(_BYTE *)(v39 + 1906);
  if ( !v48 )
  {
    v49 = 4;
    v50 = 4;
    goto LABEL_33;
  }
  if ( v48 != 1 )
LABEL_99:
    BUG();
  v49 = 12;
  v50 = 8;
  (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 536))(a1, 0xFFFFFFFFLL, 4);
LABEL_33:
  v126 = sub_E808D0(v133, 0, v39, 0);
  v51 = sub_E808D0(v44, 0, v39, 0);
  v52 = sub_E81A00(18, v126, v51, v39, 0);
  v53 = sub_E81A90(v49, v39, 0, 0);
  v54 = sub_E81A00(18, v52, v53, v39, 0);
  sub_E71DA0(a1, v54, v50, v55, v56);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, *(unsigned __int16 *)(v39 + 1904), 2);
  v57 = *(_QWORD *)(v39 + 152);
  v58 = *(_DWORD *)(v57 + 8);
  if ( *(_WORD *)(v39 + 1904) > 4u )
  {
    (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 536))(a1, 1, 1);
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, (int)v58, 1);
  }
  if ( v116 )
    sub_E9A500(a1, v116, v50, *(unsigned __int8 *)(v57 + 259));
  else
    (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 536))(a1, 0, v50);
  if ( *(_WORD *)(v39 + 1904) <= 4u )
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, (int)v58, 1);
  sub_E98EB0(a1, 1, 0);
  if ( v117 )
  {
    sub_E9A500(a1, v117, v50, *(unsigned __int8 *)(v57 + 259));
    v59 = v123;
    if ( v123 )
    {
LABEL_41:
      sub_E9A500(a1, v59, v50, 0);
      goto LABEL_42;
    }
  }
  else
  {
    (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 536))(a1, 0, v50);
    v59 = v123;
    if ( v123 )
      goto LABEL_41;
  }
  v74 = *(_QWORD **)(v39 + 1832);
  v75 = *(_QWORD *)(*v74 + 16LL);
  v76 = sub_E92830(*v74, v39);
  v77 = sub_E808D0(v75, 0, v39, 0);
  sub_E9A5B0(a1, v77, v58, 0);
  v78 = sub_E808D0(v76, 0, v39, 0);
  sub_E9A5B0(a1, v78, v58, 0);
LABEL_42:
  v137 = 0;
  v60 = *(_QWORD *)(v39 + 1744);
  v61 = v39 + 1736;
  if ( !v60 )
  {
    v62 = v39 + 1736;
LABEL_46:
    v138[0] = (unsigned int *)&v137;
    v62 = sub_E7A040((_QWORD *)(v39 + 1728), v62, v138);
    goto LABEL_47;
  }
  do
  {
    v62 = v60;
    v60 = *(_QWORD *)(v60 + 16);
  }
  while ( v60 );
  if ( v61 == v62 || *(_DWORD *)(v62 + 32) )
    goto LABEL_46;
LABEL_47:
  if ( *(_DWORD *)(v62 + 56) )
  {
    (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 512))(
      a1,
      **(_QWORD **)(v62 + 48),
      *(_QWORD *)(*(_QWORD *)(v62 + 48) + 8LL));
    v63 = *(void (__fastcall **)(__int64 *, char *))(*a1 + 512);
    v64 = sub_C81260(0);
    v63(a1, v64);
  }
  v137 = 0;
  v65 = *(_QWORD *)(v39 + 1744);
  if ( !v65 )
  {
    v66 = v39 + 1736;
LABEL_53:
    v138[0] = (unsigned int *)&v137;
    v66 = sub_E7A040((_QWORD *)(v39 + 1728), v66, v138);
    goto LABEL_54;
  }
  do
  {
    v66 = v65;
    v65 = *(_QWORD *)(v65 + 16);
  }
  while ( v65 );
  if ( v66 == v61 || *(_DWORD *)(v66 + 32) )
    goto LABEL_53;
LABEL_54:
  if ( !*(_DWORD *)(v66 + 168) )
  {
    v137 = 0;
    v100 = *(_QWORD *)(v39 + 1744);
    if ( v100 )
    {
      do
      {
        v101 = v100;
        v100 = *(_QWORD *)(v100 + 16);
      }
      while ( v100 );
      if ( v101 != v61 && !*(_DWORD *)(v101 + 32) )
        goto LABEL_90;
    }
    else
    {
      v101 = v39 + 1736;
    }
    v138[0] = (unsigned int *)&v137;
    v101 = sub_E7A040((_QWORD *)(v39 + 1728), v101, v138);
LABEL_90:
    v67 = (_QWORD *)(v101 + 472);
    goto LABEL_56;
  }
  v67 = (_QWORD *)(*(_QWORD *)(v66 + 160) + 80LL);
LABEL_56:
  (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 512))(a1, *v67, v67[1]);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
  if ( *(_QWORD *)(v39 + 1536) )
  {
    (*(void (__fastcall **)(__int64 *, _QWORD))(*a1 + 512))(a1, *(_QWORD *)(v39 + 1528));
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
  }
  if ( *(_QWORD *)(v39 + 1880) )
  {
    (*(void (__fastcall **)(__int64 *, _QWORD))(*a1 + 512))(a1, *(_QWORD *)(v39 + 1872));
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
  }
  v68 = *(_QWORD *)(v39 + 1896);
  v69 = *(const char **)(v39 + 1888);
  if ( !v68 )
  {
    v69 = "llvm-mc (based on LLVM 20.0.0)";
    v68 = 30;
  }
  (*(void (__fastcall **)(__int64 *, const char *, __int64))(*a1 + 512))(a1, v69, v68);
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
  (*(void (__fastcall **)(__int64 *, __int64, __int64))(*a1 + 536))(a1, 32769, 2);
  v70 = a1[1];
  v71 = *(_QWORD *)(v70 + 1856);
  v72 = *(_QWORD *)(v70 + 1848);
  while ( v71 != v72 )
  {
    v72 += 32;
    sub_E98EB0(a1, 2, 0);
    (*(void (__fastcall **)(__int64 *, _QWORD, _QWORD))(*a1 + 512))(a1, *(_QWORD *)(v72 - 32), *(_QWORD *)(v72 - 24));
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, *(unsigned int *)(v72 - 16), 4);
    (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, *(unsigned int *)(v72 - 12), 4);
    v73 = sub_E808D0(*(_QWORD *)(v72 - 8), 0, v39, 0);
    sub_E9A5B0(a1, v73, v58, 0);
  }
  (*(void (__fastcall **)(__int64 *, _QWORD, __int64))(*a1 + 536))(a1, 0, 1);
  return (*(__int64 (__fastcall **)(__int64 *, __int64, _QWORD))(*a1 + 208))(a1, v133, 0);
}
