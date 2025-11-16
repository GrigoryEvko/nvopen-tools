// Function: sub_2071870
// Address: 0x2071870
//
void __fastcall sub_2071870(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // ebx
  char v10; // al
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // rax
  int v15; // r8d
  __int64 v16; // r9
  __int64 v17; // r12
  int v18; // r15d
  _OWORD *v19; // rdx
  _OWORD *v20; // rax
  __int64 *v21; // rax
  int v22; // edx
  __int64 v23; // r9
  int v24; // r12d
  __int64 v25; // r13
  __int64 *v26; // r10
  int v27; // ebx
  __int64 *v29; // r14
  _OWORD *v30; // rax
  int v31; // edx
  __int64 *v32; // rcx
  _QWORD *v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 *v36; // rax
  char v37; // r13
  int v38; // r12d
  __int64 v39; // r14
  _OWORD *v40; // rax
  int v41; // edx
  _QWORD *v42; // rcx
  __int64 v43; // r11
  _QWORD *v44; // rdi
  __int64 v45; // rcx
  __int64 v46; // r8
  _QWORD *v47; // rax
  __int64 *v48; // r15
  _OWORD *v49; // r12
  __int64 v50; // r13
  __int64 v51; // rax
  __int64 v52; // r9
  __int64 v53; // r10
  const void ***v54; // rcx
  __int64 v55; // rax
  int v56; // edx
  int v57; // ebx
  bool v58; // zf
  __int64 v59; // rsi
  __int64 *v60; // rbx
  int v61; // edx
  int v62; // r12d
  __int64 *v63; // rax
  _OWORD *v64; // rdi
  _QWORD *v65; // rdi
  _QWORD *v66; // rbx
  int v67; // edx
  int v68; // r12d
  __int64 *v69; // rax
  __int64 *v70; // rax
  __int64 *v71; // r11
  int v72; // edx
  int v73; // r13d
  unsigned int v74; // r12d
  __int64 v75; // r15
  __int64 *v76; // r14
  __int64 v77; // r9
  char v78; // r13
  _OWORD *v79; // rbx
  int v80; // edx
  __int64 *v81; // rax
  __int64 v82; // rbx
  _QWORD *v83; // rdi
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 *v86; // rax
  int v87; // edx
  __int128 v88; // [rsp-10h] [rbp-1E0h]
  int v89; // [rsp+0h] [rbp-1D0h]
  __int64 *v90; // [rsp+8h] [rbp-1C8h]
  unsigned int v91; // [rsp+8h] [rbp-1C8h]
  int v92; // [rsp+10h] [rbp-1C0h]
  __int64 v93; // [rsp+10h] [rbp-1C0h]
  unsigned int v94; // [rsp+18h] [rbp-1B8h]
  __int64 *v95; // [rsp+18h] [rbp-1B8h]
  int v96; // [rsp+18h] [rbp-1B8h]
  char v97; // [rsp+20h] [rbp-1B0h]
  int v98; // [rsp+20h] [rbp-1B0h]
  int v99; // [rsp+28h] [rbp-1A8h]
  int v100; // [rsp+28h] [rbp-1A8h]
  __int64 v101; // [rsp+40h] [rbp-190h]
  _QWORD *v102; // [rsp+40h] [rbp-190h]
  unsigned int v103; // [rsp+40h] [rbp-190h]
  __int64 *v104; // [rsp+40h] [rbp-190h]
  __int64 v106; // [rsp+50h] [rbp-180h]
  __int64 v107; // [rsp+50h] [rbp-180h]
  __int64 v108; // [rsp+50h] [rbp-180h]
  __int64 v109; // [rsp+50h] [rbp-180h]
  __int64 *v110; // [rsp+50h] [rbp-180h]
  __int64 v111; // [rsp+58h] [rbp-178h]
  unsigned int v112; // [rsp+60h] [rbp-170h]
  __int64 v113; // [rsp+60h] [rbp-170h]
  const void ***v114; // [rsp+60h] [rbp-170h]
  char v115; // [rsp+68h] [rbp-168h]
  __int64 v116; // [rsp+68h] [rbp-168h]
  __int64 v117; // [rsp+68h] [rbp-168h]
  __int64 v118; // [rsp+68h] [rbp-168h]
  __int64 v119; // [rsp+98h] [rbp-138h] BYREF
  __int64 v120; // [rsp+A0h] [rbp-130h] BYREF
  int v121; // [rsp+A8h] [rbp-128h]
  unsigned __int8 *v122; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v123; // [rsp+B8h] [rbp-118h]
  _BYTE v124[64]; // [rsp+C0h] [rbp-110h] BYREF
  _BYTE *v125; // [rsp+100h] [rbp-D0h] BYREF
  __int64 v126; // [rsp+108h] [rbp-C8h]
  _BYTE v127[64]; // [rsp+110h] [rbp-C0h] BYREF
  _OWORD *v128; // [rsp+150h] [rbp-80h] BYREF
  __int64 v129; // [rsp+158h] [rbp-78h]
  _OWORD v130[7]; // [rsp+160h] [rbp-70h] BYREF

  v5 = a1;
  v6 = *(_QWORD *)a2;
  if ( *(_BYTE *)(a2 + 16) == 5 )
  {
    v106 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v101 = *(_QWORD *)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
    v111 = *(_QWORD *)v101;
    v7 = sub_1594710(a2);
    v9 = sub_20C7BE0(v6, v7, v7 + 4 * v8, 0);
  }
  else
  {
    v106 = *(_QWORD *)(a2 - 48);
    v101 = *(_QWORD *)(a2 - 24);
    v111 = *(_QWORD *)v101;
    v9 = sub_20C7BE0(v6, *(_QWORD *)(a2 + 56), *(_QWORD *)(a2 + 56) + 4LL * *(unsigned int *)(a2 + 64), 0);
  }
  v10 = *(_BYTE *)(v106 + 16);
  v122 = v124;
  v115 = v10;
  v123 = 0x400000000LL;
  v97 = *(_BYTE *)(v101 + 16);
  v11 = *(_QWORD *)(a1 + 552);
  v12 = *(_QWORD *)(v11 + 16);
  v13 = sub_1E0A0C0(*(_QWORD *)(v11 + 32));
  sub_20C7CE0(v12, v13, v6, &v122, 0, 0);
  v126 = 0x400000000LL;
  v125 = v127;
  v14 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
  sub_20C7CE0(v12, v14, v111, &v125, 0, 0);
  v17 = (unsigned int)v123;
  v129 = 0x400000000LL;
  v112 = v126;
  v18 = v123;
  v128 = v130;
  if ( (unsigned int)v123 > 4 )
  {
    sub_16CD150((__int64)&v128, v130, (unsigned int)v123, 16, v15, v16);
    v20 = v128;
    LODWORD(v129) = v17;
    v19 = &v128[v17];
  }
  else
  {
    LODWORD(v129) = v123;
    v19 = &v130[(unsigned int)v123];
    v20 = v130;
    if ( v19 == v130 )
      goto LABEL_8;
  }
  do
  {
    if ( v20 )
    {
      *(_QWORD *)v20 = 0;
      *((_DWORD *)v20 + 2) = 0;
    }
    ++v20;
  }
  while ( v20 != v19 );
LABEL_8:
  if ( !v18 )
  {
    v65 = *(_QWORD **)(a1 + 552);
    v120 = 0;
    v121 = 0;
    v66 = sub_1D2B300(v65, 0x30u, (__int64)&v120, 1u, 0, v16);
    v68 = v67;
    if ( v120 )
      sub_161E7C0((__int64)&v120, v120);
    v120 = a2;
    v69 = sub_205F5C0(v5 + 8, &v120);
    v64 = v128;
    v69[1] = (__int64)v66;
    *((_DWORD *)v69 + 4) = v68;
    if ( v64 != v130 )
      goto LABEL_30;
    goto LABEL_31;
  }
  v21 = sub_20685E0(a1, (__int64 *)v106, a3, a4, a5);
  v99 = v22;
  v23 = (__int64)v21;
  v24 = v22;
  if ( !v9 )
  {
    v26 = &v120;
    if ( !v112 )
      goto LABEL_16;
    v110 = v21;
    v74 = 0;
    v86 = sub_20685E0(a1, (__int64 *)v101, a3, a4, a5);
    v23 = (__int64)v110;
    v26 = &v120;
    v71 = v86;
    v73 = v87;
    goto LABEL_44;
  }
  v94 = v9;
  v25 = 0;
  v26 = &v120;
  v92 = v18;
  v27 = v9 + v22;
  v29 = v21;
  do
  {
    v31 = v24;
    v32 = v29;
    if ( v115 == 9 )
    {
      v33 = *(_QWORD **)(a1 + 552);
      v107 = (__int64)v26;
      v34 = *(_QWORD *)&v122[v25 * 16];
      v35 = *(_QWORD *)&v122[v25 * 16 + 8];
      v120 = 0;
      v121 = 0;
      v36 = sub_1D2B300(v33, 0x30u, (__int64)v26, v34, v35, v23);
      v26 = (__int64 *)v107;
      v32 = v36;
      if ( v120 )
      {
        v89 = v31;
        v90 = v36;
        sub_161E7C0(v107, v120);
        v31 = v89;
        v32 = v90;
        v26 = (__int64 *)v107;
      }
    }
    ++v24;
    v30 = &v128[v25++];
    *(_QWORD *)v30 = v32;
    *((_DWORD *)v30 + 2) = v31;
  }
  while ( v27 != v24 );
  v23 = (__int64)v29;
  v9 = v94;
  v5 = a1;
  v18 = v92;
  if ( v112 )
  {
    v95 = v26;
    v108 = v23;
    v70 = sub_20685E0(v5, (__int64 *)v101, a3, a4, a5);
    v112 += v9;
    v23 = v108;
    v71 = v70;
    v26 = v95;
    v73 = v72;
    if ( v9 == v112 )
    {
LABEL_50:
      v9 = v112;
      goto LABEL_16;
    }
    v74 = v9;
LABEL_44:
    v96 = v18;
    v75 = v5;
    v76 = v71;
    v93 = v23;
    v77 = v73 - v9;
    v78 = v97;
    do
    {
      v80 = v77 + v74;
      v81 = v76;
      v82 = v74;
      if ( v78 == 9 )
      {
        v83 = *(_QWORD **)(v75 + 552);
        v103 = v77;
        v84 = *(_QWORD *)&v122[16 * v74];
        v85 = *(_QWORD *)&v122[v82 * 16 + 8];
        v109 = (__int64)v26;
        v120 = 0;
        v121 = 0;
        v81 = sub_1D2B300(v83, 0x30u, (__int64)v26, v84, v85, v77);
        v26 = (__int64 *)v109;
        v77 = v103;
        if ( v120 )
        {
          v91 = v103;
          v98 = v80;
          v104 = v81;
          sub_161E7C0(v109, v120);
          v77 = v91;
          v80 = v98;
          v81 = v104;
          v26 = (__int64 *)v109;
        }
      }
      v79 = &v128[v82];
      ++v74;
      *(_QWORD *)v79 = v81;
      *((_DWORD *)v79 + 2) = v80;
    }
    while ( v74 != v112 );
    v5 = v75;
    v23 = v93;
    v18 = v96;
    goto LABEL_50;
  }
LABEL_16:
  if ( v18 != v9 )
  {
    v37 = v115;
    v38 = v99;
    v116 = v5;
    v39 = v23;
    do
    {
      v41 = v38 + v9;
      v42 = (_QWORD *)v39;
      v43 = v9;
      if ( v37 == 9 )
      {
        v113 = (__int64)v26;
        v44 = *(_QWORD **)(v116 + 552);
        v45 = *(_QWORD *)&v122[16 * v9];
        v46 = *(_QWORD *)&v122[v43 * 16 + 8];
        v120 = 0;
        v121 = 0;
        v47 = sub_1D2B300(v44, 0x30u, (__int64)v26, v45, v46, v23);
        v26 = (__int64 *)v113;
        v43 = v9;
        v42 = v47;
        if ( v120 )
        {
          v100 = v41;
          v102 = v47;
          sub_161E7C0(v113, v120);
          v41 = v100;
          v42 = v102;
          v43 = v9;
          v26 = (__int64 *)v113;
        }
      }
      ++v9;
      v40 = &v128[v43];
      *(_QWORD *)v40 = v42;
      *((_DWORD *)v40 + 2) = v41;
    }
    while ( v9 != v18 );
    v5 = v116;
  }
  v48 = *(__int64 **)(v5 + 552);
  v117 = (__int64)v26;
  v49 = v128;
  v50 = (unsigned int)v129;
  v51 = sub_1D25C30((__int64)v48, v122, (unsigned int)v123);
  v53 = v117;
  v120 = 0;
  v54 = (const void ***)v51;
  v55 = *(_QWORD *)v5;
  v57 = v56;
  v58 = *(_QWORD *)v5 == 0;
  v121 = *(_DWORD *)(v5 + 536);
  if ( !v58 && v117 != v55 + 48 )
  {
    v59 = *(_QWORD *)(v55 + 48);
    v120 = v59;
    if ( v59 )
    {
      v114 = v54;
      sub_1623A60(v117, v59, 2);
      v54 = v114;
      v53 = v117;
    }
  }
  *((_QWORD *)&v88 + 1) = v50;
  *(_QWORD *)&v88 = v49;
  v118 = v53;
  v60 = sub_1D36D80(v48, 51, v53, v54, v57, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5, v52, v88);
  v62 = v61;
  v119 = a2;
  v63 = sub_205F5C0(v5 + 8, &v119);
  v63[1] = (__int64)v60;
  *((_DWORD *)v63 + 4) = v62;
  if ( v120 )
    sub_161E7C0(v118, v120);
  v64 = v128;
  if ( v128 != v130 )
LABEL_30:
    _libc_free((unsigned __int64)v64);
LABEL_31:
  if ( v125 != v127 )
    _libc_free((unsigned __int64)v125);
  if ( v122 != v124 )
    _libc_free((unsigned __int64)v122);
}
