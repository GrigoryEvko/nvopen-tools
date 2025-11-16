// Function: sub_20548A0
// Address: 0x20548a0
//
void __fastcall sub_20548A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _QWORD *a6,
        __m128i a7,
        double a8,
        __m128i a9,
        __int64 a10)
{
  int v10; // r15d
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int8 v14; // r14
  __int64 *v15; // r13
  __int64 *v16; // rax
  int v17; // edx
  const void ***v18; // r14
  int v19; // edx
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 v22; // r9
  __int64 v23; // rdx
  int v24; // eax
  __int64 *v25; // r14
  __int64 v27; // r15
  __int64 v28; // r13
  __int64 v30; // rcx
  __int64 v32; // rdx
  __int64 (__fastcall *v33)(__int64, __int64, __int64, _QWORD, _QWORD); // r12
  __int64 v34; // rax
  __int64 v35; // r13
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // rdx
  const void **v39; // r13
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 *v45; // rax
  unsigned int v46; // edx
  int v47; // eax
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  __int64 *v50; // rax
  unsigned __int64 v51; // rdx
  unsigned __int64 v52; // r13
  __int64 v53; // r12
  __int64 *v54; // r14
  __int128 v55; // rax
  __int64 *v56; // rax
  __int64 *v57; // r12
  __int16 *v58; // rdx
  __int64 v59; // rdx
  __int64 (__fastcall *v60)(__int64, __int64, __int64, _QWORD, _QWORD); // r13
  __int64 v61; // rax
  __int64 v62; // r14
  __int64 v63; // rax
  int v64; // eax
  __int64 v65; // rdx
  const void **v66; // r14
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 *v72; // r12
  __int64 v73; // rsi
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  _QWORD *v78; // r14
  __int64 v79; // rdx
  __int64 v80; // r15
  __int64 v81; // rcx
  __int64 v82; // r8
  int v83; // r9d
  __int64 *v84; // rax
  __int16 *v85; // rdx
  __int64 *v86; // r14
  unsigned __int64 v87; // rdx
  __int64 v88; // rdx
  __int64 v89; // rcx
  __int64 v90; // r8
  __int64 v91; // r9
  __int64 *v92; // r14
  __int128 v93; // rax
  int v94; // edx
  __int64 v95; // r12
  __int64 v97; // rdx
  __int64 (__fastcall *v98)(__int64, __int64, __int64, __int64, _QWORD); // r12
  __int64 v99; // rax
  __int64 v100; // rax
  int v101; // eax
  __int64 v102; // rdx
  const void **v103; // r15
  __int64 v104; // rcx
  __int64 v105; // r8
  __int64 v106; // r9
  __int64 v107; // rax
  __int64 v108; // rdx
  unsigned int v109; // edx
  __int128 v110; // [rsp-10h] [rbp-110h]
  __int64 *v111; // [rsp+0h] [rbp-100h]
  __int64 v112; // [rsp+0h] [rbp-100h]
  __int128 v113; // [rsp+0h] [rbp-100h]
  int v114; // [rsp+10h] [rbp-F0h]
  __int128 v115; // [rsp+10h] [rbp-F0h]
  __int128 v116; // [rsp+10h] [rbp-F0h]
  __int64 v117; // [rsp+20h] [rbp-E0h]
  __int64 v118; // [rsp+20h] [rbp-E0h]
  unsigned int v119; // [rsp+20h] [rbp-E0h]
  __int128 v120; // [rsp+20h] [rbp-E0h]
  unsigned __int64 v121; // [rsp+20h] [rbp-E0h]
  __int64 v122; // [rsp+20h] [rbp-E0h]
  __int16 *v123; // [rsp+28h] [rbp-D8h]
  int v124; // [rsp+3Ch] [rbp-C4h]
  unsigned __int8 v126; // [rsp+48h] [rbp-B8h]
  __int64 v127; // [rsp+48h] [rbp-B8h]
  int v128; // [rsp+50h] [rbp-B0h]
  __int128 v129; // [rsp+50h] [rbp-B0h]
  __int64 *v130; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v131; // [rsp+58h] [rbp-A8h]
  __int64 v133; // [rsp+A0h] [rbp-60h] BYREF
  int v134; // [rsp+A8h] [rbp-58h]
  __int64 *v135; // [rsp+B0h] [rbp-50h] BYREF
  int v136; // [rsp+B8h] [rbp-48h]
  _QWORD *v137; // [rsp+C0h] [rbp-40h]
  __int64 v138; // [rsp+C8h] [rbp-38h]

  v10 = a5;
  v11 = a2;
  v12 = *(_QWORD *)a1;
  v13 = *(unsigned int *)(a1 + 536);
  v124 = a4;
  v133 = 0;
  v134 = v13;
  if ( v12 )
  {
    v13 = v12 + 48;
    if ( &v133 != (__int64 *)(v12 + 48) )
    {
      a2 = *(_QWORD *)(v12 + 48);
      v133 = a2;
      if ( a2 )
        sub_1623A60((__int64)&v133, a2, 2);
    }
  }
  v14 = *(_BYTE *)(v11 + 44);
  v15 = *(__int64 **)(a1 + 552);
  v126 = v14;
  v16 = sub_2051DF0((__int64 *)a1, *(double *)a7.m128i_i64, a8, a9, a2, v13, a4, a5, (int)a6);
  v114 = v17;
  v117 = v14;
  v111 = v16;
  v18 = (const void ***)sub_1D252B0((__int64)v15, v14, 0, 1, 0);
  v128 = v19;
  v135 = v111;
  v136 = v114;
  v137 = sub_1D2A660(v15, v10, v117, 0, v20, v117);
  v138 = v21;
  *((_QWORD *)&v110 + 1) = 2;
  *(_QWORD *)&v110 = &v135;
  *(_QWORD *)&v115 = sub_1D36D80(v15, 47, (__int64)&v133, v18, v128, *(double *)a7.m128i_i64, a8, a9, v22, v110);
  *((_QWORD *)&v115 + 1) = v23;
  v118 = *a6;
  v24 = sub_39FAC40(*a6);
  v25 = *(__int64 **)(a1 + 552);
  _RDX = v118;
  v27 = v25[2];
  if ( v24 != 1 )
  {
    v28 = v24;
    v119 = *(_DWORD *)(v11 + 24);
    if ( v119 > 0x40 )
    {
      v112 = _RDX;
      v47 = sub_16A57B0(v11 + 16);
      _RDX = v112;
      if ( v119 - v47 <= 0x40 && v28 == **(_QWORD **)(v11 + 16) )
        goto LABEL_8;
    }
    else if ( v24 == *(_QWORD *)(v11 + 16) )
    {
LABEL_8:
      _RDX = ~_RDX;
      v30 = 0;
      __asm { tzcnt   rsi, rdx }
      _RSI = (int)_RSI;
      if ( !_RDX )
        _RSI = 64;
      LOBYTE(v30) = v126;
      *(_QWORD *)&v120 = sub_1D38BB0((__int64)v25, _RSI, (__int64)&v133, v30, 0, 0, a7, a8, a9, 0);
      *((_QWORD *)&v120 + 1) = v32;
      v33 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v27 + 264LL);
      v34 = *(_QWORD *)(a1 + 552);
      v35 = *(_QWORD *)(v34 + 48);
      v36 = sub_1E0A0C0(*(_QWORD *)(v34 + 32));
      v37 = v33(v27, v36, v35, v126, 0);
      v39 = (const void **)v38;
      LODWORD(v33) = v37;
      v43 = sub_1D28D50(v25, 0x16u, v38, v40, v41, v42);
      v45 = sub_1D3A900(
              v25,
              0x89u,
              (__int64)&v133,
              (unsigned int)v33,
              v39,
              0,
              (__m128)a7,
              a8,
              a9,
              v115,
              *((__int16 **)&v115 + 1),
              v120,
              v43,
              v44);
      goto LABEL_14;
    }
    v48 = sub_1D38BB0((__int64)v25, 1, (__int64)&v133, v126, 0, 0, a7, a8, a9, 0);
    v50 = sub_1D332F0(v25, 122, (__int64)&v133, v126, 0, 0, *(double *)a7.m128i_i64, a8, a9, v48, v49, v115);
    v52 = v51;
    v53 = (__int64)v50;
    v54 = *(__int64 **)(a1 + 552);
    *(_QWORD *)&v55 = sub_1D38BB0((__int64)v54, *a6, (__int64)&v133, v126, 0, 0, a7, a8, a9, 0);
    v56 = sub_1D332F0(v54, 118, (__int64)&v133, v126, 0, 0, *(double *)a7.m128i_i64, a8, a9, v53, v52, v55);
    v57 = *(__int64 **)(a1 + 552);
    v123 = v58;
    v121 = (unsigned __int64)v56;
    *(_QWORD *)&v116 = sub_1D38BB0((__int64)v57, 0, (__int64)&v133, v126, 0, 0, a7, a8, a9, 0);
    *((_QWORD *)&v116 + 1) = v59;
    v60 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v27 + 264LL);
    v61 = *(_QWORD *)(a1 + 552);
    v62 = *(_QWORD *)(v61 + 48);
    v63 = sub_1E0A0C0(*(_QWORD *)(v61 + 32));
    v64 = v60(v27, v63, v62, v126, 0);
    v66 = (const void **)v65;
    LODWORD(v60) = v64;
    v70 = sub_1D28D50(v57, 0x16u, v65, v67, v68, v69);
    v45 = sub_1D3A900(
            v57,
            0x89u,
            (__int64)&v133,
            (unsigned int)v60,
            v66,
            0,
            (__m128)a7,
            a8,
            a9,
            v121,
            v123,
            v116,
            v70,
            v71);
LABEL_14:
    *(_QWORD *)&v129 = v45;
    *((_QWORD *)&v129 + 1) = v46;
    goto LABEL_15;
  }
  __asm { tzcnt   rsi, rdx }
  _RSI = (int)_RSI;
  if ( !v118 )
    _RSI = 64;
  *(_QWORD *)&v113 = sub_1D38BB0(*(_QWORD *)(a1 + 552), _RSI, (__int64)&v133, v126, 0, 0, a7, a8, a9, 0);
  *((_QWORD *)&v113 + 1) = v97;
  v98 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD))(*(_QWORD *)v27 + 264LL);
  v99 = *(_QWORD *)(a1 + 552);
  v122 = v126;
  v127 = *(_QWORD *)(v99 + 48);
  v100 = sub_1E0A0C0(*(_QWORD *)(v99 + 32));
  v101 = v98(v27, v100, v127, v122, 0);
  v103 = (const void **)v102;
  LODWORD(v98) = v101;
  v107 = sub_1D28D50(v25, 0x11u, v102, v104, v105, v106);
  *(_QWORD *)&v129 = sub_1D3A900(
                       v25,
                       0x89u,
                       (__int64)&v133,
                       (unsigned int)v98,
                       v103,
                       0,
                       (__m128)a7,
                       a8,
                       a9,
                       v115,
                       *((__int16 **)&v115 + 1),
                       v113,
                       v107,
                       v108);
  *((_QWORD *)&v129 + 1) = v109;
LABEL_15:
  sub_2052F00(a1, a10, a6[2], *((_DWORD *)a6 + 6));
  sub_2052F00(a1, a10, a3, v124);
  sub_1D96570(*(unsigned int **)(a10 + 112), *(unsigned int **)(a10 + 120));
  v72 = *(__int64 **)(a1 + 552);
  v73 = a6[2];
  v78 = sub_1D2A490(v72, v73, v74, v75, v76, v77);
  v80 = v79;
  v84 = sub_2051DF0((__int64 *)a1, *(double *)a7.m128i_i64, a8, a9, v73, v79, v81, v82, v83);
  v130 = sub_1D3A900(
           v72,
           0xBFu,
           (__int64)&v133,
           1u,
           0,
           0,
           (__m128)a7,
           a8,
           a9,
           (unsigned __int64)v84,
           v85,
           v129,
           (__int64)v78,
           v80);
  v86 = v130;
  v131 = v87;
  if ( a3 != sub_2054600(a1, a10) )
  {
    v92 = *(__int64 **)(a1 + 552);
    *(_QWORD *)&v93 = sub_1D2A490(v92, a3, v88, v89, v90, v91);
    v86 = sub_1D332F0(v92, 188, (__int64)&v133, 1, 0, 0, *(double *)a7.m128i_i64, a8, a9, (__int64)v130, v131, v93);
    LODWORD(v131) = v94;
  }
  v95 = *(_QWORD *)(a1 + 552);
  if ( v86 )
  {
    nullsub_686();
    *(_QWORD *)(v95 + 176) = v86;
    *(_DWORD *)(v95 + 184) = v131;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v95 + 176) = 0;
    *(_DWORD *)(v95 + 184) = v131;
  }
  if ( v133 )
    sub_161E7C0((__int64)&v133, v133);
}
