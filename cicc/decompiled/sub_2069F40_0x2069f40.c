// Function: sub_2069F40
// Address: 0x2069f40
//
void __fastcall sub_2069F40(__int64 *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5, __m128i a6)
{
  unsigned __int64 v6; // r13
  __int64 *v9; // rax
  __int64 v10; // rsi
  __int16 *v11; // rdx
  __int64 *v12; // rsi
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int16 *v15; // rdx
  __int16 *v16; // r13
  unsigned __int64 v17; // r12
  unsigned int v18; // r14d
  unsigned __int8 *v19; // rax
  const void **v20; // r8
  unsigned int v21; // r10d
  unsigned int v22; // r14d
  int v23; // eax
  __int64 *v24; // r14
  __int64 v25; // rsi
  __int128 v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // r14
  unsigned __int64 v29; // r12
  const void **v30; // r8
  __int16 *v31; // rdx
  __int16 *v32; // r13
  unsigned int v33; // r10d
  __int64 *v34; // r11
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rax
  __int64 v40; // rdx
  unsigned int v41; // edx
  __int64 *v42; // r14
  __int64 v43; // rdx
  __int64 v44; // rcx
  __int64 v45; // r8
  __int64 v46; // r9
  __int64 v47; // rax
  __int64 v48; // rdx
  unsigned int v49; // edx
  __int64 v50; // rdx
  __int64 v51; // r14
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  __int64 *v56; // r14
  __int64 v57; // rsi
  __int64 v58; // rdx
  __int64 v59; // rcx
  __int64 v60; // r8
  int v61; // r9d
  __int64 *v62; // rax
  __int16 *v63; // rdx
  __int64 *v64; // rax
  __int64 *v65; // r14
  __int64 v66; // r12
  __int64 v67; // rdx
  unsigned __int64 v68; // r13
  __int64 v69; // rcx
  __int64 v70; // r8
  __int64 v71; // r9
  __int128 v72; // rax
  __int64 *v73; // rax
  int v74; // edx
  __int64 v75; // r14
  __int64 *v76; // r12
  int v77; // ebx
  __int64 v78; // r12
  __int64 *v79; // r12
  __int64 *v80; // r13
  __int64 v81; // rdx
  __int64 v82; // r14
  __int64 v83; // rcx
  __int64 v84; // r9
  __int64 v85; // rax
  __int64 v86; // rdx
  __int64 *v87; // rax
  unsigned int v88; // edx
  __int64 v89; // r12
  __int128 v90; // rax
  __int64 v91; // rax
  __int128 v92; // rax
  __int128 v93; // [rsp-20h] [rbp-100h]
  __int128 v94; // [rsp-20h] [rbp-100h]
  unsigned __int64 v95; // [rsp+0h] [rbp-E0h]
  const void **v96; // [rsp+0h] [rbp-E0h]
  unsigned int v97; // [rsp+0h] [rbp-E0h]
  __int16 *v98; // [rsp+8h] [rbp-D8h]
  unsigned int v99; // [rsp+10h] [rbp-D0h]
  __int64 *v100; // [rsp+10h] [rbp-D0h]
  __int64 v101; // [rsp+18h] [rbp-C8h]
  __int64 *v102; // [rsp+20h] [rbp-C0h]
  const void **v103; // [rsp+20h] [rbp-C0h]
  const void **v104; // [rsp+20h] [rbp-C0h]
  __int128 v105; // [rsp+20h] [rbp-C0h]
  __int128 v106; // [rsp+20h] [rbp-C0h]
  unsigned int v107; // [rsp+20h] [rbp-C0h]
  _QWORD *v109; // [rsp+30h] [rbp-B0h]
  __int64 v110; // [rsp+38h] [rbp-A8h]
  unsigned int v111; // [rsp+40h] [rbp-A0h]
  __int64 v112; // [rsp+40h] [rbp-A0h]
  const void **v113; // [rsp+40h] [rbp-A0h]
  unsigned int v114; // [rsp+40h] [rbp-A0h]
  __int64 v115; // [rsp+80h] [rbp-60h] BYREF
  int v116; // [rsp+88h] [rbp-58h]
  __int64 v117; // [rsp+90h] [rbp-50h] BYREF
  unsigned int v118; // [rsp+98h] [rbp-48h]
  __int64 v119; // [rsp+A0h] [rbp-40h] BYREF
  unsigned int v120; // [rsp+A8h] [rbp-38h]

  v9 = sub_20685E0((__int64)a1, *(__int64 **)(a2 + 8), a4, a5, a6);
  v10 = *(_QWORD *)(a2 + 56);
  v95 = (unsigned __int64)v9;
  v102 = v9;
  v98 = v11;
  v111 = (unsigned int)v11;
  v115 = v10;
  if ( v10 )
    sub_1623A60((__int64)&v115, v10, 2);
  v12 = *(__int64 **)(a2 + 16);
  v116 = *(_DWORD *)(a2 + 64);
  v101 = *(_QWORD *)(a2 + 24);
  if ( v12 )
  {
    v112 = *(_QWORD *)(a2 + 8);
    v103 = (const void **)(v101 + 24);
    v13 = sub_20685E0((__int64)a1, v12, a4, a5, a6);
    v14 = *(_QWORD *)(a2 + 8);
    v16 = v15;
    v17 = (unsigned __int64)v13;
    v18 = *(_DWORD *)(v14 + 32);
    v19 = (unsigned __int8 *)(v13[5] + 16LL * (unsigned int)v15);
    v20 = (const void **)*((_QWORD *)v19 + 1);
    v21 = *v19;
    if ( v18 <= 0x40 )
    {
      if ( *(_QWORD *)(v14 + 24) != 1LL << ((unsigned __int8)v18 - 1) )
        goto LABEL_7;
    }
    else
    {
      v22 = v18 - 1;
      if ( (*(_QWORD *)(*(_QWORD *)(v14 + 24) + 8LL * (v22 >> 6)) & (1LL << v22)) == 0
        || (v99 = *v19,
            v96 = (const void **)*((_QWORD *)v19 + 1),
            v23 = sub_16A58A0(v14 + 24),
            v20 = v96,
            v21 = v99,
            v22 != v23) )
      {
LABEL_7:
        v24 = (__int64 *)a1[69];
        v97 = v21;
        v25 = v112 + 24;
        v100 = (__int64 *)(v112 + 24);
        v113 = v20;
        *(_QWORD *)&v26 = sub_1D38970(
                            (__int64)v24,
                            v25,
                            (__int64)&v115,
                            v21,
                            v20,
                            0,
                            a4,
                            *(double *)a5.m128i_i64,
                            a6,
                            0);
        v27 = sub_1D332F0(
                v24,
                53,
                (__int64)&v115,
                v97,
                v113,
                0,
                *(double *)a4.m128i_i64,
                *(double *)a5.m128i_i64,
                a6,
                v17,
                (unsigned __int64)v16,
                v26);
        v28 = (__int64 *)a1[69];
        v29 = (unsigned __int64)v27;
        v30 = v113;
        v32 = v31;
        v33 = v97;
        v34 = v100;
        v118 = *(_DWORD *)(v101 + 32);
        if ( v118 > 0x40 )
        {
          sub_16A4FD0((__int64)&v117, v103);
          v33 = v97;
          v34 = v100;
          v30 = v113;
        }
        else
        {
          v117 = *(_QWORD *)(v101 + 24);
        }
        v114 = v33;
        v104 = v30;
        sub_16A7590((__int64)&v117, v34);
        v120 = v118;
        v118 = 0;
        v119 = v117;
        *(_QWORD *)&v105 = sub_1D38970(
                             (__int64)v28,
                             (__int64)&v119,
                             (__int64)&v115,
                             v114,
                             v104,
                             0,
                             a4,
                             *(double *)a5.m128i_i64,
                             a6,
                             0);
        *((_QWORD *)&v105 + 1) = v35;
        v39 = sub_1D28D50(v28, 0xDu, v35, v36, v37, v38);
        v102 = sub_1D3A900(
                 v28,
                 0x89u,
                 (__int64)&v115,
                 2u,
                 0,
                 0,
                 (__m128)a4,
                 *(double *)a5.m128i_i64,
                 a6,
                 v29,
                 v32,
                 v105,
                 v39,
                 v40);
        v111 = v41;
        v6 = v41;
        if ( v120 > 0x40 && v119 )
          j_j___libc_free_0_0(v119);
        if ( v118 > 0x40 && v117 )
          j_j___libc_free_0_0(v117);
        goto LABEL_17;
      }
    }
    v42 = (__int64 *)a1[69];
    *(_QWORD *)&v106 = sub_1D38970(
                         (__int64)v42,
                         (__int64)v103,
                         (__int64)&v115,
                         v21,
                         v20,
                         0,
                         a4,
                         *(double *)a5.m128i_i64,
                         a6,
                         0);
    *((_QWORD *)&v106 + 1) = v43;
    v47 = sub_1D28D50(v42, 0x15u, v43, v44, v45, v46);
    v102 = sub_1D3A900(
             v42,
             0x89u,
             (__int64)&v115,
             2u,
             0,
             0,
             (__m128)a4,
             *(double *)a5.m128i_i64,
             a6,
             v17,
             v16,
             v106,
             v47,
             v48);
    v111 = v49;
    v6 = v49;
    goto LABEL_17;
  }
  if ( v101 != sub_159C4F0(*(__int64 **)(a1[69] + 48)) || *(_DWORD *)a2 != 17 )
  {
    v78 = *(_QWORD *)(a2 + 24);
    if ( v78 == sub_159C540(*(__int64 **)(a1[69] + 48)) && *(_DWORD *)a2 == 17 )
    {
      v89 = 16LL * v111;
      *(_QWORD *)&v90 = sub_1D38BB0(
                          a1[69],
                          1,
                          (__int64)&v115,
                          *(unsigned __int8 *)(v89 + v102[5]),
                          *(const void ***)(v89 + v102[5] + 8),
                          0,
                          a4,
                          *(double *)a5.m128i_i64,
                          a6,
                          0);
      v87 = sub_1D332F0(
              (__int64 *)a1[69],
              120,
              (__int64)&v115,
              *(unsigned __int8 *)(v102[5] + v89),
              *(const void ***)(v102[5] + v89 + 8),
              0,
              *(double *)a4.m128i_i64,
              *(double *)a5.m128i_i64,
              a6,
              v95,
              (unsigned __int64)v98,
              v90);
    }
    else
    {
      v79 = (__int64 *)a1[69];
      v107 = *(_DWORD *)a2;
      v80 = sub_20685E0((__int64)a1, *(__int64 **)(a2 + 24), a4, a5, a6);
      v82 = v81;
      v85 = sub_1D28D50(v79, v107, v81, v83, v107, v84);
      *((_QWORD *)&v94 + 1) = v82;
      *(_QWORD *)&v94 = v80;
      v87 = sub_1D3A900(
              v79,
              0x89u,
              (__int64)&v115,
              2u,
              0,
              0,
              (__m128)a4,
              *(double *)a5.m128i_i64,
              a6,
              v95,
              v98,
              v94,
              v85,
              v86);
    }
    v111 = v88;
    v102 = v87;
    v6 = v88;
  }
LABEL_17:
  sub_2052F00((__int64)a1, a3, *(_QWORD *)(a2 + 32), *(_DWORD *)(a2 + 72));
  v50 = *(_QWORD *)(a2 + 40);
  if ( *(_QWORD *)(a2 + 32) != v50 )
    sub_2052F00((__int64)a1, a3, v50, *(_DWORD *)(a2 + 76));
  sub_1D96570(*(unsigned int **)(a3 + 112), *(unsigned int **)(a3 + 120));
  v51 = *(_QWORD *)(a2 + 32);
  if ( v51 == sub_2054600((__int64)a1, a3) )
  {
    v91 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(a2 + 32) = *(_QWORD *)(a2 + 40);
    *(_QWORD *)(a2 + 40) = v91;
    *(_QWORD *)&v92 = sub_1D38BB0(
                        a1[69],
                        1,
                        (__int64)&v115,
                        *(unsigned __int8 *)(16LL * v111 + v102[5]),
                        *(const void ***)(16LL * v111 + v102[5] + 8),
                        0,
                        a4,
                        *(double *)a5.m128i_i64,
                        a6,
                        0);
    v6 = v111 | v6 & 0xFFFFFFFF00000000LL;
    v102 = sub_1D332F0(
             (__int64 *)a1[69],
             120,
             (__int64)&v115,
             *(unsigned __int8 *)(v102[5] + 16LL * v111),
             *(const void ***)(v102[5] + 16LL * v111 + 8),
             0,
             *(double *)a4.m128i_i64,
             *(double *)a5.m128i_i64,
             a6,
             (__int64)v102,
             v6,
             v92);
    v111 = v52;
  }
  v56 = (__int64 *)a1[69];
  v57 = *(_QWORD *)(a2 + 32);
  v109 = sub_1D2A490(v56, v57, v52, v53, v54, v55);
  v110 = v58;
  v62 = sub_2051DF0(a1, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v57, v58, v59, v60, v61);
  *((_QWORD *)&v93 + 1) = v111 | v6 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v93 = v102;
  v64 = sub_1D3A900(
          v56,
          0xBFu,
          (__int64)&v115,
          1u,
          0,
          0,
          (__m128)a4,
          *(double *)a5.m128i_i64,
          a6,
          (unsigned __int64)v62,
          v63,
          v93,
          (__int64)v109,
          v110);
  v65 = (__int64 *)a1[69];
  v66 = (__int64)v64;
  v68 = v67;
  *(_QWORD *)&v72 = sub_1D2A490(v65, *(_QWORD *)(a2 + 40), v67, v69, v70, v71);
  v73 = sub_1D332F0(
          v65,
          188,
          (__int64)&v115,
          1,
          0,
          0,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          a6,
          v66,
          v68,
          v72);
  v75 = a1[69];
  v76 = v73;
  v77 = v74;
  if ( v73 )
  {
    nullsub_686();
    *(_QWORD *)(v75 + 176) = v76;
    *(_DWORD *)(v75 + 184) = v77;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v75 + 176) = 0;
    *(_DWORD *)(v75 + 184) = v74;
  }
  if ( v115 )
    sub_161E7C0((__int64)&v115, v115);
}
