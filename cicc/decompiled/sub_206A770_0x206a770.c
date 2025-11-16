// Function: sub_206A770
// Address: 0x206a770
//
void __fastcall sub_206A770(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5, __m128i a6, __m128i a7)
{
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 *v11; // rax
  __int64 *v12; // r15
  unsigned __int64 v13; // rdx
  unsigned __int8 *v14; // rax
  unsigned int v15; // r13d
  __int128 v16; // rax
  __int64 *v17; // rax
  __int64 *v18; // r15
  __int16 *v19; // rdx
  __int64 v20; // rax
  unsigned int v21; // eax
  char v22; // r8
  __int64 v23; // rax
  __int64 v24; // r15
  unsigned int v25; // edx
  __int64 v26; // rax
  unsigned int v27; // eax
  __int64 v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  int v32; // r9d
  int v33; // r15d
  unsigned int v34; // edx
  __int64 *v35; // rax
  __int16 *v36; // rdx
  unsigned __int8 *v37; // rax
  __int64 v38; // r8
  __int64 v39; // r9
  __int128 v40; // rax
  __int16 *v41; // rdx
  __int64 *v42; // r15
  __int64 v43; // rdx
  __int64 (__fastcall *v44)(__int64, __int64, __int64, __int64, __int64); // r12
  __int64 v45; // rax
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r8
  __int64 v51; // r9
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // r8
  __int64 v57; // r9
  __int64 *v58; // r12
  __int64 v59; // r13
  __int64 *v60; // r15
  _QWORD *v61; // rax
  __int64 v62; // rdx
  __int64 *v63; // r15
  __int64 v64; // r12
  unsigned __int64 v65; // rdx
  __int64 v66; // rdx
  __int64 v67; // rcx
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 *v70; // r15
  __int128 v71; // rax
  int v72; // edx
  __int64 v73; // r12
  __int128 v74; // [rsp-20h] [rbp-120h]
  unsigned __int64 v75; // [rsp+0h] [rbp-100h]
  __int16 *v76; // [rsp+8h] [rbp-F8h]
  __int64 *v77; // [rsp+10h] [rbp-F0h]
  unsigned int v79; // [rsp+20h] [rbp-E0h]
  __int128 v80; // [rsp+20h] [rbp-E0h]
  __int64 v81; // [rsp+30h] [rbp-D0h]
  __int64 v82; // [rsp+30h] [rbp-D0h]
  const void **v83; // [rsp+38h] [rbp-C8h]
  __int64 v84; // [rsp+38h] [rbp-C8h]
  const void **v85; // [rsp+38h] [rbp-C8h]
  __int64 v86; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v87; // [rsp+40h] [rbp-C0h]
  __int16 *v89; // [rsp+50h] [rbp-B0h]
  unsigned __int64 v90; // [rsp+58h] [rbp-A8h]
  __int64 v91; // [rsp+60h] [rbp-A0h]
  __int64 *v92; // [rsp+60h] [rbp-A0h]
  __int64 *v93; // [rsp+60h] [rbp-A0h]
  unsigned __int64 v94; // [rsp+68h] [rbp-98h]
  __int16 *v95; // [rsp+68h] [rbp-98h]
  unsigned __int64 v96; // [rsp+68h] [rbp-98h]
  __int64 v97; // [rsp+A0h] [rbp-60h] BYREF
  int v98; // [rsp+A8h] [rbp-58h]
  __int64 v99; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v100; // [rsp+B8h] [rbp-48h]
  __int64 v101; // [rsp+C0h] [rbp-40h] BYREF
  unsigned int v102; // [rsp+C8h] [rbp-38h]

  v8 = *(_DWORD *)(a1 + 536);
  v9 = *(_QWORD *)a1;
  v97 = 0;
  v98 = v8;
  if ( v9 )
  {
    if ( &v97 != (__int64 *)(v9 + 48) )
    {
      v10 = *(_QWORD *)(v9 + 48);
      v97 = v10;
      if ( v10 )
        sub_1623A60((__int64)&v97, v10, 2);
    }
  }
  v11 = sub_20685E0(a1, *(__int64 **)(a3 + 32), a5, a6, a7);
  v12 = *(__int64 **)(a1 + 552);
  v94 = v13;
  v91 = (__int64)v11;
  v14 = (unsigned __int8 *)(v11[5] + 16LL * (unsigned int)v13);
  v15 = *v14;
  v83 = (const void **)*((_QWORD *)v14 + 1);
  *(_QWORD *)&v16 = sub_1D38970((__int64)v12, a3, (__int64)&v97, v15, v83, 0, a5, *(double *)a6.m128i_i64, a7, 0);
  v17 = sub_1D332F0(
          v12,
          53,
          (__int64)&v97,
          v15,
          v83,
          0,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128i_i64,
          a7,
          v91,
          v94,
          v16);
  v18 = *(__int64 **)(a1 + 552);
  v90 = (unsigned __int64)v17;
  v89 = v19;
  v86 = v18[2];
  v20 = sub_1E0A0C0(v18[4]);
  v21 = 8 * sub_15A9520(v20, 0);
  if ( v21 == 32 )
  {
    v22 = 5;
  }
  else if ( v21 > 0x20 )
  {
    v22 = 6;
    if ( v21 != 64 )
    {
      v22 = 0;
      if ( v21 == 128 )
        v22 = 7;
    }
  }
  else
  {
    v22 = 3;
    if ( v21 != 8 )
      v22 = 4 * (v21 == 16);
  }
  v23 = sub_1D323C0(
          v18,
          v90,
          (__int64)v89,
          (__int64)&v97,
          v22 & 7,
          0,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128i_i64,
          *(double *)a7.m128i_i64);
  v24 = *(_QWORD *)(a1 + 712);
  v81 = v23;
  v79 = v25;
  v26 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
  v27 = 8 * sub_15A9520(v26, 0);
  if ( v27 == 32 )
  {
    v28 = 5;
  }
  else if ( v27 > 0x20 )
  {
    v28 = 6;
    if ( v27 != 64 )
    {
      v28 = 0;
      if ( v27 == 128 )
        v28 = 7;
    }
  }
  else
  {
    v28 = 3;
    if ( v27 != 8 )
    {
      LOBYTE(v28) = v27 == 16;
      v28 = (unsigned int)(4 * v28);
    }
  }
  v33 = sub_1FDDF90(v24, v28);
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 544) + 504LL) - 34) <= 1 )
  {
    v28 = 5;
    v81 = sub_1D323C0(
            *(__int64 **)(a1 + 552),
            v90,
            (__int64)v89,
            (__int64)&v97,
            5,
            0,
            *(double *)a5.m128i_i64,
            *(double *)a6.m128i_i64,
            *(double *)a7.m128i_i64);
    v79 = v34;
    v33 = sub_1FDDF90(*(_QWORD *)(a1 + 712), 5u);
  }
  v77 = *(__int64 **)(a1 + 552);
  v35 = sub_2051DF0((__int64 *)a1, *(double *)a5.m128i_i64, *(double *)a6.m128i_i64, a7, v28, v29, v30, v31, v32);
  v76 = v36;
  v75 = (unsigned __int64)v35;
  v37 = (unsigned __int8 *)(*(_QWORD *)(v81 + 40) + 16LL * v79);
  *(_QWORD *)&v40 = sub_1D2A660(v77, v33, *v37, *((_QWORD *)v37 + 1), v38, v39);
  v92 = sub_1D3A900(
          v77,
          0x2Eu,
          (__int64)&v97,
          1u,
          0,
          0,
          (__m128)a5,
          *(double *)a6.m128i_i64,
          a7,
          v75,
          v76,
          v40,
          v81,
          v79 | v94 & 0xFFFFFFFF00000000LL);
  v95 = v41;
  *(_DWORD *)a2 = v33;
  v42 = *(__int64 **)(a1 + 552);
  v100 = *(_DWORD *)(a3 + 24);
  if ( v100 > 0x40 )
    sub_16A4FD0((__int64)&v99, (const void **)(a3 + 16));
  else
    v99 = *(_QWORD *)(a3 + 16);
  sub_16A7590((__int64)&v99, (__int64 *)a3);
  v102 = v100;
  v100 = 0;
  v101 = v99;
  *(_QWORD *)&v80 = sub_1D38970(
                      (__int64)v42,
                      (__int64)&v101,
                      (__int64)&v97,
                      v15,
                      v83,
                      0,
                      a5,
                      *(double *)a6.m128i_i64,
                      a7,
                      0);
  *((_QWORD *)&v80 + 1) = v43;
  v44 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v86 + 264LL);
  v45 = *(_QWORD *)(a1 + 552);
  v82 = *(_QWORD *)(*(_QWORD *)(v90 + 40) + 16LL * (unsigned int)v89 + 8);
  v46 = *(_QWORD *)(v45 + 48);
  v84 = *(unsigned __int8 *)(*(_QWORD *)(v90 + 40) + 16LL * (unsigned int)v89);
  v47 = sub_1E0A0C0(*(_QWORD *)(v45 + 32));
  v48 = v44(v86, v47, v46, v84, v82);
  v85 = (const void **)v49;
  v87 = v48;
  v52 = sub_1D28D50(v42, 0xAu, v49, v48, v50, v51);
  v58 = sub_1D3A900(
          v42,
          0x89u,
          (__int64)&v97,
          v87,
          v85,
          0,
          (__m128)a5,
          *(double *)a6.m128i_i64,
          a7,
          v90,
          v89,
          v80,
          v52,
          v53);
  v59 = v54;
  if ( v102 > 0x40 && v101 )
    j_j___libc_free_0_0(v101);
  if ( v100 > 0x40 && v99 )
    j_j___libc_free_0_0(v99);
  v60 = *(__int64 **)(a1 + 552);
  v61 = sub_1D2A490(v60, *(_QWORD *)(a2 + 16), v54, v55, v56, v57);
  *((_QWORD *)&v74 + 1) = v59;
  *(_QWORD *)&v74 = v58;
  v93 = sub_1D3A900(
          v60,
          0xBFu,
          (__int64)&v97,
          1u,
          0,
          0,
          (__m128)a5,
          *(double *)a6.m128i_i64,
          a7,
          (unsigned __int64)v92,
          v95,
          v74,
          (__int64)v61,
          v62);
  v63 = v93;
  v64 = *(_QWORD *)(a2 + 8);
  v96 = v65;
  if ( v64 != sub_2054600(a1, a4) )
  {
    v70 = *(__int64 **)(a1 + 552);
    *(_QWORD *)&v71 = sub_1D2A490(v70, *(_QWORD *)(a2 + 8), v66, v67, v68, v69);
    v63 = sub_1D332F0(
            v70,
            188,
            (__int64)&v97,
            1,
            0,
            0,
            *(double *)a5.m128i_i64,
            *(double *)a6.m128i_i64,
            a7,
            (__int64)v93,
            v96,
            v71);
    LODWORD(v96) = v72;
  }
  v73 = *(_QWORD *)(a1 + 552);
  if ( v63 )
  {
    nullsub_686();
    *(_QWORD *)(v73 + 176) = v63;
    *(_DWORD *)(v73 + 184) = v96;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v73 + 176) = 0;
    *(_DWORD *)(v73 + 184) = v96;
  }
  if ( v97 )
    sub_161E7C0((__int64)&v97, v97);
}
