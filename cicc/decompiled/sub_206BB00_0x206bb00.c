// Function: sub_206BB00
// Address: 0x206bb00
//
void __fastcall sub_206BB00(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5, __m128i a6)
{
  __int64 v8; // rax
  int v9; // edx
  __int64 v10; // rsi
  __int64 *v11; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r15
  __int64 v14; // r14
  unsigned __int8 *v15; // rax
  __int64 v16; // r13
  __int128 v17; // rax
  __int64 *v18; // rax
  __int64 v19; // r10
  __int64 *v20; // r15
  __int64 v21; // rdx
  __int64 v22; // r14
  __int128 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rcx
  __int64 v26; // r14
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 v33; // rdx
  __int128 v34; // rax
  unsigned int v35; // r11d
  unsigned int v36; // eax
  char v37; // cl
  char v38; // cl
  unsigned int v39; // edx
  unsigned __int64 *v40; // rax
  int v41; // r8d
  __int64 v42; // rdi
  __int64 v43; // rax
  unsigned int v44; // eax
  char v45; // r11
  __int64 v46; // rax
  unsigned int v47; // edx
  __int64 v48; // rsi
  int v49; // eax
  __int64 *v50; // r13
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  int v54; // r9d
  __int64 *v55; // r14
  __int16 *v56; // rdx
  __int16 *v57; // r15
  __int64 v58; // r8
  __int64 v59; // r9
  __int128 v60; // rax
  __int64 *v61; // r14
  __int16 *v62; // rdx
  __int16 *v63; // r15
  __int64 *v64; // r13
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  _QWORD *v69; // rax
  __int64 v70; // rdx
  __int64 *v71; // r12
  unsigned __int64 v72; // rdx
  unsigned __int64 v73; // r13
  __int64 *v74; // r14
  __int64 v75; // rdx
  __int64 v76; // rcx
  __int64 v77; // r8
  __int64 v78; // r9
  __int64 *v79; // r14
  __int128 v80; // rax
  int v81; // edx
  __int64 v82; // r12
  bool v83; // zf
  char v84; // al
  __int64 *v85; // [rsp+8h] [rbp-E8h]
  const void **v86; // [rsp+10h] [rbp-E0h]
  __int64 (__fastcall *v87)(__int64, __int64, __int64, __int64, __int64); // [rsp+18h] [rbp-D8h]
  unsigned __int64 v88; // [rsp+18h] [rbp-D8h]
  __int128 v89; // [rsp+20h] [rbp-D0h]
  __int128 v90; // [rsp+20h] [rbp-D0h]
  __int64 v91; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v92; // [rsp+38h] [rbp-B8h]
  __int64 v93; // [rsp+40h] [rbp-B0h]
  unsigned __int8 v94; // [rsp+4Bh] [rbp-A5h]
  unsigned int v95; // [rsp+4Ch] [rbp-A4h]
  __int64 v96; // [rsp+50h] [rbp-A0h]
  __int64 *v97; // [rsp+58h] [rbp-98h]
  __int64 v98; // [rsp+58h] [rbp-98h]
  unsigned __int8 v99; // [rsp+58h] [rbp-98h]
  int v100; // [rsp+58h] [rbp-98h]
  const void **v101; // [rsp+60h] [rbp-90h]
  __int64 v102; // [rsp+60h] [rbp-90h]
  __int64 v103; // [rsp+60h] [rbp-90h]
  int v104; // [rsp+68h] [rbp-88h]
  __int64 v106; // [rsp+B0h] [rbp-40h] BYREF
  int v107; // [rsp+B8h] [rbp-38h]

  v8 = *(_QWORD *)a1;
  v9 = *(_DWORD *)(a1 + 536);
  v106 = 0;
  v107 = v9;
  if ( v8 )
  {
    if ( &v106 != (__int64 *)(v8 + 48) )
    {
      v10 = *(_QWORD *)(v8 + 48);
      v106 = v10;
      if ( v10 )
        sub_1623A60((__int64)&v106, v10, 2);
    }
  }
  v11 = sub_20685E0(a1, *(__int64 **)(a2 + 32), a4, a5, a6);
  v13 = v12;
  v14 = (__int64)v11;
  v15 = (unsigned __int8 *)(v11[5] + 16LL * (unsigned int)v12);
  v97 = *(__int64 **)(a1 + 552);
  v94 = *v15;
  v16 = *v15;
  v101 = (const void **)*((_QWORD *)v15 + 1);
  *(_QWORD *)&v17 = sub_1D38970((__int64)v97, a2, (__int64)&v106, v16, v101, 0, a4, *(double *)a5.m128i_i64, a6, 0);
  v18 = sub_1D332F0(
          v97,
          53,
          (__int64)&v106,
          (unsigned int)v16,
          v101,
          0,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          a6,
          v14,
          v13,
          v17);
  v19 = *(_QWORD *)(a1 + 552);
  v20 = v18;
  v22 = v21;
  v96 = (__int64)v18;
  v95 = v21;
  v85 = (__int64 *)v19;
  v98 = *(_QWORD *)(v19 + 16);
  *(_QWORD *)&v23 = sub_1D38970(v19, a2 + 16, (__int64)&v106, v16, v101, 0, a4, *(double *)a5.m128i_i64, a6, 0);
  v102 = v22;
  v89 = v23;
  v92 = (unsigned __int64)v20;
  v93 = 16LL * (unsigned int)v22;
  *(_QWORD *)&v23 = v20[5] + v93;
  v24 = *(_QWORD *)(v23 + 8);
  v87 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v98 + 264LL);
  v25 = *(unsigned __int8 *)v23;
  *(_QWORD *)&v23 = *(_QWORD *)(a1 + 552);
  v26 = *(_QWORD *)(v23 + 48);
  v91 = v25;
  v27 = sub_1E0A0C0(*(_QWORD *)(v23 + 32));
  v28 = v87(v98, v27, v26, v91, v24);
  v86 = (const void **)v29;
  v88 = v28;
  v32 = sub_1D28D50(v85, 0xAu, v29, v28, v30, v31);
  *(_QWORD *)&v34 = sub_1D3A900(
                      v85,
                      0x89u,
                      (__int64)&v106,
                      v88,
                      v86,
                      0,
                      (__m128)a4,
                      *(double *)a5.m128i_i64,
                      a6,
                      v92,
                      (__int16 *)v102,
                      v89,
                      v32,
                      v33);
  v35 = v94;
  v90 = v34;
  if ( v94 && *(_QWORD *)(v98 + 8LL * v94 + 120) )
  {
    if ( !*(_DWORD *)(a2 + 72) )
      goto LABEL_18;
    v36 = sub_2045180(v94);
    v38 = v37 - v36;
    v39 = v36;
    v40 = *(unsigned __int64 **)(a2 + 64);
    v42 = (__int64)&v40[4 * (unsigned int)(v41 - 1) + 4];
    while ( v39 > 0x3F || *v40 <= 0xFFFFFFFFFFFFFFFFLL >> v38 )
    {
      v40 += 4;
      if ( v40 == (unsigned __int64 *)v42 )
        goto LABEL_18;
    }
  }
  v43 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
  v44 = 8 * sub_15A9520(v43, 0);
  if ( v44 == 32 )
  {
    v45 = 5;
  }
  else if ( v44 > 0x20 )
  {
    v45 = 6;
    if ( v44 != 64 )
    {
      v83 = v44 == 128;
      v84 = 7;
      if ( !v83 )
        v84 = 0;
      v45 = v84;
    }
  }
  else
  {
    v45 = 3;
    if ( v44 != 8 )
      v45 = 4 * (v44 == 16);
  }
  LOBYTE(v16) = v45;
  v99 = v45;
  v46 = sub_1D323C0(
          *(__int64 **)(a1 + 552),
          v92,
          v102,
          (__int64)&v106,
          (unsigned int)v16,
          0,
          *(double *)a4.m128i_i64,
          *(double *)a5.m128i_i64,
          *(double *)a6.m128i_i64);
  v35 = v99;
  v95 = v47;
  v96 = v46;
  v93 = 16LL * v47;
LABEL_18:
  *(_BYTE *)(a2 + 44) = v35;
  v48 = v35;
  v49 = sub_1FDDF90(*(_QWORD *)(a1 + 712), v35);
  *(_DWORD *)(a2 + 40) = v49;
  v50 = *(__int64 **)(a1 + 552);
  v100 = v49;
  v55 = sub_2051DF0((__int64 *)a1, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6, v48, v51, v52, v53, v54);
  v57 = v56;
  *(_QWORD *)&v60 = sub_1D2A660(
                      v50,
                      v100,
                      *(unsigned __int8 *)(*(_QWORD *)(v96 + 40) + v93),
                      *(_QWORD *)(*(_QWORD *)(v96 + 40) + v93 + 8),
                      v58,
                      v59);
  v61 = sub_1D3A900(
          v50,
          0x2Eu,
          (__int64)&v106,
          1u,
          0,
          0,
          (__m128)a4,
          *(double *)a5.m128i_i64,
          a6,
          (unsigned __int64)v55,
          v57,
          v60,
          v96,
          v95 | v102 & 0xFFFFFFFF00000000LL);
  v63 = v62;
  v103 = *(_QWORD *)(*(_QWORD *)(a2 + 64) + 8LL);
  sub_2052F00(a1, a3, *(_QWORD *)(a2 + 56), *(_DWORD *)(a2 + 180));
  sub_2052F00(a1, a3, v103, *(_DWORD *)(a2 + 176));
  sub_1D96570(*(unsigned int **)(a3 + 112), *(unsigned int **)(a3 + 120));
  v64 = *(__int64 **)(a1 + 552);
  v69 = sub_1D2A490(v64, *(_QWORD *)(a2 + 56), v65, v66, v67, v68);
  v71 = sub_1D3A900(
          v64,
          0xBFu,
          (__int64)&v106,
          1u,
          0,
          0,
          (__m128)a4,
          *(double *)a5.m128i_i64,
          a6,
          (unsigned __int64)v61,
          v63,
          v90,
          (__int64)v69,
          v70);
  v73 = v72;
  v74 = v71;
  v104 = v72;
  if ( v103 != sub_2054600(a1, a3) )
  {
    v79 = *(__int64 **)(a1 + 552);
    *(_QWORD *)&v80 = sub_1D2A490(v79, v103, v75, v76, v77, v78);
    v74 = sub_1D332F0(
            v79,
            188,
            (__int64)&v106,
            1,
            0,
            0,
            *(double *)a4.m128i_i64,
            *(double *)a5.m128i_i64,
            a6,
            (__int64)v71,
            v73,
            v80);
    v104 = v81;
  }
  v82 = *(_QWORD *)(a1 + 552);
  if ( v74 )
  {
    nullsub_686();
    *(_QWORD *)(v82 + 176) = v74;
    *(_DWORD *)(v82 + 184) = v104;
    sub_1D23870();
  }
  else
  {
    *(_QWORD *)(v82 + 176) = 0;
    *(_DWORD *)(v82 + 184) = v104;
  }
  if ( v106 )
    sub_161E7C0((__int64)&v106, v106);
}
