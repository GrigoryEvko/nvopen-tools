// Function: sub_2120FA0
// Address: 0x2120fa0
//
__int64 *__fastcall sub_2120FA0(__int64 *a1, __int64 a2, unsigned int a3, __m128i a4, double a5, __m128i a6)
{
  unsigned __int8 *v8; // rax
  __int64 v9; // r13
  __int64 v10; // r15
  unsigned __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rsi
  unsigned __int64 v15; // rdx
  __int64 v16; // r13
  char v17; // di
  __int64 v18; // rax
  char v19; // r13
  const void **v20; // rax
  int v21; // r15d
  __int64 *v22; // r13
  __int64 v23; // rax
  __int64 v24; // rax
  const void **v25; // rdx
  __int128 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  __int128 v29; // rax
  __int64 *v30; // r15
  unsigned int v31; // edx
  __int64 v32; // r13
  int v33; // r14d
  int v34; // eax
  __int64 *v35; // r10
  int v36; // r14d
  bool v37; // zf
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // rax
  const void **v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r15
  __int64 v45; // r14
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 *v48; // rax
  __int64 *v49; // r13
  unsigned __int64 v50; // rdx
  unsigned __int64 v51; // r15
  __int64 v52; // r14
  __int128 v53; // rax
  unsigned int v54; // edx
  unsigned int v55; // edx
  __int64 *v56; // rbx
  int v58; // eax
  __int64 v59; // rax
  __int64 v60; // rax
  const void **v61; // rdx
  __int128 v62; // rax
  unsigned int v63; // edx
  __int64 v64; // rax
  unsigned int v65; // edx
  __int64 v66; // rax
  __int64 *v67; // r13
  __int64 v68; // r15
  unsigned int v69; // edx
  __int64 v70; // rax
  __int64 v71; // rax
  const void **v72; // rdx
  __int128 v73; // rax
  __int64 *v74; // rax
  unsigned int v75; // edx
  __int128 v76; // [rsp-10h] [rbp-F0h]
  __int128 v77; // [rsp+0h] [rbp-E0h]
  __int128 v78; // [rsp+0h] [rbp-E0h]
  __int128 v79; // [rsp+0h] [rbp-E0h]
  __int64 v80; // [rsp+10h] [rbp-D0h]
  __int64 *v81; // [rsp+10h] [rbp-D0h]
  __int64 v82; // [rsp+10h] [rbp-D0h]
  __int64 v83; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v84; // [rsp+18h] [rbp-C8h]
  int v85; // [rsp+20h] [rbp-C0h]
  unsigned int v86; // [rsp+24h] [rbp-BCh]
  __int64 *v87; // [rsp+28h] [rbp-B8h]
  __int64 *v88; // [rsp+28h] [rbp-B8h]
  __int64 v89; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v90; // [rsp+30h] [rbp-B0h]
  __int64 *v91; // [rsp+30h] [rbp-B0h]
  unsigned __int64 v92; // [rsp+38h] [rbp-A8h]
  __int64 v93; // [rsp+40h] [rbp-A0h]
  __int128 v94; // [rsp+40h] [rbp-A0h]
  __int128 v95; // [rsp+40h] [rbp-A0h]
  __int64 *v96; // [rsp+50h] [rbp-90h]
  __int64 v97; // [rsp+70h] [rbp-70h] BYREF
  int v98; // [rsp+78h] [rbp-68h]
  unsigned int v99; // [rsp+80h] [rbp-60h] BYREF
  const void **v100; // [rsp+88h] [rbp-58h]
  unsigned int v101; // [rsp+90h] [rbp-50h] BYREF
  const void **v102; // [rsp+98h] [rbp-48h]
  __int64 v103; // [rsp+A0h] [rbp-40h]

  v8 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * a3);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  sub_1F40D10((__int64)&v101, *a1, *(_QWORD *)(a1[1] + 48), (unsigned __int8)v9, v10);
  if ( (_BYTE)v9 == (_BYTE)v102 )
  {
    if ( v10 == v103 )
    {
      if ( !(_BYTE)v9 )
        goto LABEL_2;
    }
    else if ( !(_BYTE)v9 )
    {
      goto LABEL_2;
    }
    if ( *(_QWORD *)(*a1 + 8 * v9 + 120) )
      return (__int64 *)a2;
  }
LABEL_2:
  v90 = sub_2120330((__int64)a1, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL));
  v12 = (unsigned int)v11;
  v92 = v11;
  v13 = sub_200D2A0(
          (__int64)a1,
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL),
          *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL),
          *(double *)a4.m128i_i64,
          a5,
          *(double *)a6.m128i_i64);
  v14 = *(_QWORD *)(a2 + 72);
  v80 = v13;
  v84 = v15;
  v97 = v14;
  if ( v14 )
    sub_1623A60((__int64)&v97, v14, 2);
  v16 = *(_QWORD *)(v90 + 40) + 16 * v12;
  v98 = *(_DWORD *)(a2 + 64);
  v17 = *(_BYTE *)v16;
  v100 = *(const void ***)(v16 + 8);
  v18 = *(_QWORD *)(v80 + 40) + 16LL * (unsigned int)v84;
  LOBYTE(v99) = v17;
  v19 = *(_BYTE *)v18;
  v20 = *(const void ***)(v18 + 8);
  LOBYTE(v101) = v19;
  v102 = v20;
  if ( v17 )
  {
    v85 = sub_211A7A0(v17);
    if ( v19 )
      goto LABEL_6;
  }
  else
  {
    v85 = sub_1F58D40((__int64)&v99);
    if ( v19 )
    {
LABEL_6:
      v21 = sub_211A7A0(v19);
      goto LABEL_7;
    }
  }
  v21 = sub_1F58D40((__int64)&v101);
LABEL_7:
  v22 = (__int64 *)a1[1];
  v93 = *a1;
  v23 = sub_1E0A0C0(v22[4]);
  v24 = sub_1F40B60(v93, v101, (__int64)v102, v23, 1);
  *(_QWORD *)&v26 = sub_1D38BB0((__int64)v22, (unsigned int)(v21 - 1), (__int64)&v97, v24, v25, 0, a4, a5, a6, 0);
  v94 = v26;
  v27 = sub_1D38BB0(a1[1], 1, (__int64)&v97, v101, v102, 0, a4, a5, a6, 0);
  *(_QWORD *)&v29 = sub_1D332F0(v22, 122, (__int64)&v97, v101, v102, 0, *(double *)a4.m128i_i64, a5, a6, v27, v28, v94);
  *((_QWORD *)&v94 + 1) = *((_QWORD *)&v29 + 1);
  v81 = sub_1D332F0((__int64 *)a1[1], 118, (__int64)&v97, v101, v102, 0, *(double *)a4.m128i_i64, a5, a6, v80, v84, v29);
  v30 = v81;
  v32 = v31;
  *(_QWORD *)&v95 = v81;
  v86 = v31;
  *((_QWORD *)&v95 + 1) = v31 | *((_QWORD *)&v94 + 1) & 0xFFFFFFFF00000000LL;
  if ( (_BYTE)v101 )
    v33 = sub_211A7A0(v101);
  else
    v33 = sub_1F58D40((__int64)&v101);
  if ( (_BYTE)v99 )
  {
    v34 = sub_211A7A0(v99);
    v35 = (__int64 *)a1[1];
    v36 = v33 - v34;
    v37 = v36 == 0;
    if ( v36 <= 0 )
      goto LABEL_11;
  }
  else
  {
    v58 = sub_1F58D40((__int64)&v99);
    v35 = (__int64 *)a1[1];
    v36 = v33 - v58;
    v37 = v36 == 0;
    if ( v36 <= 0 )
    {
LABEL_11:
      if ( !v37 )
      {
        *((_QWORD *)&v79 + 1) = *((_QWORD *)&v95 + 1);
        *(_QWORD *)&v79 = v81;
        v66 = sub_1D309E0(
                v35,
                144,
                (__int64)&v97,
                v99,
                v100,
                0,
                *(double *)a4.m128i_i64,
                a5,
                *(double *)a6.m128i_i64,
                v79);
        v67 = (__int64 *)a1[1];
        v68 = *a1;
        *(_QWORD *)&v95 = v66;
        v83 = v66;
        v89 = v69;
        *((_QWORD *)&v95 + 1) = v69 | *((_QWORD *)&v95 + 1) & 0xFFFFFFFF00000000LL;
        v70 = sub_1E0A0C0(v67[4]);
        v71 = sub_1F40B60(
                v68,
                *(unsigned __int8 *)(*(_QWORD *)(v83 + 40) + 16 * v89),
                *(_QWORD *)(*(_QWORD *)(v83 + 40) + 16 * v89 + 8),
                v70,
                1);
        *(_QWORD *)&v73 = sub_1D38BB0((__int64)v67, -v36, (__int64)&v97, v71, v72, 0, a4, a5, a6, 0);
        v74 = sub_1D332F0(
                v67,
                122,
                (__int64)&v97,
                v99,
                v100,
                0,
                *(double *)a4.m128i_i64,
                a5,
                a6,
                v95,
                *((unsigned __int64 *)&v95 + 1),
                v73);
        v35 = (__int64 *)a1[1];
        v81 = v74;
        v86 = v75;
      }
      goto LABEL_13;
    }
  }
  v88 = v35;
  v82 = *a1;
  v59 = sub_1E0A0C0(v35[4]);
  v60 = sub_1F40B60(v82, *(unsigned __int8 *)(v30[5] + 16 * v32), *(_QWORD *)(v30[5] + 16 * v32 + 8), v59, 1);
  *(_QWORD *)&v62 = sub_1D38BB0((__int64)v88, v36, (__int64)&v97, v60, v61, 0, a4, a5, a6, 0);
  *(_QWORD *)&v95 = sub_1D332F0(
                      v88,
                      124,
                      (__int64)&v97,
                      v101,
                      v102,
                      0,
                      *(double *)a4.m128i_i64,
                      a5,
                      a6,
                      v95,
                      *((unsigned __int64 *)&v95 + 1),
                      v62);
  *((_QWORD *)&v95 + 1) = v63 | *((_QWORD *)&v95 + 1) & 0xFFFFFFFF00000000LL;
  v64 = sub_1D309E0(
          (__int64 *)a1[1],
          145,
          (__int64)&v97,
          v99,
          v100,
          0,
          *(double *)a4.m128i_i64,
          a5,
          *(double *)a6.m128i_i64,
          v95);
  v35 = (__int64 *)a1[1];
  v81 = (__int64 *)v64;
  v86 = v65;
LABEL_13:
  v38 = *a1;
  v87 = v35;
  v39 = sub_1E0A0C0(v35[4]);
  v40 = sub_1F40B60(v38, v99, (__int64)v100, v39, 1);
  v42 = sub_1D38BB0((__int64)v87, (unsigned int)(v85 - 1), (__int64)&v97, v40, v41, 0, a4, a5, a6, 0);
  v44 = v43;
  v45 = v42;
  v46 = sub_1D38BB0(a1[1], 1, (__int64)&v97, v99, v100, 0, a4, a5, a6, 0);
  *((_QWORD *)&v76 + 1) = v44;
  *(_QWORD *)&v76 = v45;
  v48 = sub_1D332F0(v87, 122, (__int64)&v97, v99, v100, 0, *(double *)a4.m128i_i64, a5, a6, v46, v47, v76);
  v49 = (__int64 *)a1[1];
  v51 = v50;
  v52 = (__int64)v48;
  *(_QWORD *)&v53 = sub_1D38BB0((__int64)v49, 1, (__int64)&v97, v99, v100, 0, a4, a5, a6, 0);
  v96 = sub_1D332F0(v49, 53, (__int64)&v97, v99, v100, 0, *(double *)a4.m128i_i64, a5, a6, v52, v51, v53);
  *((_QWORD *)&v77 + 1) = v54 | v51 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v77 = v96;
  v91 = sub_1D332F0((__int64 *)a1[1], 118, (__int64)&v97, v99, v100, 0, *(double *)a4.m128i_i64, a5, a6, v90, v92, v77);
  *((_QWORD *)&v78 + 1) = v86 | *((_QWORD *)&v95 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v78 = v81;
  v56 = sub_1D332F0(
          (__int64 *)a1[1],
          119,
          (__int64)&v97,
          v99,
          v100,
          0,
          *(double *)a4.m128i_i64,
          a5,
          a6,
          (__int64)v91,
          v55 | v92 & 0xFFFFFFFF00000000LL,
          v78);
  if ( v97 )
    sub_161E7C0((__int64)&v97, v97);
  return v56;
}
