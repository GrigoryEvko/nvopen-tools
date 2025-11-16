// Function: sub_2048260
// Address: 0x2048260
//
__int64 __fastcall sub_2048260(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 *a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  __int128 v9; // rax
  unsigned __int64 v10; // r15
  __int64 v11; // r14
  __int128 v12; // rax
  __int128 v13; // rax
  __int64 v14; // rax
  unsigned int v15; // edx
  unsigned __int8 v16; // al
  __int128 v17; // rax
  __int64 *v18; // r14
  unsigned int v19; // edx
  unsigned __int64 v20; // r15
  __int128 v21; // rax
  __int64 *v22; // rax
  unsigned __int64 v23; // rdx
  __int128 v24; // rax
  __int64 *v25; // rax
  unsigned __int64 v26; // rdx
  __int64 *v27; // rax
  unsigned __int64 v28; // rdx
  __int128 v29; // rax
  __int64 *v30; // rax
  unsigned __int64 v31; // rdx
  __int64 *v32; // rax
  unsigned __int64 v33; // rdx
  __int128 v34; // rax
  __int64 *v35; // rax
  unsigned __int64 v36; // rdx
  __int64 *v37; // rax
  unsigned __int64 v38; // rdx
  __int128 v39; // rax
  __int64 *v40; // rax
  unsigned __int64 v41; // rdx
  __int64 *v42; // rax
  unsigned __int64 v43; // rdx
  __int128 v44; // rax
  __int64 *v45; // rax
  unsigned __int64 v46; // rdx
  __int64 *v47; // rax
  unsigned __int64 v48; // rdx
  __int128 v49; // rax
  __int64 *v50; // rax
  __int64 *v51; // r10
  __int64 v52; // rdx
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  __int128 v55; // rax
  __int128 v57; // rax
  __int64 *v58; // rax
  unsigned __int64 v59; // rdx
  __int128 v60; // rax
  __int64 *v61; // rax
  unsigned __int64 v62; // rdx
  __int64 *v63; // rax
  unsigned __int64 v64; // rdx
  __int128 v65; // rax
  __int64 *v66; // rax
  unsigned __int64 v67; // rdx
  __int64 *v68; // rax
  unsigned __int64 v69; // rdx
  __int128 v70; // rax
  __int64 *v71; // rax
  unsigned __int64 v72; // rdx
  __int128 v73; // rax
  __int64 *v74; // rax
  unsigned __int64 v75; // rdx
  __int64 *v76; // rax
  unsigned __int64 v77; // rdx
  __int128 v78; // rax
  __int128 v79; // [rsp-10h] [rbp-C0h]
  __int128 v80; // [rsp+0h] [rbp-B0h]
  __int128 v81; // [rsp+0h] [rbp-B0h]
  __int64 v82; // [rsp+20h] [rbp-90h]
  __int64 v83; // [rsp+20h] [rbp-90h]
  __int64 v84; // [rsp+20h] [rbp-90h]
  __int64 v85; // [rsp+20h] [rbp-90h]
  __int64 v86; // [rsp+20h] [rbp-90h]
  __int64 v87; // [rsp+20h] [rbp-90h]
  __int64 v88; // [rsp+20h] [rbp-90h]
  __int64 v89; // [rsp+20h] [rbp-90h]
  unsigned __int64 v90; // [rsp+28h] [rbp-88h]
  unsigned __int64 v91; // [rsp+28h] [rbp-88h]
  unsigned __int64 v92; // [rsp+28h] [rbp-88h]
  unsigned __int64 v93; // [rsp+28h] [rbp-88h]
  unsigned __int64 v94; // [rsp+28h] [rbp-88h]
  unsigned __int64 v95; // [rsp+28h] [rbp-88h]
  unsigned __int64 v96; // [rsp+28h] [rbp-88h]
  unsigned __int64 v97; // [rsp+28h] [rbp-88h]
  __int128 v98; // [rsp+30h] [rbp-80h]
  __int64 v99; // [rsp+30h] [rbp-80h]
  __int64 v100; // [rsp+30h] [rbp-80h]
  unsigned __int64 v101; // [rsp+38h] [rbp-78h]
  unsigned __int64 v102; // [rsp+38h] [rbp-78h]

  *(_QWORD *)&v9 = sub_1D309E0(a4, 152, a3, 5, 0, 0, *(double *)a5.m128i_i64, a6, *(double *)a7.m128i_i64, v80);
  v10 = *((_QWORD *)&v9 + 1);
  v11 = v9;
  *(_QWORD *)&v12 = sub_1D309E0(a4, 146, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, *(double *)a7.m128i_i64, v9);
  *(_QWORD *)&v13 = sub_1D332F0(a4, 77, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, a1, a2, v12);
  v98 = v13;
  v14 = sub_1E0A0C0(a4[4]);
  v15 = 8 * sub_15A9520(v14, 0);
  if ( v15 == 32 )
  {
    v16 = 5;
  }
  else if ( v15 > 0x20 )
  {
    v16 = 6;
    if ( v15 != 64 )
    {
      v16 = 0;
      if ( v15 == 128 )
        v16 = 7;
    }
  }
  else
  {
    v16 = 3;
    if ( v15 != 8 )
      v16 = 4 * (v15 == 16);
  }
  *(_QWORD *)&v17 = sub_1D38BB0((__int64)a4, 23, a3, v16, 0, 0, a5, a6, a7, 0);
  v18 = sub_1D332F0(a4, 122, a3, 5, 0, 0, *(double *)a5.m128i_i64, a6, a7, v11, v10, v17);
  v20 = v19 | v10 & 0xFFFFFFFF00000000LL;
  if ( (unsigned int)dword_4FCEF88 <= 6 )
  {
    *(_QWORD *)&v70 = sub_2048150((__int64)a4, 0x3E814304u, a3, *(double *)a5.m128i_i64, a6, a7);
    v71 = sub_1D332F0(a4, 78, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v98, *((unsigned __int64 *)&v98 + 1), v70);
    v97 = v72;
    v89 = (__int64)v71;
    *(_QWORD *)&v73 = sub_2048150((__int64)a4, 0x3F3C50C8u, a3, *(double *)a5.m128i_i64, a6, a7);
    v74 = sub_1D332F0(a4, 76, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v89, v97, v73);
    v76 = sub_1D332F0(a4, 78, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v74, v75, v98);
    v102 = v77;
    v100 = (__int64)v76;
    *(_QWORD *)&v78 = sub_2048150((__int64)a4, 0x3F7F5E7Eu, a3, *(double *)a5.m128i_i64, a6, a7);
    v51 = sub_1D332F0(a4, 76, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v100, v102, v78);
    v52 = (unsigned int)v52;
  }
  else
  {
    if ( (unsigned int)dword_4FCEF88 <= 0xC )
    {
      *(_QWORD *)&v57 = sub_2048150((__int64)a4, 0x3DA235E3u, a3, *(double *)a5.m128i_i64, a6, a7);
      v58 = sub_1D332F0(a4, 78, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v98, *((unsigned __int64 *)&v98 + 1), v57);
      v95 = v59;
      v87 = (__int64)v58;
      *(_QWORD *)&v60 = sub_2048150((__int64)a4, 0x3E65B8F3u, a3, *(double *)a5.m128i_i64, a6, a7);
      v61 = sub_1D332F0(a4, 76, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v87, v95, v60);
      v63 = sub_1D332F0(a4, 78, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v61, v62, v98);
      v96 = v64;
      v88 = (__int64)v63;
      *(_QWORD *)&v65 = sub_2048150((__int64)a4, 0x3F324B07u, a3, *(double *)a5.m128i_i64, a6, a7);
      v66 = sub_1D332F0(a4, 76, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v88, v96, v65);
      v68 = sub_1D332F0(a4, 78, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v66, v67, v98);
      v101 = v69;
      v99 = (__int64)v68;
      *(_QWORD *)&v49 = sub_2048150((__int64)a4, 0x3F7FF8FDu, a3, *(double *)a5.m128i_i64, a6, a7);
    }
    else
    {
      *(_QWORD *)&v21 = sub_2048150((__int64)a4, 0x3924B03Eu, a3, *(double *)a5.m128i_i64, a6, a7);
      v22 = sub_1D332F0(a4, 78, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v98, *((unsigned __int64 *)&v98 + 1), v21);
      v90 = v23;
      v82 = (__int64)v22;
      *(_QWORD *)&v24 = sub_2048150((__int64)a4, 0x3AB24B87u, a3, *(double *)a5.m128i_i64, a6, a7);
      v25 = sub_1D332F0(a4, 76, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v82, v90, v24);
      v27 = sub_1D332F0(a4, 78, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v25, v26, v98);
      v91 = v28;
      v83 = (__int64)v27;
      *(_QWORD *)&v29 = sub_2048150((__int64)a4, 0x3C1D8C17u, a3, *(double *)a5.m128i_i64, a6, a7);
      v30 = sub_1D332F0(a4, 76, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v83, v91, v29);
      v32 = sub_1D332F0(a4, 78, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v30, v31, v98);
      v92 = v33;
      v84 = (__int64)v32;
      *(_QWORD *)&v34 = sub_2048150((__int64)a4, 0x3D634A1Du, a3, *(double *)a5.m128i_i64, a6, a7);
      v35 = sub_1D332F0(a4, 76, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v84, v92, v34);
      v37 = sub_1D332F0(a4, 78, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v35, v36, v98);
      v93 = v38;
      v85 = (__int64)v37;
      *(_QWORD *)&v39 = sub_2048150((__int64)a4, 0x3E75FE14u, a3, *(double *)a5.m128i_i64, a6, a7);
      v40 = sub_1D332F0(a4, 76, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v85, v93, v39);
      v42 = sub_1D332F0(a4, 78, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v40, v41, v98);
      v94 = v43;
      v86 = (__int64)v42;
      *(_QWORD *)&v44 = sub_2048150((__int64)a4, 0x3F317234u, a3, *(double *)a5.m128i_i64, a6, a7);
      v45 = sub_1D332F0(a4, 76, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v86, v94, v44);
      v47 = sub_1D332F0(a4, 78, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v45, v46, v98);
      v101 = v48;
      v99 = (__int64)v47;
      *(_QWORD *)&v49 = sub_2048150((__int64)a4, 0x3F800000u, a3, *(double *)a5.m128i_i64, a6, a7);
    }
    v50 = sub_1D332F0(a4, 76, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, a7, v99, v101, v49);
    v51 = v50;
    v52 = (unsigned int)v52;
  }
  *((_QWORD *)&v81 + 1) = v52;
  *(_QWORD *)&v81 = v51;
  v53 = sub_1D309E0(a4, 158, a3, 5, 0, 0, *(double *)a5.m128i_i64, a6, *(double *)a7.m128i_i64, v81);
  *((_QWORD *)&v79 + 1) = v20;
  *(_QWORD *)&v79 = v18;
  *(_QWORD *)&v55 = sub_1D332F0(a4, 52, a3, 5, 0, 0, *(double *)a5.m128i_i64, a6, a7, v53, v54, v79);
  return sub_1D309E0(a4, 158, a3, 9, 0, 0, *(double *)a5.m128i_i64, a6, *(double *)a7.m128i_i64, v55);
}
