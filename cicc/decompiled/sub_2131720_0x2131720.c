// Function: sub_2131720
// Address: 0x2131720
//
unsigned __int64 __fastcall sub_2131720(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        double a6,
        __m128i a7)
{
  unsigned __int8 *v8; // rax
  unsigned __int8 v9; // r12
  __int64 v10; // r14
  __int64 v11; // rsi
  int v12; // eax
  unsigned __int64 *v13; // rax
  __int64 v14; // rdx
  unsigned __int64 result; // rax
  __int64 v16; // rax
  int v17; // ecx
  __m128i *v18; // rsi
  __int64 v19; // rax
  __int64 v20; // r11
  __m128i v21; // xmm0
  __m128i v22; // xmm1
  unsigned int v23; // eax
  __int64 v24; // r14
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // rax
  unsigned __int64 v29; // rdx
  __int128 v30; // rax
  __int64 *v31; // rax
  unsigned __int64 v32; // rdx
  __int64 *v33; // rax
  __int64 v34; // r14
  unsigned __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  unsigned int v38; // eax
  __int128 v39; // rax
  __int64 *v40; // rax
  __int64 v41; // rdx
  __int64 v42; // r13
  __int64 *v43; // r12
  __int64 *v44; // rax
  unsigned __int64 v45; // rdx
  __int128 v46; // rax
  __int64 *v47; // r14
  __int64 *v48; // rax
  unsigned __int64 v49; // rdx
  __int64 *v50; // rax
  unsigned __int64 v51; // rdx
  unsigned __int64 v52; // r13
  __int64 v53; // r12
  __int128 v54; // rax
  __int64 *v55; // rax
  __int64 *v56; // r14
  unsigned __int64 v57; // rdx
  unsigned __int64 v58; // r13
  __int64 v59; // r12
  __int64 *v60; // rax
  unsigned __int64 v61; // rdx
  __int128 v62; // rax
  __int128 v63; // rax
  __int64 *v64; // r14
  __int64 *v65; // rax
  __int64 v66; // rdx
  __int64 v67; // r13
  __int64 *v68; // r12
  __int64 *v69; // rax
  unsigned __int64 v70; // rdx
  __int64 *v71; // rax
  __int64 *v72; // r14
  unsigned __int64 v73; // rdx
  unsigned __int64 v74; // r13
  __int64 v75; // r12
  __int128 v76; // rax
  __int64 *v77; // rax
  const void **v78; // r8
  int v79; // edx
  __int64 *v80; // r14
  __int128 v81; // rax
  __int64 *v82; // rax
  unsigned __int64 v83; // rdx
  __int128 v84; // rax
  unsigned int v85; // edx
  int v86; // r14d
  __int128 v87; // [rsp-10h] [rbp-190h]
  __int128 v88; // [rsp-10h] [rbp-190h]
  __int64 v89; // [rsp+0h] [rbp-180h]
  unsigned __int64 v90; // [rsp+8h] [rbp-178h]
  __int128 v91; // [rsp+10h] [rbp-170h]
  __int128 v92; // [rsp+20h] [rbp-160h]
  __int64 v93; // [rsp+30h] [rbp-150h]
  __int64 v94; // [rsp+30h] [rbp-150h]
  unsigned __int64 v95; // [rsp+38h] [rbp-148h]
  unsigned __int64 v96; // [rsp+38h] [rbp-148h]
  __int64 v97; // [rsp+40h] [rbp-140h]
  unsigned __int64 v98; // [rsp+48h] [rbp-138h]
  __int128 v99; // [rsp+50h] [rbp-130h]
  __int128 v100; // [rsp+50h] [rbp-130h]
  __int128 v101; // [rsp+50h] [rbp-130h]
  __int128 v102; // [rsp+60h] [rbp-120h]
  __int128 v105; // [rsp+80h] [rbp-100h]
  unsigned int v106; // [rsp+B0h] [rbp-D0h] BYREF
  const void **v107; // [rsp+B8h] [rbp-C8h]
  __int64 v108; // [rsp+C0h] [rbp-C0h] BYREF
  int v109; // [rsp+C8h] [rbp-B8h]
  __int128 v110; // [rsp+D0h] [rbp-B0h] BYREF
  __int128 v111; // [rsp+E0h] [rbp-A0h] BYREF
  __int128 v112; // [rsp+F0h] [rbp-90h] BYREF
  __int128 v113; // [rsp+100h] [rbp-80h] BYREF
  _OWORD v114[2]; // [rsp+110h] [rbp-70h] BYREF
  unsigned __int64 v115; // [rsp+130h] [rbp-50h] BYREF
  unsigned __int64 v116; // [rsp+138h] [rbp-48h]
  const void **v117; // [rsp+140h] [rbp-40h]

  v8 = *(unsigned __int8 **)(a2 + 40);
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  sub_1F40D10((__int64)&v115, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), *v8, v10);
  v11 = *(_QWORD *)(a2 + 72);
  LOBYTE(v106) = v116;
  v108 = v11;
  v107 = v117;
  if ( v11 )
    sub_1623A60((__int64)&v108, v11, 2);
  v12 = *(_DWORD *)(a2 + 64);
  DWORD2(v110) = 0;
  DWORD2(v113) = 0;
  v109 = v12;
  v13 = *(unsigned __int64 **)(a2 + 32);
  DWORD2(v111) = 0;
  DWORD2(v112) = 0;
  v14 = v13[1];
  *(_QWORD *)&v110 = 0;
  *(_QWORD *)&v111 = 0;
  *(_QWORD *)&v112 = 0;
  *(_QWORD *)&v113 = 0;
  sub_20174B0(a1, *v13, v14, &v110, &v111);
  sub_20174B0(a1, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL), &v112, &v113);
  result = sub_20B7E10(
             *(_QWORD *)a1,
             a2,
             a3,
             a4,
             v106,
             v107,
             a5,
             a6,
             a7,
             *(__int64 **)(a1 + 8),
             1,
             v110,
             v111,
             v112,
             v113);
  if ( (_BYTE)result )
    goto LABEL_31;
  switch ( v9 )
  {
    case 4u:
      v16 = 13;
      v17 = 13;
      break;
    case 5u:
      v16 = 14;
      v17 = 14;
      break;
    case 6u:
      v16 = 15;
      v17 = 15;
      break;
    case 7u:
      v16 = 16;
      v17 = 16;
      break;
    default:
      goto LABEL_13;
  }
  v18 = *(__m128i **)a1;
  if ( *(_QWORD *)(*(_QWORD *)a1 + 8 * v16 + 74096) )
  {
    v19 = *(_QWORD *)(a2 + 32);
    v20 = *(_QWORD *)(a1 + 8);
    v21 = _mm_loadu_si128((const __m128i *)v19);
    v114[0] = v21;
    v22 = _mm_loadu_si128((const __m128i *)(v19 + 40));
    v114[1] = v22;
    sub_20BE530((__int64)&v115, v18, v20, v17, v9, v10, v21, v22, a7, (__int64)v114, 2u, 1u, (__int64)&v108, 0, 1);
    result = sub_200E870(a1, v115, v116, a3, (_QWORD *)a4, v21, *(double *)v22.m128i_i64, a7);
    if ( v108 )
      return sub_161E7C0((__int64)&v108, v108);
    return result;
  }
LABEL_13:
  if ( (_BYTE)v106 )
    v23 = sub_2127930(v106);
  else
    v23 = sub_1F58D40((__int64)&v106);
  LODWORD(v116) = v23;
  v24 = *(_QWORD *)(a1 + 8);
  v25 = v23 >> 1;
  if ( v23 > 0x40 )
    sub_16A4EF0((__int64)&v115, 0, 0);
  else
    v115 = 0;
  if ( (_DWORD)v25 )
  {
    if ( (unsigned int)v25 > 0x40 )
    {
      sub_16A5260(&v115, 0, v25);
    }
    else
    {
      v26 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v25);
      if ( (unsigned int)v116 > 0x40 )
        *(_QWORD *)v115 |= v26;
      else
        v115 |= v26;
    }
  }
  *(_QWORD *)&v99 = sub_1D38970(v24, (__int64)&v115, (__int64)&v108, v106, v107, 0, a5, a6, a7, 0);
  *((_QWORD *)&v99 + 1) = v27;
  if ( (unsigned int)v116 > 0x40 && v115 )
    j_j___libc_free_0_0(v115);
  v28 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          118,
          (__int64)&v108,
          v106,
          v107,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          v110,
          *((unsigned __int64 *)&v110 + 1),
          v99);
  v98 = v29;
  v97 = (__int64)v28;
  *(_QWORD *)&v30 = sub_1D332F0(
                      *(__int64 **)(a1 + 8),
                      118,
                      (__int64)&v108,
                      v106,
                      v107,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      v112,
                      *((unsigned __int64 *)&v112 + 1),
                      v99);
  v91 = v30;
  v31 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          54,
          (__int64)&v108,
          v106,
          v107,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          v97,
          v98,
          v30);
  v95 = v32;
  v93 = (__int64)v31;
  v33 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          118,
          (__int64)&v108,
          v106,
          v107,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          (__int64)v31,
          v32,
          v99);
  v34 = *(_QWORD *)a1;
  v89 = (__int64)v33;
  v90 = v35;
  v36 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL));
  LODWORD(v114[0]) = sub_1F40B60(v34, v106, (__int64)v107, v36, 1);
  *((_QWORD *)&v114[0] + 1) = v37;
  if ( LOBYTE(v114[0]) )
    v38 = sub_2127930(v114[0]);
  else
    v38 = sub_1F58D40((__int64)v114);
  LODWORD(v116) = v38;
  if ( v38 > 0x40 )
  {
    sub_16A4EF0((__int64)&v115, -1, 1);
    v86 = v116;
    if ( (unsigned int)v116 > 0x40 )
    {
      if ( v86 - (unsigned int)sub_16A57B0((__int64)&v115) > 0x40 || *(_QWORD *)v115 >= v25 )
      {
        if ( v115 )
          j_j___libc_free_0_0(v115);
        goto LABEL_30;
      }
      j_j___libc_free_0_0(v115);
      goto LABEL_29;
    }
  }
  else
  {
    v115 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v38;
  }
  if ( v115 < v25 )
  {
LABEL_29:
    LOBYTE(v114[0]) = 5;
    *((_QWORD *)&v114[0] + 1) = 0;
  }
LABEL_30:
  *(_QWORD *)&v39 = sub_1D38BB0(
                      *(_QWORD *)(a1 + 8),
                      v25,
                      (__int64)&v108,
                      LODWORD(v114[0]),
                      *((const void ***)&v114[0] + 1),
                      0,
                      a5,
                      a6,
                      a7,
                      0);
  v102 = v39;
  v40 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          124,
          (__int64)&v108,
          v106,
          v107,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          v93,
          v95,
          v39);
  v42 = v41;
  v43 = v40;
  v44 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          124,
          (__int64)&v108,
          v106,
          v107,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          v110,
          *((unsigned __int64 *)&v110 + 1),
          v102);
  v96 = v45;
  v94 = (__int64)v44;
  *(_QWORD *)&v46 = sub_1D332F0(
                      *(__int64 **)(a1 + 8),
                      124,
                      (__int64)&v108,
                      v106,
                      v107,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      v112,
                      *((unsigned __int64 *)&v112 + 1),
                      v102);
  v47 = *(__int64 **)(a1 + 8);
  v92 = v46;
  v48 = sub_1D332F0(v47, 54, (__int64)&v108, v106, v107, 0, *(double *)a5.m128i_i64, a6, a7, v94, v96, v91);
  *((_QWORD *)&v87 + 1) = v42;
  *(_QWORD *)&v87 = v43;
  v50 = sub_1D332F0(v47, 52, (__int64)&v108, v106, v107, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v48, v49, v87);
  v52 = v51;
  v53 = (__int64)v50;
  *(_QWORD *)&v54 = sub_1D332F0(
                      *(__int64 **)(a1 + 8),
                      118,
                      (__int64)&v108,
                      v106,
                      v107,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      (__int64)v50,
                      v51,
                      v99);
  v100 = v54;
  v55 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          124,
          (__int64)&v108,
          v106,
          v107,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          v53,
          v52,
          v102);
  v56 = *(__int64 **)(a1 + 8);
  v58 = v57;
  v59 = (__int64)v55;
  v60 = sub_1D332F0(v56, 54, (__int64)&v108, v106, v107, 0, *(double *)a5.m128i_i64, a6, a7, v97, v98, v92);
  *(_QWORD *)&v62 = sub_1D332F0(
                      v56,
                      52,
                      (__int64)&v108,
                      v106,
                      v107,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      (__int64)v60,
                      v61,
                      v100);
  v101 = v62;
  *(_QWORD *)&v63 = sub_1D332F0(
                      *(__int64 **)(a1 + 8),
                      124,
                      (__int64)&v108,
                      v106,
                      v107,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      v62,
                      *((unsigned __int64 *)&v62 + 1),
                      v102);
  v64 = *(__int64 **)(a1 + 8);
  v65 = sub_1D332F0(v64, 52, (__int64)&v108, v106, v107, 0, *(double *)a5.m128i_i64, a6, a7, v59, v58, v63);
  v67 = v66;
  v68 = v65;
  v69 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          54,
          (__int64)&v108,
          v106,
          v107,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          v94,
          v96,
          v92);
  *((_QWORD *)&v88 + 1) = v67;
  *(_QWORD *)&v88 = v68;
  v71 = sub_1D332F0(v64, 52, (__int64)&v108, v106, v107, 0, *(double *)a5.m128i_i64, a6, a7, (__int64)v69, v70, v88);
  v72 = *(__int64 **)(a1 + 8);
  v74 = v73;
  v75 = (__int64)v71;
  *(_QWORD *)&v76 = sub_1D332F0(
                      v72,
                      122,
                      (__int64)&v108,
                      v106,
                      v107,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      v101,
                      *((unsigned __int64 *)&v101 + 1),
                      v102);
  v77 = sub_1D332F0(v72, 52, (__int64)&v108, v106, v107, 0, *(double *)a5.m128i_i64, a6, a7, v89, v90, v76);
  v78 = v107;
  *(_QWORD *)a3 = v77;
  *(_DWORD *)(a3 + 8) = v79;
  v80 = *(__int64 **)(a1 + 8);
  *(_QWORD *)&v81 = sub_1D332F0(
                      v80,
                      54,
                      (__int64)&v108,
                      v106,
                      v78,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      v112,
                      *((unsigned __int64 *)&v112 + 1),
                      v111);
  v105 = v81;
  v82 = sub_1D332F0(
          *(__int64 **)(a1 + 8),
          54,
          (__int64)&v108,
          v106,
          v107,
          0,
          *(double *)a5.m128i_i64,
          a6,
          a7,
          v113,
          *((unsigned __int64 *)&v113 + 1),
          v110);
  *(_QWORD *)&v84 = sub_1D332F0(
                      v80,
                      52,
                      (__int64)&v108,
                      v106,
                      v107,
                      0,
                      *(double *)a5.m128i_i64,
                      a6,
                      a7,
                      (__int64)v82,
                      v83,
                      v105);
  *(_QWORD *)a4 = sub_1D332F0(v80, 52, (__int64)&v108, v106, v107, 0, *(double *)a5.m128i_i64, a6, a7, v75, v74, v84);
  result = v85;
  *(_DWORD *)(a4 + 8) = v85;
LABEL_31:
  if ( v108 )
    return sub_161E7C0((__int64)&v108, v108);
  return result;
}
