// Function: sub_196B740
// Address: 0x196b740
//
__int64 __fastcall sub_196B740(
        __int64 a1,
        __int64 *a2,
        unsigned int a3,
        unsigned int a4,
        unsigned __int8 *a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14,
        __int64 a15,
        __int64 a16,
        __int64 a17,
        char a18,
        char a19)
{
  __int64 v21; // rax
  unsigned int v22; // r12d
  __int64 v23; // r13
  unsigned __int64 v24; // r14
  _QWORD *v25; // rax
  unsigned __int8 *v26; // rsi
  __int64 v27; // rax
  __int64 v28; // rax
  unsigned __int8 *v29; // r14
  __int64 v30; // rax
  __int64 *v31; // rax
  __int64 v32; // r12
  unsigned int v33; // r12d
  _QWORD *v34; // rbx
  _QWORD *v35; // r13
  __int64 v36; // rax
  unsigned __int64 v38; // rax
  _QWORD *v39; // rcx
  __int64 v40; // r15
  unsigned __int64 v41; // rax
  __int64 *v42; // rax
  __int64 *v43; // r13
  __int64 v44; // rbx
  __int64 *v45; // r13
  __int64 v46; // rsi
  __int64 ****v47; // rax
  __int64 v48; // rdx
  __int64 ****v49; // r14
  __int64 ***v50; // r13
  __int64 ****v51; // rbx
  __int64 v52; // rax
  double v53; // xmm4_8
  double v54; // xmm5_8
  __int64 ****v55; // rax
  __int64 v56; // rsi
  unsigned __int8 *v57; // rsi
  __int64 v58; // r14
  __int64 *v59; // rax
  __int64 v60; // rax
  __int64 v61; // rdx
  _QWORD *v62; // rax
  __int64 v63; // rbx
  __int64 v64; // rax
  __int64 v66; // [rsp+8h] [rbp-2A8h]
  __int64 v67; // [rsp+8h] [rbp-2A8h]
  __int64 *v69; // [rsp+18h] [rbp-298h]
  __int64 **v70; // [rsp+20h] [rbp-290h]
  __int64 v71; // [rsp+28h] [rbp-288h]
  __int64 v72; // [rsp+28h] [rbp-288h]
  _QWORD *v73; // [rsp+38h] [rbp-278h]
  _QWORD v74[4]; // [rsp+40h] [rbp-270h] BYREF
  __int64 v75[2]; // [rsp+60h] [rbp-250h] BYREF
  _QWORD v76[4]; // [rsp+70h] [rbp-240h] BYREF
  __int64 v77[3]; // [rsp+90h] [rbp-220h] BYREF
  _QWORD *v78; // [rsp+A8h] [rbp-208h]
  __int64 v79; // [rsp+B0h] [rbp-200h]
  int v80; // [rsp+B8h] [rbp-1F8h]
  __int64 v81; // [rsp+C0h] [rbp-1F0h]
  __int64 v82; // [rsp+C8h] [rbp-1E8h]
  unsigned __int8 *v83[4]; // [rsp+E0h] [rbp-1D0h] BYREF
  _QWORD *v84; // [rsp+100h] [rbp-1B0h]
  __int64 v85; // [rsp+108h] [rbp-1A8h]
  unsigned int v86; // [rsp+110h] [rbp-1A0h]
  __int64 v87; // [rsp+118h] [rbp-198h]
  __int64 v88; // [rsp+120h] [rbp-190h]
  __int64 v89; // [rsp+128h] [rbp-188h]
  __int64 v90; // [rsp+130h] [rbp-180h]
  __int64 v91; // [rsp+138h] [rbp-178h]
  __int64 v92; // [rsp+140h] [rbp-170h]
  __int64 v93; // [rsp+148h] [rbp-168h]
  __int64 v94; // [rsp+150h] [rbp-160h]
  __int64 v95; // [rsp+158h] [rbp-158h]
  __int64 v96; // [rsp+160h] [rbp-150h]
  __int64 v97; // [rsp+168h] [rbp-148h]
  int v98; // [rsp+170h] [rbp-140h]
  __int64 v99; // [rsp+178h] [rbp-138h]
  _BYTE *v100; // [rsp+180h] [rbp-130h]
  _BYTE *v101; // [rsp+188h] [rbp-128h]
  __int64 v102; // [rsp+190h] [rbp-120h]
  int v103; // [rsp+198h] [rbp-118h]
  _BYTE v104[16]; // [rsp+1A0h] [rbp-110h] BYREF
  __int64 v105; // [rsp+1B0h] [rbp-100h]
  __int64 v106; // [rsp+1B8h] [rbp-F8h]
  __int64 v107; // [rsp+1C0h] [rbp-F0h]
  __int64 v108; // [rsp+1C8h] [rbp-E8h]
  __int64 v109; // [rsp+1D0h] [rbp-E0h]
  __int64 v110; // [rsp+1D8h] [rbp-D8h]
  __int16 v111; // [rsp+1E0h] [rbp-D0h]
  __int64 v112[5]; // [rsp+1E8h] [rbp-C8h] BYREF
  int v113; // [rsp+210h] [rbp-A0h]
  __int64 v114; // [rsp+218h] [rbp-98h]
  __int64 v115; // [rsp+220h] [rbp-90h]
  unsigned __int8 *v116; // [rsp+228h] [rbp-88h]
  _BYTE *v117; // [rsp+230h] [rbp-80h]
  __int64 v118; // [rsp+238h] [rbp-78h]
  _BYTE v119[112]; // [rsp+240h] [rbp-70h] BYREF

  v69 = 0;
  v66 = sub_14ABE30(a5);
  if ( !v66 )
    v69 = (__int64 *)sub_1969620((__int64)a5, *(_BYTE **)(a1 + 56));
  v21 = *a2;
  if ( *(_BYTE *)(*a2 + 8) == 16 )
    v21 = **(_QWORD **)(v21 + 16);
  v22 = *(_DWORD *)(v21 + 8) >> 8;
  v23 = sub_13FC520(*(_QWORD *)a1);
  v24 = sub_157EBA0(v23);
  v25 = (_QWORD *)sub_16498A0(v24);
  v81 = 0;
  v82 = 0;
  v26 = *(unsigned __int8 **)(v24 + 48);
  v78 = v25;
  v80 = 0;
  v27 = *(_QWORD *)(v24 + 40);
  v77[0] = 0;
  v77[1] = v27;
  v79 = 0;
  v77[2] = v24 + 24;
  v83[0] = v26;
  if ( v26 )
  {
    sub_1623A60((__int64)v83, (__int64)v26, 2);
    v77[0] = (__int64)v83[0];
    if ( v83[0] )
      sub_1623210((__int64)v83, v83[0], (__int64)v77);
  }
  v28 = *(_QWORD *)(a1 + 32);
  v29 = *(unsigned __int8 **)(a1 + 56);
  v100 = v104;
  v101 = v104;
  v83[2] = "loop-idiom";
  v83[0] = (unsigned __int8 *)v28;
  v83[1] = v29;
  v83[3] = 0;
  v84 = 0;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v102 = 2;
  v103 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 1;
  v30 = sub_15E0530(*(_QWORD *)(v28 + 24));
  memset(v112, 0, 24);
  v112[3] = v30;
  v117 = v119;
  v112[4] = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = v29;
  v118 = 0x800000000LL;
  v70 = (__int64 **)sub_16471D0(v78, v22);
  v71 = sub_15A9620(*(_QWORD *)(a1 + 56), (__int64)v78, v22);
  v31 = *(__int64 **)(a16 + 32);
  v32 = *v31;
  if ( a18 )
    v32 = sub_1969510(*v31, a17, v71, a3, *(_QWORD **)(a1 + 32), a7, a8);
  if ( !(unsigned __int8)sub_3870AF0(v32, *(_QWORD *)(a1 + 32)) )
    goto LABEL_11;
  v38 = sub_157EBA0(v23);
  v73 = (_QWORD *)sub_38767A0(v83, v32, v70, v38);
  if ( (unsigned __int8)sub_1969150((__int64)v73, 7u, *(_QWORD *)a1, a17, a3, *(_QWORD **)(a1 + 8), a15) )
  {
    sub_196A390((__int64)v83);
    sub_1AEB370(v73, *(_QWORD *)(a1 + 40));
LABEL_11:
    v33 = 0;
    goto LABEL_12;
  }
  v39 = *(_QWORD **)a1;
  if ( *(_BYTE *)(a1 + 64) && (unsigned int)((__int64)(v39[5] - v39[4]) >> 3) > 1 && !*v39 && !a19 )
    goto LABEL_11;
  v40 = sub_19699C0(a17, v71, a3, (__int64)v39, *(_QWORD *)(a1 + 56), *(_QWORD **)(a1 + 32), a7, a8);
  v33 = sub_3870AF0(v40, *(_QWORD *)(a1 + 32));
  if ( !(_BYTE)v33 )
    goto LABEL_11;
  v41 = sub_157EBA0(v23);
  v42 = (__int64 *)sub_38767A0(v83, v40, v71, v41);
  v43 = v42;
  if ( v66 )
  {
    v44 = (__int64)sub_15E7280(v77, v73, v66, v42, a4, 0, 0, 0, 0);
  }
  else
  {
    v58 = sub_15F2050(a6);
    v59 = (__int64 *)sub_1643270(v78);
    v75[0] = (__int64)v76;
    v76[0] = v70;
    v76[1] = v70;
    v76[2] = v71;
    v75[1] = 0x300000003LL;
    v60 = sub_1644EA0(v59, v76, 3, 0);
    v72 = sub_1632080(v58, (__int64)"memset_pattern16", 16, v60, 0);
    sub_1AB1740(v58, "memset_pattern16", 16, *(_QWORD *)(a1 + 40));
    v61 = *v69;
    v75[0] = (__int64)".memset_pattern";
    v67 = v61;
    LOWORD(v76[0]) = 259;
    v62 = sub_1648A60(88, 1u);
    v63 = (__int64)v62;
    if ( v62 )
      sub_15E51E0((__int64)v62, v58, v67, 1, 8, (__int64)v69, (__int64)v75, 0, 0, 0, 0);
    *(_BYTE *)(v63 + 32) = *(_BYTE *)(v63 + 32) & 0x3F | 0x80;
    sub_15E4CC0(v63, 0x10u);
    v64 = sub_15A4510((__int64 ***)v63, v70, 0);
    LOWORD(v76[0]) = 257;
    v74[1] = v64;
    v74[2] = v43;
    v74[0] = v73;
    v44 = sub_1285290(v77, *(_QWORD *)(*(_QWORD *)v72 + 24LL), v72, (int)v74, 3, (__int64)v75, 0);
  }
  v45 = (__int64 *)(v44 + 48);
  v46 = *(_QWORD *)(a6 + 48);
  v75[0] = v46;
  if ( !v46 )
  {
    if ( v45 == v75 )
      goto LABEL_45;
    v56 = *(_QWORD *)(v44 + 48);
    if ( !v56 )
      goto LABEL_45;
LABEL_59:
    sub_161E7C0(v44 + 48, v56);
    goto LABEL_60;
  }
  sub_1623A60((__int64)v75, v46, 2);
  if ( v45 == v75 )
  {
    if ( v75[0] )
      sub_161E7C0((__int64)v75, v75[0]);
    goto LABEL_45;
  }
  v56 = *(_QWORD *)(v44 + 48);
  if ( v56 )
    goto LABEL_59;
LABEL_60:
  v57 = (unsigned __int8 *)v75[0];
  *(_QWORD *)(v44 + 48) = v75[0];
  if ( v57 )
    sub_1623210((__int64)v75, v57, v44 + 48);
LABEL_45:
  v47 = *(__int64 *****)(a15 + 16);
  if ( v47 == *(__int64 *****)(a15 + 8) )
    v48 = *(unsigned int *)(a15 + 28);
  else
    v48 = *(unsigned int *)(a15 + 24);
  v49 = &v47[v48];
  if ( v47 != v49 )
  {
    while ( 1 )
    {
      v50 = *v47;
      v51 = v47;
      if ( (unsigned __int64)*v47 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v49 == ++v47 )
        goto LABEL_12;
    }
    while ( v49 != v51 )
    {
      v52 = sub_1599EF0(*v50);
      sub_164D160((__int64)v50, v52, (__m128)a7, *(double *)a8.m128i_i64, a9, a10, v53, v54, a13, a14);
      sub_15F20C0(v50);
      v55 = v51 + 1;
      if ( v51 + 1 == v49 )
        break;
      while ( 1 )
      {
        v50 = *v55;
        v51 = v55;
        if ( (unsigned __int64)*v55 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v49 == ++v55 )
          goto LABEL_12;
      }
    }
  }
LABEL_12:
  if ( v117 != v119 )
    _libc_free((unsigned __int64)v117);
  if ( v112[0] )
    sub_161E7C0((__int64)v112, v112[0]);
  j___libc_free_0(v108);
  if ( v101 != v100 )
    _libc_free((unsigned __int64)v101);
  j___libc_free_0(v96);
  j___libc_free_0(v92);
  j___libc_free_0(v88);
  if ( v86 )
  {
    v34 = v84;
    v35 = &v84[5 * v86];
    do
    {
      while ( *v34 == -8 )
      {
        if ( v34[1] != -8 )
          goto LABEL_21;
        v34 += 5;
        if ( v35 == v34 )
          goto LABEL_28;
      }
      if ( *v34 != -16 || v34[1] != -16 )
      {
LABEL_21:
        v36 = v34[4];
        if ( v36 != -8 && v36 != 0 && v36 != -16 )
          sub_1649B30(v34 + 2);
      }
      v34 += 5;
    }
    while ( v35 != v34 );
  }
LABEL_28:
  j___libc_free_0(v84);
  if ( v77[0] )
    sub_161E7C0((__int64)v77, v77[0]);
  return v33;
}
