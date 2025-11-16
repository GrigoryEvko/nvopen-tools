// Function: sub_1A396A0
// Address: 0x1a396a0
//
__int64 __fastcall sub_1A396A0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  _QWORD **v12; // r12
  __int64 v13; // rbx
  __int64 v15; // rax
  __int64 v16; // rcx
  _QWORD *v17; // r8
  int v18; // r9d
  unsigned __int64 v19; // rax
  _QWORD *v20; // r8
  __int64 v21; // r9
  __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  _QWORD *v24; // r13
  unsigned __int64 v25; // r15
  _QWORD *v26; // rbx
  __int64 v27; // r12
  int v28; // r9d
  double v29; // xmm4_8
  double v30; // xmm5_8
  unsigned int v31; // r13d
  unsigned int v32; // r13d
  __int64 v33; // rax
  char v34; // r15
  _BYTE *v35; // rsi
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  _QWORD *v39; // r8
  int v40; // r9d
  unsigned __int64 v41; // rax
  __int64 v42; // rax
  __m128 *v43; // rax
  _QWORD **v44; // r12
  __int64 v45; // rdx
  __int64 v46; // rbx
  __int64 *v47; // rax
  unsigned __int64 *v48; // rsi
  __int64 v49; // rax
  __int64 *v50; // r15
  __int64 v51; // r12
  unsigned __int64 v52; // r14
  unsigned __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r14
  __int64 *v56; // rdx
  unsigned __int64 *v57; // r13
  __int64 *v58; // r12
  __int64 *v59; // r13
  _QWORD *v60; // rdi
  unsigned __int64 **v61; // rax
  unsigned __int64 v62; // rdi
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rdi
  __int64 v66; // r15
  __int64 *v67; // r13
  __m128i *v68; // rsi
  unsigned __int64 v69; // rax
  unsigned __int64 *v70; // r15
  unsigned __int64 *v71; // rdi
  _BYTE *v72; // rdi
  size_t v73; // rdx
  _BYTE *v74; // rdi
  size_t v75; // rdx
  __int64 v76; // [rsp+10h] [rbp-3A0h]
  __int64 v77; // [rsp+18h] [rbp-398h]
  __int64 v78; // [rsp+28h] [rbp-388h]
  __int64 *dest; // [rsp+30h] [rbp-380h]
  char v80; // [rsp+38h] [rbp-378h]
  void *v81; // [rsp+38h] [rbp-378h]
  __int64 *v82; // [rsp+38h] [rbp-378h]
  __int64 v83; // [rsp+40h] [rbp-370h]
  __int64 v84; // [rsp+40h] [rbp-370h]
  unsigned __int8 v85; // [rsp+48h] [rbp-368h]
  char v86; // [rsp+48h] [rbp-368h]
  __int64 v87; // [rsp+48h] [rbp-368h]
  __int64 v88; // [rsp+58h] [rbp-358h] BYREF
  __m128i v89; // [rsp+60h] [rbp-350h] BYREF
  unsigned __int64 v90; // [rsp+70h] [rbp-340h]
  __int64 v91; // [rsp+80h] [rbp-330h] BYREF
  __int64 v92; // [rsp+88h] [rbp-328h]
  __int64 v93; // [rsp+90h] [rbp-320h]
  __int64 v94; // [rsp+98h] [rbp-318h]
  _BYTE *v95; // [rsp+A0h] [rbp-310h] BYREF
  __int64 v96; // [rsp+A8h] [rbp-308h]
  _BYTE v97[32]; // [rsp+B0h] [rbp-300h] BYREF
  __int64 v98; // [rsp+D0h] [rbp-2E0h]
  __int64 v99; // [rsp+D8h] [rbp-2D8h]
  __int64 v100; // [rsp+E0h] [rbp-2D0h] BYREF
  __int64 v101; // [rsp+E8h] [rbp-2C8h] BYREF
  __int64 v102; // [rsp+F0h] [rbp-2C0h]
  __int64 v103; // [rsp+F8h] [rbp-2B8h]
  _BYTE *v104; // [rsp+100h] [rbp-2B0h] BYREF
  __int64 v105; // [rsp+108h] [rbp-2A8h]
  _BYTE v106[32]; // [rsp+110h] [rbp-2A0h] BYREF
  __int64 v107; // [rsp+130h] [rbp-280h]
  __int64 v108; // [rsp+138h] [rbp-278h]
  __int64 *v109; // [rsp+140h] [rbp-270h] BYREF
  __int64 v110; // [rsp+148h] [rbp-268h]
  _BYTE v111[96]; // [rsp+150h] [rbp-260h] BYREF
  _QWORD v112[4]; // [rsp+1B0h] [rbp-200h] BYREF
  void *v113; // [rsp+1D0h] [rbp-1E0h]
  unsigned int v114; // [rsp+1D8h] [rbp-1D8h]
  char v115; // [rsp+1E0h] [rbp-1D0h] BYREF
  __int64 v116; // [rsp+200h] [rbp-1B0h]
  __int64 v117; // [rsp+208h] [rbp-1A8h]
  __int64 v118; // [rsp+210h] [rbp-1A0h]
  __int64 v119; // [rsp+218h] [rbp-198h]
  __int64 v120; // [rsp+220h] [rbp-190h]
  __int64 v121; // [rsp+228h] [rbp-188h]
  void *src; // [rsp+230h] [rbp-180h]
  unsigned int v123; // [rsp+238h] [rbp-178h]
  char v124; // [rsp+240h] [rbp-170h] BYREF
  __int64 v125; // [rsp+260h] [rbp-150h]
  __int64 v126; // [rsp+268h] [rbp-148h]

  v85 = 0;
  if ( !(3LL * *(unsigned int *)(a3 + 16)) )
    return v85;
  v12 = (_QWORD **)a1;
  v13 = a3;
  v15 = sub_15F2050(a2);
  v83 = sub_1632FA0(v15);
  v86 = sub_1A33E80(a1, a2, v13, v16, v17, v18);
  v19 = sub_12BE0A0(v83, *(_QWORD *)(a2 + 56));
  v22 = *(_QWORD *)(v13 + 8);
  v80 = 1;
  v23 = v19;
  v24 = (_QWORD *)(v22 + 24LL * *(unsigned int *)(v13 + 16));
  if ( (_QWORD *)v22 != v24 )
  {
    v25 = v19;
    v78 = v13;
    v26 = *(_QWORD **)(v13 + 8);
    do
    {
      v27 = v26[2];
      if ( (v27 & 4) != 0
        && (*v26 || v25 > v26[1])
        && (unsigned __int8)(*((_BYTE *)sub_1648700(v27 & 0xFFFFFFFFFFFFFFF8LL) + 16) - 54) <= 1u )
      {
        v80 = 0;
        v23 = v27 & 0xFFFFFFFFFFFFFFFBLL;
        v26[2] = v27 & 0xFFFFFFFFFFFFFFFBLL;
      }
      v26 += 3;
    }
    while ( v24 != v26 );
    v12 = (_QWORD **)a1;
    v13 = v78;
    if ( !v80 )
    {
      v64 = *(unsigned int *)(v78 + 16);
      v65 = *(_QWORD *)(v78 + 8);
      v66 = 24 * v64;
      v67 = (__int64 *)(v65 + 24 * v64);
      if ( (__int64 *)v65 != v67 )
      {
        v68 = (__m128i *)(v65 + 24 * v64);
        v82 = *(__int64 **)(v78 + 8);
        _BitScanReverse64(&v69, 0xAAAAAAAAAAAAAAABLL * (v66 >> 3));
        sub_1A1B940(v65, v68, 2LL * (int)(63 - (v69 ^ 0x3F)), v23, (__int64)v20, v21);
        if ( (unsigned __int64)v66 <= 0x180 )
        {
          sub_1A1AE20(v82, v67);
        }
        else
        {
          v70 = (unsigned __int64 *)(v82 + 48);
          sub_1A1AE20(v82, v82 + 48);
          if ( v67 != v82 + 48 )
          {
            do
            {
              v71 = v70;
              v70 += 3;
              sub_1A1AC50(v71);
            }
            while ( v67 != (__int64 *)v70 );
          }
        }
      }
    }
  }
  v109 = (__int64 *)v111;
  v110 = 0x400000000LL;
  sub_1A217D0(v112, v13, v22, v23, v20, v21);
  v96 = 0x400000000LL;
  v31 = v114;
  v91 = v112[0];
  v92 = v112[1];
  v93 = v112[2];
  v94 = v112[3];
  v95 = v97;
  if ( v114 )
  {
    v74 = v97;
    v75 = 8LL * v114;
    if ( v114 <= 4 || (sub_16CD150((__int64)&v95, v97, v114, 8, v114, v28), v74 = v95, (v75 = 8LL * v114) != 0) )
      memcpy(v74, v113, v75);
    LODWORD(v96) = v31;
  }
  v32 = v123;
  v98 = v116;
  v104 = v106;
  v99 = v117;
  v103 = v121;
  v100 = v118;
  v105 = 0x400000000LL;
  v101 = v119;
  v33 = v120;
  v102 = v120;
  if ( v123 )
  {
    v72 = v106;
    v73 = 8LL * v123;
    if ( v123 <= 4 || (sub_16CD150((__int64)&v104, v106, v123, 8, v123, v28), v72 = v104, (v73 = 8LL * v123) != 0) )
      memcpy(v72, src, v73);
    LODWORD(v105) = v32;
    v33 = v102;
  }
  v34 = v86;
  v107 = v125;
  v108 = v126;
  while ( v33 != v93 || ((_DWORD)v96 == 0) != ((_DWORD)v105 == 0) )
  {
    v35 = (_BYTE *)a2;
    v36 = sub_1A37040(v12, a2, v13, (__int64)&v91, a4, a5, a6, a7, v29, v30, a10, a11);
    if ( v36 )
    {
      v34 = 1;
      if ( a2 != v36 )
      {
        v87 = v36;
        v35 = (_BYTE *)sub_127FA20(v83, *(_QWORD *)(v36 + 56));
        v89.m128i_i64[0] = v87;
        v41 = 8 * (v92 - v91);
        if ( v41 > (unsigned __int64)v35 )
          v41 = (unsigned __int64)v35;
        v38 = 8 * v91;
        v89.m128i_i64[1] = 8 * v91;
        v90 = v41;
        v42 = (unsigned int)v110;
        if ( (unsigned int)v110 >= HIDWORD(v110) )
        {
          v35 = v111;
          sub_16CD150((__int64)&v109, v111, 0, 24, (int)v39, v40);
          v42 = (unsigned int)v110;
        }
        a4 = (__m128)_mm_loadu_si128(&v89);
        v34 = 1;
        v43 = (__m128 *)&v109[3 * v42];
        *v43 = a4;
        v37 = v90;
        LODWORD(v110) = v110 + 1;
        v43[1].m128_u64[0] = v90;
      }
    }
    sub_1A21450((__int64)&v91, (__int64)v35, v37, v38, v39, v40);
    v33 = v102;
  }
  v85 = v34;
  if ( v104 != v106 )
    _libc_free((unsigned __int64)v104);
  if ( v95 != v97 )
    _libc_free((unsigned __int64)v95);
  if ( src != &v124 )
    _libc_free((unsigned __int64)src);
  if ( v113 != &v115 )
    _libc_free((unsigned __int64)v113);
  sub_1AEA030(&v88);
  v44 = (_QWORD **)(v88 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v88 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
  {
    if ( (v88 & 4) != 0 )
    {
      if ( !*((_DWORD *)v44 + 2) )
      {
LABEL_72:
        if ( *v44 != v44 + 2 )
          _libc_free((unsigned __int64)*v44);
        j_j___libc_free_0(v44, 48);
        goto LABEL_75;
      }
      v44 = (_QWORD **)**v44;
    }
    v45 = *((_DWORD *)v44 + 5) & 0xFFFFFFF;
    v77 = v44[3 * (1 - v45)][3];
    v46 = v44[3 * (2 - v45)][3];
    sub_15B1130((__int64)&v89, v77);
    v47 = (__int64 *)sub_15F2050(a2);
    sub_15A5590((__int64)v112, v47, 0, 0);
    v48 = *(unsigned __int64 **)(a2 + 56);
    v49 = sub_127FA20(v83, (__int64)v48);
    v50 = v109;
    v81 = (void *)v49;
    dest = &v109[3 * (unsigned int)v110];
    if ( v109 != dest )
    {
      v76 = a2;
      while ( 1 )
      {
        v51 = v50[1];
        v52 = v50[2];
        v84 = *v50;
        if ( v52 >= (unsigned __int64)v81 )
        {
          sub_15B1350((__int64)&v100, *(unsigned __int64 **)(v46 + 24), *(unsigned __int64 **)(v46 + 32));
          if ( !(_BYTE)v102 )
            goto LABEL_78;
        }
        v48 = *(unsigned __int64 **)(v46 + 24);
        sub_15B1350((__int64)&v91, v48, *(unsigned __int64 **)(v46 + 32));
        if ( (_BYTE)v93 )
        {
          if ( v91 + v92 <= (unsigned __int64)(v51 + v92) )
            goto LABEL_39;
          v53 = v91 - v51;
          v51 += v92;
          if ( v52 > v53 )
            v52 = v53;
        }
        v48 = *(unsigned __int64 **)(v46 + 24);
        sub_15B1350((__int64)&v100, v48, *(unsigned __int64 **)(v46 + 32));
        if ( (_BYTE)v102 )
          v51 -= v101;
        if ( v89.m128i_i8[8] )
        {
          v54 = v52;
          if ( v89.m128i_i64[0] <= v52 )
            v54 = v89.m128i_i64[0];
          if ( !v54 )
            goto LABEL_39;
          v48 = (unsigned __int64 *)(v51 + v54);
          if ( v89.m128i_i64[0] < (unsigned __int64)(v51 + v54) )
            goto LABEL_39;
          if ( v89.m128i_i64[0] > v52 )
          {
            LODWORD(v52) = v54;
            goto LABEL_55;
          }
LABEL_78:
          v55 = v46;
LABEL_57:
          sub_1AEA030(&v100);
          if ( (v100 & 4) != 0 )
          {
            v56 = *(__int64 **)(v100 & 0xFFFFFFFFFFFFFFF8LL);
            v57 = (unsigned __int64 *)(v100 & 0xFFFFFFFFFFFFFFF8LL);
            v58 = &v56[*(unsigned int *)((v100 & 0xFFFFFFFFFFFFFFF8LL) + 8)];
            if ( v56 == v58 )
              goto LABEL_63;
          }
          else
          {
            if ( (v100 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
              goto LABEL_67;
            v58 = &v101;
            v56 = &v100;
          }
          v59 = v56;
          do
          {
            v60 = (_QWORD *)*v59++;
            sub_15F20C0(v60);
          }
          while ( v58 != v59 );
          if ( (v100 & 4) != 0 )
          {
            v57 = (unsigned __int64 *)(v100 & 0xFFFFFFFFFFFFFFF8LL);
LABEL_63:
            if ( v57 )
            {
              if ( (unsigned __int64 *)*v57 != v57 + 2 )
                _libc_free(*v57);
              j_j___libc_free_0(v57, 48);
            }
          }
LABEL_67:
          v61 = (unsigned __int64 **)(v88 & 0xFFFFFFFFFFFFFFF8LL);
          v62 = v88 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v88 & 4) != 0 || !v61 )
            v62 = **v61;
          v50 += 3;
          v63 = sub_15C70A0(v62 + 48);
          v48 = (unsigned __int64 *)v84;
          sub_15A7500(v112, v84, v77, v55, v63, v76);
          if ( dest == v50 )
            break;
        }
        else
        {
LABEL_55:
          v48 = (unsigned __int64 *)v46;
          sub_15C4EF0((__int64)&v100, (_QWORD *)v46, v51, v52);
          if ( (_BYTE)v101 )
          {
            v55 = v100;
            goto LABEL_57;
          }
LABEL_39:
          v50 += 3;
          if ( dest == v50 )
            break;
        }
      }
    }
    sub_129E320((__int64)v112, (__int64)v48);
    if ( (v88 & 4) != 0 )
    {
      v44 = (_QWORD **)(v88 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v88 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        goto LABEL_72;
    }
  }
LABEL_75:
  if ( v109 != (__int64 *)v111 )
    _libc_free((unsigned __int64)v109);
  return v85;
}
