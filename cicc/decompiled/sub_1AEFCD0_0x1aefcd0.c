// Function: sub_1AEFCD0
// Address: 0x1aefcd0
//
__int64 __fastcall sub_1AEFCD0(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _QWORD *v10; // r14
  const char *v11; // rax
  __int64 v12; // rdx
  _QWORD *v13; // r12
  unsigned __int64 *v14; // rcx
  unsigned __int64 v15; // rdx
  double v16; // xmm4_8
  double v17; // xmm5_8
  int v18; // r9d
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // r12
  int v22; // r12d
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  int v26; // eax
  __int64 *v27; // r14
  __int64 v28; // rdx
  __int64 *v29; // r12
  __int64 v30; // rdx
  int v31; // ecx
  unsigned __int64 v32; // r8
  __int64 *v33; // rax
  bool v34; // sf
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r12
  __int64 v38; // rax
  __int64 v39; // r14
  __int64 v40; // r13
  __int64 v41; // rax
  __int64 *v42; // rax
  __int64 v43; // rsi
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rcx
  const char *v47; // rax
  __int64 v48; // r14
  int v49; // ecx
  __int64 *v50; // rbx
  __int64 *v51; // rsi
  __int64 v52; // rdx
  __int64 *v53; // rdx
  __int64 v54; // rax
  _QWORD *v55; // rax
  double v56; // xmm4_8
  double v57; // xmm5_8
  __int64 v58; // r13
  __int64 v59; // rsi
  __int64 *v60; // r14
  _QWORD *v61; // rbx
  unsigned __int64 *v62; // rcx
  unsigned __int64 v63; // rdx
  double v64; // xmm4_8
  double v65; // xmm5_8
  __int64 *v66; // rbx
  __int64 *v67; // r12
  __int64 v68; // rdi
  __int64 v70; // rsi
  unsigned __int8 *v71; // rsi
  unsigned int v72; // [rsp+Ch] [rbp-144h]
  __int64 v74; // [rsp+20h] [rbp-130h]
  __int64 v75; // [rsp+28h] [rbp-128h]
  __int64 v76; // [rsp+30h] [rbp-120h]
  __int64 *v77; // [rsp+38h] [rbp-118h]
  int v78; // [rsp+38h] [rbp-118h]
  __int64 v79; // [rsp+48h] [rbp-108h]
  _QWORD v80[2]; // [rsp+50h] [rbp-100h] BYREF
  __int64 v81[2]; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v82; // [rsp+70h] [rbp-E0h]
  __int64 *v83; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v84; // [rsp+88h] [rbp-C8h]
  _BYTE v85[64]; // [rsp+90h] [rbp-C0h] BYREF
  __int64 *v86; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v87; // [rsp+D8h] [rbp-78h]
  _WORD v88[56]; // [rsp+E0h] [rbp-70h] BYREF

  v10 = *(_QWORD **)(a1 + 40);
  v74 = (__int64)v10;
  v11 = sub_1649960(a1);
  v86 = (__int64 *)&v83;
  v84 = v12;
  v83 = (__int64 *)v11;
  v88[0] = 773;
  v87 = (__int64)".noexc";
  v79 = sub_157FBF0(v10, (__int64 *)(a1 + 24), (__int64)&v86);
  v13 = (_QWORD *)(v10[5] & 0xFFFFFFFFFFFFFFF8LL);
  sub_157EA20((__int64)(v10 + 5), (__int64)(v13 - 3));
  v14 = (unsigned __int64 *)v13[1];
  v15 = *v13 & 0xFFFFFFFFFFFFFFF8LL;
  *v14 = v15 | *v14 & 7;
  *(_QWORD *)(v15 + 8) = v14;
  *v13 &= 7uLL;
  v13[1] = 0;
  sub_164BEC0((__int64)(v13 - 3), (__int64)(v13 - 3), v15, (__int64)v14, a3, a4, a5, a6, v16, v17, a9, a10);
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_8;
  v19 = sub_1648A40(a1);
  v21 = v19 + v20;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    if ( (unsigned int)(v21 >> 4) )
LABEL_54:
      BUG();
LABEL_8:
    v25 = -24;
    goto LABEL_9;
  }
  if ( !(unsigned int)((v21 - sub_1648A40(a1)) >> 4) )
    goto LABEL_8;
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_54;
  v22 = *(_DWORD *)(sub_1648A40(a1) + 8);
  if ( *(char *)(a1 + 23) >= 0 )
    BUG();
  v23 = sub_1648A40(a1);
  v25 = -24 - 24LL * (unsigned int)(*(_DWORD *)(v23 + v24 - 4) - v22);
LABEL_9:
  v26 = *(_DWORD *)(a1 + 20);
  v27 = (__int64 *)(a1 + v25);
  v86 = (__int64 *)v88;
  v28 = 24LL * (v26 & 0xFFFFFFF);
  v87 = 0x800000000LL;
  v29 = (__int64 *)(a1 - v28);
  v30 = v25 + v28;
  v31 = 0;
  v32 = 0xAAAAAAAAAAAAAAABLL * (v30 >> 3);
  v33 = (__int64 *)v88;
  if ( (unsigned __int64)v30 > 0xC0 )
  {
    v78 = -1431655765 * (v30 >> 3);
    sub_16CD150((__int64)&v86, v88, 0xAAAAAAAAAAAAAAABLL * (v30 >> 3), 8, v32, v18);
    v31 = v87;
    LODWORD(v32) = v78;
    v33 = &v86[(unsigned int)v87];
  }
  if ( v29 != v27 )
  {
    do
    {
      if ( v33 )
        *v33 = *v29;
      v29 += 3;
      ++v33;
    }
    while ( v27 != v29 );
    v31 = v87;
  }
  v34 = *(char *)(a1 + 23) < 0;
  v83 = (__int64 *)v85;
  LODWORD(v87) = v31 + v32;
  v84 = 0x100000000LL;
  if ( v34 )
  {
    v35 = sub_1648A40(a1);
    v37 = v35 + v36;
    if ( *(char *)(a1 + 23) >= 0 )
      v38 = v37 >> 4;
    else
      LODWORD(v38) = (v37 - sub_1648A40(a1)) >> 4;
    v39 = 0;
    v40 = 16LL * (unsigned int)v38;
    if ( (_DWORD)v38 )
    {
      do
      {
        v41 = 0;
        if ( *(char *)(a1 + 23) < 0 )
          v41 = sub_1648A40(a1);
        v42 = (__int64 *)(v39 + v41);
        v39 += 16;
        v43 = *v42;
        v44 = *((unsigned int *)v42 + 2);
        v45 = *((unsigned int *)v42 + 3);
        v46 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        v82 = v43;
        v81[0] = a1 + 24 * v44 - 8 * v46;
        v81[1] = 0xAAAAAAAAAAAAAAABLL * ((24 * v45 - 24 * v44) >> 3);
        sub_1740580((__int64)&v83, (__int64)v81);
      }
      while ( v40 != v39 );
    }
  }
  v47 = sub_1649960(a1);
  v48 = *(_QWORD *)(a1 - 24);
  v49 = 0;
  v80[0] = v47;
  v50 = v83;
  LOWORD(v82) = 261;
  v81[0] = (__int64)v80;
  v77 = v86;
  v75 = (unsigned int)v84;
  v76 = (unsigned int)v87;
  v51 = &v83[7 * (unsigned int)v84];
  v80[1] = v52;
  v53 = v83;
  if ( v83 == v51 )
  {
    v49 = 0;
  }
  else
  {
    do
    {
      v54 = v53[5] - v53[4];
      v53 += 7;
      v49 += v54 >> 3;
    }
    while ( v51 != v53 );
  }
  v72 = v87 + v49 + 3;
  v55 = sub_1648AB0(72, v72, 16 * (int)v84);
  v58 = (__int64)v55;
  if ( v55 )
  {
    sub_15F1F50(
      (__int64)v55,
      **(_QWORD **)(*(_QWORD *)(*(_QWORD *)v48 + 24LL) + 16LL),
      5,
      (__int64)&v55[-3 * v72],
      v72,
      v74);
    *(_QWORD *)(v58 + 56) = 0;
    sub_15F6500(v58, *(_QWORD *)(*(_QWORD *)v48 + 24LL), v48, v79, a2, (__int64)v81, v77, v76, v50, v75);
  }
  v59 = *(_QWORD *)(a1 + 48);
  v60 = (__int64 *)(v58 + 48);
  v81[0] = v59;
  if ( !v59 )
  {
    if ( v60 == v81 )
      goto LABEL_31;
    v70 = *(_QWORD *)(v58 + 48);
    if ( !v70 )
      goto LABEL_31;
LABEL_44:
    sub_161E7C0(v58 + 48, v70);
    goto LABEL_45;
  }
  sub_1623A60((__int64)v81, v59, 2);
  if ( v60 == v81 )
  {
    if ( v81[0] )
      sub_161E7C0((__int64)v81, v81[0]);
    goto LABEL_31;
  }
  v70 = *(_QWORD *)(v58 + 48);
  if ( v70 )
    goto LABEL_44;
LABEL_45:
  v71 = (unsigned __int8 *)v81[0];
  *(_QWORD *)(v58 + 48) = v81[0];
  if ( v71 )
    sub_1623210((__int64)v81, v71, v58 + 48);
LABEL_31:
  *(_WORD *)(v58 + 18) = *(_WORD *)(v58 + 18) & 0x8000
                       | *(_WORD *)(v58 + 18) & 3
                       | (4 * ((*(_WORD *)(a1 + 18) >> 2) & 0xDFFF));
  *(_QWORD *)(v58 + 56) = *(_QWORD *)(a1 + 56);
  sub_164D160(a1, v58, a3, a4, a5, a6, v56, v57, a9, a10);
  v61 = *(_QWORD **)(v79 + 48);
  sub_157EA20(v79 + 40, (__int64)(v61 - 3));
  v62 = (unsigned __int64 *)v61[1];
  v63 = *v61 & 0xFFFFFFFFFFFFFFF8LL;
  *v62 = v63 | *v62 & 7;
  *(_QWORD *)(v63 + 8) = v62;
  *v61 &= 7uLL;
  v61[1] = 0;
  sub_164BEC0((__int64)(v61 - 3), (__int64)(v61 - 3), v63, (__int64)v62, a3, a4, a5, a6, v64, v65, a9, a10);
  v66 = v83;
  v67 = &v83[7 * (unsigned int)v84];
  if ( v83 != v67 )
  {
    do
    {
      v68 = *(v67 - 3);
      v67 -= 7;
      if ( v68 )
        j_j___libc_free_0(v68, v67[6] - v68);
      if ( (__int64 *)*v67 != v67 + 2 )
        j_j___libc_free_0(*v67, v67[2] + 1);
    }
    while ( v66 != v67 );
    v67 = v83;
  }
  if ( v67 != (__int64 *)v85 )
    _libc_free((unsigned __int64)v67);
  if ( v86 != (__int64 *)v88 )
    _libc_free((unsigned __int64)v86);
  return v79;
}
