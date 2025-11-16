// Function: sub_1AF0320
// Address: 0x1af0320
//
void __fastcall sub_1AF0320(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10,
        __int64 a11,
        __int64 a12,
        int a13,
        int a14)
{
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // rbx
  int v18; // ebx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // eax
  __int64 *v23; // r13
  __int64 v24; // rdx
  __int64 *v25; // rbx
  __int64 v26; // rdx
  int v27; // ecx
  unsigned __int64 v28; // r14
  __int64 *v29; // rax
  bool v30; // sf
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rbx
  __int64 v34; // rbx
  __int64 v35; // r13
  __int64 v36; // rbx
  __int64 v37; // rax
  __int64 *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rsi
  __int64 v41; // rax
  __int64 v42; // rcx
  __int64 *v43; // rbx
  __int64 *v44; // r13
  __int64 *v45; // rdx
  int v46; // edi
  __int64 v47; // rax
  _QWORD *v48; // rax
  int v49; // r8d
  __int64 v50; // r14
  __int64 v51; // rax
  __int64 *v52; // r10
  __int64 v53; // r11
  __int64 v54; // rsi
  __int64 v55; // rdx
  int v56; // r8d
  __int64 v57; // rcx
  __int64 *v58; // r13
  double v59; // xmm4_8
  double v60; // xmm5_8
  __int64 v61; // rsi
  __int64 v62; // r13
  _QWORD *v63; // rdi
  __int64 v64; // r13
  __int64 v65; // r14
  __int64 *v66; // rbx
  __int64 *v67; // r12
  __int64 v68; // rdi
  __int64 v69; // rsi
  unsigned __int8 *v70; // rsi
  _QWORD *v71; // rax
  __int64 v72; // [rsp+0h] [rbp-140h]
  __int64 *v73; // [rsp+0h] [rbp-140h]
  __int64 v74; // [rsp+0h] [rbp-140h]
  __int64 v75; // [rsp+8h] [rbp-138h]
  __int64 v76; // [rsp+10h] [rbp-130h]
  __int64 v77; // [rsp+10h] [rbp-130h]
  __int64 v78; // [rsp+10h] [rbp-130h]
  int v80; // [rsp+28h] [rbp-118h]
  __int64 v81; // [rsp+28h] [rbp-118h]
  int v82; // [rsp+28h] [rbp-118h]
  __int64 *v83; // [rsp+30h] [rbp-110h]
  __int64 v84; // [rsp+38h] [rbp-108h]
  __int64 v85; // [rsp+40h] [rbp-100h]
  __int64 v86[2]; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v87; // [rsp+60h] [rbp-E0h]
  __int64 *v88; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v89; // [rsp+78h] [rbp-C8h]
  _BYTE v90[64]; // [rsp+80h] [rbp-C0h] BYREF
  __int64 *v91; // [rsp+C0h] [rbp-80h] BYREF
  __int64 v92; // [rsp+C8h] [rbp-78h]
  _BYTE v93[112]; // [rsp+D0h] [rbp-70h] BYREF

  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_8;
  v15 = sub_1648A40(a1);
  v17 = v15 + v16;
  if ( *(char *)(a1 + 23) >= 0 )
  {
    if ( (unsigned int)(v17 >> 4) )
LABEL_63:
      BUG();
LABEL_8:
    v21 = -72;
    goto LABEL_9;
  }
  if ( !(unsigned int)((v17 - sub_1648A40(a1)) >> 4) )
    goto LABEL_8;
  if ( *(char *)(a1 + 23) >= 0 )
    goto LABEL_63;
  v18 = *(_DWORD *)(sub_1648A40(a1) + 8);
  if ( *(char *)(a1 + 23) >= 0 )
    BUG();
  v19 = sub_1648A40(a1);
  v21 = -72 - 24LL * (unsigned int)(*(_DWORD *)(v19 + v20 - 4) - v18);
LABEL_9:
  v22 = *(_DWORD *)(a1 + 20);
  v23 = (__int64 *)(a1 + v21);
  v91 = (__int64 *)v93;
  v24 = 24LL * (v22 & 0xFFFFFFF);
  v92 = 0x800000000LL;
  v25 = (__int64 *)(a1 - v24);
  v26 = v21 + v24;
  v27 = 0;
  v28 = 0xAAAAAAAAAAAAAAABLL * (v26 >> 3);
  v29 = (__int64 *)v93;
  if ( (unsigned __int64)v26 > 0xC0 )
  {
    sub_16CD150((__int64)&v91, v93, 0xAAAAAAAAAAAAAAABLL * (v26 >> 3), 8, a13, a14);
    v27 = v92;
    v29 = &v91[(unsigned int)v92];
  }
  if ( v25 != v23 )
  {
    do
    {
      if ( v29 )
        *v29 = *v25;
      v25 += 3;
      ++v29;
    }
    while ( v23 != v25 );
    v27 = v92;
  }
  v30 = *(char *)(a1 + 23) < 0;
  v88 = (__int64 *)v90;
  LODWORD(v92) = v27 + v28;
  v89 = 0x100000000LL;
  if ( v30 )
  {
    v31 = sub_1648A40(a1);
    v33 = v31 + v32;
    if ( *(char *)(a1 + 23) < 0 )
      v33 -= sub_1648A40(a1);
    v34 = v33 >> 4;
    if ( (_DWORD)v34 )
    {
      v35 = 0;
      v36 = 16LL * (unsigned int)v34;
      do
      {
        v37 = 0;
        if ( *(char *)(a1 + 23) < 0 )
          v37 = sub_1648A40(a1);
        v38 = (__int64 *)(v35 + v37);
        v35 += 16;
        v39 = *((unsigned int *)v38 + 2);
        v40 = *v38;
        v41 = *((unsigned int *)v38 + 3);
        v42 = 3LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
        v87 = v40;
        v39 *= 24;
        v86[0] = a1 + v39 - 8 * v42;
        v86[1] = 0xAAAAAAAAAAAAAAABLL * ((24 * v41 - v39) >> 3);
        sub_1740580((__int64)&v88, (__int64)v86);
      }
      while ( v36 != v35 );
    }
  }
  LOWORD(v87) = 257;
  v43 = v88;
  v83 = v91;
  v84 = *(_QWORD *)(a1 - 72);
  v85 = *(_QWORD *)(*(_QWORD *)v84 + 24LL);
  v44 = &v88[7 * (unsigned int)v89];
  if ( v88 == v44 )
  {
    v78 = (unsigned int)v92;
    v74 = (unsigned int)v89;
    v82 = v92 + 1;
    v71 = sub_1648AB0(72, (int)v92 + 1, 16 * (int)v89);
    v56 = v82;
    v57 = v78;
    v50 = (__int64)v71;
    if ( v71 )
    {
      v81 = (__int64)v71;
      v51 = v78;
      v52 = v43;
      v53 = v74;
LABEL_31:
      v77 = v51;
      v73 = v52;
      v75 = v53;
      sub_15F1EA0(v50, **(_QWORD **)(v85 + 16), 54, v50 - 24 * v57 - 24, v56, a1);
      *(_QWORD *)(v50 + 56) = 0;
      sub_15F5B40(v50, v85, v84, v83, v77, (__int64)v86, v73, v75);
      goto LABEL_32;
    }
  }
  else
  {
    v45 = v88;
    v46 = 0;
    do
    {
      v47 = v45[5] - v45[4];
      v45 += 7;
      v46 += v47 >> 3;
    }
    while ( v44 != v45 );
    v76 = (unsigned int)v92;
    v72 = (unsigned int)v89;
    v80 = v92 + 1;
    v48 = sub_1648AB0(72, v46 + (int)v92 + 1, 16 * (int)v89);
    v49 = v80;
    v50 = (__int64)v48;
    if ( v48 )
    {
      v81 = (__int64)v48;
      v51 = v76;
      v52 = v43;
      v53 = v72;
      LODWORD(v54) = 0;
      do
      {
        v55 = v43[5] - v43[4];
        v43 += 7;
        v54 = (unsigned int)(v55 >> 3) + (unsigned int)v54;
      }
      while ( v44 != v43 );
      v56 = v54 + v49;
      v57 = v54 + v76;
      goto LABEL_31;
    }
  }
  v81 = 0;
  v50 = 0;
LABEL_32:
  v58 = (__int64 *)(v50 + 48);
  sub_164B7C0(v81, a1);
  *(_WORD *)(v50 + 18) = *(_WORD *)(v50 + 18) & 0x8000
                       | *(_WORD *)(v50 + 18) & 3
                       | (4 * ((*(_WORD *)(a1 + 18) >> 2) & 0xDFFF));
  *(_QWORD *)(v50 + 56) = *(_QWORD *)(a1 + 56);
  v61 = *(_QWORD *)(a1 + 48);
  v86[0] = v61;
  if ( !v61 )
  {
    if ( v58 == v86 )
      goto LABEL_36;
    v69 = *(_QWORD *)(v50 + 48);
    if ( !v69 )
      goto LABEL_36;
LABEL_53:
    sub_161E7C0(v50 + 48, v69);
    goto LABEL_54;
  }
  sub_1623A60((__int64)v86, v61, 2);
  if ( v58 == v86 )
  {
    if ( v86[0] )
      sub_161E7C0((__int64)v86, v86[0]);
    goto LABEL_36;
  }
  v69 = *(_QWORD *)(v50 + 48);
  if ( v69 )
    goto LABEL_53;
LABEL_54:
  v70 = (unsigned __int8 *)v86[0];
  *(_QWORD *)(v50 + 48) = v86[0];
  if ( v70 )
    sub_1623210((__int64)v86, v70, v50 + 48);
LABEL_36:
  sub_164D160(a1, v50, a3, a4, a5, a6, v59, v60, a9, a10);
  v62 = *(_QWORD *)(a1 - 48);
  v63 = sub_1648A60(56, 1u);
  if ( v63 )
    sub_15F8320((__int64)v63, v62, a1);
  v64 = *(_QWORD *)(a1 + 40);
  v65 = *(_QWORD *)(a1 - 24);
  sub_157F2D0(v65, v64, 0);
  sub_15F20C0((_QWORD *)a1);
  if ( a2 )
    sub_15CDBF0(a2, v64, v65);
  v66 = v88;
  v67 = &v88[7 * (unsigned int)v89];
  if ( v88 != v67 )
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
    v67 = v88;
  }
  if ( v67 != (__int64 *)v90 )
    _libc_free((unsigned __int64)v67);
  if ( v91 != (__int64 *)v93 )
    _libc_free((unsigned __int64)v91);
}
