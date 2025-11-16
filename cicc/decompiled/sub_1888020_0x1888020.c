// Function: sub_1888020
// Address: 0x1888020
//
__int64 __fastcall sub_1888020(
        __int64 **a1,
        __int64 a2,
        char a3,
        __m128 a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  __int64 v11; // r12
  char v12; // bl
  char *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // rax
  __int64 v17; // r14
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 v20; // r13
  _QWORD *v21; // rbx
  __int64 *v22; // r8
  __int64 v23; // r12
  __int64 v24; // rax
  __int64 v25; // r14
  double v26; // xmm4_8
  double v27; // xmm5_8
  int v28; // r8d
  int v29; // r9d
  __int64 v30; // rax
  __int64 *v31; // rbx
  __int64 *v32; // r13
  _QWORD *v33; // rdi
  char v34; // al
  __int64 v36; // rax
  __int64 *v37; // rbx
  __int64 v38; // r14
  __int64 v39; // rax
  __int64 v40; // r13
  __int64 *v41; // rbx
  __int64 *v42; // r14
  unsigned __int64 v43; // rax
  unsigned __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // r13
  __int64 v47; // rax
  __int64 v48; // [rsp+0h] [rbp-E0h]
  __int64 v49; // [rsp+20h] [rbp-C0h]
  __int64 v50; // [rsp+20h] [rbp-C0h]
  __int64 v52; // [rsp+30h] [rbp-B0h]
  _BYTE *v54; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v55; // [rsp+48h] [rbp-98h]
  _BYTE v56[16]; // [rsp+50h] [rbp-90h] BYREF
  __int64 v57[2]; // [rsp+60h] [rbp-80h] BYREF
  __int16 v58; // [rsp+70h] [rbp-70h] BYREF
  __int64 *v59; // [rsp+80h] [rbp-60h] BYREF
  __int64 v60; // [rsp+88h] [rbp-58h]
  _WORD v61[40]; // [rsp+90h] [rbp-50h] BYREF

  v11 = a2;
  v12 = (*(_BYTE *)(a2 + 32) >> 4) & 3;
  v13 = (char *)sub_1649960(a2);
  if ( v13 )
  {
    v54 = v56;
    sub_18736F0((__int64 *)&v54, v13, (__int64)&v13[v14]);
    if ( (*(_BYTE *)(a2 + 32) & 0xF) != 1 )
      goto LABEL_3;
LABEL_35:
    if ( !a3 )
      goto LABEL_50;
    goto LABEL_36;
  }
  v56[0] = 0;
  v54 = v56;
  v36 = *(unsigned __int8 *)(a2 + 32);
  v55 = 0;
  if ( (v36 & 0xF) == 1 )
    goto LABEL_35;
LABEL_3:
  if ( sub_15E4F60(a2) )
  {
    if ( !a3 )
    {
      if ( (*(_BYTE *)(a2 + 32) & 0xF) == 1 || sub_15E4F60(a2) )
        goto LABEL_50;
      return sub_2240A30(&v54);
    }
LABEL_36:
    if ( (*(_BYTE *)(a2 + 33) & 0x40) != 0 )
    {
      v37 = *a1;
      v59 = (__int64 *)v61;
      sub_1872C70((__int64 *)&v59, v54, (__int64)&v54[v55]);
      sub_2241520(&v59, ".cfi");
      v57[0] = (__int64)&v59;
      v38 = *(_QWORD *)(a2 + 24);
      v58 = 260;
      v39 = sub_1648B60(120);
      v40 = v39;
      if ( v39 )
        sub_15E2490(v39, v38, 0, (__int64)v57, (__int64)v37);
      sub_2240A30(&v59);
      *(_BYTE *)(v40 + 32) = *(_BYTE *)(v40 + 32) & 0xCF | 0x10;
      sub_15A5120(v40);
      v41 = *(__int64 **)(a2 + 8);
      while ( v41 )
      {
        v42 = v41;
        v41 = (__int64 *)v41[1];
        v43 = (unsigned __int64)sub_1648700((__int64)v42);
        if ( *(_BYTE *)(v43 + 16) == 78 && v42 == (__int64 *)((v43 & 0xFFFFFFFFFFFFFFF8LL) - 24) )
        {
          if ( *v42 )
          {
            v44 = v42[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v44 = v41;
            if ( v41 )
              v41[2] = v41[2] & 3 | v44;
          }
          *v42 = v40;
          v45 = *(_QWORD *)(v40 + 8);
          v42[1] = v45;
          if ( v45 )
            *(_QWORD *)(v45 + 16) = (unsigned __int64)(v42 + 1) | *(_QWORD *)(v45 + 16) & 3LL;
          v42[2] = (v40 + 8) | v42[2] & 3;
          *(_QWORD *)(v40 + 8) = v42;
        }
      }
    }
    return sub_2240A30(&v54);
  }
  if ( (*(_BYTE *)(a2 + 32) & 0xF) == 1 || sub_15E4F60(a2) )
  {
    if ( a3 )
      goto LABEL_7;
LABEL_50:
    v46 = (__int64)*a1;
    v57[0] = (__int64)&v58;
    sub_1872C70(v57, v54, (__int64)&v54[v55]);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v57[1]) > 6 )
    {
      sub_2241490(v57, ".cfi_jt", 7);
      v59 = v57;
      v61[0] = 260;
      v50 = *(_QWORD *)(a2 + 24);
      v47 = sub_1648B60(120);
      v17 = v47;
      if ( v47 )
        sub_15E2490(v47, v50, 0, (__int64)&v59, v46);
      sub_2240A30(v57);
      *(_BYTE *)(v17 + 32) = *(_BYTE *)(v17 + 32) & 0xCF | 0x10;
      sub_15A5120(v17);
      goto LABEL_25;
    }
LABEL_56:
    sub_4262D8((__int64)"basic_string::append");
  }
  if ( !a3 )
    return sub_2240A30(&v54);
LABEL_7:
  v59 = (__int64 *)v61;
  sub_1872C70((__int64 *)&v59, v54, (__int64)&v54[v55]);
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v60) <= 3 )
    goto LABEL_56;
  sub_2241490(&v59, ".cfi", 4);
  v57[0] = (__int64)&v59;
  v58 = 260;
  sub_164B780(a2, v57);
  sub_2240A30(&v59);
  *(_BYTE *)(a2 + 32) &= 0xF0u;
  sub_15A5120(a2);
  v61[0] = 260;
  v15 = *(_QWORD *)(a2 + 24);
  v49 = (__int64)*a1;
  v59 = (__int64 *)&v54;
  v16 = sub_1648B60(120);
  v17 = v16;
  if ( v16 )
    sub_15E2490(v16, v15, 0, (__int64)&v59, v49);
  *(_BYTE *)(v17 + 32) = (16 * (v12 & 3)) | *(_BYTE *)(v17 + 32) & 0xCF;
  sub_15A5120(v17);
  v20 = *(_QWORD *)(a2 + 8);
  v59 = (__int64 *)v61;
  v60 = 0x400000000LL;
  if ( v20 )
  {
    v48 = v17;
    do
    {
      v21 = sub_1648700(v20);
      if ( *((_BYTE *)v21 + 16) == 1 )
      {
        v22 = *a1;
        v58 = 257;
        v52 = (__int64)v22;
        v23 = *(_QWORD *)(a2 + 24);
        v24 = sub_1648B60(120);
        v25 = v24;
        if ( v24 )
          sub_15E2490(v24, v23, 0, (__int64)v57, v52);
        sub_164B7C0(v25, (__int64)v21);
        sub_164D160((__int64)v21, v25, a4, a5, a6, a7, v26, v27, a10, a11);
        v30 = (unsigned int)v60;
        if ( (unsigned int)v60 >= HIDWORD(v60) )
        {
          sub_16CD150((__int64)&v59, v61, 0, 8, v28, v29);
          v30 = (unsigned int)v60;
        }
        v59[v30] = (__int64)v21;
        LODWORD(v60) = v60 + 1;
      }
      v20 = *(_QWORD *)(v20 + 8);
    }
    while ( v20 );
    v31 = v59;
    v17 = v48;
    v11 = a2;
    v32 = &v59[(unsigned int)v60];
    if ( v59 != v32 )
    {
      do
      {
        v33 = (_QWORD *)*v31++;
        sub_15E58C0(v33);
      }
      while ( v32 != v31 );
      v32 = v59;
    }
    if ( v32 != (__int64 *)v61 )
      _libc_free((unsigned __int64)v32);
  }
  v12 = 1;
LABEL_25:
  v34 = *(_BYTE *)(v11 + 32) & 0xF;
  if ( ((v34 + 14) & 0xFu) > 3 && ((v34 + 7) & 0xFu) > 1 )
    sub_1887680(v11, v17, a3, *(double *)a4.m128_u64, a5, a6);
  else
    sub_1887A80(a1, v11, (__int64 *)v17, a3, a4, a5, a6, a7, v18, v19, a10, a11);
  *(_BYTE *)(v11 + 32) = (16 * (v12 & 3)) | *(_BYTE *)(v11 + 32) & 0xCF;
  sub_15A5120(v11);
  return sub_2240A30(&v54);
}
