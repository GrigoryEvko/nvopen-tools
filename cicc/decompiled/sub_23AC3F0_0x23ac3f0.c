// Function: sub_23AC3F0
// Address: 0x23ac3f0
//
unsigned __int64 *__fastcall sub_23AC3F0(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, int a4)
{
  _BYTE *v5; // rbx
  unsigned int v6; // ecx
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // r9
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // r15
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // rdi
  int *v18; // r15
  int *v19; // rbx
  int *v20; // r15
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // r15
  unsigned __int64 v24; // rdi
  unsigned __int64 *v25; // r8
  unsigned __int64 v26; // r15
  unsigned __int64 v27; // rbx
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  _QWORD *v30; // rax
  _BOOL8 v31; // rsi
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  _QWORD *v41; // rax
  __int64 v42; // r9
  _QWORD *v43; // rax
  _QWORD *v44; // rax
  _QWORD *v45; // rax
  _QWORD *v46; // rax
  _QWORD *v47; // rax
  _QWORD *v48; // rax
  _QWORD *v49; // rax
  int v51; // eax
  _QWORD *v52; // rax
  _QWORD *v53; // rax
  __int64 v54; // rax
  _QWORD *v55; // rax
  _QWORD *v56; // rax
  _QWORD *v57; // rax
  _QWORD *v58; // rax
  _QWORD *v59; // rax
  _QWORD *v60; // rax
  _QWORD *v61; // rax
  _QWORD *v62; // rax
  _QWORD *v63; // rax
  _QWORD *v64; // rax
  __int64 v65; // rax
  unsigned __int64 v66; // rax
  _BYTE *v67; // [rsp+18h] [rbp-4D8h]
  unsigned __int64 v68; // [rsp+28h] [rbp-4C8h]
  __int16 v69; // [rsp+28h] [rbp-4C8h]
  bool v70; // [rsp+37h] [rbp-4B9h]
  unsigned int v72; // [rsp+44h] [rbp-4ACh]
  __int64 v74[4]; // [rsp+60h] [rbp-490h] BYREF
  unsigned __int64 v75[6]; // [rsp+80h] [rbp-470h] BYREF
  unsigned __int64 v76[34]; // [rsp+B0h] [rbp-440h] BYREF
  _QWORD *v77; // [rsp+1C0h] [rbp-330h] BYREF
  unsigned __int64 v78[2]; // [rsp+1C8h] [rbp-328h] BYREF
  __int64 v79; // [rsp+1D8h] [rbp-318h]
  int *v80; // [rsp+1E0h] [rbp-310h]
  __int64 v81; // [rsp+1E8h] [rbp-308h]
  int v82; // [rsp+1F0h] [rbp-300h] BYREF
  __int64 v83; // [rsp+1F8h] [rbp-2F8h]
  unsigned int v84; // [rsp+208h] [rbp-2E8h]
  char *v85; // [rsp+210h] [rbp-2E0h]
  char v86; // [rsp+220h] [rbp-2D0h] BYREF
  unsigned __int64 v87; // [rsp+270h] [rbp-280h]
  __int64 v88; // [rsp+298h] [rbp-258h]
  unsigned int v89; // [rsp+2A8h] [rbp-248h]
  char *v90; // [rsp+2B0h] [rbp-240h]
  char v91; // [rsp+2C0h] [rbp-230h] BYREF

  v5 = (_BYTE *)a2;
  v6 = a4 & 0xFFFFFFFD;
  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  v70 = v6 == 1;
  v68 = HIDWORD(a3);
  v72 = v6;
  if ( byte_4FDDA08 )
  {
    v56 = (_QWORD *)sub_22077B0(0x10u);
    if ( v56 )
      *v56 = &unk_4A0DAF8;
    v77 = v56;
    sub_23A2230(a1, (unsigned __int64 *)&v77);
    sub_23501E0((__int64 *)&v77);
  }
  if ( v72 != 1 )
  {
    v55 = (_QWORD *)sub_22077B0(0x10u);
    if ( v55 )
      *v55 = &unk_4A0D238;
    v77 = v55;
    sub_23A2230(a1, (unsigned __int64 *)&v77);
    sub_23501E0((__int64 *)&v77);
  }
  if ( (_BYTE)qword_4FDCB28 )
  {
    v53 = (_QWORD *)sub_22077B0(0x10u);
    if ( v53 )
      *v53 = &unk_4A0D538;
    v77 = v53;
    sub_23A2230(a1, (unsigned __int64 *)&v77);
    sub_23501E0((__int64 *)&v77);
  }
  v7 = (_QWORD *)sub_22077B0(0x10u);
  if ( v7 )
    *v7 = &unk_4A0E0B8;
  v77 = v7;
  sub_23A2230(a1, (unsigned __int64 *)&v77);
  sub_23501E0((__int64 *)&v77);
  if ( v72 != 1 && *(_BYTE *)(a2 + 192) )
  {
    v51 = *(_DWORD *)(a2 + 172);
    if ( v51 == 1 )
    {
      v66 = *(_QWORD *)(a2 + 184);
      v75[0] = v66;
      if ( v66 )
        _InterlockedAdd((volatile signed __int32 *)(v66 + 8), 1u);
      sub_2241BD0((__int64 *)&v77, a2 + 104);
      sub_2241BD0((__int64 *)v76, a2 + 72);
      sub_23A2D30(a2, a1, a3, 1, 1u, *(_BYTE *)(a2 + 182), v76, (__int64)&v77, (__int64 *)v75);
    }
    else
    {
      if ( v51 != 2 )
        goto LABEL_8;
      v75[0] = *(_QWORD *)(a2 + 184);
      sub_239EC60(v75[0]);
      sub_2241BD0((__int64 *)&v77, a2 + 104);
      sub_2241BD0((__int64 *)v76, a2 + 40);
      sub_23A2D30(a2, a1, a3, 0, 1u, *(_BYTE *)(a2 + 182), v76, (__int64)&v77, (__int64 *)v75);
    }
    sub_2240A30(v76);
    sub_2240A30((unsigned __int64 *)&v77);
    if ( v75[0] )
      sub_23569D0((volatile signed __int32 *)(v75[0] + 8));
  }
LABEL_8:
  if ( byte_4FDDAE8 )
  {
    v60 = (_QWORD *)sub_22077B0(0x10u);
    if ( v60 )
      *v60 = &unk_4A0DFB8;
    v77 = v60;
    sub_23A2230(a1, (unsigned __int64 *)&v77);
    sub_23501E0((__int64 *)&v77);
  }
  sub_23A1080(a2, (__int64)a1, a3, a4);
  memset(v75, 0, 40);
  if ( byte_4FDC548 )
  {
    v57 = (_QWORD *)sub_22077B0(0x10u);
    if ( v57 )
      *v57 = &unk_4A12278;
    v77 = v57;
    v74[0] = 0;
    v78[0] = 0;
    v78[1] = 0;
    v79 = 0;
    v80 = 0;
    v81 = 0;
    v82 = 0;
    v58 = (_QWORD *)sub_22077B0(0x10u);
    if ( v58 )
      *v58 = &unk_4A0B640;
    v76[0] = (unsigned __int64)v58;
    sub_23A1F40(v78, v76);
    sub_233EFE0((__int64 *)v76);
    v59 = (_QWORD *)sub_22077B0(0x10u);
    if ( v59 )
      *v59 = &unk_4A0B680;
    v76[0] = (unsigned __int64)v59;
    sub_23A1F40(v78, v76);
    sub_233EFE0((__int64 *)v76);
    sub_233F7D0(v74);
    sub_2353940(v75, (__int64 *)&v77);
    sub_233F7F0((__int64)v78);
    sub_233F7D0((__int64 *)&v77);
    v76[0] = *(_QWORD *)(a2 + 16);
    LOWORD(v76[1]) = 1;
    sub_2356430((__int64)&v77, (__int64 *)v76, 1, 0, 0);
    sub_2353940(v75, (__int64 *)&v77);
    sub_233F7F0((__int64)v78);
    sub_233F7D0((__int64 *)&v77);
  }
  memset(v76, 0, 0x108u);
  memset(&v76[5], 0, 40);
  v76[4] = (unsigned __int64)&v76[6];
  v76[10] = (unsigned __int64)&v76[12];
  v76[11] = 0x800000000LL;
  v76[23] = (unsigned __int64)&v76[21];
  v76[24] = (unsigned __int64)&v76[21];
  v76[30] = (unsigned __int64)&v76[32];
  LODWORD(v76[21]) = 0;
  v76[22] = 0;
  memset(&v76[25], 0, 36);
  v76[31] = 0;
  sub_2365C20((__int64)&v77, (__int64)v76, v8, 0, v9, v10);
  v11 = (_QWORD *)sub_22077B0(0x110u);
  v15 = (__int64)v11;
  if ( v11 )
  {
    *v11 = &unk_4A0F6B8;
    sub_2365C20((__int64)(v11 + 1), (__int64)&v77, (__int64)&unk_4A0F6B8, v12, v13, v14);
  }
  v74[0] = v15;
  sub_23A1F40(v75, (unsigned __int64 *)v74);
  sub_233EFE0(v74);
  if ( v90 != &v91 )
    _libc_free((unsigned __int64)v90);
  sub_C7D6A0(v88, 16LL * v89, 8);
  v16 = v87;
  while ( v16 )
  {
    sub_239F7A0(*(_QWORD *)(v16 + 24));
    v17 = v16;
    v16 = *(_QWORD *)(v16 + 16);
    j_j___libc_free_0(v17);
  }
  if ( v85 != &v86 )
    _libc_free((unsigned __int64)v85);
  sub_C7D6A0(v83, 8LL * v84, 8);
  v18 = &v80[10 * (unsigned int)v81];
  if ( v80 != v18 )
  {
    v19 = &v80[10 * (unsigned int)v81];
    v20 = v80;
    do
    {
      v19 -= 10;
      if ( (unsigned int)v19[8] > 0x40 )
      {
        v21 = *((_QWORD *)v19 + 3);
        if ( v21 )
          j_j___libc_free_0_0(v21);
      }
      if ( (unsigned int)v19[4] > 0x40 )
      {
        v22 = *((_QWORD *)v19 + 1);
        if ( v22 )
          j_j___libc_free_0_0(v22);
      }
    }
    while ( v20 != v19 );
    v5 = (_BYTE *)a2;
    v18 = v80;
  }
  if ( v18 != &v82 )
    _libc_free((unsigned __int64)v18);
  sub_C7D6A0(v78[0], 16LL * (unsigned int)v79, 8);
  if ( (unsigned __int64 *)v76[30] != &v76[32] )
    _libc_free(v76[30]);
  sub_C7D6A0(v76[27], 16LL * LODWORD(v76[29]), 8);
  v23 = v76[22];
  while ( v23 )
  {
    sub_239F7A0(*(_QWORD *)(v23 + 24));
    v24 = v23;
    v23 = *(_QWORD *)(v23 + 16);
    j_j___libc_free_0(v24);
  }
  if ( (unsigned __int64 *)v76[10] != &v76[12] )
    _libc_free(v76[10]);
  sub_C7D6A0(v76[7], 8LL * LODWORD(v76[9]), 8);
  v25 = (unsigned __int64 *)(v76[4] + 40LL * LODWORD(v76[5]));
  if ( (unsigned __int64 *)v76[4] != v25 )
  {
    v67 = v5;
    v26 = v76[4];
    v27 = v76[4] + 40LL * LODWORD(v76[5]);
    do
    {
      v27 -= 40LL;
      if ( *(_DWORD *)(v27 + 32) > 0x40u )
      {
        v28 = *(_QWORD *)(v27 + 24);
        if ( v28 )
          j_j___libc_free_0_0(v28);
      }
      if ( *(_DWORD *)(v27 + 16) > 0x40u )
      {
        v29 = *(_QWORD *)(v27 + 8);
        if ( v29 )
          j_j___libc_free_0_0(v29);
      }
    }
    while ( v26 != v27 );
    v5 = v67;
    v25 = (unsigned __int64 *)v76[4];
  }
  if ( v25 != &v76[6] )
    _libc_free((unsigned __int64)v25);
  sub_C7D6A0(v76[1], 16LL * LODWORD(v76[3]), 8);
  v30 = (_QWORD *)sub_22077B0(0x10u);
  if ( v30 )
    *v30 = &unk_4A0FE78;
  v77 = v30;
  sub_23A1F40(v75, (unsigned __int64 *)&v77);
  sub_233EFE0((__int64 *)&v77);
  if ( byte_4FDCA48 )
  {
    v54 = sub_22077B0(0x10u);
    if ( v54 )
    {
      *(_BYTE *)(v54 + 8) = 0;
      *(_QWORD *)v54 = &unk_4A11A78;
    }
    v77 = (_QWORD *)v54;
    sub_23A1F40(v75, (unsigned __int64 *)&v77);
    sub_233EFE0((__int64 *)&v77);
    LOBYTE(v77) = 0;
    sub_23A2060(v75, (char *)&v77);
  }
  if ( (_BYTE)qword_4FDCCE8 && (_DWORD)v68 == dword_5033EF0[1] && (_DWORD)a3 == dword_5033EF0[0] )
  {
    sub_23FD580(v76);
    v63 = (_QWORD *)sub_22077B0(0x10u);
    if ( v63 )
      *v63 = &unk_4A0F038;
    v77 = v63;
    sub_23A1F40(v75, (unsigned __int64 *)&v77);
    sub_233EFE0((__int64 *)&v77);
  }
  sub_23A0FA0((__int64)v5, (__int64)v75, a3);
  v31 = 1;
  v76[0] = (unsigned __int64)&v76[2];
  v76[1] = 0x600000000LL;
  LODWORD(v76[8]) = 0;
  memset(&v76[9], 0, 48);
  if ( !byte_4FDD4C8 && (_DWORD)v68 == HIDWORD(qword_5033EE0) )
    v31 = (_DWORD)qword_5033EE0 != (_DWORD)a3;
  sub_28448C0(v74, v31, v70);
  sub_2332320((__int64)v76, 0, v32, v33, v34, v35);
  v69 = v74[0];
  v36 = sub_22077B0(0x10u);
  if ( v36 )
  {
    *(_WORD *)(v36 + 8) = v69;
    *(_QWORD *)v36 = &unk_4A124B8;
  }
  v77 = (_QWORD *)v36;
  sub_23A46F0(&v76[9], (unsigned __int64 *)&v77);
  sub_233F7D0((__int64 *)&v77);
  sub_2332320((__int64)v76, 0, v37, v38, v39, v40);
  v41 = (_QWORD *)sub_22077B0(0x10u);
  if ( v41 )
    *v41 = &unk_4A12038;
  v77 = v41;
  sub_23A46F0(&v76[9], (unsigned __int64 *)&v77);
  sub_233F7D0((__int64 *)&v77);
  sub_23A20C0((__int64)&v77, (__int64)v76, 0, 0, 0, v42);
  sub_2353940(v75, (__int64 *)&v77);
  sub_233F7F0((__int64)v78);
  sub_233F7D0((__int64 *)&v77);
  v43 = (_QWORD *)sub_22077B0(0x10u);
  if ( v43 )
    *v43 = &unk_4A0FCF8;
  v77 = v43;
  sub_23A1F40(v75, (unsigned __int64 *)&v77);
  sub_233EFE0((__int64 *)&v77);
  v44 = (_QWORD *)sub_22077B0(0x10u);
  if ( v44 )
    *v44 = &unk_4A0F8F8;
  v77 = v44;
  sub_23A1F40(v75, (unsigned __int64 *)&v77);
  sub_233EFE0((__int64 *)&v77);
  sub_23A95C0((__int64)v5, a3, v75, 0);
  sub_23A1010((__int64)v5, (__int64)v75, a3);
  v45 = (_QWORD *)sub_22077B0(0x10u);
  if ( v45 )
    *v45 = &unk_4A0FDB8;
  v77 = v45;
  sub_23A1F40(v75, (unsigned __int64 *)&v77);
  sub_233EFE0((__int64 *)&v77);
  v46 = (_QWORD *)sub_22077B0(0x10u);
  if ( v46 )
    *v46 = &unk_4A0F9B8;
  v77 = v46;
  sub_23A1F40(v75, (unsigned __int64 *)&v77);
  sub_233EFE0((__int64 *)&v77);
  v47 = (_QWORD *)sub_22077B0(0x10u);
  if ( v47 )
    *v47 = &unk_4A0F2F8;
  v77 = v47;
  sub_23A1F40(v75, (unsigned __int64 *)&v77);
  sub_233EFE0((__int64 *)&v77);
  v48 = (_QWORD *)sub_22077B0(0x10u);
  if ( v48 )
    *v48 = &unk_4A10F78;
  v77 = v48;
  sub_23A1F40(v75, (unsigned __int64 *)&v77);
  sub_233EFE0((__int64 *)&v77);
  v74[0] = 0x100010000000001LL;
  v74[1] = 0x101000101000100LL;
  v74[2] = 0;
  sub_29744D0(&v77, v74);
  sub_23A1F80(v75, (__int64 *)&v77);
  sub_234AAB0((__int64)&v77, (__int64 *)v75, v5[32]);
  sub_23571D0(a1, (__int64 *)&v77);
  sub_233EFE0((__int64 *)&v77);
  sub_23A1110((__int64)v5, (__int64)a1, a3, a4);
  if ( (_BYTE)qword_4FDD308 == 1 && !v70 )
  {
    v64 = (_QWORD *)sub_22077B0(0x10u);
    if ( v64 )
      *v64 = &unk_4A0D438;
    v77 = v64;
    sub_23A2230(a1, (unsigned __int64 *)&v77);
    sub_23501E0((__int64 *)&v77);
  }
  if ( (_BYTE)qword_4FDD228 )
  {
    v62 = (_QWORD *)sub_22077B0(0x10u);
    if ( v62 )
      *v62 = &unk_4A0D678;
    v77 = v62;
    sub_23A2230(a1, (unsigned __int64 *)&v77);
    sub_23501E0((__int64 *)&v77);
  }
  sub_23A0BA0((__int64)&v77, 0);
  sub_23A2670(a1, (__int64)&v77);
  sub_233AAF0((__int64)&v77);
  v49 = (_QWORD *)sub_22077B0(0x10u);
  if ( v49 )
    *v49 = &unk_4A0CF78;
  v77 = v49;
  sub_23A2230(a1, (unsigned __int64 *)&v77);
  sub_23501E0((__int64 *)&v77);
  if ( v5[26] )
  {
    v61 = (_QWORD *)sub_22077B0(0x10u);
    if ( v61 )
      *v61 = &unk_4A0D8F8;
    v77 = v61;
    sub_23A2230(a1, (unsigned __int64 *)&v77);
    sub_23501E0((__int64 *)&v77);
  }
  if ( v5[24] )
  {
    if ( v72 == 1 )
      goto LABEL_80;
    v65 = sub_22077B0(0x10u);
    if ( v65 )
    {
      *(_BYTE *)(v65 + 8) = ((a4 - 2) & 0xFFFFFFFD) == 0;
      *(_QWORD *)v65 = &unk_4A0E738;
    }
    v77 = (_QWORD *)v65;
    sub_23A2230(a1, (unsigned __int64 *)&v77);
    sub_23501E0((__int64 *)&v77);
  }
  else if ( v72 == 1 )
  {
    goto LABEL_80;
  }
  v52 = (_QWORD *)sub_22077B0(0x10u);
  if ( v52 )
    *v52 = &unk_4A0DFF8;
  v77 = v52;
  sub_23A2230(a1, (unsigned __int64 *)&v77);
  sub_23501E0((__int64 *)&v77);
LABEL_80:
  sub_2337B30(v76);
  sub_233F7F0((__int64)v75);
  return a1;
}
