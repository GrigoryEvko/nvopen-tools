// Function: sub_3137B20
// Address: 0x3137b20
//
void __fastcall sub_3137B20(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v8; // rbx
  int v9; // esi
  _BYTE *v10; // rax
  __int64 v11; // r13
  unsigned __int8 *v12; // r12
  unsigned __int8 *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 **v16; // rcx
  __int64 v17; // r14
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 **v20; // r15
  unsigned int v21; // r14d
  unsigned int v22; // eax
  bool v23; // cc
  unsigned __int64 v24; // r14
  __int64 v25; // rax
  int v26; // edx
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r14
  __int64 v31; // rax
  int v32; // r14d
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r8
  __int64 v36; // rdx
  __int64 v37; // r15
  unsigned __int8 *v38; // r14
  __int64 v39; // rdx
  __int64 v40; // r15
  unsigned __int64 v41; // r9
  char *v42; // rax
  unsigned int v43; // r15d
  __int64 **v44; // r14
  __int64 v45; // r8
  int v46; // ecx
  unsigned __int64 *v47; // rax
  unsigned __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rdi
  __int64 v52; // r15
  __int64 v53; // rax
  _QWORD *v54; // rax
  __int64 v55; // r14
  __int64 v56; // rax
  __int64 v57; // r15
  __int64 v58; // r15
  __int64 v59; // rbx
  __int64 v60; // rdx
  unsigned int v61; // esi
  __int64 v62; // rax
  char v63; // al
  _QWORD *v64; // rax
  __int64 v65; // r9
  __int64 v66; // r15
  __int64 v67; // r13
  __int64 v68; // rbx
  __int64 v69; // rdx
  unsigned int v70; // esi
  _QWORD **v71; // rbx
  __int64 v72; // r12
  _QWORD *v73; // rdi
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // r15
  __int64 v77; // rax
  unsigned __int64 v78; // rdx
  __int64 *v79; // r12
  __int64 v80; // rax
  unsigned __int64 v81; // rax
  __int64 v82; // r8
  __int64 v83; // r9
  unsigned __int64 v84; // rcx
  unsigned __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // [rsp-10h] [rbp-180h]
  __int64 v88; // [rsp+0h] [rbp-170h]
  unsigned int v89; // [rsp+Ch] [rbp-164h]
  unsigned __int64 v90; // [rsp+10h] [rbp-160h]
  __int64 v93; // [rsp+30h] [rbp-140h]
  __int64 v95; // [rsp+40h] [rbp-130h]
  char v96; // [rsp+40h] [rbp-130h]
  char v97; // [rsp+40h] [rbp-130h]
  __int64 v99; // [rsp+48h] [rbp-128h]
  int v100[8]; // [rsp+50h] [rbp-120h] BYREF
  __int16 v101; // [rsp+70h] [rbp-100h]
  __int64 v102[4]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v103; // [rsp+A0h] [rbp-D0h]
  unsigned __int64 v104; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v105; // [rsp+B8h] [rbp-B8h]
  _QWORD v106[2]; // [rsp+C0h] [rbp-B0h] BYREF
  unsigned __int64 v107; // [rsp+D0h] [rbp-A0h]

  v8 = a1;
  v9 = 8;
  if ( !a4 )
    v9 = 7;
  v10 = sub_3135910(a1, v9);
  v95 = (__int64)v10;
  if ( !v10 )
    BUG();
  v90 = *((_QWORD *)v10 + 3);
  if ( !*v10 && ((v10[7] & 0x20) == 0 || !sub_B91C10((__int64)v10, 26)) )
  {
    *(_QWORD *)v100 = sub_B2BE50(v95);
    v79 = *(__int64 **)v100;
    v104 = -1;
    v102[0] = sub_B8C960(v100, 2u, (int *)&v104, 2, 1u);
    v80 = sub_B9C770(v79, v102, (__int64 *)1, 0, 1);
    sub_B994D0(v95, 26, v80);
  }
  v11 = a1 + 512;
  sub_B2D3C0(a2, 0, 22);
  sub_B2D3C0(a2, 1, 22);
  sub_B2CD30(a2, 41);
  v12 = *(unsigned __int8 **)(*(_QWORD *)(a2 + 16) + 24LL);
  v89 = *(_QWORD *)(a2 + 104) - 2;
  v13 = (unsigned __int8 *)*((_QWORD *)v12 + 5);
  v104 = (unsigned __int64)"omp_parallel";
  LOWORD(v107) = 259;
  sub_BD6B50(v13, (const char **)&v104);
  sub_D5F1F0(v8 + 512, (__int64)v12);
  v14 = sub_BCB2D0(*(_QWORD **)(v8 + 584));
  v15 = sub_ACD640(v14, v89, 0);
  v16 = *(__int64 ***)(v8 + 2928);
  v17 = v15;
  LOWORD(v107) = 257;
  v18 = 49;
  v107 = sub_31223E0((__int64 *)(v8 + 512), 0x31u, a2, v16, (__int64)&v104, 0, v102[0], 0);
  v105 = 0x1000000003LL;
  v104 = (unsigned __int64)v106;
  v106[0] = a3;
  v106[1] = v17;
  if ( a4 )
  {
    v19 = *(_QWORD *)(a4 + 8);
    v20 = *(__int64 ***)(v8 + 2632);
    v103 = 257;
    v21 = sub_BCB060(v19);
    v22 = sub_BCB060((__int64)v20);
    v23 = v21 <= v22;
    if ( v21 < v22 )
    {
      v81 = sub_31223E0((__int64 *)(v8 + 512), 0x28u, a4, v20, (__int64)v102, 0, v100[0], 0);
      v84 = HIDWORD(v105);
      v18 = v87;
      v24 = v81;
      v25 = (unsigned int)v105;
      v85 = (unsigned int)v105 + 1LL;
    }
    else
    {
      v24 = a4;
      v25 = 3;
      if ( v23 )
      {
LABEL_10:
        *(_QWORD *)(v104 + 8 * v25) = v24;
        LODWORD(v105) = v105 + 1;
        goto LABEL_11;
      }
      v18 = a4;
      v86 = sub_A82DA0((unsigned int **)(v8 + 512), a4, (__int64)v20, (__int64)v102, 0, 0);
      v84 = HIDWORD(v105);
      v24 = v86;
      v25 = (unsigned int)v105;
      v85 = (unsigned int)v105 + 1LL;
    }
    if ( v84 < v85 )
    {
      v18 = (__int64)v106;
      sub_C8D5F0((__int64)&v104, v106, v85, 8u, v82, v83);
      v25 = (unsigned int)v105;
    }
    goto LABEL_10;
  }
LABEL_11:
  v26 = *v12;
  if ( v26 == 40 )
  {
    v27 = -32 - 32LL * (unsigned int)sub_B491D0((__int64)v12);
  }
  else
  {
    v27 = -32;
    if ( v26 != 85 )
    {
      v27 = -96;
      if ( v26 != 34 )
        BUG();
    }
  }
  if ( (v12[7] & 0x80u) != 0 )
  {
    v28 = sub_BD2BC0((__int64)v12);
    v30 = v28 + v29;
    v31 = 0;
    if ( (v12[7] & 0x80u) != 0 )
      v31 = sub_BD2BC0((__int64)v12);
    if ( (unsigned int)((v30 - v31) >> 4) )
    {
      if ( (v12[7] & 0x80u) == 0 )
        BUG();
      v32 = *(_DWORD *)(sub_BD2BC0((__int64)v12) + 8);
      if ( (v12[7] & 0x80u) == 0 )
        BUG();
      v33 = sub_BD2BC0((__int64)v12);
      v27 -= 32LL * (unsigned int)(*(_DWORD *)(v33 + v34 - 4) - v32);
    }
  }
  v35 = (__int64)&v12[v27];
  v36 = 64 - 32LL * (*((_DWORD *)v12 + 1) & 0x7FFFFFF);
  v37 = v27 - v36;
  v38 = &v12[v36];
  v39 = (unsigned int)v105;
  v40 = v37 >> 5;
  v41 = v40 + (unsigned int)v105;
  if ( v41 > HIDWORD(v105) )
  {
    v18 = (__int64)v106;
    v88 = v35;
    sub_C8D5F0((__int64)&v104, v106, v40 + (unsigned int)v105, 8u, v35, v41);
    v39 = (unsigned int)v105;
    v35 = v88;
  }
  v42 = (char *)(v104 + 8 * v39);
  if ( v38 != (unsigned __int8 *)v35 )
  {
    do
    {
      if ( v42 )
        *(_QWORD *)v42 = *(_QWORD *)v38;
      v38 += 32;
      v42 += 8;
    }
    while ( v38 != (unsigned __int8 *)v35 );
    LODWORD(v39) = v105;
  }
  v43 = v39 + v40;
  v44 = *(__int64 ***)(v8 + 2704);
  LODWORD(v105) = v43;
  v45 = v43;
  if ( a4 )
  {
    if ( !v89 )
    {
      v76 = sub_AD6530((__int64)v44, v18);
      v77 = (unsigned int)v105;
      v78 = (unsigned int)v105 + 1LL;
      if ( v78 > HIDWORD(v105) )
      {
        sub_C8D5F0((__int64)&v104, v106, v78, 8u, v74, v75);
        v77 = (unsigned int)v105;
      }
      *(_QWORD *)(v104 + 8 * v77) = v76;
      v45 = (unsigned int)(v105 + 1);
      LODWORD(v105) = v105 + 1;
    }
    v46 = v104;
    v47 = (unsigned __int64 *)(v104 + 8 * v45 - 8);
    if ( v44 != *(__int64 ***)(*v47 + 8) )
    {
      v103 = 257;
      v48 = sub_31223E0((__int64 *)(v8 + 512), 0x31u, *v47, v44, (__int64)v102, 0, v100[0], 0);
      *(_QWORD *)(v104 + 8LL * (unsigned int)v105 - 8) = v48;
      v46 = v104;
      LODWORD(v45) = v105;
    }
  }
  else
  {
    v46 = v104;
    LODWORD(v45) = v43;
  }
  v103 = 257;
  sub_921880((unsigned int **)(v8 + 512), v90, v95, v46, v45, (__int64)v102, 0);
  sub_D5F1F0(v8 + 512, a5);
  if ( (*(_BYTE *)(a2 + 2) & 1) != 0 )
    sub_B2C6D0(a2, a5, v49, v50);
  v51 = *(_QWORD *)(v8 + 560);
  v52 = *(_QWORD *)(v8 + 2632);
  v101 = 257;
  v93 = *(_QWORD *)(a2 + 96);
  v53 = sub_AA4E30(v51);
  v96 = sub_AE5020(v53, v52);
  v103 = 257;
  v54 = sub_BD2C40(80, 1u);
  v55 = (__int64)v54;
  if ( v54 )
    sub_B4D190((__int64)v54, v52, v93, (__int64)v102, 0, v96, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(v8 + 600) + 16LL))(
    *(_QWORD *)(v8 + 600),
    v55,
    v100,
    *(_QWORD *)(v8 + 568),
    *(_QWORD *)(v8 + 576));
  v56 = *(_QWORD *)(v8 + 512);
  v57 = 16LL * *(unsigned int *)(v8 + 520);
  if ( v56 != v56 + v57 )
  {
    v99 = v8;
    v58 = v56 + v57;
    v59 = *(_QWORD *)(v8 + 512);
    do
    {
      v60 = *(_QWORD *)(v59 + 8);
      v61 = *(_DWORD *)v59;
      v59 += 16;
      sub_B99FD0(v55, v61, v60);
    }
    while ( v58 != v59 );
    v8 = v99;
  }
  v62 = sub_AA4E30(*(_QWORD *)(v8 + 560));
  v63 = sub_AE5020(v62, *(_QWORD *)(v55 + 8));
  v103 = 257;
  v97 = v63;
  v64 = sub_BD2C40(80, unk_3F10A10);
  v66 = (__int64)v64;
  if ( v64 )
    sub_B4D3C0((__int64)v64, v55, a6, 0, v97, v65, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v8 + 600) + 16LL))(
    *(_QWORD *)(v8 + 600),
    v66,
    v102,
    *(_QWORD *)(v11 + 56),
    *(_QWORD *)(v11 + 64));
  v67 = *(_QWORD *)(v8 + 512);
  v68 = v67 + 16LL * *(unsigned int *)(v8 + 520);
  while ( v68 != v67 )
  {
    v69 = *(_QWORD *)(v67 + 8);
    v70 = *(_DWORD *)v67;
    v67 += 16;
    sub_B99FD0(v66, v70, v69);
  }
  sub_B43D60(v12);
  v71 = *(_QWORD ***)a7;
  v72 = *(_QWORD *)a7 + 8LL * *(unsigned int *)(a7 + 8);
  if ( *(_QWORD *)a7 != v72 )
  {
    do
    {
      v73 = *v71++;
      sub_B43D60(v73);
    }
    while ( (_QWORD **)v72 != v71 );
  }
  if ( (_QWORD *)v104 != v106 )
    _libc_free(v104);
}
