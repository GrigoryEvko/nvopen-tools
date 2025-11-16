// Function: sub_94E0D0
// Address: 0x94e0d0
//
__int64 __fastcall sub_94E0D0(__int64 a1, __int64 a2, int a3, __int64 a4)
{
  __int64 v4; // rax
  char v6; // r14
  __int64 *v7; // r13
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // r12
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  int v13; // eax
  __int64 v14; // rdi
  int v15; // r13d
  __int64 v16; // rax
  char v17; // al
  int v18; // r15d
  __int64 v19; // rax
  __int64 v20; // r9
  __int64 v21; // r12
  unsigned int *v22; // r15
  unsigned int *v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rsi
  __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // r12
  unsigned __int64 v29; // rax
  __int64 v30; // rcx
  int v31; // eax
  __int64 v32; // rdi
  int v33; // r13d
  __int64 v34; // rax
  char v35; // al
  int v36; // r15d
  __int64 v37; // rax
  __int64 v38; // r12
  unsigned int *v39; // r15
  unsigned int *v40; // r13
  __int64 v41; // rdx
  __int64 v42; // rsi
  __int64 v43; // rax
  unsigned __int64 v44; // rdx
  __int64 v45; // r12
  unsigned __int64 v46; // rax
  __int64 v47; // rcx
  int v48; // eax
  __int64 v49; // rdi
  int v50; // r13d
  __int64 v51; // rax
  int v52; // r15d
  __int64 v53; // rax
  __int64 v54; // r12
  unsigned int *v55; // r15
  unsigned int *v56; // r13
  __int64 v57; // rdx
  __int64 v58; // rsi
  __int64 v59; // rax
  unsigned __int64 v60; // rdx
  __int64 v61; // rax
  unsigned __int64 v62; // rsi
  __int64 v63; // rax
  __int64 v64; // r12
  unsigned __int64 v65; // rax
  __int64 v66; // rcx
  __int64 v67; // rax
  int v68; // r15d
  __int64 v69; // r12
  __int64 v70; // rax
  char v71; // al
  __int16 v72; // cx
  __int64 v73; // rax
  int v74; // r9d
  __int64 v75; // r13
  __int64 v76; // rsi
  unsigned int *v77; // r15
  unsigned int *v78; // r12
  __int64 v79; // rdx
  _QWORD *v80; // rdi
  __int64 v82; // rax
  __m128i *v83; // r12
  __int64 v84; // rax
  unsigned __int64 v85; // rdx
  __int64 v86; // rdx
  __int64 v87; // r12
  unsigned __int64 v88; // rax
  __int64 v89; // rcx
  __int64 v90; // [rsp-8h] [rbp-248h]
  __int64 *v92; // [rsp+8h] [rbp-238h]
  __int64 *v93; // [rsp+10h] [rbp-230h]
  unsigned int v94; // [rsp+28h] [rbp-218h]
  int v95; // [rsp+2Ch] [rbp-214h]
  __m128i *v96; // [rsp+30h] [rbp-210h]
  __m128i *v97; // [rsp+38h] [rbp-208h]
  __m128i *v98; // [rsp+40h] [rbp-200h]
  __m128i *v99; // [rsp+48h] [rbp-1F8h]
  int v100; // [rsp+50h] [rbp-1F0h]
  int v101; // [rsp+54h] [rbp-1ECh]
  __int64 *v102; // [rsp+58h] [rbp-1E8h]
  int v103; // [rsp+60h] [rbp-1E0h]
  unsigned __int16 v104; // [rsp+66h] [rbp-1DAh]
  unsigned __int16 v105; // [rsp+68h] [rbp-1D8h]
  __int64 v106; // [rsp+70h] [rbp-1D0h]
  __int64 v107; // [rsp+70h] [rbp-1D0h]
  __int64 v108; // [rsp+70h] [rbp-1D0h]
  __int64 v109; // [rsp+70h] [rbp-1D0h]
  unsigned __int16 v110; // [rsp+78h] [rbp-1C8h]
  __int16 v111; // [rsp+7Ah] [rbp-1C6h]
  signed int i; // [rsp+7Ch] [rbp-1C4h]
  unsigned int v113; // [rsp+7Ch] [rbp-1C4h]
  unsigned int v114; // [rsp+7Ch] [rbp-1C4h]
  unsigned int j; // [rsp+7Ch] [rbp-1C4h]
  char v116; // [rsp+80h] [rbp-1C0h]
  _DWORD *v117; // [rsp+88h] [rbp-1B8h]
  unsigned int v118; // [rsp+9Ch] [rbp-1A4h] BYREF
  _BYTE v119[32]; // [rsp+A0h] [rbp-1A0h] BYREF
  __int16 v120; // [rsp+C0h] [rbp-180h]
  _BYTE v121[32]; // [rsp+D0h] [rbp-170h] BYREF
  __int16 v122; // [rsp+F0h] [rbp-150h]
  _QWORD *v123; // [rsp+100h] [rbp-140h] BYREF
  __int64 v124; // [rsp+108h] [rbp-138h]
  _QWORD v125[38]; // [rsp+110h] [rbp-130h] BYREF

  v4 = (unsigned int)(a3 - 678);
  if ( (unsigned int)v4 > 0x1D )
  {
    v82 = (unsigned int)(a3 - 708);
    if ( (unsigned int)v82 > 0x17 )
    {
      v86 = (unsigned int)(a3 - 732);
      if ( (unsigned int)v86 > 0xC )
        sub_91B980("unexpected WMMA intrinsic!", 0);
      v116 = 1;
      v6 = 0;
      v94 = dword_3F147A0[v86];
    }
    else
    {
      v116 = 0;
      v6 = 0;
      v94 = dword_3F147E0[v82];
    }
  }
  else
  {
    v116 = 0;
    v6 = 1;
    v94 = dword_3F14840[v4];
  }
  v117 = (_DWORD *)(a4 + 36);
  v7 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(a4 + 72) + 16LL) + 16LL);
  v102 = *(__int64 **)(*(_QWORD *)(a4 + 72) + 16LL);
  v93 = (__int64 *)v7[2];
  v8 = *(_QWORD *)(v93[2] + 16);
  v92 = (__int64 *)v93[2];
  v9 = *(_QWORD *)(v8 + 16);
  sub_9480A0(v8, 3u, "unexpected 'rowcol' operand", "'rowcol' operand can be 0, 1, 2, or 3 only", (_DWORD *)(a4 + 36));
  v99 = sub_92F410(a2, (__int64)v102);
  v98 = sub_92F410(a2, (__int64)v7);
  v97 = sub_92F410(a2, (__int64)v93);
  v96 = sub_92F410(a2, (__int64)v92);
  v125[0] = sub_92F410(a2, v8);
  v123 = v125;
  v124 = 0x2000000001LL;
  if ( v94 != 8279 )
  {
    sub_9480A0(v9, 1u, "unexpected 'satf' operand", "'satf' operand can be 0, or 1 only", v117);
    v83 = sub_92F410(a2, v9);
    v84 = (unsigned int)v124;
    v85 = (unsigned int)v124 + 1LL;
    if ( v85 > HIDWORD(v124) )
    {
      sub_C8D5F0(&v123, v125, v85, 8);
      v84 = (unsigned int)v124;
    }
    v123[v84] = v83;
    LODWORD(v124) = v124 + 1;
    if ( !v6 )
    {
      if ( v116 )
        goto LABEL_6;
      if ( v94 > 0x22C1 )
      {
        if ( v94 - 8904 <= 1 )
        {
          v103 = 8;
          v101 = 8;
          v95 = 4;
          v100 = 1;
          goto LABEL_7;
        }
        goto LABEL_67;
      }
      if ( v94 > 0x22BF )
      {
        v87 = *(_QWORD *)(a2 + 32) + 8LL;
        v88 = sub_8D46C0(*v7);
        v103 = 8;
        v106 = sub_91A390(v87, v88, 0, v89);
        v101 = 8;
        v95 = 1;
        v100 = 4;
        goto LABEL_8;
      }
LABEL_43:
      if ( v94 - 8888 <= 1 )
      {
        v103 = 8;
        v101 = 8;
        v95 = 2;
        v100 = 2;
        goto LABEL_7;
      }
LABEL_67:
      sub_91B980("unexpected imma_mma intrinsic call!", 0);
    }
    if ( ((v94 - 8834) & 0xFFFFFFFD) == 0 || (v94 & 0xFFFFFFFD) == 0x2290 )
      goto LABEL_51;
    goto LABEL_62;
  }
  if ( v6 )
  {
LABEL_62:
    v101 = 4;
    if ( ((v94 - 8858) & 0xFFFFFFFD) != 0 )
    {
LABEL_52:
      if ( v94 - 8833 > 0x19 )
      {
        v103 = 8;
        v95 = 8;
        v100 = 8;
      }
      else
      {
        v95 = 8;
        v100 = 8;
        v103 = ((0x300C003uLL >> ((unsigned __int8)v94 + 127)) & 1) != 0 ? 4 : 8;
      }
      goto LABEL_7;
    }
LABEL_51:
    v101 = 8;
    goto LABEL_52;
  }
  if ( !v116 )
    goto LABEL_43;
LABEL_6:
  v103 = 2;
  v101 = 2;
  v95 = 1;
  v100 = 1;
LABEL_7:
  v10 = *(_QWORD *)(a2 + 32) + 8LL;
  v11 = sub_8D46C0(*v7);
  v106 = sub_91A390(v10, v11, 0, v12);
LABEL_8:
  for ( i = 0; i < v100; ++i )
  {
    v122 = 257;
    v13 = sub_94B2B0((unsigned int **)(a2 + 48), v106, (__int64)v98, i, (__int64)v121);
    v14 = *(_QWORD *)(a2 + 96);
    v120 = 257;
    v15 = v13;
    v16 = sub_AA4E30(v14);
    v17 = sub_AE5020(v16, v106);
    v18 = v105;
    v122 = 257;
    LOBYTE(v18) = v17;
    v105 = v18;
    v19 = sub_BD2C40(80, unk_3F10A14);
    v21 = v19;
    if ( v19 )
    {
      sub_B4D190(v19, v106, v15, (unsigned int)v121, 0, v18, 0, 0);
      v20 = v90;
    }
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a2 + 136) + 16LL))(
      *(_QWORD *)(a2 + 136),
      v21,
      v119,
      *(_QWORD *)(a2 + 104),
      *(_QWORD *)(a2 + 112),
      v20);
    v22 = *(unsigned int **)(a2 + 48);
    v23 = &v22[4 * *(unsigned int *)(a2 + 56)];
    while ( v23 != v22 )
    {
      v24 = *((_QWORD *)v22 + 1);
      v25 = *v22;
      v22 += 4;
      sub_B99FD0(v21, v25, v24);
    }
    v26 = (unsigned int)v124;
    v27 = (unsigned int)v124 + 1LL;
    if ( v27 > HIDWORD(v124) )
    {
      sub_C8D5F0(&v123, v125, v27, 8);
      v26 = (unsigned int)v124;
    }
    v123[v26] = v21;
    LODWORD(v124) = v124 + 1;
  }
  v28 = *(_QWORD *)(a2 + 32) + 8LL;
  v29 = sub_8D46C0(*v93);
  v113 = 0;
  v107 = sub_91A390(v28, v29, 0, v30);
  do
  {
    v122 = 257;
    v31 = sub_94B2B0((unsigned int **)(a2 + 48), v107, (__int64)v97, v113, (__int64)v121);
    v32 = *(_QWORD *)(a2 + 96);
    v120 = 257;
    v33 = v31;
    v34 = sub_AA4E30(v32);
    v35 = sub_AE5020(v34, v107);
    v36 = v110;
    v122 = 257;
    LOBYTE(v36) = v35;
    v110 = v36;
    v37 = sub_BD2C40(80, unk_3F10A14);
    v38 = v37;
    if ( v37 )
      sub_B4D190(v37, v107, v33, (unsigned int)v121, 0, v36, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
      *(_QWORD *)(a2 + 136),
      v38,
      v119,
      *(_QWORD *)(a2 + 104),
      *(_QWORD *)(a2 + 112));
    v39 = *(unsigned int **)(a2 + 48);
    v40 = &v39[4 * *(unsigned int *)(a2 + 56)];
    while ( v40 != v39 )
    {
      v41 = *((_QWORD *)v39 + 1);
      v42 = *v39;
      v39 += 4;
      sub_B99FD0(v38, v42, v41);
    }
    v43 = (unsigned int)v124;
    v44 = (unsigned int)v124 + 1LL;
    if ( v44 > HIDWORD(v124) )
    {
      sub_C8D5F0(&v123, v125, v44, 8);
      v43 = (unsigned int)v124;
    }
    ++v113;
    v123[v43] = v38;
    LODWORD(v124) = v124 + 1;
  }
  while ( v95 != v113 );
  v45 = *(_QWORD *)(a2 + 32) + 8LL;
  v46 = sub_8D46C0(*v92);
  v114 = 0;
  v108 = sub_91A390(v45, v46, 0, v47);
  do
  {
    v122 = 257;
    v48 = sub_94B2B0((unsigned int **)(a2 + 48), v108, (__int64)v96, v114, (__int64)v121);
    v49 = *(_QWORD *)(a2 + 96);
    v50 = v48;
    v120 = 257;
    v51 = sub_AA4E30(v49);
    v52 = v104;
    LOBYTE(v52) = sub_AE5020(v51, v108);
    v122 = 257;
    v104 = v52;
    v53 = sub_BD2C40(80, unk_3F10A14);
    v54 = v53;
    if ( v53 )
      sub_B4D190(v53, v108, v50, (unsigned int)v121, 0, v52, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
      *(_QWORD *)(a2 + 136),
      v54,
      v119,
      *(_QWORD *)(a2 + 104),
      *(_QWORD *)(a2 + 112));
    v55 = *(unsigned int **)(a2 + 48);
    v56 = &v55[4 * *(unsigned int *)(a2 + 56)];
    while ( v56 != v55 )
    {
      v57 = *((_QWORD *)v55 + 1);
      v58 = *v55;
      v55 += 4;
      sub_B99FD0(v54, v58, v57);
    }
    v59 = (unsigned int)v124;
    v60 = (unsigned int)v124 + 1LL;
    if ( v60 > HIDWORD(v124) )
    {
      sub_C8D5F0(&v123, v125, v60, 8);
      v59 = (unsigned int)v124;
    }
    ++v114;
    v123[v59] = v54;
    LODWORD(v124) = v124 + 1;
  }
  while ( v101 != v114 );
  v61 = sub_90A810(*(__int64 **)(a2 + 32), v94, 0, 0);
  v62 = 0;
  v122 = 257;
  if ( v61 )
    v62 = *(_QWORD *)(v61 + 24);
  v109 = sub_921880((unsigned int **)(a2 + 48), v62, v61, (int)v123, v124, (__int64)v121, 0);
  for ( j = 0; j != v103; ++j )
  {
    v63 = *(_QWORD *)(a2 + 32);
    v122 = 257;
    v64 = v63 + 8;
    v65 = sub_8D46C0(*v102);
    v67 = sub_91A390(v64, v65, 0, v66);
    v68 = sub_94B2B0((unsigned int **)(a2 + 48), v67, (__int64)v99, j, (__int64)v121);
    v120 = 257;
    v118 = j;
    v69 = sub_94D3D0((unsigned int **)(a2 + 48), v109, (__int64)&v118, 1, (__int64)v119);
    v70 = sub_AA4E30(*(_QWORD *)(a2 + 96));
    v71 = sub_AE5020(v70, *(_QWORD *)(v69 + 8));
    HIBYTE(v72) = HIBYTE(v111);
    v122 = 257;
    LOBYTE(v72) = v71;
    v111 = v72;
    v73 = sub_BD2C40(80, unk_3F10A10);
    v75 = v73;
    if ( v73 )
      sub_B4D3C0(v73, v69, v68, 0, (unsigned __int8)v111, v74, 0, 0);
    v76 = v75;
    (*(void (__fastcall **)(_QWORD, __int64, _BYTE *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 136) + 16LL))(
      *(_QWORD *)(a2 + 136),
      v75,
      v121,
      *(_QWORD *)(a2 + 104),
      *(_QWORD *)(a2 + 112));
    v77 = *(unsigned int **)(a2 + 48);
    v78 = &v77[4 * *(unsigned int *)(a2 + 56)];
    while ( v78 != v77 )
    {
      v79 = *((_QWORD *)v77 + 1);
      v76 = *v77;
      v77 += 4;
      sub_B99FD0(v75, v76, v79);
    }
  }
  v80 = v123;
  *(_BYTE *)(a1 + 12) &= ~1u;
  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 16) = 0;
  if ( v80 != v125 )
    _libc_free(v80, v76);
  return a1;
}
