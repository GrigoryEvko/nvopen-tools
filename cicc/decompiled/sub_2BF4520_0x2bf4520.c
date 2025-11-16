// Function: sub_2BF4520
// Address: 0x2bf4520
//
__int64 __fastcall sub_2BF4520(__int64 *a1, __int64 a2)
{
  __int64 v4; // r13
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // rcx
  unsigned int v8; // r8d
  __int64 v9; // rsi
  unsigned int v10; // edi
  unsigned int v11; // eax
  __int64 *v12; // rdx
  __int64 v13; // r9
  __int64 *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // r12
  unsigned int v17; // r8d
  __int64 *v18; // rdx
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // r14
  unsigned int v22; // eax
  __int64 v23; // rax
  bool v24; // r13
  __int64 v25; // rax
  __int64 v26; // r12
  __int64 v27; // r12
  __int64 v28; // rax
  __int16 v29; // ax
  __int64 v30; // rax
  unsigned int **v31; // rdi
  __int64 v32; // rax
  __int64 v33; // r15
  __int64 v34; // rdi
  __int64 v35; // rax
  __int16 v36; // cx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rsi
  __int64 v41; // r8
  unsigned __int64 v42; // rsi
  _QWORD *v43; // rax
  int v44; // ecx
  _QWORD *v45; // rdx
  unsigned __int64 v47; // rax
  __int64 v48; // r12
  __int64 v49; // rax
  __int64 v50; // rdi
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rsi
  int v53; // edx
  int v54; // r10d
  __int64 v55; // rdx
  int v56; // edx
  int v57; // r10d
  __int64 v58; // [rsp+8h] [rbp-B8h]
  __int64 v59[4]; // [rsp+20h] [rbp-A0h] BYREF
  char v60; // [rsp+40h] [rbp-80h]
  char v61; // [rsp+41h] [rbp-7Fh]
  __int64 v62; // [rsp+50h] [rbp-70h]
  _QWORD v63[2]; // [rsp+58h] [rbp-68h] BYREF
  __int64 v64; // [rsp+68h] [rbp-58h]
  __int64 v65; // [rsp+70h] [rbp-50h]
  __int16 v66; // [rsp+78h] [rbp-48h]
  __int64 v67[8]; // [rsp+80h] [rbp-40h] BYREF

  if ( !sub_2BF04A0(a1[1]) )
    goto LABEL_26;
  v4 = *a1;
  v5 = sub_2BF3F10(*(_QWORD **)(*a1 + 920));
  v6 = v5;
  if ( v5 )
  {
    if ( *(_DWORD *)(v5 + 64) == 1 )
      v6 = **(_QWORD **)(v5 + 56);
    else
      v6 = 0;
  }
  v7 = *(_QWORD *)(sub_2BF04A0(a1[1]) + 80);
  if ( v7 == v6 )
    goto LABEL_62;
  v8 = *(_DWORD *)(v4 + 1136);
  v9 = *(_QWORD *)(v4 + 1120);
  if ( !v8 )
    goto LABEL_74;
  v10 = v8 - 1;
  v11 = (v8 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v12 = (__int64 *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( v6 != *v12 )
  {
    v56 = 1;
    while ( v13 != -4096 )
    {
      v57 = v56 + 1;
      v11 = v10 & (v56 + v11);
      v12 = (__int64 *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( v6 == *v12 )
        goto LABEL_8;
      v56 = v57;
    }
LABEL_74:
    v14 = (__int64 *)(v9 + 16LL * v8);
LABEL_75:
    v16 = 0;
    v10 = v8 - 1;
    if ( !v8 )
      goto LABEL_77;
    goto LABEL_76;
  }
LABEL_8:
  v14 = (__int64 *)(v9 + 16LL * v8);
  if ( v14 == v12 )
    goto LABEL_75;
  v15 = *((unsigned int *)v12 + 2);
  if ( *(_DWORD *)(v4 + 1056) > (unsigned int)v15 )
  {
    v16 = *(_QWORD *)(*(_QWORD *)(v4 + 1048) + 8 * v15);
    goto LABEL_11;
  }
LABEL_76:
  v16 = 0;
LABEL_11:
  v17 = v10 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
  v18 = (__int64 *)(v9 + 16LL * v17);
  v19 = *v18;
  if ( v7 != *v18 )
  {
    v53 = 1;
    while ( v19 != -4096 )
    {
      v54 = v53 + 1;
      v55 = v10 & (v17 + v53);
      v17 = v55;
      v18 = (__int64 *)(v9 + 16 * v55);
      v19 = *v18;
      if ( v7 == *v18 )
        goto LABEL_12;
      v53 = v54;
    }
    goto LABEL_77;
  }
LABEL_12:
  if ( v18 == v14 || (v20 = *((unsigned int *)v18 + 2), *(_DWORD *)(v4 + 1056) <= (unsigned int)v20) )
  {
LABEL_77:
    v24 = v16 == 0;
    goto LABEL_27;
  }
  v21 = *(_QWORD *)(*(_QWORD *)(v4 + 1048) + 8 * v20);
  if ( v21 == v16 || !v16 )
    goto LABEL_26;
  if ( !v21 )
  {
LABEL_62:
    v24 = 0;
    goto LABEL_27;
  }
  if ( *(_QWORD *)(v16 + 8) == v21 )
  {
LABEL_26:
    v24 = 1;
    goto LABEL_27;
  }
  if ( *(_QWORD *)(v21 + 8) == v16 || *(_DWORD *)(v21 + 16) >= *(_DWORD *)(v16 + 16) )
    goto LABEL_62;
  if ( *(_BYTE *)(v4 + 1160) )
    goto LABEL_61;
  v22 = *(_DWORD *)(v4 + 1164) + 1;
  *(_DWORD *)(v4 + 1164) = v22;
  if ( v22 > 0x20 )
  {
    sub_2BF23E0(v4 + 1024);
LABEL_61:
    if ( *(_DWORD *)(v16 + 72) >= *(_DWORD *)(v21 + 72) )
    {
      v24 = *(_DWORD *)(v16 + 76) <= *(_DWORD *)(v21 + 76);
      goto LABEL_27;
    }
    goto LABEL_62;
  }
  do
  {
    v23 = v16;
    v16 = *(_QWORD *)(v16 + 8);
  }
  while ( v16 && *(_DWORD *)(v21 + 16) <= *(_DWORD *)(v16 + 16) );
  v24 = v23 == v21;
LABEL_27:
  v25 = *a1;
  if ( !*(_BYTE *)(*a1 + 12) )
  {
    v26 = a2;
    if ( *(_DWORD *)(v25 + 8) == 1 )
      return v26;
  }
  v27 = *(_QWORD *)(v25 + 904);
  v63[0] = 0;
  v63[1] = 0;
  v28 = *(_QWORD *)(v27 + 48);
  v62 = v27;
  v64 = v28;
  if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
    sub_BD73F0((__int64)v63);
  v29 = *(_WORD *)(v27 + 64);
  v65 = *(_QWORD *)(v27 + 56);
  v66 = v29;
  sub_B33910(v67, (__int64 *)v27);
  if ( v24 )
  {
    v48 = *a1 + 120;
    v49 = sub_2BF3F10(*(_QWORD **)(*a1 + 920));
    if ( v49 )
    {
      if ( *(_DWORD *)(v49 + 64) == 1 )
        v49 = **(_QWORD **)(v49 + 56);
      else
        v49 = 0;
    }
    v59[0] = v49;
    v50 = *sub_2BF2B80(v48, v59);
    if ( v50 )
    {
      v51 = sub_986580(v50);
      sub_D5F1F0(*(_QWORD *)(*a1 + 904), v51);
    }
  }
  v30 = *a1;
  v31 = *(unsigned int ***)(*a1 + 904);
  v59[0] = (__int64)"broadcast";
  v61 = 1;
  v60 = 3;
  v32 = sub_B37620(v31, *(_QWORD *)(v30 + 8), a2, v59);
  v33 = v62;
  v34 = v65;
  v26 = v32;
  v35 = v64;
  v36 = v66;
  if ( v64 )
  {
    *(_QWORD *)(v62 + 48) = v64;
    *(_QWORD *)(v33 + 56) = v34;
    *(_WORD *)(v33 + 64) = v36;
    if ( v34 != v35 + 48 )
    {
      if ( v34 )
        v34 -= 24;
      v38 = *(_QWORD *)sub_B46C60(v34);
      v59[0] = v38;
      if ( v38 )
      {
        sub_B96E90((__int64)v59, v38, 1);
        v38 = v59[0];
      }
      sub_F80810(v33, 0, v38, v37, v38, v39);
      if ( v59[0] )
        sub_B91220((__int64)v59, v59[0]);
      v40 = v67[0];
      v33 = v62;
      v59[0] = v67[0];
      if ( !v67[0] )
        goto LABEL_42;
      goto LABEL_47;
    }
  }
  else
  {
    *(_QWORD *)(v62 + 48) = 0;
    *(_QWORD *)(v33 + 56) = 0;
    *(_WORD *)(v33 + 64) = 0;
  }
  v40 = v67[0];
  v59[0] = v67[0];
  if ( !v67[0] )
  {
LABEL_42:
    sub_93FB40(v33, 0);
    v41 = v59[0];
    goto LABEL_43;
  }
LABEL_47:
  sub_B96E90((__int64)v59, v40, 1);
  v41 = v59[0];
  if ( !v59[0] )
    goto LABEL_42;
  v42 = *(unsigned int *)(v33 + 8);
  v43 = *(_QWORD **)v33;
  v44 = *(_DWORD *)(v33 + 8);
  v45 = (_QWORD *)(*(_QWORD *)v33 + 16 * v42);
  if ( *(_QWORD **)v33 != v45 )
  {
    while ( *(_DWORD *)v43 )
    {
      v43 += 2;
      if ( v45 == v43 )
        goto LABEL_63;
    }
    v43[1] = v59[0];
    goto LABEL_53;
  }
LABEL_63:
  v47 = *(unsigned int *)(v33 + 12);
  if ( v42 >= v47 )
  {
    v52 = v42 + 1;
    if ( v47 < v52 )
    {
      v58 = v59[0];
      sub_C8D5F0(v33, (const void *)(v33 + 16), v52, 0x10u, v59[0], v33 + 16);
      v41 = v58;
      v45 = (_QWORD *)(*(_QWORD *)v33 + 16LL * *(unsigned int *)(v33 + 8));
    }
    *v45 = 0;
    v45[1] = v41;
    ++*(_DWORD *)(v33 + 8);
    v41 = v59[0];
  }
  else
  {
    if ( v45 )
    {
      *(_DWORD *)v45 = 0;
      v45[1] = v41;
      v44 = *(_DWORD *)(v33 + 8);
      v41 = v59[0];
    }
    *(_DWORD *)(v33 + 8) = v44 + 1;
  }
LABEL_43:
  if ( v41 )
LABEL_53:
    sub_B91220((__int64)v59, v41);
  if ( v67[0] )
    sub_B91220((__int64)v67, v67[0]);
  if ( v64 != 0 && v64 != -4096 && v64 != -8192 )
    sub_BD60C0(v63);
  return v26;
}
