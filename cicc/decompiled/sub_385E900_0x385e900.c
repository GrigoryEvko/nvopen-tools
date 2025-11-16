// Function: sub_385E900
// Address: 0x385e900
//
__int64 __fastcall sub_385E900(__int64 *a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 *v8; // rbx
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 *v12; // r9
  _QWORD *v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  int v16; // eax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // edx
  __int64 v21; // r15
  __int64 *v22; // rcx
  __int64 *v23; // rax
  unsigned __int64 v24; // r10
  unsigned int v25; // r12d
  __int64 *v27; // rdx
  __int64 v28; // rcx
  __int64 *v29; // rax
  bool v30; // r9
  __int64 v31; // rax
  int v32; // edx
  __int64 *v33; // rax
  char *v34; // r14
  unsigned __int64 v35; // r12
  char *v36; // rdx
  __int64 v37; // rax
  unsigned int *v38; // r8
  __int64 v39; // rcx
  __int64 v40; // rax
  __int64 v41; // rsi
  unsigned int *v42; // rax
  unsigned int *v43; // rsi
  __int64 v44; // rax
  __int64 *v45; // r13
  __int64 *v46; // rbx
  __int64 *v47; // r14
  __int64 v48; // rcx
  __int64 *i; // r15
  __int64 v50; // rdi
  __int64 *v51; // rax
  char v52; // r13
  __int64 v53; // rax
  bool v54; // zf
  unsigned int v55; // eax
  __int64 v56; // rax
  __int64 *v57; // r10
  __int64 v58; // rdx
  __int64 *v59; // rax
  bool v60; // r10
  __int64 v61; // rax
  __int64 v62; // rax
  unsigned int *v63; // r13
  _DWORD *v64; // rax
  __int64 v65; // rdx
  __int64 v66; // rax
  __int64 v67; // rdx
  unsigned __int64 v68; // rax
  __int64 v69; // rcx
  __int64 v70; // rcx
  __int64 v71; // r15
  char *v72; // rax
  __int64 v73; // [rsp+0h] [rbp-160h]
  unsigned __int64 v74; // [rsp+8h] [rbp-158h]
  __int64 *v75; // [rsp+10h] [rbp-150h]
  __int64 v76; // [rsp+18h] [rbp-148h]
  char v77; // [rsp+20h] [rbp-140h]
  __int64 *v78; // [rsp+20h] [rbp-140h]
  __int64 *v79; // [rsp+20h] [rbp-140h]
  __int64 *v81; // [rsp+30h] [rbp-130h]
  unsigned __int64 v82; // [rsp+30h] [rbp-130h]
  __int64 v83; // [rsp+30h] [rbp-130h]
  char v84; // [rsp+30h] [rbp-130h]
  __int64 *v85; // [rsp+30h] [rbp-130h]
  __int64 *v86; // [rsp+60h] [rbp-100h]
  __int64 v87; // [rsp+68h] [rbp-F8h]
  void *src; // [rsp+70h] [rbp-F0h]
  unsigned int *srca; // [rsp+70h] [rbp-F0h]
  __int64 *v90; // [rsp+78h] [rbp-E8h]
  _BYTE *v91; // [rsp+80h] [rbp-E0h] BYREF
  __int64 v92; // [rsp+88h] [rbp-D8h]
  _BYTE v93[64]; // [rsp+90h] [rbp-D0h] BYREF
  __int64 *v94; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v95; // [rsp+D8h] [rbp-88h]
  _BYTE v96[40]; // [rsp+E0h] [rbp-80h] BYREF
  __int64 v97; // [rsp+108h] [rbp-58h] BYREF
  __int64 *v98; // [rsp+110h] [rbp-50h]
  __int64 *v99; // [rsp+118h] [rbp-48h]
  __int64 *v100; // [rsp+120h] [rbp-40h]
  __int64 v101; // [rsp+128h] [rbp-38h]

  v8 = a1;
  v91 = v93;
  v92 = 0x400000000LL;
  if ( a2 > 4 )
    sub_16CD150((__int64)&v91, v93, a2, 16, a5, a6);
  v9 = *a1;
  v90 = (__int64 *)*a1;
  v87 = sub_146F1B0(a4, *a1);
  v10 = sub_14AD280(v9, a3, 6u);
  LODWORD(v97) = 0;
  src = (void *)v10;
  v94 = (__int64 *)v96;
  v95 = 0x400000000LL;
  v98 = 0;
  v99 = &v97;
  v100 = &v97;
  v101 = 0;
  v86 = &a1[a2];
  if ( a1 == v86 )
    goto LABEL_41;
  do
  {
    v13 = (_QWORD *)*v8;
    v14 = *(_QWORD *)*v8;
    if ( *(_BYTE *)(v14 + 8) == 16 )
      v14 = **(_QWORD **)(v14 + 16);
    v15 = *v90;
    v16 = *(_DWORD *)(v14 + 8) >> 8;
    if ( *(_BYTE *)(*v90 + 8) == 16 )
      v15 = **(_QWORD **)(v15 + 16);
    if ( *(_DWORD *)(v15 + 8) >> 8 != v16
      || src != (void *)sub_14AD280(*v8, a3, 6u)
      || (v17 = sub_146F1B0(a4, (__int64)v13), v18 = sub_14806B0(a4, v17, v87, 0, 0), *(_WORD *)(v18 + 24)) )
    {
LABEL_18:
      v24 = (unsigned __int64)v98;
      goto LABEL_19;
    }
    v19 = *(_QWORD *)(v18 + 32);
    v20 = *(_DWORD *)(v19 + 32);
    v11 = *(_QWORD *)(v19 + 24);
    if ( v20 > 0x40 )
    {
      v21 = *(_QWORD *)v11;
      if ( !v101 )
        goto LABEL_13;
LABEL_26:
      v27 = v98;
      v24 = (unsigned __int64)v98;
      if ( !v98 )
      {
        v27 = &v97;
        if ( v99 == &v97 )
        {
          v27 = &v97;
          v30 = 1;
LABEL_36:
          v77 = v30;
          v81 = v27;
          v31 = sub_22077B0(0x28u);
          *(_QWORD *)(v31 + 32) = v21;
          sub_220F040(v77, v31, v81, &v97);
          ++v101;
          goto LABEL_37;
        }
LABEL_54:
        v82 = (unsigned __int64)v98;
        v78 = v27;
        v44 = sub_220EF80((__int64)v27);
        v24 = v82;
        if ( v21 <= *(_QWORD *)(v44 + 32) )
          goto LABEL_19;
        v27 = v78;
        if ( !v78 )
          goto LABEL_19;
LABEL_34:
        v30 = 1;
        if ( v27 != &v97 )
          v30 = v21 < v27[4];
        goto LABEL_36;
      }
      while ( 1 )
      {
        v28 = v27[4];
        v29 = (__int64 *)v27[3];
        if ( v21 < v28 )
          v29 = (__int64 *)v27[2];
        if ( !v29 )
          break;
        v27 = v29;
      }
      if ( v21 < v28 )
      {
        if ( v99 == v27 )
          goto LABEL_34;
        goto LABEL_54;
      }
      if ( v21 > v28 )
        goto LABEL_34;
LABEL_19:
      v25 = 0;
      goto LABEL_20;
    }
    v11 = v11 << (64 - (unsigned __int8)v20) >> (64 - (unsigned __int8)v20);
    v21 = v11;
    if ( v101 )
      goto LABEL_26;
LABEL_13:
    v22 = &v94[(unsigned int)v95];
    if ( v94 != v22 )
    {
      v23 = v94;
      while ( v21 != *v23 )
      {
        if ( v22 == ++v23 )
          goto LABEL_57;
      }
      if ( v22 != v23 )
        goto LABEL_18;
    }
LABEL_57:
    if ( (unsigned int)v95 <= 3uLL )
    {
      if ( (unsigned int)v95 >= HIDWORD(v95) )
      {
        sub_16CD150((__int64)&v94, v96, 0, 8, v11, (int)v12);
        v22 = &v94[(unsigned int)v95];
      }
      *v22 = v21;
      LODWORD(v95) = v95 + 1;
      goto LABEL_37;
    }
    v73 = a4;
    v45 = v98;
    v75 = v8;
    v46 = v98;
    v74 = a3;
    v47 = &v94[(unsigned int)v95 - 1];
    v76 = v21;
    if ( v98 )
    {
LABEL_59:
      v48 = *v47;
      for ( i = v45; ; i = v51 )
      {
        v50 = i[4];
        v51 = (__int64 *)i[3];
        if ( v48 < v50 )
          v51 = (__int64 *)i[2];
        LOBYTE(v11) = v48 < v50;
        if ( !v51 )
          break;
      }
      if ( v48 < v50 )
      {
        if ( v99 != i )
          goto LABEL_74;
      }
      else if ( v48 <= v50 )
      {
        goto LABEL_69;
      }
LABEL_66:
      v52 = 1;
      if ( i != &v97 )
        v52 = v48 < i[4];
LABEL_68:
      v53 = sub_22077B0(0x28u);
      *(_QWORD *)(v53 + 32) = *v47;
      sub_220F040(v52, v53, i, &v97);
      ++v101;
      v45 = v98;
      v46 = v98;
      goto LABEL_69;
    }
    while ( 1 )
    {
      i = &v97;
      if ( v99 == &v97 )
      {
        v52 = 1;
        goto LABEL_68;
      }
      v48 = *v47;
LABEL_74:
      v83 = v48;
      v56 = sub_220EF80((__int64)i);
      v48 = v83;
      if ( v83 > *(_QWORD *)(v56 + 32) )
        goto LABEL_66;
LABEL_69:
      v54 = (_DWORD)v95 == 1;
      v55 = v95 - 1;
      LODWORD(v95) = v95 - 1;
      if ( v54 )
        break;
      v47 = &v94[v55 - 1];
      if ( v45 )
        goto LABEL_59;
    }
    v57 = v45;
    v12 = v46;
    v21 = v76;
    v8 = v75;
    a3 = v74;
    a4 = v73;
    if ( v57 )
    {
      while ( 1 )
      {
        v58 = v12[4];
        v59 = (__int64 *)v12[3];
        if ( v76 < v58 )
          v59 = (__int64 *)v12[2];
        if ( !v59 )
          break;
        v12 = v59;
      }
      if ( v76 < v58 )
      {
        if ( v99 != v12 )
          goto LABEL_89;
      }
      else if ( v76 <= v58 )
      {
        goto LABEL_37;
      }
LABEL_84:
      v60 = 1;
      if ( v12 != &v97 )
        v60 = v76 < v12[4];
LABEL_86:
      v79 = v12;
      v84 = v60;
      v61 = sub_22077B0(0x28u);
      *(_QWORD *)(v61 + 32) = v76;
      sub_220F040(v84, v61, v79, &v97);
      ++v101;
      v32 = v92;
      if ( (unsigned int)v92 >= HIDWORD(v92) )
        goto LABEL_87;
      goto LABEL_38;
    }
    v12 = &v97;
    if ( v99 == &v97 )
    {
      v12 = &v97;
      v60 = 1;
      goto LABEL_86;
    }
LABEL_89:
    v85 = v12;
    if ( v76 > *(_QWORD *)(sub_220EF80((__int64)v12) + 32) )
    {
      v12 = v85;
      if ( v85 )
        goto LABEL_84;
    }
LABEL_37:
    v32 = v92;
    if ( (unsigned int)v92 < HIDWORD(v92) )
      goto LABEL_38;
LABEL_87:
    sub_16CD150((__int64)&v91, v93, 0, 16, v11, (int)v12);
    v32 = v92;
LABEL_38:
    v33 = (__int64 *)&v91[16 * v32];
    if ( v33 )
    {
      *v33 = v21;
      v33[1] = (__int64)v13;
      v32 = v92;
    }
    ++v8;
    LODWORD(v92) = v32 + 1;
  }
  while ( v86 != v8 );
LABEL_41:
  *(_DWORD *)(a5 + 8) = 0;
  if ( !a2 )
  {
    v34 = *(char **)a5;
    srca = *(unsigned int **)a5;
    goto LABEL_43;
  }
  v62 = 0;
  if ( a2 > *(unsigned int *)(a5 + 12) )
  {
    sub_16CD150(a5, (const void *)(a5 + 16), a2, 4, v11, (int)v12);
    v62 = 4LL * *(unsigned int *)(a5 + 8);
  }
  v63 = *(unsigned int **)a5;
  v64 = (_DWORD *)(*(_QWORD *)a5 + v62);
  v65 = *(_QWORD *)a5 + 4 * a2;
  if ( v64 != (_DWORD *)v65 )
  {
    do
    {
      if ( v64 )
        *v64 = 0;
      ++v64;
    }
    while ( (_DWORD *)v65 != v64 );
    v63 = *(unsigned int **)a5;
  }
  *(_DWORD *)(a5 + 8) = a2;
  v66 = 4LL * (unsigned int)a2;
  if ( v66 )
  {
    v67 = 0;
    v68 = (unsigned __int64)(v66 - 4) >> 2;
    do
    {
      v69 = v67;
      v63[v67] = v67;
      ++v67;
    }
    while ( v68 != v69 );
    v70 = *(unsigned int *)(a5 + 8);
    v63 = *(unsigned int **)a5;
    v66 = 4 * v70;
  }
  else
  {
    v70 = 0;
  }
  srca = v63;
  v34 = (char *)&v63[(unsigned __int64)v66 / 4];
  if ( v66 )
  {
    v71 = v70;
    while ( 1 )
    {
      v72 = (char *)sub_2207800(4 * v71);
      v35 = (unsigned __int64)v72;
      if ( v72 )
        break;
      v71 >>= 1;
      if ( !v71 )
        goto LABEL_43;
    }
    sub_385CDA0(v63, v34, v72, (unsigned int *)v71, &v91);
  }
  else
  {
LABEL_43:
    v35 = 0;
    sub_385C520(srca, v34, &v91);
  }
  j_j___libc_free_0(v35);
  v36 = *(char **)a5;
  v37 = 4LL * *(unsigned int *)(a5 + 8);
  v38 = (unsigned int *)(*(_QWORD *)a5 + v37);
  v39 = v37 >> 2;
  v40 = v37 >> 4;
  if ( !v40 )
  {
    v42 = *(unsigned int **)a5;
LABEL_114:
    if ( v39 != 2 )
    {
      if ( v39 != 3 )
      {
        if ( v39 != 1 )
          goto LABEL_118;
LABEL_117:
        if ( *v42 != *(_DWORD *)&v36[4 * *v42] )
          goto LABEL_51;
        goto LABEL_118;
      }
      if ( *v42 != *(_DWORD *)&v36[4 * *v42] )
        goto LABEL_51;
      ++v42;
    }
    if ( *v42 != *(_DWORD *)&v36[4 * *v42] )
      goto LABEL_51;
    ++v42;
    goto LABEL_117;
  }
  v41 = 16 * v40;
  v42 = *(unsigned int **)a5;
  v43 = (unsigned int *)&v36[v41];
  while ( *v42 == *(_DWORD *)&v36[4 * *v42] )
  {
    if ( v42[1] != *(_DWORD *)&v36[4 * v42[1]] )
    {
      ++v42;
      break;
    }
    if ( v42[2] != *(_DWORD *)&v36[4 * v42[2]] )
    {
      v42 += 2;
      break;
    }
    if ( v42[3] != *(_DWORD *)&v36[4 * v42[3]] )
    {
      v42 += 3;
      break;
    }
    v42 += 4;
    if ( v43 == v42 )
    {
      v39 = v38 - v42;
      goto LABEL_114;
    }
  }
LABEL_51:
  if ( v38 != v42 )
  {
    v24 = (unsigned __int64)v98;
    v25 = 1;
    goto LABEL_20;
  }
LABEL_118:
  v24 = (unsigned __int64)v98;
  v25 = 1;
  *(_DWORD *)(a5 + 8) = 0;
LABEL_20:
  sub_385BE40(v24);
  if ( v94 != (__int64 *)v96 )
    _libc_free((unsigned __int64)v94);
  if ( v91 != v93 )
    _libc_free((unsigned __int64)v91);
  return v25;
}
