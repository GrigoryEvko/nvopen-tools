// Function: sub_1D931A0
// Address: 0x1d931a0
//
__int64 __fastcall sub_1D931A0(
        __int64 a1,
        _QWORD *a2,
        _QWORD *a3,
        _QWORD *a4,
        _QWORD *a5,
        _DWORD *a6,
        _DWORD *a7,
        __int64 a8,
        __int64 a9,
        char a10)
{
  _QWORD *v11; // rcx
  _QWORD *v12; // rax
  _QWORD *i; // rsi
  _QWORD *v18; // rdi
  _QWORD *v19; // rax
  __int64 v20; // r8
  _QWORD *v21; // rdi
  __int64 (*v22)(); // rax
  __int16 v23; // ax
  _QWORD *v24; // rax
  __int64 v25; // rdi
  _QWORD *v27; // rdx
  unsigned __int64 v28; // r13
  __int64 v29; // rax
  unsigned __int64 v30; // rax
  unsigned __int64 v31; // r14
  __int64 v32; // rax
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // r8
  __int64 v38; // rax
  unsigned __int64 v39; // r15
  unsigned __int64 v40; // rax
  __int16 v41; // dx
  __int16 v42; // si
  char v43; // al
  char v44; // r10
  char v45; // al
  char v46; // r9
  __int64 v47; // rax
  _QWORD *v48; // rax
  _QWORD *v49; // rdx
  __int64 v50; // rax
  unsigned __int64 v51; // rax
  __int16 v52; // dx
  __int16 v53; // si
  char v54; // al
  char v55; // r10
  char v56; // al
  __int64 v57; // r9
  char v58; // r9
  __int64 v59; // rax
  _QWORD *v60; // rax
  _QWORD *v61; // rdx
  __int64 v62; // rax
  unsigned __int64 v63; // rax
  __int16 v64; // ax
  char v65; // al
  _QWORD *v66; // rax
  _QWORD *v67; // rdx
  __int64 v68; // rax
  unsigned __int64 v69; // rax
  unsigned __int64 v70; // rdx
  __int64 v71; // rax
  unsigned __int64 v72; // rax
  _QWORD *v73; // rax
  _QWORD *v74; // rdx
  __int64 v75; // rax
  unsigned __int64 v76; // rax
  _QWORD *v77; // rax
  _QWORD *v78; // rdx
  __int64 v79; // rax
  unsigned __int64 v80; // rax
  unsigned __int64 v81; // [rsp+10h] [rbp-60h]
  unsigned __int64 v82; // [rsp+10h] [rbp-60h]
  unsigned __int64 v83; // [rsp+10h] [rbp-60h]
  unsigned __int64 v84; // [rsp+10h] [rbp-60h]
  unsigned __int64 v86; // [rsp+18h] [rbp-58h]
  char v87; // [rsp+18h] [rbp-58h]
  char v88; // [rsp+18h] [rbp-58h]
  unsigned __int64 v89; // [rsp+18h] [rbp-58h]
  char v90; // [rsp+18h] [rbp-58h]
  char v91; // [rsp+18h] [rbp-58h]
  unsigned __int64 v92; // [rsp+18h] [rbp-58h]
  _QWORD v93[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v94; // [rsp+30h] [rbp-40h]

  v11 = (_QWORD *)*a4;
  v12 = (_QWORD *)*a2;
  if ( v11 == (_QWORD *)*a2 )
    return 1;
  while ( 1 )
  {
    if ( *a5 == *a3 )
      return 1;
    for ( ; v12 != v11; v12 = (_QWORD *)v12[1] )
    {
      if ( (unsigned __int16)(*(_WORD *)v12[2] - 12) > 1u )
        break;
      if ( (*(_BYTE *)v12 & 4) == 0 )
      {
        while ( (*((_BYTE *)v12 + 46) & 8) != 0 )
          v12 = (_QWORD *)v12[1];
      }
    }
    *a2 = v12;
    for ( i = (_QWORD *)*a3; i != (_QWORD *)*a5; i = (_QWORD *)i[1] )
    {
      if ( (unsigned __int16)(*(_WORD *)i[2] - 12) > 1u )
        break;
      if ( (*(_BYTE *)i & 4) == 0 )
      {
        while ( (*((_BYTE *)i + 46) & 8) != 0 )
          i = (_QWORD *)i[1];
      }
    }
    *a3 = i;
    v18 = (_QWORD *)*a2;
    v19 = (_QWORD *)*a4;
    if ( *a4 == *a2 )
      return 1;
    if ( (_QWORD *)*a5 == i )
      goto LABEL_44;
    if ( !(unsigned __int8)sub_1E15D60(v18, i, 0) )
      break;
    v20 = *(_QWORD *)(a1 + 544);
    v93[0] = 0;
    v93[1] = 0;
    v21 = (_QWORD *)*a2;
    v94 = 0;
    v22 = *(__int64 (**)())(*(_QWORD *)v20 + 712LL);
    if ( v22 != sub_1D918E0 )
    {
      if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD *, _QWORD *))v22)(v20, v21, v93) )
      {
        if ( v93[0] )
          j_j___libc_free_0(v93[0], v94 - v93[0]);
        return 0;
      }
      v21 = (_QWORD *)*a2;
    }
    v23 = *((_WORD *)v21 + 23);
    if ( (v23 & 4) == 0 && (v23 & 8) != 0 )
    {
      if ( !(unsigned __int8)sub_1E15D00(v21, 128, 1) )
LABEL_18:
        ++*a6;
      v21 = (_QWORD *)*a2;
      if ( !*a2 )
        BUG();
      goto LABEL_20;
    }
    if ( *(char *)(v21[2] + 8LL) >= 0 )
      goto LABEL_18;
LABEL_20:
    if ( (*(_BYTE *)v21 & 4) == 0 )
    {
      while ( (*((_BYTE *)v21 + 46) & 8) != 0 )
        v21 = (_QWORD *)v21[1];
    }
    *a2 = v21[1];
    v24 = (_QWORD *)*a3;
    if ( !*a3 )
      BUG();
    if ( (*(_BYTE *)v24 & 4) == 0 )
    {
      while ( (*((_BYTE *)v24 + 46) & 8) != 0 )
        v24 = (_QWORD *)v24[1];
    }
    v25 = v93[0];
    *a3 = v24[1];
    if ( v25 )
      j_j___libc_free_0(v25, v94 - v25);
    v12 = (_QWORD *)*a2;
    v11 = (_QWORD *)*a4;
    if ( *a4 == *a2 )
      return 1;
  }
  v19 = (_QWORD *)*a4;
  v18 = (_QWORD *)*a2;
  if ( *a2 == *a4 )
    return 1;
LABEL_44:
  v27 = (_QWORD *)*a3;
  if ( *a3 == *a5 )
    return 1;
  v28 = *v19 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v28 )
    BUG();
  v29 = *(_QWORD *)v28;
  if ( (*(_QWORD *)v28 & 4) == 0 && (*(_BYTE *)(v28 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      v30 = v29 & 0xFFFFFFFFFFFFFFF8LL;
      v28 = v30;
      if ( (*(_BYTE *)(v30 + 46) & 4) == 0 )
        break;
      v29 = *(_QWORD *)v30;
    }
  }
  v31 = *(_QWORD *)*a5 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v31 )
    BUG();
  v32 = *(_QWORD *)v31;
  if ( (*(_QWORD *)v31 & 4) == 0 && (*(_BYTE *)(v31 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      v33 = v32 & 0xFFFFFFFFFFFFFFF8LL;
      v31 = v33;
      if ( (*(_BYTE *)(v33 + 46) & 4) == 0 )
        break;
      v32 = *(_QWORD *)v33;
    }
  }
  v34 = *v18 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v34 )
    BUG();
  v35 = *(_QWORD *)v34;
  if ( (*(_QWORD *)v34 & 4) == 0 && (*(_BYTE *)(v34 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      v36 = v35 & 0xFFFFFFFFFFFFFFF8LL;
      v34 = v36;
      if ( (*(_BYTE *)(v36 + 46) & 4) == 0 )
        break;
      v35 = *(_QWORD *)v36;
    }
  }
  v37 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v37 )
    BUG();
  v38 = *(_QWORD *)v37;
  v39 = *v27 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (*(_QWORD *)v37 & 4) == 0 && (*(_BYTE *)(v37 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      v40 = v38 & 0xFFFFFFFFFFFFFFF8LL;
      v39 = v40;
      if ( (*(_BYTE *)(v40 + 46) & 4) == 0 )
        break;
      v38 = *(_QWORD *)v40;
    }
  }
  if ( *(_QWORD *)(a8 + 96) == *(_QWORD *)(a8 + 88) )
  {
    if ( *(_QWORD *)(a9 + 96) == *(_QWORD *)(a9 + 88) || !a10 )
      goto LABEL_126;
  }
  else if ( !a10 )
  {
    goto LABEL_126;
  }
  if ( v28 == v34 )
  {
    if ( v31 != v39 )
      goto LABEL_90;
    goto LABEL_146;
  }
  while ( 2 )
  {
    v41 = *(_WORD *)(v28 + 46);
    v42 = v41 & 4;
    if ( (v41 & 4) != 0 || (v41 & 8) == 0 )
    {
      v44 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v28 + 16) + 8LL) >> 7;
    }
    else
    {
      v86 = v34;
      v43 = sub_1E15D00(v28, 128, 1);
      v41 = *(_WORD *)(v28 + 46);
      v34 = v86;
      v44 = v43;
      v42 = v41 & 4;
    }
    if ( v42 || (v41 & 8) == 0 )
    {
      v46 = v44 & ((*(_QWORD *)(*(_QWORD *)(v28 + 16) + 8LL) & 0x20LL) != 0);
      if ( v42 )
        goto LABEL_152;
LABEL_78:
      if ( (v41 & 8) == 0 )
        goto LABEL_152;
      v82 = v34;
      v88 = v46;
      LOBYTE(v47) = sub_1E15D00(v28, 256, 1);
      v46 = v88;
      v34 = v82;
    }
    else
    {
      v81 = v34;
      v87 = v44;
      v45 = sub_1E15D00(v28, 32, 1);
      v41 = *(_WORD *)(v28 + 46);
      v34 = v81;
      v46 = v87 & v45;
      if ( (v41 & 4) == 0 )
        goto LABEL_78;
LABEL_152:
      v47 = (*(_QWORD *)(*(_QWORD *)(v28 + 16) + 8LL) >> 8) & 1LL;
    }
    if ( (_BYTE)v47 != 1 && v46 )
    {
      v48 = (_QWORD *)(*(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL);
      v49 = v48;
      if ( !v48 )
        BUG();
      v28 = *(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL;
      v50 = *v48;
      if ( (v50 & 4) == 0 && (*((_BYTE *)v49 + 46) & 4) != 0 )
      {
        while ( 1 )
        {
          v51 = v50 & 0xFFFFFFFFFFFFFFF8LL;
          v28 = v51;
          if ( (*(_BYTE *)(v51 + 46) & 4) == 0 )
            break;
          v50 = *(_QWORD *)v51;
        }
      }
      if ( v28 != v34 )
        continue;
    }
    break;
  }
  while ( v31 != v39 )
  {
LABEL_90:
    v52 = *(_WORD *)(v31 + 46);
    v53 = v52 & 4;
    if ( (v52 & 4) != 0 || (v52 & 8) == 0 )
    {
      v55 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v31 + 16) + 8LL) >> 7;
    }
    else
    {
      v89 = v34;
      v54 = sub_1E15D00(v31, 128, 1);
      v52 = *(_WORD *)(v31 + 46);
      v34 = v89;
      v55 = v54;
      v53 = v52 & 4;
    }
    if ( v53 || (v52 & 8) == 0 )
    {
      v57 = (*(_QWORD *)(*(_QWORD *)(v31 + 16) + 8LL) >> 5) & 1LL;
    }
    else
    {
      v83 = v34;
      v90 = v55;
      v56 = sub_1E15D00(v31, 32, 1);
      v52 = *(_WORD *)(v31 + 46);
      v55 = v90;
      v34 = v83;
      LOBYTE(v57) = v56;
      v53 = v52 & 4;
    }
    v58 = v55 & v57;
    if ( v53 || (v52 & 8) == 0 )
    {
      v59 = (*(_QWORD *)(*(_QWORD *)(v31 + 16) + 8LL) >> 8) & 1LL;
    }
    else
    {
      v84 = v34;
      v91 = v58;
      LOBYTE(v59) = sub_1E15D00(v31, 256, 1);
      v58 = v91;
      v34 = v84;
    }
    if ( (_BYTE)v59 == 1 || !v58 )
      break;
    v60 = (_QWORD *)(*(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL);
    v61 = v60;
    if ( !v60 )
      BUG();
    v31 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
    v62 = *v60;
    if ( (v62 & 4) == 0 && (*((_BYTE *)v61 + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        v63 = v62 & 0xFFFFFFFFFFFFFFF8LL;
        v31 = v63;
        if ( (*(_BYTE *)(v63 + 46) & 4) == 0 )
          break;
        v62 = *(_QWORD *)v63;
      }
    }
  }
LABEL_126:
  while ( v34 != v28 )
  {
    if ( v31 == v39 )
      goto LABEL_145;
    while ( v34 != v28 )
    {
      if ( (unsigned __int16)(**(_WORD **)(v28 + 16) - 12) > 1u )
        break;
      v73 = (_QWORD *)(*(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL);
      v74 = v73;
      if ( !v73 )
        BUG();
      v28 = *(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL;
      v75 = *v73;
      if ( (v75 & 4) == 0 && (*((_BYTE *)v74 + 46) & 4) != 0 )
      {
        while ( 1 )
        {
          v76 = v75 & 0xFFFFFFFFFFFFFFF8LL;
          v28 = v76;
          if ( (*(_BYTE *)(v76 + 46) & 4) == 0 )
            break;
          v75 = *(_QWORD *)v76;
        }
      }
    }
    while ( (unsigned __int16)(**(_WORD **)(v31 + 16) - 12) <= 1u )
    {
      v77 = (_QWORD *)(*(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL);
      v78 = v77;
      if ( !v77 )
        BUG();
      v31 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
      v79 = *v77;
      if ( (v79 & 4) == 0 && (*((_BYTE *)v78 + 46) & 4) != 0 )
      {
        while ( 1 )
        {
          v80 = v79 & 0xFFFFFFFFFFFFFFF8LL;
          v31 = v80;
          if ( (*(_BYTE *)(v80 + 46) & 4) == 0 )
            break;
          v79 = *(_QWORD *)v80;
        }
      }
      if ( v39 == v31 )
        goto LABEL_145;
    }
    if ( v28 == v34 || v39 == v31 || (v92 = v34, !(unsigned __int8)sub_1E15D60(v28, v31, 0)) )
    {
LABEL_145:
      if ( !v28 )
        BUG();
      break;
    }
    v64 = *(_WORD *)(v28 + 46);
    v34 = v92;
    if ( (v64 & 4) != 0 || (v64 & 8) == 0 )
    {
      v65 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(v28 + 16) + 8LL) >> 7;
    }
    else
    {
      v65 = sub_1E15D00(v28, 128, 1);
      v34 = v92;
    }
    if ( !v65 )
      ++*a7;
    v66 = (_QWORD *)(*(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL);
    v67 = v66;
    if ( !v66 )
      BUG();
    v28 = *(_QWORD *)v28 & 0xFFFFFFFFFFFFFFF8LL;
    v68 = *v66;
    if ( (v68 & 4) == 0 && (*((_BYTE *)v67 + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        v69 = v68 & 0xFFFFFFFFFFFFFFF8LL;
        v28 = v69;
        if ( (*(_BYTE *)(v69 + 46) & 4) == 0 )
          break;
        v68 = *(_QWORD *)v69;
      }
    }
    v70 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v70 )
      BUG();
    v71 = *(_QWORD *)v70;
    v31 = *(_QWORD *)v31 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_QWORD *)v70 & 4) == 0 && (*(_BYTE *)(v70 + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        v72 = v71 & 0xFFFFFFFFFFFFFFF8LL;
        v31 = v72;
        if ( (*(_BYTE *)(v72 + 46) & 4) == 0 )
          break;
        v71 = *(_QWORD *)v72;
      }
    }
  }
LABEL_146:
  if ( (*(_BYTE *)v28 & 4) == 0 && (*(_BYTE *)(v28 + 46) & 8) != 0 )
  {
    do
      v28 = *(_QWORD *)(v28 + 8);
    while ( (*(_BYTE *)(v28 + 46) & 8) != 0 );
  }
  *a4 = *(_QWORD *)(v28 + 8);
  if ( !v31 )
    BUG();
  if ( (*(_BYTE *)v31 & 4) == 0 && (*(_BYTE *)(v31 + 46) & 8) != 0 )
  {
    do
      v31 = *(_QWORD *)(v31 + 8);
    while ( (*(_BYTE *)(v31 + 46) & 8) != 0 );
  }
  *a5 = *(_QWORD *)(v31 + 8);
  return 1;
}
