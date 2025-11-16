// Function: sub_2BB5860
// Address: 0x2bb5860
//
__int64 __fastcall sub_2BB5860(__int64 a1, __int64 *a2)
{
  __int64 v4; // r14
  char v5; // cl
  __int64 v6; // rdi
  int v7; // esi
  unsigned int v8; // edx
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rax
  unsigned int v13; // esi
  unsigned int v14; // eax
  __int64 v15; // r13
  int v16; // edx
  unsigned int v17; // edi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int64 v22; // rdx
  char *v23; // r14
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // r8
  int v26; // esi
  __int64 v27; // rdx
  bool v28; // zf
  _QWORD *v29; // rax
  _QWORD *v30; // r12
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rax
  _BYTE *v36; // r12
  unsigned __int64 v37; // r14
  unsigned __int64 v38; // rdi
  __int64 v39; // r15
  unsigned __int64 v40; // r12
  unsigned __int64 v41; // rdi
  int v42; // r9d
  __int64 v43; // rdi
  char *v44; // r14
  __int64 v45; // rsi
  int v46; // edx
  unsigned int v47; // eax
  __int64 v48; // rcx
  __int64 v49; // rsi
  int v50; // edx
  unsigned int v51; // eax
  __int64 v52; // rcx
  int v53; // r8d
  __int64 v54; // rdi
  int v55; // edx
  int v56; // edx
  int v57; // r8d
  __int64 v58; // [rsp+0h] [rbp-270h]
  __int64 *v59; // [rsp+8h] [rbp-268h]
  __int64 v60[34]; // [rsp+10h] [rbp-260h] BYREF
  __int64 v61; // [rsp+120h] [rbp-150h] BYREF
  __int64 v62; // [rsp+128h] [rbp-148h] BYREF
  __int64 v63; // [rsp+130h] [rbp-140h]
  __int64 v64; // [rsp+138h] [rbp-138h]
  unsigned int v65; // [rsp+140h] [rbp-130h]
  __int64 v66; // [rsp+148h] [rbp-128h]
  _BYTE *v67; // [rsp+158h] [rbp-118h] BYREF
  __int64 v68; // [rsp+160h] [rbp-110h]
  _BYTE v69[264]; // [rsp+168h] [rbp-108h] BYREF

  v4 = *a2;
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( v5 )
  {
    v6 = a1 + 16;
    v7 = 7;
  }
  else
  {
    v13 = *(_DWORD *)(a1 + 24);
    v6 = *(_QWORD *)(a1 + 16);
    if ( !v13 )
    {
      v14 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v15 = 0;
      v16 = (v14 >> 1) + 1;
LABEL_9:
      v17 = 3 * v13;
      goto LABEL_10;
    }
    v7 = v13 - 1;
  }
  v8 = v7 & (((0xBF58476D1CE4E5B9LL * v4) >> 31) ^ (484763065 * v4));
  v9 = v6 + 16LL * v8;
  v10 = *(_QWORD *)v9;
  if ( v4 == *(_QWORD *)v9 )
  {
LABEL_4:
    v11 = *(unsigned int *)(v9 + 8);
    return *(_QWORD *)(a1 + 144) + 280 * v11 + 8;
  }
  v42 = 1;
  v15 = 0;
  while ( v10 != -1 )
  {
    if ( !v15 && v10 == -2 )
      v15 = v9;
    v8 = v7 & (v42 + v8);
    v9 = v6 + 16LL * v8;
    v10 = *(_QWORD *)v9;
    if ( v4 == *(_QWORD *)v9 )
      goto LABEL_4;
    ++v42;
  }
  if ( !v15 )
    v15 = v9;
  v14 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v16 = (v14 >> 1) + 1;
  if ( !v5 )
  {
    v13 = *(_DWORD *)(a1 + 24);
    goto LABEL_9;
  }
  v17 = 24;
  v13 = 8;
LABEL_10:
  if ( 4 * v16 >= v17 )
  {
    sub_2BB5410(a1, 2 * v13);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v45 = a1 + 16;
      v46 = 7;
    }
    else
    {
      v55 = *(_DWORD *)(a1 + 24);
      v45 = *(_QWORD *)(a1 + 16);
      if ( !v55 )
        goto LABEL_87;
      v46 = v55 - 1;
    }
    v47 = v46 & (((0xBF58476D1CE4E5B9LL * v4) >> 31) ^ (484763065 * v4));
    v15 = v45 + 16LL * v47;
    v48 = *(_QWORD *)v15;
    if ( *(_QWORD *)v15 != v4 )
    {
      v57 = 1;
      v54 = 0;
      while ( v48 != -1 )
      {
        if ( !v54 && v48 == -2 )
          v54 = v15;
        v47 = v46 & (v57 + v47);
        v15 = v45 + 16LL * v47;
        v48 = *(_QWORD *)v15;
        if ( v4 == *(_QWORD *)v15 )
          goto LABEL_56;
        ++v57;
      }
      goto LABEL_62;
    }
LABEL_56:
    v14 = *(_DWORD *)(a1 + 8);
    goto LABEL_12;
  }
  if ( v13 - *(_DWORD *)(a1 + 12) - v16 <= v13 >> 3 )
  {
    sub_2BB5410(a1, v13);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v49 = a1 + 16;
      v50 = 7;
      goto LABEL_59;
    }
    v56 = *(_DWORD *)(a1 + 24);
    v49 = *(_QWORD *)(a1 + 16);
    if ( v56 )
    {
      v50 = v56 - 1;
LABEL_59:
      v51 = v50 & (((0xBF58476D1CE4E5B9LL * v4) >> 31) ^ (484763065 * v4));
      v15 = v49 + 16LL * v51;
      v52 = *(_QWORD *)v15;
      if ( v4 != *(_QWORD *)v15 )
      {
        v53 = 1;
        v54 = 0;
        while ( v52 != -1 )
        {
          if ( !v54 && v52 == -2 )
            v54 = v15;
          v51 = v50 & (v53 + v51);
          v15 = v49 + 16LL * v51;
          v52 = *(_QWORD *)v15;
          if ( v4 == *(_QWORD *)v15 )
            goto LABEL_56;
          ++v53;
        }
LABEL_62:
        if ( v54 )
          v15 = v54;
        goto LABEL_56;
      }
      goto LABEL_56;
    }
LABEL_87:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_12:
  *(_DWORD *)(a1 + 8) = (2 * (v14 >> 1) + 2) | v14 & 1;
  if ( *(_QWORD *)v15 != -1 )
    --*(_DWORD *)(a1 + 12);
  *(_QWORD *)v15 = v4;
  *(_DWORD *)(v15 + 8) = 0;
  v18 = *a2;
  memset(v60, 0, sizeof(v60));
  LOBYTE(v60[1]) = 1;
  v61 = v18;
  v60[2] = -1;
  v60[4] = -1;
  v59 = &v60[8];
  v60[6] = (__int64)&v60[8];
  v60[7] = 0x200000000LL;
  v62 = 0;
  v63 = 1;
  v64 = -1;
  v66 = -1;
  sub_2B48B90((__int64)&v62, (__int64)v60);
  v67 = v69;
  v68 = 0x200000000LL;
  if ( LODWORD(v60[7]) )
    sub_2B49010((__int64)&v67, &v60[6], LODWORD(v60[7]), v19, v20, v21);
  v22 = *(unsigned int *)(a1 + 152);
  v23 = (char *)&v61;
  v24 = *(_QWORD *)(a1 + 144);
  v25 = v22 + 1;
  v26 = *(_DWORD *)(a1 + 152);
  if ( v22 + 1 > *(unsigned int *)(a1 + 156) )
  {
    v43 = a1 + 144;
    if ( v24 > (unsigned __int64)&v61 || (v22 = v24 + 280 * v22, (unsigned __int64)&v61 >= v22) )
    {
      sub_2B49530(v43, v25, v22, v19, v25, v21);
      v22 = *(unsigned int *)(a1 + 152);
      v24 = *(_QWORD *)(a1 + 144);
      v26 = *(_DWORD *)(a1 + 152);
    }
    else
    {
      v44 = (char *)&v61 - v24;
      sub_2B49530(v43, v25, v22, v19, v25, v21);
      v24 = *(_QWORD *)(a1 + 144);
      v22 = *(unsigned int *)(a1 + 152);
      v23 = &v44[v24];
      v26 = *(_DWORD *)(a1 + 152);
    }
  }
  v27 = 280 * v22;
  v28 = v27 + v24 == 0;
  v29 = (_QWORD *)(v27 + v24);
  v30 = v29;
  if ( !v28 )
  {
    v31 = (__int64)(v29 + 7);
    *(_QWORD *)(v31 - 56) = *(_QWORD *)v23;
    v30[1] = 0;
    v30[2] = 1;
    v30[3] = -1;
    v30[5] = -1;
    v58 = v31;
    sub_2B48B90((__int64)(v30 + 1), (__int64)(v23 + 8));
    v30[7] = v30 + 9;
    v30[8] = 0x200000000LL;
    if ( *((_DWORD *)v23 + 16) )
      sub_2B49010(v58, (__int64 *)v23 + 7, (__int64)(v30 + 9), v32, v33, v34);
    v26 = *(_DWORD *)(a1 + 152);
  }
  v35 = (unsigned int)v68;
  v36 = v67;
  *(_DWORD *)(a1 + 152) = v26 + 1;
  v37 = (unsigned __int64)&v36[104 * v35];
  if ( v36 != (_BYTE *)v37 )
  {
    do
    {
      v37 -= 104LL;
      v38 = *(_QWORD *)(v37 + 56);
      if ( v38 != v37 + 72 )
        _libc_free(v38);
      if ( (*(_BYTE *)(v37 + 16) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v37 + 24), 16LL * *(unsigned int *)(v37 + 32), 8);
    }
    while ( v36 != (_BYTE *)v37 );
    v37 = (unsigned __int64)v67;
  }
  if ( (_BYTE *)v37 != v69 )
    _libc_free(v37);
  if ( (v63 & 1) == 0 )
    sub_C7D6A0(v64, 16LL * v65, 8);
  v39 = v60[6];
  v40 = v60[6] + 104LL * LODWORD(v60[7]);
  if ( v60[6] != v40 )
  {
    do
    {
      v40 -= 104LL;
      v41 = *(_QWORD *)(v40 + 56);
      if ( v41 != v40 + 72 )
        _libc_free(v41);
      if ( (*(_BYTE *)(v40 + 16) & 1) == 0 )
        sub_C7D6A0(*(_QWORD *)(v40 + 24), 16LL * *(unsigned int *)(v40 + 32), 8);
    }
    while ( v39 != v40 );
    v40 = v60[6];
  }
  if ( (__int64 *)v40 != v59 )
    _libc_free(v40);
  if ( (v60[1] & 1) == 0 )
    sub_C7D6A0(v60[2], 16LL * LODWORD(v60[3]), 8);
  v11 = (unsigned int)(*(_DWORD *)(a1 + 152) - 1);
  *(_DWORD *)(v15 + 8) = v11;
  return *(_QWORD *)(a1 + 144) + 280 * v11 + 8;
}
