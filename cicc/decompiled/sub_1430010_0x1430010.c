// Function: sub_1430010
// Address: 0x1430010
//
void __fastcall sub_1430010(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 *v8; // r14
  __int64 *v9; // r15
  __int64 v10; // r14
  unsigned int v11; // esi
  __int64 v12; // rdi
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // rax
  char v16; // r15
  char v17; // r14
  unsigned __int8 v18; // r15
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // r14
  __int64 v22; // r15
  __int64 v23; // rdi
  __int64 *v24; // r14
  __int64 *v25; // r15
  __int64 v26; // rdi
  __int64 v27; // rdi
  __int64 v28; // rax
  char v29; // cl
  int v30; // r11d
  __int64 *v31; // r10
  int v32; // eax
  int v33; // eax
  int v34; // edx
  int v35; // edx
  __int64 v36; // rdi
  unsigned int v37; // ecx
  __int64 v38; // rsi
  int v39; // r11d
  __int64 *v40; // r9
  int v41; // edx
  int v42; // edx
  __int64 v43; // rsi
  int v44; // r9d
  __int64 *v45; // r8
  unsigned int v46; // r15d
  __int64 v47; // rcx
  char v48; // [rsp-1A5h] [rbp-1A5h]
  char v49; // [rsp-1A4h] [rbp-1A4h]
  int v50; // [rsp-1A4h] [rbp-1A4h]
  __int64 v51[2]; // [rsp-198h] [rbp-198h] BYREF
  __int64 v52; // [rsp-188h] [rbp-188h]
  _QWORD v53[2]; // [rsp-178h] [rbp-178h] BYREF
  __int64 v54; // [rsp-168h] [rbp-168h]
  _QWORD v55[2]; // [rsp-158h] [rbp-158h] BYREF
  __int64 v56; // [rsp-148h] [rbp-148h]
  __int64 v57[2]; // [rsp-138h] [rbp-138h] BYREF
  __int64 v58; // [rsp-128h] [rbp-128h]
  __int64 v59[2]; // [rsp-118h] [rbp-118h] BYREF
  __int64 v60; // [rsp-108h] [rbp-108h]
  __int64 *v61; // [rsp-F8h] [rbp-F8h] BYREF
  __int64 *v62; // [rsp-F0h] [rbp-F0h]
  __int64 v63; // [rsp-E8h] [rbp-E8h] BYREF
  __int64 v64; // [rsp-D8h] [rbp-D8h] BYREF
  __int64 v65; // [rsp-D0h] [rbp-D0h]
  __int64 v66; // [rsp-C8h] [rbp-C8h]

  if ( (a4 & 6) != 0 )
    return;
  **(_BYTE **)a1 = 1;
  v5 = sub_1632000(*(_QWORD *)(a1 + 8));
  v6 = v5;
  if ( !v5 )
    return;
  v7 = *(_QWORD *)(a1 + 16);
  v49 = (*(_BYTE *)(v5 + 33) & 0x40) != 0;
  sub_15E4EB0(&v61);
  v8 = v62;
  v9 = v61;
  sub_16C1840(&v64);
  sub_16C1A90(&v64, v9, v8);
  sub_16C1AA0(&v64, v59);
  v10 = v59[0];
  if ( v61 != &v63 )
    j_j___libc_free_0(v61, v63 + 1);
  v11 = *(_DWORD *)(v7 + 24);
  if ( !v11 )
  {
    ++*(_QWORD *)v7;
    goto LABEL_51;
  }
  v12 = *(_QWORD *)(v7 + 8);
  v13 = (v11 - 1) & (37 * v10);
  v14 = (__int64 *)(v12 + 8LL * v13);
  v15 = *v14;
  if ( v10 != *v14 )
  {
    v30 = 1;
    v31 = 0;
    while ( v15 != -1 )
    {
      if ( v15 != -2 || v31 )
        v14 = v31;
      v13 = (v11 - 1) & (v30 + v13);
      v15 = *(_QWORD *)(v12 + 8LL * v13);
      if ( v10 == v15 )
        goto LABEL_7;
      ++v30;
      v31 = v14;
      v14 = (__int64 *)(v12 + 8LL * v13);
    }
    v32 = *(_DWORD *)(v7 + 16);
    if ( !v31 )
      v31 = v14;
    ++*(_QWORD *)v7;
    v33 = v32 + 1;
    if ( 4 * v33 < 3 * v11 )
    {
      if ( v11 - *(_DWORD *)(v7 + 20) - v33 > v11 >> 3 )
      {
LABEL_47:
        *(_DWORD *)(v7 + 16) = v33;
        if ( *v31 != -1 )
          --*(_DWORD *)(v7 + 20);
        *v31 = v10;
        goto LABEL_7;
      }
      sub_142F750(v7, v11);
      v41 = *(_DWORD *)(v7 + 24);
      if ( v41 )
      {
        v42 = v41 - 1;
        v43 = *(_QWORD *)(v7 + 8);
        v44 = 1;
        v45 = 0;
        v46 = v42 & (37 * v10);
        v31 = (__int64 *)(v43 + 8LL * v46);
        v47 = *v31;
        v33 = *(_DWORD *)(v7 + 16) + 1;
        if ( *v31 != v10 )
        {
          while ( v47 != -1 )
          {
            if ( v47 == -2 && !v45 )
              v45 = v31;
            v46 = v42 & (v44 + v46);
            v31 = (__int64 *)(v43 + 8LL * v46);
            v47 = *v31;
            if ( v10 == *v31 )
              goto LABEL_47;
            ++v44;
          }
          if ( v45 )
            v31 = v45;
        }
        goto LABEL_47;
      }
LABEL_79:
      ++*(_DWORD *)(v7 + 16);
      BUG();
    }
LABEL_51:
    sub_142F750(v7, 2 * v11);
    v34 = *(_DWORD *)(v7 + 24);
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *(_QWORD *)(v7 + 8);
      v37 = v35 & (37 * v10);
      v31 = (__int64 *)(v36 + 8LL * v37);
      v38 = *v31;
      v33 = *(_DWORD *)(v7 + 16) + 1;
      if ( *v31 != v10 )
      {
        v39 = 1;
        v40 = 0;
        while ( v38 != -1 )
        {
          if ( !v40 && v38 == -2 )
            v40 = v31;
          v37 = v35 & (v39 + v37);
          v31 = (__int64 *)(v36 + 8LL * v37);
          v38 = *v31;
          if ( v10 == *v31 )
            goto LABEL_47;
          ++v39;
        }
        if ( v40 )
          v31 = v40;
      }
      goto LABEL_47;
    }
    goto LABEL_79;
  }
LABEL_7:
  if ( *(_BYTE *)(v6 + 16) )
  {
    v28 = sub_22077B0(64);
    if ( v28 )
    {
      v29 = *(_BYTE *)(v28 + 12);
      *(_DWORD *)(v28 + 8) = 2;
      *(_QWORD *)(v28 + 16) = 0;
      *(_QWORD *)(v28 + 24) = 0;
      *(_QWORD *)(v28 + 32) = 0;
      *(_QWORD *)(v28 + 40) = 0;
      *(_BYTE *)(v28 + 12) = v29 & 0x80 | (v49 << 6) | 0x37;
      *(_QWORD *)(v28 + 48) = 0;
      *(_QWORD *)(v28 + 56) = 0;
      *(_QWORD *)v28 = &unk_49EB4D8;
    }
    v64 = v28;
    v27 = *(_QWORD *)(a1 + 24);
  }
  else
  {
    v16 = sub_1560180(v6 + 112, 36);
    v48 = sub_1560180(v6 + 112, 37);
    v17 = sub_1560180(v6 + 112, 27);
    v51[0] = 0;
    v51[1] = 0;
    v52 = 0;
    v53[0] = 0;
    v53[1] = 0;
    v50 = (unsigned __int8)(v49 << 6) | 0x37;
    v54 = 0;
    v18 = ((8 * sub_1560260(v6 + 112, 0, 20)) | (4 * v17) | (2 * v48) | v16) & 0xF;
    v55[0] = 0;
    v55[1] = 0;
    v56 = 0;
    v57[0] = 0;
    v57[1] = 0;
    v58 = 0;
    v59[0] = 0;
    v59[1] = 0;
    v60 = 0;
    v61 = 0;
    v62 = 0;
    v63 = 0;
    v64 = 0;
    v65 = 0;
    v66 = 0;
    v19 = sub_22077B0(104);
    v20 = v19;
    if ( v19 )
      sub_142CF20(v19, v50, 0, v18, v51, v53, v55, v57, v59, (__int64 *)&v61, &v64);
    v21 = v65;
    v22 = v64;
    if ( v65 != v64 )
    {
      do
      {
        v23 = *(_QWORD *)(v22 + 16);
        if ( v23 )
          j_j___libc_free_0(v23, *(_QWORD *)(v22 + 32) - v23);
        v22 += 40;
      }
      while ( v21 != v22 );
      v22 = v64;
    }
    if ( v22 )
      j_j___libc_free_0(v22, v66 - v22);
    v24 = v62;
    v25 = v61;
    if ( v62 != v61 )
    {
      do
      {
        v26 = v25[2];
        if ( v26 )
          j_j___libc_free_0(v26, v25[4] - v26);
        v25 += 5;
      }
      while ( v24 != v25 );
      v25 = v61;
    }
    if ( v25 )
      j_j___libc_free_0(v25, v63 - (_QWORD)v25);
    if ( v59[0] )
      j_j___libc_free_0(v59[0], v60 - v59[0]);
    if ( v57[0] )
      j_j___libc_free_0(v57[0], v58 - v57[0]);
    if ( v55[0] )
      j_j___libc_free_0(v55[0], v56 - v55[0]);
    if ( v53[0] )
      j_j___libc_free_0(v53[0], v54 - v53[0]);
    if ( v51[0] )
      j_j___libc_free_0(v51[0], v52 - v51[0]);
    v64 = v20;
    v27 = *(_QWORD *)(a1 + 24);
  }
  sub_142ED30(v27, v6, &v64);
  if ( v64 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v64 + 8LL))(v64);
}
