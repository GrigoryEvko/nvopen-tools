// Function: sub_2BB6310
// Address: 0x2bb6310
//
__int64 __fastcall sub_2BB6310(__int64 a1, __int64 *a2)
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
  __int64 v15; // r12
  int v16; // edx
  unsigned int v17; // edi
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // r9
  __int64 v21; // rdx
  char *v22; // r15
  unsigned __int64 v23; // rsi
  __int64 v24; // r9
  int v25; // eax
  _QWORD *v26; // r13
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r9
  _BYTE *v31; // rdi
  unsigned __int64 v32; // rdi
  int v33; // r9d
  __int64 v34; // rdi
  char *v35; // r15
  __int64 v36; // rsi
  int v37; // edx
  unsigned int v38; // eax
  __int64 v39; // rcx
  __int64 v40; // rsi
  int v41; // edx
  unsigned int v42; // eax
  __int64 v43; // rcx
  int v44; // r8d
  __int64 v45; // rdi
  int v46; // edx
  int v47; // edx
  int v48; // r8d
  __int64 v49; // [rsp+10h] [rbp-100h] BYREF
  __int64 v50; // [rsp+18h] [rbp-F8h]
  __int64 v51; // [rsp+20h] [rbp-F0h]
  unsigned int v52; // [rsp+28h] [rbp-E8h]
  __int64 v53; // [rsp+30h] [rbp-E0h]
  _BYTE *v54; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v55; // [rsp+48h] [rbp-C8h]
  _BYTE v56[32]; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v57; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v58; // [rsp+78h] [rbp-98h] BYREF
  __int64 v59; // [rsp+80h] [rbp-90h]
  __int64 v60; // [rsp+88h] [rbp-88h]
  unsigned int v61; // [rsp+90h] [rbp-80h]
  __int64 v62; // [rsp+98h] [rbp-78h]
  _QWORD v63[2]; // [rsp+A8h] [rbp-68h] BYREF
  _BYTE v64[88]; // [rsp+B8h] [rbp-58h] BYREF

  v4 = *a2;
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( v5 )
  {
    v6 = a1 + 16;
    v7 = 1;
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
    return *(_QWORD *)(a1 + 48) + 104 * v11 + 8;
  }
  v33 = 1;
  v15 = 0;
  while ( v10 != -1 )
  {
    if ( !v15 && v10 == -2 )
      v15 = v9;
    v8 = v7 & (v33 + v8);
    v9 = v6 + 16LL * v8;
    v10 = *(_QWORD *)v9;
    if ( v4 == *(_QWORD *)v9 )
      goto LABEL_4;
    ++v33;
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
  v17 = 6;
  v13 = 2;
LABEL_10:
  if ( 4 * v16 < v17 )
  {
    if ( v13 - *(_DWORD *)(a1 + 12) - v16 > v13 >> 3 )
      goto LABEL_12;
    sub_2BB5EE0(a1, v13);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v40 = a1 + 16;
      v41 = 1;
      goto LABEL_46;
    }
    v47 = *(_DWORD *)(a1 + 24);
    v40 = *(_QWORD *)(a1 + 16);
    if ( v47 )
    {
      v41 = v47 - 1;
LABEL_46:
      v42 = v41 & (((0xBF58476D1CE4E5B9LL * v4) >> 31) ^ (484763065 * v4));
      v15 = v40 + 16LL * v42;
      v43 = *(_QWORD *)v15;
      if ( v4 != *(_QWORD *)v15 )
      {
        v44 = 1;
        v45 = 0;
        while ( v43 != -1 )
        {
          if ( !v45 && v43 == -2 )
            v45 = v15;
          v42 = v41 & (v44 + v42);
          v15 = v40 + 16LL * v42;
          v43 = *(_QWORD *)v15;
          if ( v4 == *(_QWORD *)v15 )
            goto LABEL_43;
          ++v44;
        }
LABEL_49:
        if ( v45 )
          v15 = v45;
        goto LABEL_43;
      }
      goto LABEL_43;
    }
LABEL_74:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
  sub_2BB5EE0(a1, 2 * v13);
  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v36 = a1 + 16;
    v37 = 1;
  }
  else
  {
    v46 = *(_DWORD *)(a1 + 24);
    v36 = *(_QWORD *)(a1 + 16);
    if ( !v46 )
      goto LABEL_74;
    v37 = v46 - 1;
  }
  v38 = v37 & (((0xBF58476D1CE4E5B9LL * v4) >> 31) ^ (484763065 * v4));
  v15 = v36 + 16LL * v38;
  v39 = *(_QWORD *)v15;
  if ( v4 != *(_QWORD *)v15 )
  {
    v48 = 1;
    v45 = 0;
    while ( v39 != -1 )
    {
      if ( !v45 && v39 == -2 )
        v45 = v15;
      v38 = v37 & (v48 + v38);
      v15 = v36 + 16LL * v38;
      v39 = *(_QWORD *)v15;
      if ( v4 == *(_QWORD *)v15 )
        goto LABEL_43;
      ++v48;
    }
    goto LABEL_49;
  }
LABEL_43:
  v14 = *(_DWORD *)(a1 + 8);
LABEL_12:
  *(_DWORD *)(a1 + 8) = (2 * (v14 >> 1) + 2) | v14 & 1;
  if ( *(_QWORD *)v15 != -1 )
    --*(_DWORD *)(a1 + 12);
  *(_QWORD *)v15 = v4;
  *(_DWORD *)(v15 + 8) = 0;
  v18 = *a2;
  v49 = 0;
  v50 = 1;
  v51 = -4096;
  v53 = -4096;
  v54 = v56;
  v55 = 0x200000000LL;
  v57 = v18;
  v58 = 0;
  v59 = 1;
  v60 = -4096;
  v62 = -4096;
  sub_2B48D10((__int64)&v58, (__int64)&v49);
  v63[1] = 0x200000000LL;
  v63[0] = v64;
  if ( (_DWORD)v55 )
    sub_2B0BCF0((__int64)v63, (__int64)&v54, (unsigned int)v55, v19, (__int64)v64, v20);
  v21 = *(unsigned int *)(a1 + 56);
  v22 = (char *)&v57;
  v23 = *(_QWORD *)(a1 + 48);
  v24 = v21 + 1;
  v25 = *(_DWORD *)(a1 + 56);
  if ( v21 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
  {
    v34 = a1 + 48;
    if ( v23 > (unsigned __int64)&v57 || (unsigned __int64)&v57 >= v23 + 104 * v21 )
    {
      sub_2B48E80(v34, v21 + 1, v21, v19, (__int64)v64, v24);
      v21 = *(unsigned int *)(a1 + 56);
      v23 = *(_QWORD *)(a1 + 48);
      v25 = *(_DWORD *)(a1 + 56);
    }
    else
    {
      v35 = (char *)&v57 - v23;
      sub_2B48E80(v34, v21 + 1, v21, v19, (__int64)v64, v24);
      v23 = *(_QWORD *)(a1 + 48);
      v21 = *(unsigned int *)(a1 + 56);
      v22 = &v35[v23];
      v25 = *(_DWORD *)(a1 + 56);
    }
  }
  v26 = (_QWORD *)(v23 + 104 * v21);
  if ( v26 )
  {
    v27 = *(_QWORD *)v22;
    v26[1] = 0;
    v26[2] = 1;
    *v26 = v27;
    v26[3] = -4096;
    v26[5] = -4096;
    sub_2B48D10((__int64)(v26 + 1), (__int64)(v22 + 8));
    v26[7] = v26 + 9;
    v26[8] = 0x200000000LL;
    if ( *((_DWORD *)v22 + 16) )
      sub_2B0BCF0((__int64)(v26 + 7), (__int64)(v22 + 56), v28, v29, (__int64)v64, v30);
    v25 = *(_DWORD *)(a1 + 56);
  }
  v31 = (_BYTE *)v63[0];
  *(_DWORD *)(a1 + 56) = v25 + 1;
  if ( v31 != v64 )
    _libc_free((unsigned __int64)v31);
  if ( (v59 & 1) == 0 )
  {
    sub_C7D6A0(v60, 16LL * v61, 8);
    v32 = (unsigned __int64)v54;
    if ( v54 == v56 )
      goto LABEL_26;
    goto LABEL_25;
  }
  v32 = (unsigned __int64)v54;
  if ( v54 != v56 )
LABEL_25:
    _libc_free(v32);
LABEL_26:
  if ( (v50 & 1) == 0 )
    sub_C7D6A0(v51, 16LL * v52, 8);
  v11 = (unsigned int)(*(_DWORD *)(a1 + 56) - 1);
  *(_DWORD *)(v15 + 8) = v11;
  return *(_QWORD *)(a1 + 48) + 104 * v11 + 8;
}
