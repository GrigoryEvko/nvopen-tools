// Function: sub_2652050
// Address: 0x2652050
//
__int64 __fastcall sub_2652050(char *a1, char *a2, char *a3, char *a4, __int64 a5, __int64 a6)
{
  char *v9; // r12
  unsigned __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rax
  char *v15; // rcx
  char *v16; // rdi
  const void *v17; // rsi
  signed __int64 v18; // r9
  unsigned __int64 v19; // rax
  unsigned __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // r15
  unsigned __int64 v26; // r13
  __int64 v27; // r12
  unsigned __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rax
  _QWORD *v35; // rdx
  char *v36; // rax
  __int64 v37; // rcx
  unsigned __int64 v38; // r13
  __int64 v39; // r15
  unsigned __int64 v40; // rdi
  __int64 v41; // rsi
  __int64 v42; // rdi
  __int64 v43; // rcx
  __int64 v44; // rdx
  __int64 v45; // r13
  __int64 v46; // [rsp+8h] [rbp-48h]
  __int64 v48[7]; // [rsp+18h] [rbp-38h] BYREF

  v48[0] = a6;
  if ( a2 == a1 )
    goto LABEL_12;
  v9 = a1;
  while ( a4 != a3 )
  {
    v15 = (char *)*((_QWORD *)a3 + 2);
    v16 = (char *)*((_QWORD *)a3 + 1);
    v17 = (const void *)*((_QWORD *)v9 + 1);
    v18 = v15 - v16;
    v19 = *((_QWORD *)v9 + 2) - (_QWORD)v17;
    if ( v19 < v15 - v16 )
      goto LABEL_9;
    if ( v18 == v19 )
    {
      v35 = (_QWORD *)*((_QWORD *)v9 + 1);
      if ( v15 != v16 )
      {
        v36 = (char *)*((_QWORD *)a3 + 1);
        while ( *(_QWORD *)v36 >= *v35 )
        {
          if ( *(_QWORD *)v36 > *v35 )
            goto LABEL_27;
          v36 += 8;
          ++v35;
          if ( v15 == v36 )
            goto LABEL_26;
        }
        goto LABEL_9;
      }
LABEL_26:
      if ( *((_QWORD **)v9 + 2) == v35 )
      {
LABEL_27:
        if ( (!v18 || !memcmp(v16, v17, *((_QWORD *)a3 + 2) - (_QWORD)v16))
          && sub_2650CE0(v48, (__int64)a3, (__int64)v9) )
        {
          goto LABEL_9;
        }
        goto LABEL_4;
      }
LABEL_9:
      v20 = *(_QWORD *)(a5 + 8);
      *(_QWORD *)a5 = *(_QWORD *)a3;
      *(_QWORD *)(a5 + 8) = *((_QWORD *)a3 + 1);
      *(_QWORD *)(a5 + 16) = *((_QWORD *)a3 + 2);
      *(_QWORD *)(a5 + 24) = *((_QWORD *)a3 + 3);
      *((_QWORD *)a3 + 1) = 0;
      *((_QWORD *)a3 + 2) = 0;
      *((_QWORD *)a3 + 3) = 0;
      if ( v20 )
        j_j___libc_free_0(v20);
      v21 = *(unsigned int *)(a5 + 64);
      v22 = *(_QWORD *)(a5 + 48);
      *(_QWORD *)(a5 + 32) = *((_QWORD *)a3 + 4);
      sub_C7D6A0(v22, 4 * v21, 4);
      *(_DWORD *)(a5 + 64) = 0;
      *(_QWORD *)(a5 + 48) = 0;
      *(_DWORD *)(a5 + 56) = 0;
      *(_DWORD *)(a5 + 60) = 0;
      ++*(_QWORD *)(a5 + 40);
      v23 = *((_QWORD *)a3 + 6);
      a5 += 72;
      ++*((_QWORD *)a3 + 5);
      v24 = *(_QWORD *)(a5 - 24);
      a3 += 72;
      *(_QWORD *)(a5 - 24) = v23;
      LODWORD(v23) = *((_DWORD *)a3 - 4);
      *((_QWORD *)a3 - 3) = v24;
      LODWORD(v24) = *(_DWORD *)(a5 - 16);
      *(_DWORD *)(a5 - 16) = v23;
      LODWORD(v23) = *((_DWORD *)a3 - 3);
      *((_DWORD *)a3 - 4) = v24;
      LODWORD(v24) = *(_DWORD *)(a5 - 12);
      *(_DWORD *)(a5 - 12) = v23;
      LODWORD(v23) = *((_DWORD *)a3 - 2);
      *((_DWORD *)a3 - 3) = v24;
      LODWORD(v24) = *(_DWORD *)(a5 - 8);
      *(_DWORD *)(a5 - 8) = v23;
      *((_DWORD *)a3 - 2) = v24;
      if ( a2 == v9 )
        goto LABEL_12;
    }
    else
    {
LABEL_4:
      v10 = *(_QWORD *)(a5 + 8);
      *(_QWORD *)a5 = *(_QWORD *)v9;
      *(_QWORD *)(a5 + 8) = *((_QWORD *)v9 + 1);
      *(_QWORD *)(a5 + 16) = *((_QWORD *)v9 + 2);
      *(_QWORD *)(a5 + 24) = *((_QWORD *)v9 + 3);
      *((_QWORD *)v9 + 1) = 0;
      *((_QWORD *)v9 + 2) = 0;
      *((_QWORD *)v9 + 3) = 0;
      if ( v10 )
        j_j___libc_free_0(v10);
      v11 = *(unsigned int *)(a5 + 64);
      v12 = *(_QWORD *)(a5 + 48);
      *(_QWORD *)(a5 + 32) = *((_QWORD *)v9 + 4);
      sub_C7D6A0(v12, 4 * v11, 4);
      *(_DWORD *)(a5 + 64) = 0;
      *(_QWORD *)(a5 + 48) = 0;
      *(_DWORD *)(a5 + 56) = 0;
      *(_DWORD *)(a5 + 60) = 0;
      ++*(_QWORD *)(a5 + 40);
      v13 = *((_QWORD *)v9 + 6);
      a5 += 72;
      ++*((_QWORD *)v9 + 5);
      v14 = *(_QWORD *)(a5 - 24);
      v9 += 72;
      *(_QWORD *)(a5 - 24) = v13;
      LODWORD(v13) = *((_DWORD *)v9 - 4);
      *((_QWORD *)v9 - 3) = v14;
      LODWORD(v14) = *(_DWORD *)(a5 - 16);
      *(_DWORD *)(a5 - 16) = v13;
      LODWORD(v13) = *((_DWORD *)v9 - 3);
      *((_DWORD *)v9 - 4) = v14;
      LODWORD(v14) = *(_DWORD *)(a5 - 12);
      *(_DWORD *)(a5 - 12) = v13;
      LODWORD(v13) = *((_DWORD *)v9 - 2);
      *((_DWORD *)v9 - 3) = v14;
      LODWORD(v14) = *(_DWORD *)(a5 - 8);
      *(_DWORD *)(a5 - 8) = v13;
      *((_DWORD *)v9 - 2) = v14;
      if ( a2 == v9 )
        goto LABEL_12;
    }
  }
  v46 = a2 - v9;
  v37 = a2 - v9;
  v38 = 0x8E38E38E38E38E39LL * ((a2 - v9) >> 3);
  if ( v37 <= 0 )
    return a5;
  v39 = a5;
  do
  {
    v40 = *(_QWORD *)(v39 + 8);
    *(_QWORD *)v39 = *(_QWORD *)v9;
    *(_QWORD *)(v39 + 8) = *((_QWORD *)v9 + 1);
    *(_QWORD *)(v39 + 16) = *((_QWORD *)v9 + 2);
    *(_QWORD *)(v39 + 24) = *((_QWORD *)v9 + 3);
    *((_QWORD *)v9 + 1) = 0;
    *((_QWORD *)v9 + 2) = 0;
    *((_QWORD *)v9 + 3) = 0;
    if ( v40 )
      j_j___libc_free_0(v40);
    v41 = *(unsigned int *)(v39 + 64);
    v42 = *(_QWORD *)(v39 + 48);
    *(_QWORD *)(v39 + 32) = *((_QWORD *)v9 + 4);
    sub_C7D6A0(v42, 4 * v41, 4);
    *(_DWORD *)(v39 + 64) = 0;
    *(_QWORD *)(v39 + 48) = 0;
    *(_DWORD *)(v39 + 56) = 0;
    *(_DWORD *)(v39 + 60) = 0;
    ++*(_QWORD *)(v39 + 40);
    v43 = *((_QWORD *)v9 + 6);
    v39 += 72;
    ++*((_QWORD *)v9 + 5);
    v44 = *(_QWORD *)(v39 - 24);
    v9 += 72;
    *(_QWORD *)(v39 - 24) = v43;
    LODWORD(v43) = *((_DWORD *)v9 - 4);
    *((_QWORD *)v9 - 3) = v44;
    LODWORD(v44) = *(_DWORD *)(v39 - 16);
    *(_DWORD *)(v39 - 16) = v43;
    LODWORD(v43) = *((_DWORD *)v9 - 3);
    *((_DWORD *)v9 - 4) = v44;
    LODWORD(v44) = *(_DWORD *)(v39 - 12);
    *(_DWORD *)(v39 - 12) = v43;
    LODWORD(v43) = *((_DWORD *)v9 - 2);
    *((_DWORD *)v9 - 3) = v44;
    LODWORD(v44) = *(_DWORD *)(v39 - 8);
    *(_DWORD *)(v39 - 8) = v43;
    *((_DWORD *)v9 - 2) = v44;
    --v38;
  }
  while ( v38 );
  v45 = 72;
  if ( v46 > 0 )
    v45 = v46;
  a5 += v45;
LABEL_12:
  v25 = a4 - a3;
  v26 = 0x8E38E38E38E38E39LL * ((a4 - a3) >> 3);
  if ( a4 - a3 <= 0 )
    return a5;
  v27 = a5;
  do
  {
    v28 = *(_QWORD *)(v27 + 8);
    *(_QWORD *)v27 = *(_QWORD *)a3;
    *(_QWORD *)(v27 + 8) = *((_QWORD *)a3 + 1);
    *(_QWORD *)(v27 + 16) = *((_QWORD *)a3 + 2);
    *(_QWORD *)(v27 + 24) = *((_QWORD *)a3 + 3);
    *((_QWORD *)a3 + 1) = 0;
    *((_QWORD *)a3 + 2) = 0;
    *((_QWORD *)a3 + 3) = 0;
    if ( v28 )
      j_j___libc_free_0(v28);
    v29 = *(unsigned int *)(v27 + 64);
    v30 = *(_QWORD *)(v27 + 48);
    *(_QWORD *)(v27 + 32) = *((_QWORD *)a3 + 4);
    sub_C7D6A0(v30, 4 * v29, 4);
    ++*(_QWORD *)(v27 + 40);
    v27 += 72;
    *(_DWORD *)(v27 - 8) = 0;
    *(_QWORD *)(v27 - 24) = 0;
    *(_DWORD *)(v27 - 16) = 0;
    *(_DWORD *)(v27 - 12) = 0;
    v31 = *((_QWORD *)a3 + 6);
    ++*((_QWORD *)a3 + 5);
    v32 = *(_QWORD *)(v27 - 24);
    a3 += 72;
    *(_QWORD *)(v27 - 24) = v31;
    LODWORD(v31) = *((_DWORD *)a3 - 4);
    *((_QWORD *)a3 - 3) = v32;
    LODWORD(v32) = *(_DWORD *)(v27 - 16);
    *(_DWORD *)(v27 - 16) = v31;
    LODWORD(v31) = *((_DWORD *)a3 - 3);
    *((_DWORD *)a3 - 4) = v32;
    LODWORD(v32) = *(_DWORD *)(v27 - 12);
    *(_DWORD *)(v27 - 12) = v31;
    LODWORD(v31) = *((_DWORD *)a3 - 2);
    *((_DWORD *)a3 - 3) = v32;
    LODWORD(v32) = *(_DWORD *)(v27 - 8);
    *(_DWORD *)(v27 - 8) = v31;
    *((_DWORD *)a3 - 2) = v32;
    --v26;
  }
  while ( v26 );
  v33 = 72;
  if ( v25 > 0 )
    v33 = v25;
  return a5 + v33;
}
