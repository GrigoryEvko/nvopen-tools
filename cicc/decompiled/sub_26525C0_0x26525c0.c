// Function: sub_26525C0
// Address: 0x26525c0
//
__int64 __fastcall sub_26525C0(char *a1, char *a2, char *a3, char *a4, __int64 a5, __int64 a6)
{
  char *v8; // r12
  char *v9; // rbx
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
  __int64 v25; // rcx
  unsigned __int64 v26; // r13
  __int64 v27; // r14
  unsigned __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // r13
  __int64 v34; // r14
  unsigned __int64 v35; // r13
  __int64 v36; // r12
  unsigned __int64 v37; // rdi
  __int64 v38; // rsi
  __int64 v39; // rdi
  __int64 v40; // rdx
  __int64 v41; // rax
  _QWORD *v43; // rdx
  char *v44; // rax
  __int64 v45; // [rsp+8h] [rbp-48h]
  __int64 v47[7]; // [rsp+18h] [rbp-38h] BYREF

  v8 = a1;
  v9 = a3;
  v47[0] = a6;
  if ( a1 != a2 && a3 != a4 )
  {
    do
    {
      v15 = (char *)*((_QWORD *)v9 + 2);
      v16 = (char *)*((_QWORD *)v9 + 1);
      v17 = (const void *)*((_QWORD *)v8 + 1);
      v18 = v15 - v16;
      v19 = *((_QWORD *)v8 + 2) - (_QWORD)v17;
      if ( v19 < v15 - v16 )
        goto LABEL_10;
      if ( v18 == v19 )
      {
        v43 = (_QWORD *)*((_QWORD *)v8 + 1);
        if ( v15 != v16 )
        {
          v44 = (char *)*((_QWORD *)v9 + 1);
          while ( *(_QWORD *)v44 >= *v43 )
          {
            if ( *(_QWORD *)v44 > *v43 )
              goto LABEL_37;
            v44 += 8;
            ++v43;
            if ( v15 == v44 )
              goto LABEL_36;
          }
LABEL_10:
          v20 = *(_QWORD *)(a5 + 8);
          *(_QWORD *)a5 = *(_QWORD *)v9;
          *(_QWORD *)(a5 + 8) = *((_QWORD *)v9 + 1);
          *(_QWORD *)(a5 + 16) = *((_QWORD *)v9 + 2);
          *(_QWORD *)(a5 + 24) = *((_QWORD *)v9 + 3);
          *((_QWORD *)v9 + 1) = 0;
          *((_QWORD *)v9 + 2) = 0;
          *((_QWORD *)v9 + 3) = 0;
          if ( v20 )
            j_j___libc_free_0(v20);
          v21 = *(unsigned int *)(a5 + 64);
          v22 = *(_QWORD *)(a5 + 48);
          *(_QWORD *)(a5 + 32) = *((_QWORD *)v9 + 4);
          sub_C7D6A0(v22, 4 * v21, 4);
          *(_DWORD *)(a5 + 64) = 0;
          *(_QWORD *)(a5 + 48) = 0;
          *(_DWORD *)(a5 + 56) = 0;
          *(_DWORD *)(a5 + 60) = 0;
          ++*(_QWORD *)(a5 + 40);
          v23 = *((_QWORD *)v9 + 6);
          a5 += 72;
          ++*((_QWORD *)v9 + 5);
          v24 = *(_QWORD *)(a5 - 24);
          v9 += 72;
          *(_QWORD *)(a5 - 24) = v23;
          LODWORD(v23) = *((_DWORD *)v9 - 4);
          *((_QWORD *)v9 - 3) = v24;
          LODWORD(v24) = *(_DWORD *)(a5 - 16);
          *(_DWORD *)(a5 - 16) = v23;
          LODWORD(v23) = *((_DWORD *)v9 - 3);
          *((_DWORD *)v9 - 4) = v24;
          LODWORD(v24) = *(_DWORD *)(a5 - 12);
          *(_DWORD *)(a5 - 12) = v23;
          LODWORD(v23) = *((_DWORD *)v9 - 2);
          *((_DWORD *)v9 - 3) = v24;
          LODWORD(v24) = *(_DWORD *)(a5 - 8);
          *(_DWORD *)(a5 - 8) = v23;
          *((_DWORD *)v9 - 2) = v24;
          if ( v8 == a2 )
            break;
          continue;
        }
LABEL_36:
        if ( *((_QWORD **)v8 + 2) != v43 )
          goto LABEL_10;
LABEL_37:
        if ( (!v18 || !memcmp(v16, v17, *((_QWORD *)v9 + 2) - (_QWORD)v16))
          && sub_2650CE0(v47, (__int64)v9, (__int64)v8) )
        {
          goto LABEL_10;
        }
      }
      v10 = *(_QWORD *)(a5 + 8);
      *(_QWORD *)a5 = *(_QWORD *)v8;
      *(_QWORD *)(a5 + 8) = *((_QWORD *)v8 + 1);
      *(_QWORD *)(a5 + 16) = *((_QWORD *)v8 + 2);
      *(_QWORD *)(a5 + 24) = *((_QWORD *)v8 + 3);
      *((_QWORD *)v8 + 1) = 0;
      *((_QWORD *)v8 + 2) = 0;
      *((_QWORD *)v8 + 3) = 0;
      if ( v10 )
        j_j___libc_free_0(v10);
      v11 = *(unsigned int *)(a5 + 64);
      v12 = *(_QWORD *)(a5 + 48);
      *(_QWORD *)(a5 + 32) = *((_QWORD *)v8 + 4);
      sub_C7D6A0(v12, 4 * v11, 4);
      *(_DWORD *)(a5 + 64) = 0;
      *(_QWORD *)(a5 + 48) = 0;
      *(_DWORD *)(a5 + 56) = 0;
      *(_DWORD *)(a5 + 60) = 0;
      ++*(_QWORD *)(a5 + 40);
      v13 = *((_QWORD *)v8 + 6);
      a5 += 72;
      ++*((_QWORD *)v8 + 5);
      v14 = *(_QWORD *)(a5 - 24);
      v8 += 72;
      *(_QWORD *)(a5 - 24) = v13;
      LODWORD(v13) = *((_DWORD *)v8 - 4);
      *((_QWORD *)v8 - 3) = v14;
      LODWORD(v14) = *(_DWORD *)(a5 - 16);
      *(_DWORD *)(a5 - 16) = v13;
      LODWORD(v13) = *((_DWORD *)v8 - 3);
      *((_DWORD *)v8 - 4) = v14;
      LODWORD(v14) = *(_DWORD *)(a5 - 12);
      *(_DWORD *)(a5 - 12) = v13;
      LODWORD(v13) = *((_DWORD *)v8 - 2);
      *((_DWORD *)v8 - 3) = v14;
      LODWORD(v14) = *(_DWORD *)(a5 - 8);
      *(_DWORD *)(a5 - 8) = v13;
      *((_DWORD *)v8 - 2) = v14;
      if ( v8 == a2 )
        break;
    }
    while ( v9 != a4 );
  }
  v45 = a2 - v8;
  v25 = a2 - v8;
  v26 = 0x8E38E38E38E38E39LL * ((a2 - v8) >> 3);
  if ( v25 > 0 )
  {
    v27 = a5;
    do
    {
      v28 = *(_QWORD *)(v27 + 8);
      *(_QWORD *)v27 = *(_QWORD *)v8;
      *(_QWORD *)(v27 + 8) = *((_QWORD *)v8 + 1);
      *(_QWORD *)(v27 + 16) = *((_QWORD *)v8 + 2);
      *(_QWORD *)(v27 + 24) = *((_QWORD *)v8 + 3);
      *((_QWORD *)v8 + 1) = 0;
      *((_QWORD *)v8 + 2) = 0;
      *((_QWORD *)v8 + 3) = 0;
      if ( v28 )
        j_j___libc_free_0(v28);
      v29 = *(unsigned int *)(v27 + 64);
      v30 = *(_QWORD *)(v27 + 48);
      *(_QWORD *)(v27 + 32) = *((_QWORD *)v8 + 4);
      sub_C7D6A0(v30, 4 * v29, 4);
      *(_DWORD *)(v27 + 64) = 0;
      *(_QWORD *)(v27 + 48) = 0;
      *(_DWORD *)(v27 + 56) = 0;
      *(_DWORD *)(v27 + 60) = 0;
      ++*(_QWORD *)(v27 + 40);
      v31 = *((_QWORD *)v8 + 6);
      v27 += 72;
      ++*((_QWORD *)v8 + 5);
      v32 = *(_QWORD *)(v27 - 24);
      v8 += 72;
      *(_QWORD *)(v27 - 24) = v31;
      LODWORD(v31) = *((_DWORD *)v8 - 4);
      *((_QWORD *)v8 - 3) = v32;
      LODWORD(v32) = *(_DWORD *)(v27 - 16);
      *(_DWORD *)(v27 - 16) = v31;
      LODWORD(v31) = *((_DWORD *)v8 - 3);
      *((_DWORD *)v8 - 4) = v32;
      LODWORD(v32) = *(_DWORD *)(v27 - 12);
      *(_DWORD *)(v27 - 12) = v31;
      LODWORD(v31) = *((_DWORD *)v8 - 2);
      *((_DWORD *)v8 - 3) = v32;
      LODWORD(v32) = *(_DWORD *)(v27 - 8);
      *(_DWORD *)(v27 - 8) = v31;
      *((_DWORD *)v8 - 2) = v32;
      --v26;
    }
    while ( v26 );
    v33 = 72;
    if ( v45 > 0 )
      v33 = v45;
    a5 += v33;
  }
  v34 = a4 - v9;
  v35 = 0x8E38E38E38E38E39LL * ((a4 - v9) >> 3);
  if ( a4 - v9 > 0 )
  {
    v36 = a5;
    do
    {
      v37 = *(_QWORD *)(v36 + 8);
      *(_QWORD *)v36 = *(_QWORD *)v9;
      *(_QWORD *)(v36 + 8) = *((_QWORD *)v9 + 1);
      *(_QWORD *)(v36 + 16) = *((_QWORD *)v9 + 2);
      *(_QWORD *)(v36 + 24) = *((_QWORD *)v9 + 3);
      *((_QWORD *)v9 + 1) = 0;
      *((_QWORD *)v9 + 2) = 0;
      *((_QWORD *)v9 + 3) = 0;
      if ( v37 )
        j_j___libc_free_0(v37);
      v38 = *(unsigned int *)(v36 + 64);
      v39 = *(_QWORD *)(v36 + 48);
      *(_QWORD *)(v36 + 32) = *((_QWORD *)v9 + 4);
      sub_C7D6A0(v39, 4 * v38, 4);
      ++*(_QWORD *)(v36 + 40);
      v36 += 72;
      *(_DWORD *)(v36 - 8) = 0;
      *(_QWORD *)(v36 - 24) = 0;
      *(_DWORD *)(v36 - 16) = 0;
      *(_DWORD *)(v36 - 12) = 0;
      v40 = *((_QWORD *)v9 + 6);
      ++*((_QWORD *)v9 + 5);
      v41 = *(_QWORD *)(v36 - 24);
      v9 += 72;
      *(_QWORD *)(v36 - 24) = v40;
      LODWORD(v40) = *((_DWORD *)v9 - 4);
      *((_QWORD *)v9 - 3) = v41;
      LODWORD(v41) = *(_DWORD *)(v36 - 16);
      *(_DWORD *)(v36 - 16) = v40;
      LODWORD(v40) = *((_DWORD *)v9 - 3);
      *((_DWORD *)v9 - 4) = v41;
      LODWORD(v41) = *(_DWORD *)(v36 - 12);
      *(_DWORD *)(v36 - 12) = v40;
      LODWORD(v40) = *((_DWORD *)v9 - 2);
      *((_DWORD *)v9 - 3) = v41;
      LODWORD(v41) = *(_DWORD *)(v36 - 8);
      *(_DWORD *)(v36 - 8) = v40;
      *((_DWORD *)v9 - 2) = v41;
      --v35;
    }
    while ( v35 );
    if ( v34 <= 0 )
      v34 = 72;
    a5 += v34;
  }
  return a5;
}
