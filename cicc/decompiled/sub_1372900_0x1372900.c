// Function: sub_1372900
// Address: 0x1372900
//
char __fastcall sub_1372900(__int64 a1, __int64 *a2, _DWORD *a3, __int64 a4)
{
  __int64 v7; // rax
  _DWORD *v8; // rdi
  int *v9; // rax
  char v10; // si
  __int64 v11; // rax
  __int64 v12; // rcx
  int v13; // r8d
  unsigned int v14; // r9d
  int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // r12
  _QWORD *v18; // rax
  __int64 v19; // rax
  __int64 v20; // r13
  int **v21; // rdx
  __int64 v22; // rdx
  __int64 v23; // r13
  _QWORD *v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // eax
  int v29; // r10d

  if ( a4 )
  {
    v7 = *(unsigned int *)(a4 + 12);
    v8 = *(_DWORD **)(a4 + 96);
    if ( (unsigned int)v7 > 1 )
    {
      LOBYTE(v9) = sub_1369030(v8, &v8[v7], a3);
      if ( (_BYTE)v9 )
        return (char)v9;
      v10 = *(_BYTE *)(a1 + 56) & 1;
      if ( !v10 )
        goto LABEL_5;
LABEL_19:
      v12 = a1 + 64;
      v13 = 3;
      goto LABEL_7;
    }
    LODWORD(v9) = *v8;
    if ( *a3 == *v8 )
      return (char)v9;
  }
  v10 = *(_BYTE *)(a1 + 56) & 1;
  if ( v10 )
    goto LABEL_19;
LABEL_5:
  v11 = *(unsigned int *)(a1 + 72);
  v12 = *(_QWORD *)(a1 + 64);
  if ( !(_DWORD)v11 )
    goto LABEL_24;
  v13 = v11 - 1;
LABEL_7:
  v14 = v13 & (37 * *a3);
  v9 = (int *)(v12 + 16LL * v14);
  v15 = *v9;
  if ( *a3 == *v9 )
    goto LABEL_8;
  v28 = 1;
  while ( v15 != -1 )
  {
    v29 = v28 + 1;
    v14 = v13 & (v28 + v14);
    v9 = (int *)(v12 + 16LL * v14);
    v15 = *v9;
    if ( *a3 == *v9 )
      goto LABEL_8;
    v28 = v29;
  }
  if ( v10 )
  {
    v19 = 64;
    goto LABEL_25;
  }
  v11 = *(unsigned int *)(a1 + 72);
LABEL_24:
  v19 = 16 * v11;
LABEL_25:
  v9 = (int *)(v12 + v19);
LABEL_8:
  v16 = 64;
  if ( !v10 )
    v16 = 16LL * *(unsigned int *)(a1 + 72);
  if ( v9 != (int *)(v16 + v12) )
  {
    v17 = *((_QWORD *)v9 + 1);
    v18 = (_QWORD *)a2[7];
    if ( v18 == (_QWORD *)(a2[9] - 8) )
    {
      v23 = a2[10];
      if ( (((__int64)v18 - a2[8]) >> 3) + ((((v23 - a2[6]) >> 3) - 1) << 6) + ((a2[5] - a2[3]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
        goto LABEL_42;
      if ( (unsigned __int64)(a2[2] - ((v23 - a2[1]) >> 3)) <= 1 )
      {
        sub_1372780(a2 + 1, 1u, 0);
        v23 = a2[10];
      }
      *(_QWORD *)(v23 + 8) = sub_22077B0(512);
      v24 = (_QWORD *)a2[7];
      if ( v24 )
        *v24 = v17;
      v25 = (__int64 *)(a2[10] + 8);
      a2[10] = (__int64)v25;
      v26 = *v25;
      v27 = *v25 + 512;
      a2[8] = v26;
      a2[9] = v27;
      a2[7] = v26;
    }
    else
    {
      if ( v18 )
      {
        *v18 = v17;
        v18 = (_QWORD *)a2[7];
      }
      a2[7] = (__int64)(v18 + 1);
    }
    v9 = *(int **)(v17 + 24);
    if ( v9 != *(int **)(v17 + 32) )
    {
      *((_QWORD *)v9 - 1) = a2;
      *(_QWORD *)(v17 + 24) -= 8LL;
LABEL_17:
      ++*(_DWORD *)(v17 + 4);
      return (char)v9;
    }
    v20 = *(_QWORD *)(v17 + 48);
    if ( ((((*(_QWORD *)(v17 + 80) - v20) >> 3) - 1) << 6)
       + ((__int64)(*(_QWORD *)(v17 + 56) - *(_QWORD *)(v17 + 64)) >> 3)
       + ((__int64)(*(_QWORD *)(v17 + 40) - (_QWORD)v9) >> 3) != 0xFFFFFFFFFFFFFFFLL )
    {
      if ( v20 == *(_QWORD *)(v17 + 8) )
      {
        sub_1372780((__int64 *)(v17 + 8), 1u, 1);
        v20 = *(_QWORD *)(v17 + 48);
      }
      *(_QWORD *)(v20 - 8) = sub_22077B0(512);
      v21 = (int **)(*(_QWORD *)(v17 + 48) - 8LL);
      *(_QWORD *)(v17 + 48) = v21;
      v9 = *v21;
      v22 = (__int64)(*v21 + 128);
      *(_QWORD *)(v17 + 32) = v9;
      *(_QWORD *)(v17 + 40) = v22;
      *(_QWORD *)(v17 + 24) = v9 + 126;
      *((_QWORD *)v9 + 63) = a2;
      goto LABEL_17;
    }
LABEL_42:
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  }
  return (char)v9;
}
