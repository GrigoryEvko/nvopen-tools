// Function: sub_FEB360
// Address: 0xfeb360
//
char __fastcall sub_FEB360(__int64 a1, __int64 *a2, _DWORD *a3, __int64 a4)
{
  __int64 v7; // rax
  _DWORD *v8; // rdi
  int *v9; // rax
  char v10; // cl
  __int64 v11; // rax
  __int64 v12; // rsi
  int v13; // r8d
  unsigned int v14; // r9d
  int v15; // edx
  __int64 v16; // rdx
  __int64 v17; // r12
  _QWORD *v18; // rax
  __int64 v19; // r13
  _QWORD *v20; // rax
  __int64 *v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r13
  int **v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rax
  int v28; // eax
  int v29; // r10d

  if ( a4 )
  {
    v7 = *(unsigned int *)(a4 + 12);
    v8 = *(_DWORD **)(a4 + 96);
    if ( (unsigned int)v7 > 1 )
    {
      LOBYTE(v9) = sub_FDC990(v8, &v8[v7], a3);
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
    goto LABEL_34;
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
    v27 = 64;
    goto LABEL_35;
  }
  v11 = *(unsigned int *)(a1 + 72);
LABEL_34:
  v27 = 16 * v11;
LABEL_35:
  v9 = (int *)(v12 + v27);
LABEL_8:
  v16 = 64;
  if ( !v10 )
    v16 = 16LL * *(unsigned int *)(a1 + 72);
  if ( v9 != (int *)(v12 + v16) )
  {
    v17 = *((_QWORD *)v9 + 1);
    v18 = (_QWORD *)a2[7];
    if ( v18 == (_QWORD *)(a2[9] - 8) )
    {
      v19 = a2[10];
      if ( (((__int64)v18 - a2[8]) >> 3) + ((((v19 - a2[6]) >> 3) - 1) << 6) + ((a2[5] - a2[3]) >> 3) == 0xFFFFFFFFFFFFFFFLL )
        goto LABEL_42;
      if ( (unsigned __int64)(a2[2] - ((v19 - a2[1]) >> 3)) <= 1 )
      {
        sub_FEB1E0(a2 + 1, 1u, 0);
        v19 = a2[10];
      }
      *(_QWORD *)(v19 + 8) = sub_22077B0(512);
      v20 = (_QWORD *)a2[7];
      if ( v20 )
        *v20 = v17;
      v21 = (__int64 *)(a2[10] + 8);
      a2[10] = (__int64)v21;
      v22 = *v21;
      v23 = *v21 + 512;
      a2[8] = v22;
      a2[9] = v23;
      a2[7] = v22;
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
    v24 = *(_QWORD *)(v17 + 48);
    if ( ((((*(_QWORD *)(v17 + 80) - v24) >> 3) - 1) << 6)
       + ((__int64)(*(_QWORD *)(v17 + 56) - *(_QWORD *)(v17 + 64)) >> 3)
       + ((__int64)(*(_QWORD *)(v17 + 40) - (_QWORD)v9) >> 3) != 0xFFFFFFFFFFFFFFFLL )
    {
      if ( v24 == *(_QWORD *)(v17 + 8) )
      {
        sub_FEB1E0((__int64 *)(v17 + 8), 1u, 1);
        v24 = *(_QWORD *)(v17 + 48);
      }
      *(_QWORD *)(v24 - 8) = sub_22077B0(512);
      v25 = (int **)(*(_QWORD *)(v17 + 48) - 8LL);
      *(_QWORD *)(v17 + 48) = v25;
      v9 = *v25;
      v26 = (__int64)(*v25 + 128);
      *(_QWORD *)(v17 + 32) = v9;
      *(_QWORD *)(v17 + 40) = v26;
      *(_QWORD *)(v17 + 24) = v9 + 126;
      *((_QWORD *)v9 + 63) = a2;
      goto LABEL_17;
    }
LABEL_42:
    sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
  }
  return (char)v9;
}
