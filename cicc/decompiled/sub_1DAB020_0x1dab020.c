// Function: sub_1DAB020
// Address: 0x1dab020
//
void __fastcall sub_1DAB020(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // r12
  __int64 v9; // rcx
  __int64 v10; // rsi
  unsigned int v11; // r14d
  __int64 v12; // r8
  __int64 v13; // rsi
  int v14; // r9d
  _QWORD *v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // rsi
  __int64 v18; // rcx
  __int64 v19; // rcx
  unsigned __int64 *v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // rdx
  __int64 v25; // rcx
  _QWORD *v26; // rax
  int v27; // r8d
  unsigned int v28; // eax
  __int64 v29; // rsi
  __int64 v30; // rdx
  unsigned int v31; // eax
  _DWORD *v32; // rax
  _DWORD *v33; // rcx
  __int64 v34; // r13
  int v35; // edx
  int v36; // ecx
  unsigned int v37; // eax
  __int64 v38; // r14
  __int64 v39; // rcx
  __int64 v40; // rdx

  v8 = a1 + 8;
  v9 = *(_QWORD *)a1;
  v10 = *(_QWORD *)(a1 + 8);
  v11 = a2 - 1;
  if ( !v11 )
  {
    v27 = *(_DWORD *)(v9 + 84);
    v28 = *(_DWORD *)(v10 + 12) + 1;
    if ( v27 != v28 )
    {
      do
      {
        v29 = v28;
        v30 = v28++ - 1;
        *(_QWORD *)(v9 + 8 * v30 + 8) = *(_QWORD *)(v9 + 8 * v29 + 8);
        *(_QWORD *)(v9 + 8 * v30 + 40) = *(_QWORD *)(v9 + 8 * v29 + 40);
      }
      while ( v27 != v28 );
      v28 = *(_DWORD *)(v9 + 84);
    }
    v31 = v28 - 1;
    *(_DWORD *)(v9 + 84) = v31;
    *(_DWORD *)(*(_QWORD *)(a1 + 8) + 8LL) = v31;
    if ( v31 )
      goto LABEL_8;
    *(_DWORD *)(v9 + 80) = 0;
    v32 = (_DWORD *)(v9 + 64);
    v33 = (_DWORD *)(v9 + 80);
    *((_OWORD *)v33 - 5) = 0;
    *((_OWORD *)v33 - 4) = 0;
    *((_OWORD *)v33 - 3) = 0;
    *((_OWORD *)v33 - 2) = 0;
    *((_OWORD *)v33 - 1) = 0;
    do
      *v32++ = 0;
    while ( v33 != v32 );
    v34 = *(_QWORD *)a1;
    v35 = *(_DWORD *)(a1 + 20);
    v36 = *(_DWORD *)(*(_QWORD *)a1 + 80LL);
    v37 = *(_DWORD *)(*(_QWORD *)a1 + 84LL);
    *(_DWORD *)(a1 + 16) = 0;
    if ( v36 )
    {
      v34 += 8;
      v38 = v37;
      v39 = 0;
      if ( v35 )
      {
LABEL_23:
        v40 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(v40 + v39) = v34;
        *(_QWORD *)(v40 + v39 + 8) = v38;
        ++*(_DWORD *)(a1 + 16);
        return;
      }
    }
    else
    {
      v38 = v37;
      v39 = 0;
      if ( v35 )
        goto LABEL_23;
    }
    sub_16CD150(v8, (const void *)(a1 + 24), 0, 16, v27, a6);
    v39 = 16LL * *(unsigned int *)(a1 + 16);
    goto LABEL_23;
  }
  v12 = 16LL * v11;
  v13 = v12 + v10;
  v14 = *(_DWORD *)(v13 + 8);
  v15 = *(_QWORD **)v13;
  if ( v14 == 1 )
  {
    v26 = *(_QWORD **)(v9 + 88);
    *v15 = *v26;
    *v26 = v15;
    sub_1DAB020(a1, v11);
  }
  else
  {
    v16 = *(_DWORD *)(v13 + 12) + 1;
    if ( v14 != v16 )
    {
      do
      {
        v17 = v16;
        v18 = v16++ - 1;
        v15[v18] = v15[v17];
        v15[v18 + 12] = v15[v17 + 12];
      }
      while ( v14 != v16 );
      v13 = v12 + *(_QWORD *)(a1 + 8);
      v16 = *(_DWORD *)(v13 + 8);
    }
    *(_DWORD *)(v13 + 8) = v16 - 1;
    v19 = *(_QWORD *)(a1 + 8) + 16LL * (a2 - 2);
    v20 = (unsigned __int64 *)(*(_QWORD *)v19 + 8LL * *(unsigned int *)(v19 + 12));
    v21 = v16 - 2;
    *v20 = v21 | *v20 & 0xFFFFFFFFFFFFFFC0LL;
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 8) + v12 + 12) == v16 - 1 )
    {
      sub_1DA99F0(a1, v11, v15[v21 + 12]);
      sub_39460A0(v8, v11);
    }
  }
LABEL_8:
  if ( *(_DWORD *)(a1 + 16) )
  {
    v22 = *(_QWORD *)(a1 + 8);
    if ( *(_DWORD *)(v22 + 12) < *(_DWORD *)(v22 + 8) )
    {
      v23 = 16LL * a2;
      v24 = v22 + v23;
      v25 = *(_QWORD *)(*(_QWORD *)(v22 + 16LL * v11) + 8LL * *(unsigned int *)(v22 + 16LL * v11 + 12));
      *(_QWORD *)v24 = v25 & 0xFFFFFFFFFFFFFFC0LL;
      *(_DWORD *)(v24 + 8) = (v25 & 0x3F) + 1;
      *(_DWORD *)(*(_QWORD *)(a1 + 8) + v23 + 12) = 0;
    }
  }
}
