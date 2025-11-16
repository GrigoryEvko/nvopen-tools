// Function: sub_25F6560
// Address: 0x25f6560
//
__int64 __fastcall sub_25F6560(_QWORD *a1, unsigned int a2, __int64 a3, int a4, __int64 a5)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // r10
  unsigned int v9; // edi
  int *v10; // r9
  int v11; // r11d
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned int v14; // esi
  int *v15; // rdx
  int v16; // r9d
  __int64 v18; // r11
  __int64 v19; // rax
  unsigned int v20; // edx
  int *v21; // r9
  int v22; // r10d
  int v23; // r9d
  int v24; // r12d
  int v25; // edx
  int v26; // r11d
  int v27; // r9d
  int v28; // r12d

  v5 = a1[37];
  if ( *(_DWORD *)(v5 + 212) < a2 )
  {
    v18 = *(_QWORD *)(v5 + 224);
    v19 = *(unsigned int *)(v5 + 240);
    if ( (_DWORD)v19 )
    {
      v20 = (v19 - 1) & (37 * a2);
      v21 = (int *)(v18 + 40LL * v20);
      v22 = *v21;
      if ( *v21 == a2 )
      {
LABEL_13:
        a2 = **((_DWORD **)v21 + 2);
        goto LABEL_2;
      }
      v27 = 1;
      while ( v22 != -1 )
      {
        v28 = v27 + 1;
        v20 = (v19 - 1) & (v27 + v20);
        v21 = (int *)(v18 + 40LL * v20);
        v22 = *v21;
        if ( *v21 == a2 )
          goto LABEL_13;
        v27 = v28;
      }
    }
    v21 = (int *)(v18 + 40 * v19);
    goto LABEL_13;
  }
LABEL_2:
  v6 = *a1;
  v7 = *(unsigned int *)(*a1 + 144LL);
  v8 = *(_QWORD *)(*a1 + 128LL);
  if ( (_DWORD)v7 )
  {
    v9 = (v7 - 1) & (37 * a2);
    v10 = (int *)(v8 + 8LL * v9);
    v11 = *v10;
    if ( *v10 == a2 )
    {
LABEL_4:
      if ( v10 != (int *)(v8 + 8 * v7) )
        a4 = v10[1];
    }
    else
    {
      v23 = 1;
      while ( v11 != -1 )
      {
        v24 = v23 + 1;
        v9 = (v7 - 1) & (v23 + v9);
        v10 = (int *)(v8 + 8LL * v9);
        v11 = *v10;
        if ( *v10 == a2 )
          goto LABEL_4;
        v23 = v24;
      }
    }
  }
  v12 = *(unsigned int *)(v6 + 80);
  v13 = *(_QWORD *)(v6 + 64);
  if ( (_DWORD)v12 )
  {
    v14 = (v12 - 1) & (37 * a4);
    v15 = (int *)(v13 + 16LL * v14);
    v16 = *v15;
    if ( *v15 == a4 )
    {
LABEL_8:
      if ( v15 != (int *)(v13 + 16 * v12) )
        return *((_QWORD *)v15 + 1);
    }
    else
    {
      v25 = 1;
      while ( v16 != -1 )
      {
        v26 = v25 + 1;
        v14 = (v12 - 1) & (v25 + v14);
        v15 = (int *)(v13 + 16LL * v14);
        v16 = *v15;
        if ( *v15 == a4 )
          goto LABEL_8;
        v25 = v26;
      }
    }
  }
  return a5;
}
