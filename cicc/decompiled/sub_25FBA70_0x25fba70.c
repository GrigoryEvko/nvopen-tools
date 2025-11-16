// Function: sub_25FBA70
// Address: 0x25fba70
//
__int64 __fastcall sub_25FBA70(__int64 *a1, __int64 *a2, __int64 a3)
{
  int v3; // ebx
  int v4; // r13d
  int v7; // edx
  __int64 v8; // rdi
  __int64 v9; // rax
  unsigned int v10; // ecx
  int *v11; // rsi
  int v12; // r8d
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdi
  unsigned int v16; // ecx
  int *v17; // rsi
  int v18; // r8d
  __int64 v19; // rax
  __int64 v20; // rcx
  unsigned int v21; // esi
  int *v22; // rdx
  int v23; // edi
  int v25; // edx
  int v26; // esi
  int v27; // r10d
  int v28; // esi
  int v29; // r10d
  int v30; // r9d

  v7 = sub_25FB8A0(*a1, a3);
  v8 = *(_QWORD *)(*a1 + 96);
  v9 = *(unsigned int *)(*a1 + 112);
  if ( (_DWORD)v9 )
  {
    v10 = (v9 - 1) & (37 * v7);
    v11 = (int *)(v8 + 8LL * v10);
    v12 = *v11;
    if ( v7 == *v11 )
    {
LABEL_3:
      if ( v11 != (int *)(v8 + 8 * v9) )
        v4 = v11[1];
    }
    else
    {
      v28 = 1;
      while ( v12 != -1 )
      {
        v29 = v28 + 1;
        v10 = (v9 - 1) & (v28 + v10);
        v11 = (int *)(v8 + 8LL * v10);
        v12 = *v11;
        if ( v7 == *v11 )
          goto LABEL_3;
        v28 = v29;
      }
    }
  }
  v13 = *a2;
  v14 = *(unsigned int *)(*a2 + 144);
  v15 = *(_QWORD *)(*a2 + 128);
  if ( (_DWORD)v14 )
  {
    v16 = (v14 - 1) & (37 * v4);
    v17 = (int *)(v15 + 8LL * v16);
    v18 = *v17;
    if ( v4 == *v17 )
    {
LABEL_7:
      if ( v17 != (int *)(v15 + 8 * v14) )
        v3 = v17[1];
    }
    else
    {
      v26 = 1;
      while ( v18 != -1 )
      {
        v27 = v26 + 1;
        v16 = (v14 - 1) & (v26 + v16);
        v17 = (int *)(v15 + 8LL * v16);
        v18 = *v17;
        if ( v4 == *v17 )
          goto LABEL_7;
        v26 = v27;
      }
    }
  }
  v19 = *(unsigned int *)(v13 + 80);
  v20 = *(_QWORD *)(v13 + 64);
  if ( (_DWORD)v19 )
  {
    v21 = (v19 - 1) & (37 * v3);
    v22 = (int *)(v20 + 16LL * v21);
    v23 = *v22;
    if ( *v22 == v3 )
    {
LABEL_11:
      if ( v22 != (int *)(v20 + 16 * v19) )
        return *((_QWORD *)v22 + 1);
    }
    else
    {
      v25 = 1;
      while ( v23 != -1 )
      {
        v30 = v25 + 1;
        v21 = (v19 - 1) & (v25 + v21);
        v22 = (int *)(v20 + 16LL * v21);
        v23 = *v22;
        if ( v3 == *v22 )
          goto LABEL_11;
        v25 = v30;
      }
    }
  }
  return 0;
}
