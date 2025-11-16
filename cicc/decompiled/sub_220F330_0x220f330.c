// Function: sub_220F330
// Address: 0x220f330
//
int *__fastcall sub_220F330(int *a1, _QWORD *a2)
{
  int *v2; // r8
  int *v4; // rdi
  int *v5; // rax
  int *v6; // rdx
  int *v7; // rcx
  int *v8; // rax
  int *v9; // r8
  __int64 v10; // rdi
  int v11; // edi
  __int64 v12; // rdi
  _DWORD *v13; // r8
  __int64 v14; // rdx
  __int64 v15; // rdi
  _DWORD *v16; // r8
  __int64 v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v21; // rdx
  int *v22; // rdx
  int *v23; // r8
  __int64 v24; // rdi
  __int64 v25; // r8
  __int64 v26; // rdi
  __int64 v27; // r8
  int *v28; // rdx
  int *v29; // rdi
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // r8
  __int64 v34; // r8
  __int64 v35; // r8
  __int64 v36; // r8

  v2 = (int *)*((_QWORD *)a1 + 2);
  v4 = (int *)*((_QWORD *)a1 + 3);
  if ( !v2 )
    goto LABEL_75;
  if ( !v4 )
  {
    v8 = (int *)*((_QWORD *)a1 + 1);
    goto LABEL_43;
  }
  v5 = v4;
  do
  {
    v6 = v5;
    v5 = (int *)*((_QWORD *)v5 + 2);
  }
  while ( v5 );
  v7 = (int *)*((_QWORD *)v6 + 3);
  if ( v6 == a1 )
  {
    v4 = (int *)*((_QWORD *)v6 + 3);
LABEL_75:
    v8 = (int *)*((_QWORD *)a1 + 1);
    if ( !v4 )
    {
      v7 = 0;
      goto LABEL_44;
    }
    v2 = v4;
LABEL_43:
    *((_QWORD *)v2 + 1) = v8;
    v7 = v2;
LABEL_44:
    if ( (int *)a2[1] == a1 )
    {
      a2[1] = v7;
    }
    else
    {
      v21 = *((_QWORD *)a1 + 1);
      if ( *(int **)(v21 + 16) == a1 )
        *(_QWORD *)(v21 + 16) = v7;
      else
        *(_QWORD *)(v21 + 24) = v7;
    }
    if ( (int *)a2[2] == a1 )
    {
      v28 = v7;
      if ( *((_QWORD *)a1 + 3) )
      {
        do
        {
          v29 = v28;
          v28 = (int *)*((_QWORD *)v28 + 2);
        }
        while ( v28 );
        a2[2] = v29;
      }
      else
      {
        a2[2] = *((_QWORD *)a1 + 1);
      }
    }
    v11 = *a1;
    if ( (int *)a2[3] == a1 )
    {
      v22 = v7;
      if ( *((_QWORD *)a1 + 2) )
      {
        do
        {
          v23 = v22;
          v22 = (int *)*((_QWORD *)v22 + 3);
        }
        while ( v22 );
        a2[3] = v23;
      }
      else
      {
        a2[3] = *((_QWORD *)a1 + 1);
      }
    }
    goto LABEL_14;
  }
  *((_QWORD *)v2 + 1) = v6;
  v8 = v4;
  *((_QWORD *)v6 + 2) = v2;
  if ( v6 != v4 )
  {
    v8 = (int *)*((_QWORD *)v6 + 1);
    v9 = v8;
    if ( v7 )
    {
      *((_QWORD *)v7 + 1) = v8;
      v9 = (int *)*((_QWORD *)v6 + 1);
    }
    *((_QWORD *)v9 + 2) = v7;
    *((_QWORD *)v6 + 3) = v4;
    *(_QWORD *)(*((_QWORD *)a1 + 3) + 8LL) = v6;
  }
  if ( (int *)a2[1] == a1 )
  {
    a2[1] = v6;
    v10 = *((_QWORD *)a1 + 1);
  }
  else
  {
    v10 = *((_QWORD *)a1 + 1);
    if ( *(int **)(v10 + 16) == a1 )
      *(_QWORD *)(v10 + 16) = v6;
    else
      *(_QWORD *)(v10 + 24) = v6;
  }
  *((_QWORD *)v6 + 1) = v10;
  v11 = *v6;
  *v6 = *a1;
  *a1 = v11;
LABEL_14:
  if ( !v11 )
    return a1;
  if ( (int *)a2[1] == v7 )
  {
LABEL_40:
    if ( v7 )
      goto LABEL_66;
    return a1;
  }
  while ( 1 )
  {
    if ( v7 && *v7 != 1 )
      goto LABEL_66;
    v14 = *((_QWORD *)v8 + 2);
    if ( (int *)v14 == v7 )
    {
      v14 = *((_QWORD *)v8 + 3);
      if ( !*(_DWORD *)v14 )
      {
        v26 = *(_QWORD *)(v14 + 16);
        *(_DWORD *)v14 = 1;
        *v8 = 0;
        *((_QWORD *)v8 + 3) = v26;
        if ( v26 )
          *(_QWORD *)(v26 + 8) = v8;
        v27 = *((_QWORD *)v8 + 1);
        *(_QWORD *)(v14 + 8) = v27;
        if ( v8 == (int *)a2[1] )
        {
          a2[1] = v14;
          v26 = *((_QWORD *)v8 + 3);
        }
        else if ( v8 == *(int **)(v27 + 16) )
        {
          *(_QWORD *)(v27 + 16) = v14;
        }
        else
        {
          *(_QWORD *)(v27 + 24) = v14;
          v26 = *((_QWORD *)v8 + 3);
        }
        *(_QWORD *)(v14 + 16) = v8;
        *((_QWORD *)v8 + 1) = v14;
        v14 = v26;
      }
      v15 = *(_QWORD *)(v14 + 16);
      if ( !v15 || *(_DWORD *)v15 == 1 )
      {
        v16 = *(_DWORD **)(v14 + 24);
        if ( v16 && *v16 != 1 )
          goto LABEL_32;
        goto LABEL_22;
      }
      v16 = *(_DWORD **)(v14 + 24);
      if ( v16 && *v16 != 1 )
      {
LABEL_32:
        v17 = *((_QWORD *)v8 + 3);
        *(_DWORD *)v14 = *v8;
        *v8 = 1;
      }
      else
      {
        v35 = *(_QWORD *)(v15 + 24);
        *(_DWORD *)v15 = 1;
        *(_DWORD *)v14 = 0;
        *(_QWORD *)(v14 + 16) = v35;
        if ( v35 )
          *(_QWORD *)(v35 + 8) = v14;
        v36 = *(_QWORD *)(v14 + 8);
        *(_QWORD *)(v15 + 8) = v36;
        if ( v14 == a2[1] )
        {
          a2[1] = v15;
        }
        else if ( v14 == *(_QWORD *)(v36 + 24) )
        {
          *(_QWORD *)(v36 + 24) = v15;
        }
        else
        {
          *(_QWORD *)(v36 + 16) = v15;
        }
        *(_QWORD *)(v15 + 24) = v14;
        *(_QWORD *)(v14 + 8) = v15;
        v17 = *((_QWORD *)v8 + 3);
        v16 = *(_DWORD **)(v17 + 24);
        *(_DWORD *)v17 = *v8;
        *v8 = 1;
        if ( !v16 )
          goto LABEL_34;
      }
      *v16 = 1;
LABEL_34:
      v18 = *(_QWORD *)(v17 + 16);
      *((_QWORD *)v8 + 3) = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 8) = v8;
      v19 = *((_QWORD *)v8 + 1);
      *(_QWORD *)(v17 + 8) = v19;
      if ( v8 == (int *)a2[1] )
      {
        a2[1] = v17;
      }
      else if ( v8 == *(int **)(v19 + 16) )
      {
        *(_QWORD *)(v19 + 16) = v17;
      }
      else
      {
        *(_QWORD *)(v19 + 24) = v17;
      }
      *(_QWORD *)(v17 + 16) = v8;
      *((_QWORD *)v8 + 1) = v17;
      goto LABEL_40;
    }
    if ( !*(_DWORD *)v14 )
    {
      v24 = *(_QWORD *)(v14 + 24);
      *(_DWORD *)v14 = 1;
      *v8 = 0;
      *((_QWORD *)v8 + 2) = v24;
      if ( v24 )
        *(_QWORD *)(v24 + 8) = v8;
      v25 = *((_QWORD *)v8 + 1);
      *(_QWORD *)(v14 + 8) = v25;
      if ( v8 == (int *)a2[1] )
      {
        a2[1] = v14;
        v24 = *((_QWORD *)v8 + 2);
      }
      else if ( v8 == *(int **)(v25 + 24) )
      {
        *(_QWORD *)(v25 + 24) = v14;
      }
      else
      {
        *(_QWORD *)(v25 + 16) = v14;
        v24 = *((_QWORD *)v8 + 2);
      }
      *(_QWORD *)(v14 + 24) = v8;
      *((_QWORD *)v8 + 1) = v14;
      v14 = v24;
    }
    v12 = *(_QWORD *)(v14 + 24);
    if ( v12 )
    {
      if ( *(_DWORD *)v12 != 1 )
        break;
    }
    v13 = *(_DWORD **)(v14 + 16);
    if ( v13 && *v13 != 1 )
      goto LABEL_82;
LABEL_22:
    *(_DWORD *)v14 = 0;
    v7 = v8;
    if ( (int *)a2[1] == v8 )
      goto LABEL_67;
    v8 = (int *)*((_QWORD *)v8 + 1);
  }
  v13 = *(_DWORD **)(v14 + 16);
  if ( v13 && *v13 != 1 )
  {
LABEL_82:
    v30 = *((_QWORD *)v8 + 2);
    *(_DWORD *)v14 = *v8;
    *v8 = 1;
    goto LABEL_83;
  }
  v33 = *(_QWORD *)(v12 + 16);
  *(_DWORD *)v12 = 1;
  *(_DWORD *)v14 = 0;
  *(_QWORD *)(v14 + 24) = v33;
  if ( v33 )
    *(_QWORD *)(v33 + 8) = v14;
  v34 = *(_QWORD *)(v14 + 8);
  *(_QWORD *)(v12 + 8) = v34;
  if ( v14 == a2[1] )
  {
    a2[1] = v12;
  }
  else if ( v14 == *(_QWORD *)(v34 + 16) )
  {
    *(_QWORD *)(v34 + 16) = v12;
  }
  else
  {
    *(_QWORD *)(v34 + 24) = v12;
  }
  *(_QWORD *)(v12 + 16) = v14;
  *(_QWORD *)(v14 + 8) = v12;
  v30 = *((_QWORD *)v8 + 2);
  v13 = *(_DWORD **)(v30 + 16);
  *(_DWORD *)v30 = *v8;
  *v8 = 1;
  if ( v13 )
LABEL_83:
    *v13 = 1;
  v31 = *(_QWORD *)(v30 + 24);
  *((_QWORD *)v8 + 2) = v31;
  if ( v31 )
    *(_QWORD *)(v31 + 8) = v8;
  v32 = *((_QWORD *)v8 + 1);
  *(_QWORD *)(v30 + 8) = v32;
  if ( v8 == (int *)a2[1] )
  {
    a2[1] = v30;
  }
  else if ( v8 == *(int **)(v32 + 24) )
  {
    *(_QWORD *)(v32 + 24) = v30;
  }
  else
  {
    *(_QWORD *)(v32 + 16) = v30;
  }
  *(_QWORD *)(v30 + 24) = v8;
  *((_QWORD *)v8 + 1) = v30;
  if ( !v7 )
    return a1;
LABEL_66:
  v8 = v7;
LABEL_67:
  *v8 = 1;
  return a1;
}
