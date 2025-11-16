// Function: sub_25350C0
// Address: 0x25350c0
//
__int64 __fastcall sub_25350C0(int **a1, unsigned __int8 *a2)
{
  __int64 v2; // r8
  int v3; // edx
  unsigned int v5; // eax
  int v6; // r9d
  __int64 v7; // r10
  int v8; // edx
  int *v9; // rcx
  int v10; // eax
  unsigned int v11; // r10d
  __int64 v12; // rdx
  int v13; // esi
  __int64 v14; // rcx
  int *v15; // rcx
  int v16; // edx
  int v17; // eax
  __int64 v18; // r10
  unsigned int v19; // ecx
  __int64 v20; // rcx
  int v21; // eax

  LODWORD(v2) = 1;
  v3 = *a2;
  if ( (unsigned int)(v3 - 12) <= 1 )
    return (unsigned int)v2;
  v2 = *((_QWORD *)a2 + 1);
  v5 = *(unsigned __int8 *)(v2 + 8) - 17;
  if ( (_BYTE)v3 != 22 )
  {
    v9 = a1[1];
LABEL_18:
    if ( v5 <= 1 )
    {
      v7 = *(_QWORD *)(v2 + 16);
LABEL_20:
      v8 = *(_DWORD *)(*(_QWORD *)v7 + 8LL) >> 8;
      goto LABEL_7;
    }
    v11 = *(_DWORD *)(v2 + 8);
LABEL_22:
    v8 = v11 >> 8;
LABEL_7:
    v10 = v9[25];
    LOBYTE(v2) = v10 == v8;
    if ( v10 == -1 )
    {
      v9[25] = v8;
      LODWORD(v2) = 1;
    }
    return (unsigned int)v2;
  }
  v6 = **a1;
  if ( v5 <= 1 )
  {
    v7 = *(_QWORD *)(v2 + 16);
    v8 = *(_DWORD *)(*(_QWORD *)v7 + 8LL) >> 8;
    if ( v8 != v6 )
    {
LABEL_6:
      v9 = a1[1];
      goto LABEL_7;
    }
    v12 = *((_QWORD *)a2 + 2);
    if ( !v12 )
    {
      v9 = a1[1];
      goto LABEL_20;
    }
    goto LABEL_11;
  }
  v11 = *(_DWORD *)(v2 + 8);
  v8 = v11 >> 8;
  if ( v11 >> 8 != v6 )
    goto LABEL_6;
  v12 = *((_QWORD *)a2 + 2);
  if ( !v12 )
  {
    v9 = a1[1];
    goto LABEL_22;
  }
LABEL_11:
  v13 = **a1;
  do
  {
    v14 = *(_QWORD *)(v12 + 24);
    if ( *(_BYTE *)v14 != 79 )
    {
      v15 = a1[1];
      if ( v5 <= 1 )
        v2 = **(_QWORD **)(v2 + 16);
      v16 = v15[25];
      v17 = *(_DWORD *)(v2 + 8) >> 8;
      LOBYTE(v2) = v17 == v16;
      if ( v16 == -1 )
      {
        v15[25] = v17;
        LODWORD(v2) = 1;
      }
      return (unsigned int)v2;
    }
    v18 = *(_QWORD *)(v14 + 8);
    v19 = *(unsigned __int8 *)(v18 + 8) - 17;
    if ( v13 == v6 )
    {
      if ( v19 <= 1 )
      {
        v20 = *(_QWORD *)(v18 + 16);
LABEL_28:
        v13 = *(_DWORD *)(*(_QWORD *)v20 + 8LL) >> 8;
        goto LABEL_29;
      }
      v13 = *(_DWORD *)(v18 + 8) >> 8;
    }
    else
    {
      if ( v19 <= 1 )
      {
        v20 = *(_QWORD *)(v18 + 16);
        if ( *(_DWORD *)(*(_QWORD *)v20 + 8LL) >> 8 != v13 )
          goto LABEL_36;
        goto LABEL_28;
      }
      if ( *(_DWORD *)(v18 + 8) >> 8 != v13 )
      {
LABEL_36:
        LODWORD(v2) = 0;
        return (unsigned int)v2;
      }
    }
LABEL_29:
    v12 = *(_QWORD *)(v12 + 8);
  }
  while ( v12 );
  v9 = a1[1];
  if ( v13 == v6 )
    goto LABEL_18;
  v21 = v9[25];
  LOBYTE(v2) = v13 == v21;
  if ( v21 == -1 )
  {
    v9[25] = v13;
    LODWORD(v2) = 1;
  }
  return (unsigned int)v2;
}
