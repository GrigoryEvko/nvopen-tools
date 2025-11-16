// Function: sub_19932F0
// Address: 0x19932f0
//
void __fastcall sub_19932F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // rdx
  __int64 v4; // rdx
  _QWORD *v5; // rax
  _QWORD *v6; // r10
  __int64 v7; // r9
  __int64 v8; // rdx
  _QWORD *v9; // r8
  _QWORD *v10; // rdx
  __int64 v11; // r9
  __int64 v12; // r9
  __int64 v13; // r9
  unsigned int v14; // edx
  __int64 v15; // rdx
  _QWORD *v16; // r9
  __int64 v17; // r8
  __int64 v18; // rdx
  _QWORD *v19; // rdx
  __int64 v20; // r8
  __int64 v21; // r8
  __int64 v22; // r8
  bool v23; // dl

  v2 = *(_QWORD *)(a1 + 80);
  if ( !v2 )
  {
    if ( *(_DWORD *)(a1 + 40) <= 1u )
      return;
    v5 = *(_QWORD **)(a1 + 32);
    v14 = *(_DWORD *)(a1 + 40);
    v2 = v5[v14 - 1];
    *(_QWORD *)(a1 + 24) = 1;
    *(_QWORD *)(a1 + 80) = v2;
    *(_DWORD *)(a1 + 40) = v14 - 1;
    goto LABEL_22;
  }
  if ( *(_QWORD *)(a1 + 24) != 1 )
    return;
  v3 = *(unsigned int *)(a1 + 40);
  if ( !(_DWORD)v3 )
  {
    if ( *(_WORD *)(v2 + 24) != 7 )
      return;
LABEL_23:
    if ( a2 == *(_QWORD *)(v2 + 48) )
      return;
    v5 = *(_QWORD **)(a1 + 32);
LABEL_25:
    v15 = 8LL * *(unsigned int *)(a1 + 40);
    v16 = &v5[(unsigned __int64)v15 / 8];
    v17 = v15 >> 3;
    v18 = v15 >> 5;
    if ( v18 )
    {
      v19 = &v5[4 * v18];
      while ( 1 )
      {
        if ( *(_WORD *)(*v5 + 24LL) == 7 )
        {
          if ( a2 == *(_QWORD *)(*v5 + 48LL) )
            goto LABEL_38;
          v20 = v5[1];
          if ( *(_WORD *)(v20 + 24) != 7 )
            goto LABEL_29;
        }
        else
        {
          v20 = v5[1];
          if ( *(_WORD *)(v20 + 24) != 7 )
            goto LABEL_29;
        }
        if ( a2 == *(_QWORD *)(v20 + 48) )
        {
          ++v5;
          goto LABEL_38;
        }
LABEL_29:
        v21 = v5[2];
        if ( *(_WORD *)(v21 + 24) == 7 && a2 == *(_QWORD *)(v21 + 48) )
        {
          v5 += 2;
          goto LABEL_38;
        }
        v22 = v5[3];
        if ( *(_WORD *)(v22 + 24) == 7 && a2 == *(_QWORD *)(v22 + 48) )
        {
          v5 += 3;
          goto LABEL_38;
        }
        v5 += 4;
        if ( v19 == v5 )
        {
          v17 = v16 - v5;
          break;
        }
      }
    }
    if ( v17 != 2 )
    {
      if ( v17 != 3 )
      {
        if ( v17 != 1 )
          return;
        goto LABEL_36;
      }
      if ( *(_WORD *)(*v5 + 24LL) == 7 && a2 == *(_QWORD *)(*v5 + 48LL) )
        goto LABEL_38;
      ++v5;
    }
    if ( *(_WORD *)(*v5 + 24LL) == 7 && a2 == *(_QWORD *)(*v5 + 48LL) )
      goto LABEL_38;
    ++v5;
LABEL_36:
    if ( *(_WORD *)(*v5 + 24LL) != 7 || a2 != *(_QWORD *)(*v5 + 48LL) )
      return;
LABEL_38:
    if ( v5 != v16 )
    {
      *(_QWORD *)(a1 + 80) = *v5;
      *v5 = v2;
    }
    return;
  }
  if ( *(_WORD *)(v2 + 24) == 7 && a2 == *(_QWORD *)(v2 + 48) )
    return;
  v4 = 8 * v3;
  v5 = *(_QWORD **)(a1 + 32);
  v6 = &v5[(unsigned __int64)v4 / 8];
  v7 = v4 >> 3;
  v8 = v4 >> 5;
  v9 = v5;
  if ( !v8 )
    goto LABEL_14;
  v10 = &v5[4 * v8];
  do
  {
    if ( *(_WORD *)(*v9 + 24LL) == 7 && a2 == *(_QWORD *)(*v9 + 48LL) )
      goto LABEL_49;
    v11 = v9[1];
    if ( *(_WORD *)(v11 + 24) == 7 && a2 == *(_QWORD *)(v11 + 48) )
    {
      v23 = v6 == v9 + 1;
      goto LABEL_50;
    }
    v12 = v9[2];
    if ( *(_WORD *)(v12 + 24) == 7 && a2 == *(_QWORD *)(v12 + 48) )
    {
      v23 = v6 == v9 + 2;
      goto LABEL_50;
    }
    v13 = v9[3];
    if ( *(_WORD *)(v13 + 24) == 7 && a2 == *(_QWORD *)(v13 + 48) )
    {
      v23 = v6 == v9 + 3;
LABEL_50:
      if ( v23 )
        return;
LABEL_22:
      if ( *(_WORD *)(v2 + 24) != 7 )
        goto LABEL_25;
      goto LABEL_23;
    }
    v9 += 4;
  }
  while ( v9 != v10 );
  v7 = v6 - v9;
LABEL_14:
  if ( v7 == 2 )
  {
LABEL_67:
    if ( *(_WORD *)(*v9 + 24LL) != 7 || a2 != *(_QWORD *)(*v9 + 48LL) )
    {
      ++v9;
      goto LABEL_17;
    }
    goto LABEL_49;
  }
  if ( v7 == 3 )
  {
    if ( *(_WORD *)(*v9 + 24LL) == 7 && a2 == *(_QWORD *)(*v9 + 48LL) )
      goto LABEL_49;
    ++v9;
    goto LABEL_67;
  }
  if ( v7 != 1 )
    return;
LABEL_17:
  if ( *(_WORD *)(*v9 + 24LL) == 7 && a2 == *(_QWORD *)(*v9 + 48LL) )
  {
LABEL_49:
    v23 = v6 == v9;
    goto LABEL_50;
  }
}
