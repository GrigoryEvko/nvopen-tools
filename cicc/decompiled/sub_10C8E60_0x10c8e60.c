// Function: sub_10C8E60
// Address: 0x10c8e60
//
bool __fastcall sub_10C8E60(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rcx
  char *v7; // rdx
  __int64 v8; // rax
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rcx
  __int64 v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  v4 = *((_QWORD *)a3 - 8);
  if ( !v4 )
    goto LABEL_22;
  **a1 = v4;
  v5 = *((_QWORD *)a3 - 4);
  v6 = *(_QWORD *)(v5 + 16);
  if ( !v6 || *(_QWORD *)(v6 + 8) )
    goto LABEL_5;
  if ( *(_BYTE *)v5 == 42 )
  {
    v12 = *(_QWORD *)(v5 - 64);
    if ( v12 )
    {
      *a1[1] = v12;
      v13 = *(_QWORD *)(v5 - 32);
      if ( v13 == *a1[2] )
        return 1;
      if ( !v13 )
      {
LABEL_21:
        if ( *(_BYTE *)v5 == 44 )
        {
          v11 = *(_QWORD *)(v5 - 64);
          if ( v11 )
            goto LABEL_15;
        }
LABEL_22:
        v5 = *((_QWORD *)a3 - 4);
        if ( !v5 )
          return 0;
LABEL_5:
        **a1 = v5;
        v7 = (char *)*((_QWORD *)a3 - 8);
        v8 = *((_QWORD *)v7 + 2);
        if ( !v8 || *(_QWORD *)(v8 + 8) )
          return 0;
        v9 = *v7;
        if ( *v7 != 42 )
          goto LABEL_8;
        v14 = *((_QWORD *)v7 - 8);
        if ( v14 )
        {
          *a1[1] = v14;
          v15 = *((_QWORD *)v7 - 4);
          if ( v15 == *a1[2] )
            return 1;
          if ( !v15 )
            goto LABEL_28;
        }
        else
        {
          v15 = *((_QWORD *)v7 - 4);
          if ( !v15 )
            return 0;
        }
        *a1[1] = v15;
        if ( *((_QWORD *)v7 - 8) == *a1[2] )
          return 1;
LABEL_28:
        v9 = *v7;
LABEL_8:
        if ( v9 == 44 )
        {
          v10 = *((_QWORD *)v7 - 8);
          if ( v10 )
          {
            *a1[3] = v10;
            return *a1[4] == *((_QWORD *)v7 - 4);
          }
        }
        return 0;
      }
    }
    else
    {
      v13 = *(_QWORD *)(v5 - 32);
      if ( !v13 )
        goto LABEL_5;
    }
    *a1[1] = v13;
    if ( *(_QWORD *)(v5 - 64) == *a1[2] )
      return 1;
    goto LABEL_21;
  }
  if ( *(_BYTE *)v5 != 44 )
    goto LABEL_5;
  v11 = *(_QWORD *)(v5 - 64);
  if ( !v11 )
    goto LABEL_5;
LABEL_15:
  *a1[3] = v11;
  if ( *(_QWORD *)(v5 - 32) != *a1[4] )
    goto LABEL_22;
  return 1;
}
