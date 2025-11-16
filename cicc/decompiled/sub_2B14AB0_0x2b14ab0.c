// Function: sub_2B14AB0
// Address: 0x2b14ab0
//
char *__fastcall sub_2B14AB0(_DWORD *a1, char *a2, int *a3)
{
  char *v3; // r8
  __int64 v4; // r11
  __int64 v5; // rax
  int *v6; // rax
  _DWORD *v7; // r10
  _DWORD *v8; // r9
  char *v9; // r11
  int v10; // ecx
  int v12; // ecx
  int v13; // ecx
  int v14; // ecx
  int v15; // ecx
  int v16; // ecx
  int v17; // eax
  int v18; // ecx
  int v19; // ecx
  int v20; // ecx
  char *v21; // rdx

  v3 = (char *)a1;
  v4 = (a2 - (char *)a1) >> 4;
  v5 = (a2 - (char *)a1) >> 2;
  if ( v4 > 0 )
  {
    v6 = a1 + 3;
    v7 = a1 + 2;
    v8 = a1 + 1;
    v9 = (char *)&a1[4 * v4];
    while ( 1 )
    {
      v10 = *(v6 - 3);
      if ( *a3 != -1 )
        break;
      if ( v10 != -1 )
      {
        *a3 = v10;
        v12 = *(v6 - 2);
        if ( v12 != -1 )
        {
LABEL_9:
          if ( *a3 != v12 )
            return (char *)v8;
        }
LABEL_29:
        v14 = *(v6 - 1);
        if ( v14 != -1 )
          goto LABEL_13;
        goto LABEL_30;
      }
      v13 = *(v6 - 2);
      if ( v13 != -1 )
      {
        *a3 = v13;
        v14 = *(v6 - 1);
        if ( v14 != -1 )
        {
LABEL_13:
          if ( v14 != *a3 )
            return (char *)v7;
        }
LABEL_30:
        v16 = *v6;
        goto LABEL_17;
      }
      v15 = *(v6 - 1);
      if ( v15 != -1 )
      {
        *a3 = v15;
        v16 = *v6;
LABEL_17:
        if ( v16 != -1 && *a3 != v16 )
          return (char *)v6;
        goto LABEL_22;
      }
      if ( *v6 != -1 )
        *a3 = *v6;
LABEL_22:
      v3 += 16;
      v6 += 4;
      v7 += 4;
      v8 += 4;
      if ( v3 == v9 )
      {
        v5 = (a2 - v3) >> 2;
        goto LABEL_24;
      }
    }
    if ( v10 != -1 && v10 != *a3 )
      return v3;
    v12 = *(v6 - 2);
    if ( v12 != -1 )
      goto LABEL_9;
    goto LABEL_29;
  }
LABEL_24:
  switch ( v5 )
  {
    case 2LL:
      v17 = *a3;
      v20 = *(_DWORD *)v3;
      if ( *a3 != -1 )
      {
LABEL_38:
        if ( v20 != -1 && v20 != v17 )
          return v3;
        v21 = v3;
LABEL_41:
        v18 = *((_DWORD *)v21 + 1);
        v3 += 4;
        goto LABEL_42;
      }
      goto LABEL_47;
    case 3LL:
      v17 = *a3;
      v19 = *(_DWORD *)v3;
      if ( *a3 != -1 )
      {
        if ( v19 != -1 && v19 != v17 )
          return v3;
        goto LABEL_37;
      }
      if ( v19 != -1 )
      {
        *a3 = v19;
        v17 = v19;
LABEL_37:
        v20 = *((_DWORD *)v3 + 1);
        v3 += 4;
        goto LABEL_38;
      }
      v20 = *((_DWORD *)v3 + 1);
      v3 += 4;
LABEL_47:
      if ( v20 == -1 )
      {
        v18 = *((_DWORD *)v3 + 1);
LABEL_32:
        v3 = a2;
        if ( v18 != -1 )
          *a3 = v18;
        return v3;
      }
      *a3 = v20;
      v17 = v20;
      v21 = v3;
      goto LABEL_41;
    case 1LL:
      v17 = *a3;
      v18 = *(_DWORD *)v3;
      if ( *a3 == -1 )
        goto LABEL_32;
LABEL_42:
      if ( v18 != -1 )
      {
        if ( v18 == v17 )
          return a2;
        return v3;
      }
      break;
  }
  return a2;
}
