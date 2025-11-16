// Function: sub_2B09380
// Address: 0x2b09380
//
char *__fastcall sub_2B09380(char *a1, char *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  char *v4; // rax
  _BYTE *v5; // rcx
  char *result; // rax
  _BYTE *v7; // rdx
  _BYTE *v8; // rcx
  _BYTE *v9; // rdx
  _BYTE *v10; // rcx
  _BYTE *v11; // rdx
  _BYTE *v12; // rcx
  _BYTE *v13; // rcx
  _BYTE *v14; // rcx
  _BYTE *v15; // rax

  v2 = (a2 - a1) >> 6;
  v3 = (a2 - a1) >> 4;
  if ( v2 > 0 )
  {
    v4 = &a1[64 * v2];
    while ( 1 )
    {
      if ( **(_BYTE **)a1 > 0x15u )
      {
        v5 = (_BYTE *)*((_QWORD *)a1 + 1);
        if ( *v5 > 0x15u && *(_BYTE **)a1 != v5 )
          return a1;
      }
      v7 = (_BYTE *)*((_QWORD *)a1 + 2);
      if ( *v7 > 0x15u )
      {
        v8 = (_BYTE *)*((_QWORD *)a1 + 3);
        if ( *v8 > 0x15u && v7 != v8 )
          return a1 + 16;
      }
      v9 = (_BYTE *)*((_QWORD *)a1 + 4);
      if ( *v9 > 0x15u )
      {
        v10 = (_BYTE *)*((_QWORD *)a1 + 5);
        if ( *v10 > 0x15u && v9 != v10 )
          return a1 + 32;
      }
      v11 = (_BYTE *)*((_QWORD *)a1 + 6);
      if ( *v11 > 0x15u )
      {
        v12 = (_BYTE *)*((_QWORD *)a1 + 7);
        if ( *v12 > 0x15u && v11 != v12 )
          return a1 + 48;
      }
      a1 += 64;
      if ( a1 == v4 )
      {
        v3 = (a2 - a1) >> 4;
        break;
      }
    }
  }
  if ( v3 == 2 )
  {
LABEL_30:
    if ( **(_BYTE **)a1 > 0x15u )
    {
      v14 = (_BYTE *)*((_QWORD *)a1 + 1);
      if ( *v14 > 0x15u )
      {
        result = a1;
        if ( *(_BYTE **)a1 != v14 )
          return result;
      }
    }
    a1 += 16;
    goto LABEL_34;
  }
  if ( v3 == 3 )
  {
    if ( **(_BYTE **)a1 > 0x15u )
    {
      v13 = (_BYTE *)*((_QWORD *)a1 + 1);
      if ( *v13 > 0x15u )
      {
        result = a1;
        if ( *(_BYTE **)a1 != v13 )
          return result;
      }
    }
    a1 += 16;
    goto LABEL_30;
  }
  if ( v3 != 1 )
    return a2;
LABEL_34:
  result = a2;
  if ( **(_BYTE **)a1 > 0x15u )
  {
    v15 = (_BYTE *)*((_QWORD *)a1 + 1);
    if ( *v15 > 0x15u && *(_BYTE **)a1 != v15 )
      return a1;
    return a2;
  }
  return result;
}
