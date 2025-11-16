// Function: sub_1186E90
// Address: 0x1186e90
//
bool __fastcall sub_1186E90(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  char v6; // al
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // r14
  _BYTE *v10; // rdi
  __int64 v11; // r13
  _QWORD *v12; // r13
  __int64 v13; // rcx
  __int64 v14; // rbx
  __int64 v15; // rdi
  __int64 v16; // r13
  _BYTE *v17; // rdi
  __int64 v18; // rbx
  _QWORD *v19; // rbx

  if ( a2 + 29 != *a3 )
    return 0;
  v6 = sub_995B10(a1, *((_QWORD *)a3 - 8));
  v7 = *((_QWORD *)a3 - 4);
  if ( v6 && *(_BYTE *)v7 > 0x1Cu )
  {
    v8 = *(_QWORD *)(v7 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
      v8 = **(_QWORD **)(v8 + 16);
    result = sub_BCAC40(v8, 1);
    if ( result )
    {
      if ( *(_BYTE *)v7 == 57 )
      {
        if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
          v12 = *(_QWORD **)(v7 - 8);
        else
          v12 = (_QWORD *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
        if ( *v12 )
        {
          v13 = v12[4];
          *a1[1] = *v12;
          if ( v13 )
            goto LABEL_37;
        }
      }
      else if ( *(_BYTE *)v7 == 86 )
      {
        v9 = *(_QWORD *)(v7 - 96);
        if ( *(_QWORD *)(v9 + 8) == *(_QWORD *)(v7 + 8) )
        {
          v10 = *(_BYTE **)(v7 - 32);
          if ( *v10 <= 0x15u )
          {
            v11 = *(_QWORD *)(v7 - 64);
            result = sub_AC30F0((__int64)v10);
            if ( result )
            {
              *a1[1] = v9;
              if ( v11 )
              {
                *a1[2] = v11;
                return result;
              }
            }
          }
        }
      }
    }
    v7 = *((_QWORD *)a3 - 4);
  }
  if ( !(unsigned __int8)sub_995B10(a1, v7) )
    return 0;
  v14 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v14 <= 0x1Cu )
    return 0;
  v15 = *(_QWORD *)(v14 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
    v15 = **(_QWORD **)(v15 + 16);
  result = sub_BCAC40(v15, 1);
  if ( !result )
    return 0;
  if ( *(_BYTE *)v14 != 57 )
  {
    if ( *(_BYTE *)v14 == 86 )
    {
      v16 = *(_QWORD *)(v14 - 96);
      if ( *(_QWORD *)(v16 + 8) == *(_QWORD *)(v14 + 8) )
      {
        v17 = *(_BYTE **)(v14 - 32);
        if ( *v17 <= 0x15u )
        {
          v18 = *(_QWORD *)(v14 - 64);
          result = sub_AC30F0((__int64)v17);
          if ( result )
          {
            *a1[1] = v16;
            if ( v18 )
            {
              *a1[2] = v18;
              return result;
            }
          }
        }
      }
    }
    return 0;
  }
  v19 = (*(_BYTE *)(v14 + 7) & 0x40) != 0
      ? *(_QWORD **)(v14 - 8)
      : (_QWORD *)(v14 - 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF));
  if ( !*v19 )
    return 0;
  v13 = v19[4];
  *a1[1] = *v19;
  if ( !v13 )
    return 0;
LABEL_37:
  *a1[2] = v13;
  return result;
}
