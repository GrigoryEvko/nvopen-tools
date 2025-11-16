// Function: sub_10E7DB0
// Address: 0x10e7db0
//
bool __fastcall sub_10E7DB0(_QWORD **a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  char v6; // al
  unsigned __int8 *v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // rdx
  __int64 v12; // r14
  _BYTE *v13; // rdi
  __int64 v14; // r13
  unsigned __int8 *v15; // r13
  __int64 v16; // rcx
  unsigned __int8 *v17; // rbx
  __int64 v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // rdx
  __int64 v22; // r13
  _BYTE *v23; // rdi
  __int64 v24; // rbx
  unsigned __int8 *v25; // rbx

  if ( a2 + 29 != *a3 )
    return 0;
  v6 = sub_995B10(a1, *((_QWORD *)a3 - 8));
  v7 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
  if ( v6 && *v7 > 0x1Cu )
  {
    v8 = *((_QWORD *)v7 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v8 + 8) - 17 <= 1 )
      v8 = **(_QWORD **)(v8 + 16);
    result = sub_BCAC40(v8, 1);
    if ( result )
    {
      v11 = *v7;
      if ( (_BYTE)v11 == 58 )
      {
        if ( (v7[7] & 0x40) != 0 )
          v15 = (unsigned __int8 *)*((_QWORD *)v7 - 1);
        else
          v15 = &v7[-32 * (*((_DWORD *)v7 + 1) & 0x7FFFFFF)];
        if ( *(_QWORD *)v15 )
        {
          v16 = *((_QWORD *)v15 + 4);
          *a1[1] = *(_QWORD *)v15;
          if ( v16 )
            goto LABEL_37;
        }
      }
      else if ( (_BYTE)v11 == 86 )
      {
        v12 = *((_QWORD *)v7 - 12);
        if ( *(_QWORD *)(v12 + 8) == *((_QWORD *)v7 + 1) )
        {
          v13 = (_BYTE *)*((_QWORD *)v7 - 8);
          if ( *v13 <= 0x15u )
          {
            v14 = *((_QWORD *)v7 - 4);
            result = sub_AD7A80(v13, 1, v11, v9, v10);
            if ( result )
            {
              *a1[1] = v12;
              if ( v14 )
              {
                *a1[2] = v14;
                return result;
              }
            }
          }
        }
      }
    }
    v7 = (unsigned __int8 *)*((_QWORD *)a3 - 4);
  }
  if ( !(unsigned __int8)sub_995B10(a1, (__int64)v7) )
    return 0;
  v17 = (unsigned __int8 *)*((_QWORD *)a3 - 8);
  if ( *v17 <= 0x1Cu )
    return 0;
  v18 = *((_QWORD *)v17 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v18 + 8) - 17 <= 1 )
    v18 = **(_QWORD **)(v18 + 16);
  result = sub_BCAC40(v18, 1);
  if ( !result )
    return 0;
  v21 = *v17;
  if ( (_BYTE)v21 != 58 )
  {
    if ( (_BYTE)v21 == 86 )
    {
      v22 = *((_QWORD *)v17 - 12);
      if ( *(_QWORD *)(v22 + 8) == *((_QWORD *)v17 + 1) )
      {
        v23 = (_BYTE *)*((_QWORD *)v17 - 8);
        if ( *v23 <= 0x15u )
        {
          v24 = *((_QWORD *)v17 - 4);
          result = sub_AD7A80(v23, 1, v21, v19, v20);
          if ( result )
          {
            *a1[1] = v22;
            if ( v24 )
            {
              *a1[2] = v24;
              return result;
            }
          }
        }
      }
    }
    return 0;
  }
  v25 = (v17[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)v17 - 1) : &v17[-32 * (*((_DWORD *)v17 + 1) & 0x7FFFFFF)];
  if ( !*(_QWORD *)v25 )
    return 0;
  v16 = *((_QWORD *)v25 + 4);
  *a1[1] = *(_QWORD *)v25;
  if ( !v16 )
    return 0;
LABEL_37:
  *a1[2] = v16;
  return result;
}
