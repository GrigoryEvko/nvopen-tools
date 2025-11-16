// Function: sub_10C4D50
// Address: 0x10c4d50
//
bool __fastcall sub_10C4D50(_QWORD **a1, unsigned __int8 *a2)
{
  __int64 v3; // rdi
  bool result; // al
  __int64 v5; // rdi
  __int64 v6; // r13
  __int64 v7; // r14
  unsigned __int8 *v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // rdx
  __int64 v14; // r13
  unsigned __int8 *v15; // rbx
  __int64 v16; // rcx
  _BYTE *v17; // rdi
  __int64 v18; // rbx

  if ( !a2 )
    return 0;
  v3 = *((_QWORD *)a2 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  result = sub_BCAC40(v3, 1);
  if ( !result )
    goto LABEL_16;
  if ( *a2 == 57 )
  {
    if ( (a2[7] & 0x40) != 0 )
      v8 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    else
      v8 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v9 = *(_QWORD *)v8;
    if ( *(_QWORD *)v8 )
    {
      v10 = *((_QWORD *)v8 + 4);
      **a1 = v9;
      if ( v10 )
      {
        *a1[1] = v10;
        return result;
      }
    }
    goto LABEL_16;
  }
  v5 = *((_QWORD *)a2 + 1);
  if ( *a2 == 86 )
  {
    v6 = *((_QWORD *)a2 - 12);
    if ( *(_QWORD *)(v6 + 8) == v5 && **((_BYTE **)a2 - 4) <= 0x15u )
    {
      v7 = *((_QWORD *)a2 - 8);
      result = sub_AC30F0(*((_QWORD *)a2 - 4));
      if ( result )
      {
        **a1 = v6;
        if ( v7 )
        {
          *a1[1] = v7;
          return result;
        }
      }
LABEL_16:
      v5 = *((_QWORD *)a2 + 1);
    }
  }
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  result = sub_BCAC40(v5, 1);
  if ( !result )
    return 0;
  v13 = *a2;
  if ( (_BYTE)v13 == 58 )
  {
    if ( (a2[7] & 0x40) != 0 )
      v15 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
    else
      v15 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    if ( *(_QWORD *)v15 )
    {
      v16 = *((_QWORD *)v15 + 4);
      *a1[2] = *(_QWORD *)v15;
      if ( v16 )
      {
        *a1[3] = v16;
        return result;
      }
    }
    return 0;
  }
  if ( (_BYTE)v13 != 86 )
    return 0;
  v14 = *((_QWORD *)a2 - 12);
  if ( *(_QWORD *)(v14 + 8) != *((_QWORD *)a2 + 1) )
    return 0;
  v17 = (_BYTE *)*((_QWORD *)a2 - 8);
  if ( *v17 > 0x15u )
    return 0;
  v18 = *((_QWORD *)a2 - 4);
  result = sub_AD7A80(v17, 1, v13, v11, v12);
  if ( !result )
    return 0;
  *a1[2] = v14;
  if ( !v18 )
    return 0;
  *a1[3] = v18;
  return result;
}
