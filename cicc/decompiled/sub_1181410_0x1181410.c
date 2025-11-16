// Function: sub_1181410
// Address: 0x1181410
//
bool __fastcall sub_1181410(_QWORD **a1, unsigned __int8 *a2)
{
  __int64 v3; // rdi
  bool result; // al
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // rdx
  __int64 v8; // r13
  _BYTE *v9; // rdi
  __int64 v10; // rbx
  unsigned __int8 *v11; // rbx
  __int64 v12; // rcx

  if ( *a2 <= 0x1Cu )
    return 0;
  v3 = *((_QWORD *)a2 + 1);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  result = sub_BCAC40(v3, 1);
  if ( !result )
    return 0;
  v7 = *a2;
  if ( (_BYTE)v7 != 58 )
  {
    if ( (_BYTE)v7 == 86 )
    {
      v8 = *((_QWORD *)a2 - 12);
      if ( *((_QWORD *)a2 + 1) == *(_QWORD *)(v8 + 8) )
      {
        v9 = (_BYTE *)*((_QWORD *)a2 - 8);
        if ( *v9 <= 0x15u )
        {
          v10 = *((_QWORD *)a2 - 4);
          result = sub_AD7A80(v9, 1, v7, v5, v6);
          if ( result )
          {
            **a1 = v8;
            if ( v10 )
            {
              *a1[1] = v10;
              return result;
            }
          }
        }
      }
    }
    return 0;
  }
  v11 = (a2[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a2 - 1) : &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( !*(_QWORD *)v11 )
    return 0;
  v12 = *((_QWORD *)v11 + 4);
  **a1 = *(_QWORD *)v11;
  if ( !v12 )
    return 0;
  *a1[1] = v12;
  return result;
}
