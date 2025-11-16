// Function: sub_10C4BA0
// Address: 0x10c4ba0
//
bool __fastcall sub_10C4BA0(_QWORD **a1, unsigned __int8 *a2)
{
  __int64 v2; // rax
  bool result; // al
  __int64 v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // rdx
  __int64 v9; // r13
  _BYTE *v10; // rdi
  __int64 v11; // rbx
  unsigned __int8 *v12; // rbx
  __int64 v13; // rcx

  v2 = *((_QWORD *)a2 + 2);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *a2 > 0x1Cu )
  {
    v5 = *((_QWORD *)a2 + 1);
    if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
      v5 = **(_QWORD **)(v5 + 16);
    result = sub_BCAC40(v5, 1);
    if ( result )
    {
      v8 = *a2;
      if ( (_BYTE)v8 == 58 )
      {
        if ( (a2[7] & 0x40) != 0 )
          v12 = (unsigned __int8 *)*((_QWORD *)a2 - 1);
        else
          v12 = &a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
        if ( *(_QWORD *)v12 )
        {
          v13 = *((_QWORD *)v12 + 4);
          **a1 = *(_QWORD *)v12;
          if ( v13 )
          {
            *a1[1] = v13;
            return result;
          }
        }
      }
      else if ( (_BYTE)v8 == 86 )
      {
        v9 = *((_QWORD *)a2 - 12);
        if ( *(_QWORD *)(v9 + 8) == *((_QWORD *)a2 + 1) )
        {
          v10 = (_BYTE *)*((_QWORD *)a2 - 8);
          if ( *v10 <= 0x15u )
          {
            v11 = *((_QWORD *)a2 - 4);
            result = sub_AD7A80(v10, 1, v8, v6, v7);
            if ( result )
            {
              **a1 = v9;
              if ( v11 )
              {
                *a1[1] = v11;
                return result;
              }
            }
          }
        }
      }
    }
  }
  return 0;
}
