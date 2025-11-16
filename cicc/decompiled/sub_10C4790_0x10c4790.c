// Function: sub_10C4790
// Address: 0x10c4790
//
bool __fastcall sub_10C4790(_QWORD **a1, __int64 a2)
{
  __int64 v2; // rax
  bool result; // al
  __int64 v5; // rdi
  __int64 v6; // r13
  _BYTE *v7; // rdi
  __int64 v8; // rbx
  _QWORD *v9; // rbx
  __int64 v10; // rcx

  v2 = *(_QWORD *)(a2 + 16);
  if ( !v2 || *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    v5 = *(_QWORD *)(a2 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
      v5 = **(_QWORD **)(v5 + 16);
    result = sub_BCAC40(v5, 1);
    if ( result )
    {
      if ( *(_BYTE *)a2 == 57 )
      {
        if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
          v9 = *(_QWORD **)(a2 - 8);
        else
          v9 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
        if ( *v9 )
        {
          v10 = v9[4];
          **a1 = *v9;
          if ( v10 )
          {
            *a1[1] = v10;
            return result;
          }
        }
      }
      else if ( *(_BYTE *)a2 == 86 )
      {
        v6 = *(_QWORD *)(a2 - 96);
        if ( *(_QWORD *)(v6 + 8) == *(_QWORD *)(a2 + 8) )
        {
          v7 = *(_BYTE **)(a2 - 32);
          if ( *v7 <= 0x15u )
          {
            v8 = *(_QWORD *)(a2 - 64);
            result = sub_AC30F0((__int64)v7);
            if ( result )
            {
              **a1 = v6;
              if ( v8 )
              {
                *a1[1] = v8;
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
