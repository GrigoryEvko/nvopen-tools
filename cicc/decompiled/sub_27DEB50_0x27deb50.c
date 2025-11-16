// Function: sub_27DEB50
// Address: 0x27deb50
//
bool __fastcall sub_27DEB50(_QWORD **a1, __int64 a2)
{
  __int64 v3; // rdi
  bool result; // al
  __int64 v5; // r13
  _BYTE *v6; // rdi
  __int64 v7; // rbx
  _QWORD *v8; // rbx
  __int64 v9; // rcx

  if ( !a2 )
    return 0;
  v3 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v3 + 8) - 17 <= 1 )
    v3 = **(_QWORD **)(v3 + 16);
  result = sub_BCAC40(v3, 1);
  if ( !result )
    return 0;
  if ( *(_BYTE *)a2 != 57 )
  {
    if ( *(_BYTE *)a2 == 86 )
    {
      v5 = *(_QWORD *)(a2 - 96);
      if ( *(_QWORD *)(a2 + 8) == *(_QWORD *)(v5 + 8) )
      {
        v6 = *(_BYTE **)(a2 - 32);
        if ( *v6 <= 0x15u )
        {
          v7 = *(_QWORD *)(a2 - 64);
          result = sub_AC30F0((__int64)v6);
          if ( result )
          {
            **a1 = v5;
            if ( v7 )
            {
              *a1[1] = v7;
              return result;
            }
          }
        }
      }
    }
    return 0;
  }
  v8 = (*(_BYTE *)(a2 + 7) & 0x40) != 0
     ? *(_QWORD **)(a2 - 8)
     : (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( !*v8 )
    return 0;
  v9 = v8[4];
  **a1 = *v8;
  if ( !v9 )
    return 0;
  *a1[1] = v9;
  return result;
}
