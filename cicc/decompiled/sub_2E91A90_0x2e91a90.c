// Function: sub_2E91A90
// Address: 0x2e91a90
//
__int64 __fastcall sub_2E91A90(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // r13
  _BYTE *v7; // rax
  _BYTE *i; // rcx
  char v9; // dl

  if ( *(_QWORD *)(a1 + 216) && !(*(unsigned __int8 (__fastcall **)(__int64))(a1 + 224))(a1 + 200) )
    return 0;
  v2 = *(_QWORD *)(a2 + 328);
  result = 0;
  if ( v2 == a2 + 320 )
    return 0;
  do
  {
    v4 = *(_QWORD *)(v2 + 56);
    v5 = v2 + 48;
    if ( v4 != v2 + 48 )
    {
      while ( 1 )
      {
        v6 = *(_QWORD *)(v4 + 8);
        if ( *(_WORD *)(v4 + 68) == 21 )
        {
          for ( ; v5 != v6; v6 = *(_QWORD *)(v6 + 8) )
          {
            if ( (*(_BYTE *)(v6 + 44) & 4) == 0 )
              break;
            sub_2E89050((__int64 *)v6);
            v7 = *(_BYTE **)(v6 + 32);
            for ( i = &v7[40 * (*(_DWORD *)(v6 + 40) & 0xFFFFFF)]; i != v7; v7 += 40 )
            {
              if ( !*v7 )
              {
                v9 = v7[4];
                if ( (v9 & 2) != 0 )
                  v7[4] = v9 & 0xFD;
              }
            }
          }
          sub_2E88E20(v4);
          result = 1;
        }
        if ( v6 == v5 )
          break;
        v4 = v6;
      }
    }
    v2 = *(_QWORD *)(v2 + 8);
  }
  while ( a2 + 320 != v2 );
  return result;
}
