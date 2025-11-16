// Function: sub_2DADE10
// Address: 0x2dade10
//
_BOOL8 __fastcall sub_2DADE10(__int64 a1, int a2)
{
  __int64 v3; // rcx
  unsigned __int64 v4; // rdi
  unsigned __int64 i; // rsi
  _BYTE *v6; // rbx
  _BYTE *v7; // r13
  _BYTE *v8; // r15
  _BOOL4 v9; // r14d
  _BYTE *v10; // rbx
  unsigned __int64 j; // rdx
  _BYTE *v13; // rbx
  _BYTE *v14; // r15
  _BYTE *v15; // rbx

  if ( a2 < 0 )
    v3 = *(_QWORD *)(*(_QWORD *)(a1 + 56) + 16LL * (a2 & 0x7FFFFFFF) + 8);
  else
    v3 = *(_QWORD *)(*(_QWORD *)(a1 + 304) + 8LL * (unsigned int)a2);
  if ( v3 )
  {
    if ( (*(_BYTE *)(v3 + 3) & 0x10) != 0 || (v3 = *(_QWORD *)(v3 + 32)) != 0 && (*(_BYTE *)(v3 + 3) & 0x10) != 0 )
    {
      v4 = *(_QWORD *)(v3 + 16);
      for ( i = v4; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
        ;
      while ( 1 )
      {
        v3 = *(_QWORD *)(v3 + 32);
        if ( !v3 || (*(_BYTE *)(v3 + 3) & 0x10) == 0 )
          break;
        for ( j = *(_QWORD *)(v3 + 16); (*(_BYTE *)(j + 44) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
          ;
        if ( i != j )
          return 0;
      }
      for ( ; (*(_BYTE *)(v4 + 44) & 4) != 0; v4 = *(_QWORD *)v4 & 0xFFFFFFFFFFFFFFF8LL )
        ;
      v6 = *(_BYTE **)(v4 + 32);
      v7 = &v6[40 * (*(_DWORD *)(v4 + 40) & 0xFFFFFF)];
      if ( v6 != v7 )
      {
        while ( 1 )
        {
          v8 = v6;
          v9 = sub_2DADC00(v6);
          if ( v9 )
            break;
          v6 += 40;
          if ( v7 == v6 )
            return 0;
        }
        while ( v7 != v8 )
        {
          if ( *((_DWORD *)v8 + 2) == a2 )
          {
            if ( v7 == v8 )
              break;
            v13 = v8 + 40;
            if ( v7 != v8 + 40 )
            {
              while ( 1 )
              {
                v14 = v13;
                if ( sub_2DADC00(v13) )
                  break;
                v13 += 40;
                if ( v7 == v13 )
                  return v9;
              }
              while ( v7 != v14 )
              {
                if ( a2 == *((_DWORD *)v14 + 2) )
                {
                  LOBYTE(v9) = v7 == v14;
                  return v9;
                }
                v15 = v14 + 40;
                if ( v7 == v14 + 40 )
                  return v9;
                while ( 1 )
                {
                  v14 = v15;
                  if ( sub_2DADC00(v15) )
                    break;
                  v15 += 40;
                  if ( v7 == v15 )
                    return v9;
                }
              }
            }
            return v9;
          }
          v10 = v8 + 40;
          if ( v7 == v8 + 40 )
            break;
          while ( 1 )
          {
            v8 = v10;
            if ( sub_2DADC00(v10) )
              break;
            v10 += 40;
            if ( v7 == v10 )
              return 0;
          }
        }
      }
    }
  }
  return 0;
}
