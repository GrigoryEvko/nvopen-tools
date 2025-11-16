// Function: sub_D22920
// Address: 0xd22920
//
__int64 __fastcall sub_D22920(__int64 a1, __int64 *a2, __int64 *a3, _QWORD *a4, _QWORD *a5)
{
  __int64 result; // rax
  __int64 v6; // r12
  __int64 v7; // r9
  char v8; // al
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rdi
  __int64 v16; // rcx

  result = 0;
  if ( *(_BYTE *)a1 == 31 && (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) == 3 )
  {
    v6 = *(_QWORD *)(a1 - 96);
    v7 = *(_QWORD *)(v6 + 16);
    if ( v7 )
    {
      if ( !*(_QWORD *)(v7 + 8) )
      {
        *a4 = *(_QWORD *)(a1 - 32);
        *a5 = *(_QWORD *)(a1 - 64);
        v8 = *(_BYTE *)v6;
        if ( *(_BYTE *)v6 <= 0x1Cu )
          return 0;
        if ( v8 != 85 )
        {
          if ( v8 == 57 )
          {
            v9 = *(_QWORD *)(v6 - 64);
            if ( v9 )
            {
              v10 = *(_QWORD *)(v6 - 32);
              if ( v10 )
              {
                if ( *(_BYTE *)v9 == 85 )
                {
                  v15 = *(_QWORD *)(v9 - 32);
                  if ( v15 )
                  {
                    if ( !*(_BYTE *)v15 && *(_QWORD *)(v15 + 24) == *(_QWORD *)(v9 + 80) && *(_DWORD *)(v15 + 36) == 169 )
                    {
                      v16 = *(_QWORD *)(v9 + 16);
                      if ( v16 )
                      {
                        if ( !*(_QWORD *)(v16 + 8) )
                        {
                          *a3 = sub_986520(v6);
                          *a2 = sub_986520(v6) + 32;
                          return 1;
                        }
                      }
                    }
                  }
                }
                if ( *(_BYTE *)v10 == 85 )
                {
                  v11 = *(_QWORD *)(v10 - 32);
                  if ( v11 )
                  {
                    if ( !*(_BYTE *)v11
                      && *(_QWORD *)(v11 + 24) == *(_QWORD *)(v10 + 80)
                      && *(_DWORD *)(v11 + 36) == 169 )
                    {
                      v12 = *(_QWORD *)(v10 + 16);
                      if ( v12 )
                      {
                        if ( !*(_QWORD *)(v12 + 8) )
                        {
                          *a3 = sub_986520(v6) + 32;
                          *a2 = sub_986520(v6);
                          return 1;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          return 0;
        }
        v13 = *(_QWORD *)(v6 - 32);
        if ( !v13 || *(_BYTE *)v13 || *(_QWORD *)(v13 + 24) != *(_QWORD *)(v6 + 80) || *(_DWORD *)(v13 + 36) != 169 )
          return 0;
        if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
          v14 = *(_QWORD *)(a1 - 8);
        else
          v14 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
        *a3 = v14;
        *a2 = 0;
        return 1;
      }
    }
  }
  return result;
}
