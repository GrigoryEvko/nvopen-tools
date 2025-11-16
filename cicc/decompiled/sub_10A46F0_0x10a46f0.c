// Function: sub_10A46F0
// Address: 0x10a46f0
//
bool __fastcall sub_10A46F0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  bool result; // al
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx

  if ( !a2 )
    return 0;
  v2 = *(_QWORD *)(a2 - 64);
  v3 = *(_QWORD *)(v2 + 16);
  if ( !v3
    || *(_QWORD *)(v3 + 8)
    || *(_BYTE *)v2 != 85
    || (v7 = *(_QWORD *)(v2 - 32)) == 0
    || *(_BYTE *)v7
    || *(_QWORD *)(v7 + 24) != *(_QWORD *)(v2 + 80)
    || *(_DWORD *)(v7 + 36) != *(_DWORD *)a1
    || (v8 = *(_QWORD *)(v2 + 32 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(v2 + 4) & 0x7FFFFFF)))) == 0
    || (**(_QWORD **)(a1 + 16) = v8, *(_BYTE *)v2 != 85)
    || (v9 = *(_QWORD *)(v2 + 32 * (*(unsigned int *)(a1 + 24) - (unsigned __int64)(*(_DWORD *)(v2 + 4) & 0x7FFFFFF)))) == 0 )
  {
    v4 = *(_QWORD *)(a2 - 32);
LABEL_4:
    v5 = *(_QWORD *)(v4 + 16);
    if ( v5 )
    {
      if ( !*(_QWORD *)(v5 + 8) && *(_BYTE *)v4 == 85 )
      {
        v10 = *(_QWORD *)(v4 - 32);
        if ( v10 )
        {
          if ( !*(_BYTE *)v10 && *(_QWORD *)(v10 + 24) == *(_QWORD *)(v4 + 80) && *(_DWORD *)(v10 + 36) == *(_DWORD *)a1 )
          {
            v11 = *(_QWORD *)(v4 + 32
                                 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
            if ( v11 )
            {
              **(_QWORD **)(a1 + 16) = v11;
              if ( *(_BYTE *)v4 == 85 )
              {
                v12 = *(_QWORD *)(v4
                                + 32
                                * (*(unsigned int *)(a1 + 24) - (unsigned __int64)(*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
                if ( v12 )
                {
                  **(_QWORD **)(a1 + 32) = v12;
                  return **(_QWORD **)(a1 + 40) == *(_QWORD *)(a2 - 64);
                }
              }
            }
          }
        }
      }
    }
    return 0;
  }
  **(_QWORD **)(a1 + 32) = v9;
  result = 1;
  v4 = *(_QWORD *)(a2 - 32);
  if ( **(_QWORD **)(a1 + 40) != v4 )
    goto LABEL_4;
  return result;
}
