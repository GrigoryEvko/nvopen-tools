// Function: sub_11594B0
// Address: 0x11594b0
//
char __fastcall sub_11594B0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  char result; // al
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rcx

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
    || (**(_QWORD **)(a1 + 16) = v8, *(_BYTE *)v2 != 85) )
  {
    v4 = *(_QWORD *)(a2 - 32);
    goto LABEL_4;
  }
  result = sub_993A50(
             (_QWORD **)(a1 + 32),
             *(_QWORD *)(v2 + 32 * (*(unsigned int *)(a1 + 24) - (unsigned __int64)(*(_DWORD *)(v2 + 4) & 0x7FFFFFF))));
  v4 = *(_QWORD *)(a2 - 32);
  if ( !result )
  {
LABEL_4:
    v5 = *(_QWORD *)(v4 + 16);
    if ( v5 )
    {
LABEL_18:
      if ( !*(_QWORD *)(v5 + 8) && *(_BYTE *)v4 == 85 )
      {
        v9 = *(_QWORD *)(v4 - 32);
        if ( v9 )
        {
          if ( !*(_BYTE *)v9 && *(_QWORD *)(v9 + 24) == *(_QWORD *)(v4 + 80) && *(_DWORD *)(v9 + 36) == *(_DWORD *)a1 )
          {
            v10 = *(_QWORD *)(v4 + 32
                                 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(v4 + 4) & 0x7FFFFFF)));
            if ( v10 )
            {
              **(_QWORD **)(a1 + 16) = v10;
              if ( *(_BYTE *)v4 == 85 )
              {
                if ( (unsigned __int8)sub_993A50(
                                        (_QWORD **)(a1 + 32),
                                        *(_QWORD *)(v4
                                                  + 32
                                                  * (*(unsigned int *)(a1 + 24)
                                                   - (unsigned __int64)(*(_DWORD *)(v4 + 4) & 0x7FFFFFF)))) )
                  return **(_QWORD **)(a1 + 40) == *(_QWORD *)(a2 - 64);
              }
            }
          }
        }
      }
    }
    return 0;
  }
  if ( **(_QWORD **)(a1 + 40) != v4 )
  {
    v5 = *(_QWORD *)(v4 + 16);
    if ( v5 )
      goto LABEL_18;
    return 0;
  }
  return result;
}
