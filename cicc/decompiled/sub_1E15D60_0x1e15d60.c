// Function: sub_1E15D60
// Address: 0x1e15d60
//
__int64 __fastcall sub_1E15D60(__int64 a1, __int64 a2, unsigned int a3)
{
  __int16 v4; // ax
  __int64 v5; // rdx
  __int64 result; // rax
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // r15
  __int64 v14; // r14
  __int16 v15; // ax
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // [rsp+0h] [rbp-40h]
  __int64 v19; // [rsp+0h] [rbp-40h]

  v4 = **(_WORD **)(a1 + 16);
  if ( **(_WORD **)(a2 + 16) != v4 )
    return 0;
  v5 = *(unsigned int *)(a1 + 40);
  if ( *(_DWORD *)(a2 + 40) != (_DWORD)v5 )
    return 0;
  if ( v4 == 16 )
  {
    v13 = a1;
    v14 = a2;
    while ( 1 )
    {
      v15 = *(_WORD *)(v14 + 46);
      if ( (*(_BYTE *)(v13 + 46) & 8) == 0 )
        break;
      if ( (v15 & 8) != 0 )
      {
        v13 = *(_QWORD *)(v13 + 8);
        v14 = *(_QWORD *)(v14 + 8);
        if ( (unsigned __int8)sub_1E15D60(v13, v14, a3) )
          continue;
      }
      return 0;
    }
    if ( (v15 & 8) != 0 )
      return 0;
    v5 = *(unsigned int *)(a1 + 40);
  }
  if ( (_DWORD)v5 )
  {
    v9 = 0;
    v10 = 40 * v5;
    while ( 1 )
    {
      v11 = v9 + *(_QWORD *)(a1 + 32);
      v12 = v9 + *(_QWORD *)(a2 + 32);
      if ( *(_BYTE *)v11 )
        break;
      if ( (*(_BYTE *)(v11 + 3) & 0x10) != 0 )
      {
        if ( a3 != 2 )
        {
          if ( a3 == 3 )
          {
            if ( *(int *)(v11 + 8) >= 0 || *(int *)(v12 + 8) >= 0 )
              break;
          }
          else
          {
            v19 = v9 + *(_QWORD *)(a2 + 32);
            if ( !(unsigned __int8)sub_1E31610(v11, v12)
              || a3 == 1
              && (((*(_BYTE *)(v19 + 3) & 0x10) != 0) & (*(_BYTE *)(v19 + 3) >> 6)) != ((*(_BYTE *)(v11 + 3) >> 6)
                                                                                      & ((*(_BYTE *)(v11 + 3) & 0x10) != 0)) )
            {
              return 0;
            }
          }
        }
      }
      else
      {
        v18 = v9 + *(_QWORD *)(a2 + 32);
        if ( !(unsigned __int8)sub_1E31610(v11, v12)
          || a3 == 1
          && (((*(_BYTE *)(v18 + 3) & 0x40) != 0) & ((*(_BYTE *)(v18 + 3) >> 4) ^ 1)) != (((*(_BYTE *)(v11 + 3) >> 4)
                                                                                         ^ 1)
                                                                                        & ((*(_BYTE *)(v11 + 3) & 0x40) != 0)) )
        {
          return 0;
        }
      }
LABEL_13:
      v9 += 40;
      if ( v10 == v9 )
        goto LABEL_31;
    }
    if ( !(unsigned __int8)sub_1E31610(v11, v12) )
      return 0;
    goto LABEL_13;
  }
LABEL_31:
  if ( (unsigned __int16)(**(_WORD **)(a1 + 16) - 12) > 1u )
    return 1;
  v16 = *(_QWORD *)(a1 + 64);
  result = 1;
  if ( v16 )
  {
    v17 = *(_QWORD *)(a2 + 64);
    LOBYTE(result) = v16 != v17;
    LOBYTE(v16) = v17 != 0;
    return (unsigned int)v16 & (unsigned int)result ^ 1;
  }
  return result;
}
