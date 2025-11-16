// Function: sub_2E88AF0
// Address: 0x2e88af0
//
bool __fastcall sub_2E88AF0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int16 v4; // dx
  __int64 v7; // rax
  bool result; // al
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rdi
  __int64 v12; // rsi
  __int64 v13; // r15
  __int64 v14; // r14
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rbx
  char v20; // r8
  int v21; // ebx
  __int64 v22; // [rsp+0h] [rbp-40h]
  __int64 v23; // [rsp+0h] [rbp-40h]

  v4 = *(_WORD *)(a1 + 68);
  if ( *(_WORD *)(a2 + 68) != v4 )
    return 0;
  v7 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( (*(_DWORD *)(a2 + 40) & 0xFFFFFF) != (_DWORD)v7 )
    return 0;
  if ( v4 == 21 )
  {
    v13 = a1;
    v14 = a2;
    while ( 1 )
    {
      v15 = *(_DWORD *)(v14 + 44);
      if ( (*(_BYTE *)(v13 + 44) & 8) == 0 )
        break;
      if ( (v15 & 8) != 0 )
      {
        v13 = *(_QWORD *)(v13 + 8);
        v14 = *(_QWORD *)(v14 + 8);
        if ( (unsigned __int8)sub_2E88AF0(v13, v14, a3) )
          continue;
      }
      return 0;
    }
    if ( (v15 & 8) != 0 )
      return 0;
    v7 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  }
  if ( (_DWORD)v7 )
  {
    v9 = 0;
    v10 = 40 * v7;
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
            v23 = v9 + *(_QWORD *)(a2 + 32);
            if ( !(unsigned __int8)sub_2EAB6C0(v11, v12)
              || a3 == 1
              && (((*(_BYTE *)(v23 + 3) & 0x10) != 0) & (*(_BYTE *)(v23 + 3) >> 6)) != ((*(_BYTE *)(v11 + 3) >> 6)
                                                                                      & ((*(_BYTE *)(v11 + 3) & 0x10) != 0)) )
            {
              return 0;
            }
          }
        }
      }
      else
      {
        v22 = v9 + *(_QWORD *)(a2 + 32);
        if ( !(unsigned __int8)sub_2EAB6C0(v11, v12)
          || a3 == 1
          && (((*(_BYTE *)(v22 + 3) & 0x40) != 0) & ((*(_BYTE *)(v22 + 3) >> 4) ^ 1)) != (((*(_BYTE *)(v11 + 3) >> 4)
                                                                                         ^ 1)
                                                                                        & ((*(_BYTE *)(v11 + 3) & 0x40) != 0)) )
        {
          return 0;
        }
      }
LABEL_13:
      v9 += 40;
      if ( v9 == v10 )
        goto LABEL_31;
    }
    if ( !(unsigned __int8)sub_2EAB6C0(v11, v12) )
      return 0;
    goto LABEL_13;
  }
LABEL_31:
  if ( (unsigned __int16)(*(_WORD *)(a1 + 68) - 14) <= 4u )
  {
    v16 = *(_QWORD *)(a1 + 56);
    if ( v16 )
    {
      v17 = *(_QWORD *)(a2 + 56);
      if ( v17 )
      {
        if ( v16 != v17 )
          return 0;
      }
    }
  }
  v18 = sub_2E864E0(a1);
  if ( v18 != sub_2E864E0(a2) )
    return 0;
  v19 = sub_2E86530(a1);
  if ( v19 != sub_2E86530(a2) )
    return 0;
  v20 = sub_2E50190(a1, 7, 1);
  result = 1;
  if ( v20 )
  {
    v21 = sub_2E86580(a1);
    return v21 == (unsigned int)sub_2E86580(a2);
  }
  return result;
}
