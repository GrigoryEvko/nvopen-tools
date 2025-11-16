// Function: sub_1738BE0
// Address: 0x1738be0
//
char __fastcall sub_1738BE0(__int64 **a1, unsigned __int64 a2, unsigned __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rbx
  char v5; // al
  __int64 v7; // r14
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rsi
  unsigned __int8 v12; // al
  unsigned int v13; // r12d
  bool v14; // al
  __int64 v15; // rax
  unsigned int v16; // r13d
  bool v17; // al
  unsigned int v18; // r15d
  int v19; // r13d
  __int64 v20; // rax
  unsigned int v21; // esi
  unsigned int v22; // [rsp+Ch] [rbp-34h]

  v4 = a2;
  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 == 52 )
  {
    v7 = *(_QWORD *)(a2 - 24);
    v8 = **a1;
    if ( v8 == *(_QWORD *)(a2 - 48) )
    {
      v12 = *(_BYTE *)(v7 + 16);
      if ( v12 == 13 )
      {
        v13 = *(_DWORD *)(v7 + 32);
        if ( v13 <= 0x40 )
        {
          a4 = 64 - v13;
          v14 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v13) == *(_QWORD *)(v7 + 24);
        }
        else
        {
          v14 = v13 == (unsigned int)sub_16A58F0(v7 + 24);
        }
        if ( v14 )
          return 1;
      }
      else
      {
        a3 = *(_QWORD *)v7;
        if ( *(_BYTE *)(*(_QWORD *)v7 + 8LL) == 16 && v12 <= 0x10u )
        {
          v15 = sub_15A1020(*(_BYTE **)(a2 - 24), a2, a3, a4);
          if ( v15 && *(_BYTE *)(v15 + 16) == 13 )
          {
            v16 = *(_DWORD *)(v15 + 32);
            if ( v16 <= 0x40 )
            {
              a4 = 64 - v16;
              a3 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v16);
              v17 = a3 == *(_QWORD *)(v15 + 24);
            }
            else
            {
              v17 = v16 == (unsigned int)sub_16A58F0(v15 + 24);
            }
            if ( v17 )
              return 1;
          }
          else
          {
            v18 = 0;
            v19 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
            if ( !v19 )
              return 1;
            while ( 1 )
            {
              a2 = v18;
              v20 = sub_15A0A60(v7, v18);
              if ( !v20 )
                break;
              a4 = *(unsigned __int8 *)(v20 + 16);
              if ( (_BYTE)a4 != 9 )
              {
                if ( (_BYTE)a4 != 13 )
                  break;
                v21 = *(_DWORD *)(v20 + 32);
                if ( v21 <= 0x40 )
                {
                  a4 = 64 - v21;
                  a2 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v21);
                  if ( *(_QWORD *)(v20 + 24) != a2 )
                    break;
                }
                else
                {
                  v22 = *(_DWORD *)(v20 + 32);
                  a2 = v22;
                  if ( v22 != (unsigned int)sub_16A58F0(v20 + 24) )
                    break;
                }
              }
              if ( v19 == ++v18 )
                return 1;
            }
          }
          v7 = *(_QWORD *)(v4 - 24);
          v8 = **a1;
        }
      }
    }
    if ( v7 == v8 )
      return sub_17279D0(*(_BYTE **)(v4 - 48), a2, a3, a4);
    return 0;
  }
  if ( v5 != 5 || *(_WORD *)(a2 + 18) != 28 )
    return 0;
  v9 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v10 = **a1;
  v11 = 4 * v9;
  if ( v10 != *(_QWORD *)(v4 - 24 * v9) )
  {
LABEL_9:
    if ( *(_QWORD *)(v4 + 24 * (1 - v9)) == v10 )
      return sub_1727B40(*(_BYTE **)(v4 - 24 * v9), v11, 4 * v9, v10);
    return 0;
  }
  if ( !sub_1727B40(*(_BYTE **)(v4 + 24 * (1 - v9)), v11, 1 - v9, v10) )
  {
    v9 = *(_DWORD *)(v4 + 20) & 0xFFFFFFF;
    v10 = **a1;
    goto LABEL_9;
  }
  return 1;
}
