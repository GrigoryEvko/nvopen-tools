// Function: sub_1733C00
// Address: 0x1733c00
//
char __fastcall sub_1733C00(_QWORD **a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v4; // rbx
  char v5; // al
  unsigned __int64 v7; // rdx
  __int64 v8; // r12
  unsigned __int8 v9; // al
  unsigned int v10; // ebx
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned int v15; // r12d
  bool v16; // al
  unsigned int v17; // r15d
  int v18; // r14d
  __int64 v19; // rax
  unsigned int v20; // esi
  bool v21; // al
  unsigned int v22; // [rsp+Ch] [rbp-34h]

  v4 = a2;
  v5 = *(_BYTE *)(a2 + 16);
  if ( v5 != 52 )
  {
    if ( v5 != 5 || *(_WORD *)(a2 + 18) != 28 )
      return 0;
    v11 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    v12 = *(_QWORD *)(a2 - 24 * v11);
    if ( (unsigned __int8)(*(_BYTE *)(v12 + 16) - 35) <= 0x11u )
    {
      **a1 = v12;
      if ( sub_1727B40(
             *(_BYTE **)(a2 + 24 * (1LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))),
             a2,
             *(_DWORD *)(a2 + 20) & 0xFFFFFFF,
             v12) )
      {
        return 1;
      }
      v11 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
    }
    v13 = *(_QWORD *)(a2 + 24 * (1 - v11));
    if ( (unsigned __int8)(*(_BYTE *)(v13 + 16) - 35) <= 0x11u )
    {
      **a1 = v13;
      return sub_1727B40(
               *(_BYTE **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)),
               a2,
               4LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF),
               v12);
    }
    return 0;
  }
  v7 = *(_QWORD *)(a2 - 48);
  if ( (unsigned __int8)(*(_BYTE *)(v7 + 16) - 35) > 0x11u )
    goto LABEL_6;
  **a1 = v7;
  v8 = *(_QWORD *)(a2 - 24);
  v9 = *(_BYTE *)(v8 + 16);
  if ( v9 != 13 )
  {
    v7 = *(_QWORD *)v8;
    if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) != 16 || v9 > 0x10u )
    {
LABEL_7:
      if ( (unsigned __int8)(v9 - 35) <= 0x11u )
      {
        **a1 = v8;
        return sub_17279D0(*(_BYTE **)(v4 - 48), a2, v7, a4);
      }
      return 0;
    }
    v14 = sub_15A1020(*(_BYTE **)(a2 - 24), a2, v7, a4);
    if ( v14 && *(_BYTE *)(v14 + 16) == 13 )
    {
      v15 = *(_DWORD *)(v14 + 32);
      if ( v15 <= 0x40 )
      {
        a4 = 64 - v15;
        v7 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v15);
        v16 = v7 == *(_QWORD *)(v14 + 24);
      }
      else
      {
        v16 = v15 == (unsigned int)sub_16A58F0(v14 + 24);
      }
      if ( v16 )
        return 1;
    }
    else
    {
      v17 = 0;
      v18 = *(_QWORD *)(*(_QWORD *)v8 + 32LL);
      if ( !v18 )
        return 1;
      while ( 1 )
      {
        a2 = v17;
        v19 = sub_15A0A60(v8, v17);
        if ( !v19 )
          break;
        a4 = *(unsigned __int8 *)(v19 + 16);
        if ( (_BYTE)a4 != 9 )
        {
          if ( (_BYTE)a4 != 13 )
            break;
          v20 = *(_DWORD *)(v19 + 32);
          if ( v20 <= 0x40 )
          {
            a4 = 64 - v20;
            a2 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v20);
            v21 = a2 == *(_QWORD *)(v19 + 24);
          }
          else
          {
            v22 = *(_DWORD *)(v19 + 32);
            a2 = v22;
            v21 = v22 == (unsigned int)sub_16A58F0(v19 + 24);
          }
          if ( !v21 )
            break;
        }
        if ( v18 == ++v17 )
          return 1;
      }
    }
LABEL_6:
    v8 = *(_QWORD *)(v4 - 24);
    v9 = *(_BYTE *)(v8 + 16);
    goto LABEL_7;
  }
  v10 = *(_DWORD *)(v8 + 32);
  if ( v10 <= 0x40 )
    return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v10) == *(_QWORD *)(v8 + 24);
  else
    return v10 == (unsigned int)sub_16A58F0(v8 + 24);
}
