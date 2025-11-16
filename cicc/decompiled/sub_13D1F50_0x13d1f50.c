// Function: sub_13D1F50
// Address: 0x13d1f50
//
char __fastcall sub_13D1F50(__int64 *a1, __int64 a2)
{
  char v3; // al
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // r12
  __int64 v8; // r15
  unsigned int v9; // r13d
  bool v10; // al
  __int64 v11; // rax
  unsigned int v12; // r12d
  bool v13; // al
  int v14; // r12d
  unsigned int v15; // r14d
  __int64 v16; // rax
  char v17; // cl
  unsigned int v18; // esi
  int v19; // [rsp+Ch] [rbp-34h]

  v3 = *(_BYTE *)(a2 + 16);
  if ( v3 == 52 )
  {
    v5 = *a1;
    if ( *a1 == *(_QWORD *)(a2 - 48) )
    {
      if ( (unsigned __int8)sub_13CC520(*(_QWORD *)(a2 - 24)) )
        return 1;
      v5 = *a1;
    }
    if ( *(_QWORD *)(a2 - 24) == v5 )
      return sub_13CC520(*(_QWORD *)(a2 - 48));
    return 0;
  }
  if ( v3 != 5 || *(_WORD *)(a2 + 18) != 28 )
    return 0;
  v6 = *a1;
  v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v8 = *(_QWORD *)(a2 + 24 * (1 - v7));
  if ( *a1 != *(_QWORD *)(a2 - 24 * v7) )
  {
LABEL_9:
    if ( v8 == v6 )
      return sub_13CC690(*(_QWORD *)(a2 - 24 * v7));
    return 0;
  }
  if ( *(_BYTE *)(v8 + 16) == 13 )
  {
    v9 = *(_DWORD *)(v8 + 32);
    if ( v9 <= 0x40 )
      v10 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v9) == *(_QWORD *)(v8 + 24);
    else
      v10 = v9 == (unsigned int)sub_16A58F0(v8 + 24);
    if ( !v10 )
      goto LABEL_9;
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)v8 + 8LL) != 16 )
      goto LABEL_9;
    v11 = sub_15A1020(*(_QWORD *)(a2 + 24 * (1 - v7)));
    if ( v11 && *(_BYTE *)(v11 + 16) == 13 )
    {
      v12 = *(_DWORD *)(v11 + 32);
      if ( v12 <= 0x40 )
        v13 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v12) == *(_QWORD *)(v11 + 24);
      else
        v13 = v12 == (unsigned int)sub_16A58F0(v11 + 24);
      if ( !v13 )
      {
LABEL_24:
        v6 = *a1;
        v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        v8 = *(_QWORD *)(a2 + 24 * (1 - v7));
        goto LABEL_9;
      }
    }
    else
    {
      v14 = *(_QWORD *)(*(_QWORD *)v8 + 32LL);
      if ( v14 )
      {
        v15 = 0;
        while ( 1 )
        {
          v16 = sub_15A0A60(v8, v15);
          if ( !v16 )
            break;
          v17 = *(_BYTE *)(v16 + 16);
          if ( v17 != 9 )
          {
            if ( v17 != 13 )
              goto LABEL_24;
            v18 = *(_DWORD *)(v16 + 32);
            if ( v18 <= 0x40 )
            {
              if ( *(_QWORD *)(v16 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v18) )
                break;
            }
            else
            {
              v19 = *(_DWORD *)(v16 + 32);
              if ( v19 != (unsigned int)sub_16A58F0(v16 + 24) )
                break;
            }
          }
          if ( v14 == ++v15 )
            return 1;
        }
        v6 = *a1;
        v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
        v8 = *(_QWORD *)(a2 + 24 * (1 - v7));
        goto LABEL_9;
      }
    }
  }
  return 1;
}
