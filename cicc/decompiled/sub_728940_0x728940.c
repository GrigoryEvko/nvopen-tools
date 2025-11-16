// Function: sub_728940
// Address: 0x728940
//
_BOOL8 __fastcall sub_728940(__int64 a1, __int64 a2, char a3)
{
  _BOOL8 result; // rax
  __int64 v4; // rcx
  unsigned int v5; // r9d
  unsigned int v6; // eax
  bool v7; // zf
  __int64 v8; // rdx
  __int64 v9; // r8
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rsi

  result = 0;
  if ( ((*(_BYTE *)(a1 + 89) ^ *(_BYTE *)(a2 + 89)) & 4) != 0
    || ((*(_BYTE *)(a2 + 90) ^ *(_BYTE *)(a1 + 90)) & 0x10) != 0
    || (a3 & 3) != 0 && ((*(_BYTE *)(a2 + 90) ^ *(_BYTE *)(a1 + 90)) & 0x20) != 0 )
  {
    return result;
  }
  result = 1;
  if ( (*(_WORD *)(a1 + 176) & 0x1FF) == 3 )
    return result;
  v4 = *(_QWORD *)(a2 + 40);
  v5 = a3 & 3;
  if ( (a3 & 3) != 0 )
    v5 = 64;
  v6 = v5;
  v7 = (a3 & 4) == 0;
  v8 = *(_QWORD *)(a1 + 40);
  if ( !v7 )
  {
    BYTE1(v6) = BYTE1(v5) | 1;
    v5 = v6;
  }
  v9 = *(_BYTE *)(a1 + 89) & 4;
  if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
  {
    v10 = *(_QWORD *)(v8 + 32);
    v11 = *(_QWORD *)(v4 + 32);
    result = 1;
    if ( v10 != v11 )
      return (unsigned int)sub_8D97D0(v10, v11, v5, v4, v9) != 0;
    return result;
  }
  if ( !v8 || *(_BYTE *)(v8 + 28) != 3 )
  {
    result = 1;
    if ( !v4 )
      return result;
    v12 = 0;
    if ( *(_BYTE *)(v4 + 28) != 3 )
      return result;
    goto LABEL_19;
  }
  v12 = *(_QWORD *)(v8 + 32);
  v13 = 0;
  if ( v4 && *(_BYTE *)(v4 + 28) == 3 )
LABEL_19:
    v13 = *(_QWORD *)(v4 + 32);
  if ( v13 == v12 )
    return 1;
  result = 0;
  if ( *qword_4D03FD0 && v8 && *(_BYTE *)(v8 + 28) == 3 )
  {
    v14 = *(_QWORD *)(v8 + 32);
    if ( v4 && v14 )
    {
      if ( *(_BYTE *)(v4 + 28) == 3 )
      {
        v15 = *(_QWORD *)(v4 + 32);
        if ( v15 )
          return (unsigned int)sub_8C7EB0(v14, v15) != 0;
      }
    }
    else
    {
      return 0;
    }
  }
  return result;
}
