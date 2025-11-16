// Function: sub_1C51340
// Address: 0x1c51340
//
bool __fastcall sub_1C51340(__int64 a1)
{
  int v2; // edx
  __int16 v3; // ax
  __int64 v4; // rdi
  unsigned int v5; // esi
  __int64 v6; // rax
  bool result; // al
  __int64 v8; // r12
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int8 v11; // dl
  __int64 v12; // rax
  int v13; // eax
  unsigned int v14; // ecx

  while ( 1 )
  {
    v2 = *(unsigned __int16 *)(a1 + 24);
    v3 = *(_WORD *)(a1 + 24);
    if ( (unsigned __int16)(v2 - 7) <= 2u || (unsigned int)(v2 - 4) <= 1 )
      break;
    while ( 1 )
    {
      if ( !v3 )
      {
        v4 = *(_QWORD *)(a1 + 32);
        v5 = *(_DWORD *)(v4 + 32);
        v6 = *(_QWORD *)(v4 + 24);
        if ( v5 > 0x40 )
          v6 = *(_QWORD *)(v6 + 8LL * ((v5 - 1) >> 6));
        return (v6 & (1LL << ((unsigned __int8)v5 - 1))) != 0;
      }
      if ( (unsigned int)(v2 - 1) > 2 )
        break;
      a1 = *(_QWORD *)(a1 + 32);
      v2 = *(unsigned __int16 *)(a1 + 24);
      v3 = *(_WORD *)(a1 + 24);
      if ( (unsigned int)(v2 - 4) <= 1 || (unsigned __int16)(v2 - 7) <= 2u )
        goto LABEL_11;
    }
    if ( v3 != 6 )
    {
      if ( v3 != 10 )
        return 1;
      v10 = sub_1CCAE90(*(_QWORD *)(a1 - 8), 1);
      v11 = *(_BYTE *)(v10 + 16);
      if ( v11 > 0x17u )
      {
        if ( v11 != 78 )
          return 1;
        v12 = *(_QWORD *)(v10 - 24);
        if ( *(_BYTE *)(v12 + 16) || (*(_BYTE *)(v12 + 33) & 0x20) == 0 )
          return 1;
        v13 = *(_DWORD *)(v12 + 36);
        if ( (unsigned int)(v13 - 3778) <= 1 )
          return 0;
        v14 = v13 - 4286;
        result = 1;
        if ( v14 <= 0x3E )
          return ((0x5C07380000000007uLL >> v14) & 1) == 0;
        return result;
      }
      return 0;
    }
    if ( (unsigned __int8)sub_1C51340(*(_QWORD *)(a1 + 32)) )
      return 1;
    a1 = *(_QWORD *)(a1 + 40);
  }
LABEL_11:
  v8 = 0;
  v9 = 8LL * (unsigned int)*(_QWORD *)(a1 + 40);
  if ( !(unsigned int)*(_QWORD *)(a1 + 40) )
    return 0;
  while ( !(unsigned __int8)sub_1C51340(*(_QWORD *)(*(_QWORD *)(a1 + 32) + v8)) )
  {
    v8 += 8;
    if ( v9 == v8 )
      return 0;
  }
  return 1;
}
