// Function: sub_17AE1C0
// Address: 0x17ae1c0
//
_BOOL8 __fastcall sub_17AE1C0(__int64 a1, unsigned __int8 a2)
{
  unsigned __int8 v2; // al
  __int64 v4; // r13
  int v5; // r14d
  unsigned int v6; // ebx
  int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax

  while ( 1 )
  {
    v2 = *(_BYTE *)(a1 + 16);
    if ( v2 <= 0x10u )
      break;
    if ( v2 <= 0x17u )
      return 0;
    v7 = v2;
    if ( v2 == 84 && a2 )
    {
      if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
        v10 = *(_QWORD *)(a1 - 8);
      else
        v10 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
      if ( *(_BYTE *)(*(_QWORD *)(v10 + 48) + 16LL) == 13 )
        return 1;
      goto LABEL_28;
    }
LABEL_13:
    if ( v2 == 54 )
    {
      v9 = *(_QWORD *)(a1 + 8);
      if ( !v9 )
        return 0;
      return !*(_QWORD *)(v9 + 8);
    }
    if ( (unsigned int)(v7 - 35) <= 0x11 )
    {
      v8 = *(_QWORD *)(a1 + 8);
      if ( !v8 || *(_QWORD *)(v8 + 8) )
        return 0;
      if ( (unsigned __int8)sub_17AE1C0(*(_QWORD *)(a1 - 48), a2)
        || (unsigned __int8)sub_17AE1C0(*(_QWORD *)(a1 - 24), a2) )
      {
        return 1;
      }
      v2 = *(_BYTE *)(a1 + 16);
    }
LABEL_28:
    if ( (unsigned __int8)(v2 - 75) > 1u )
      return 0;
    v11 = *(_QWORD *)(a1 + 8);
    if ( !v11 || *(_QWORD *)(v11 + 8) )
      return 0;
    if ( (unsigned __int8)sub_17AE1C0(*(_QWORD *)(a1 - 48), a2) )
      return 1;
    a1 = *(_QWORD *)(a1 - 24);
  }
  if ( a2 )
    return 1;
  v4 = sub_15A0A60(a1, 0);
  if ( !v4 )
  {
    v7 = *(unsigned __int8 *)(a1 + 16);
    v2 = *(_BYTE *)(a1 + 16);
    if ( (unsigned __int8)v7 <= 0x17u )
      return 0;
    goto LABEL_13;
  }
  v5 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( v5 == 1 )
    return 1;
  v6 = 1;
  while ( v4 == sub_15A0A60(a1, v6) )
  {
    if ( ++v6 == v5 )
      return 1;
  }
  return 0;
}
