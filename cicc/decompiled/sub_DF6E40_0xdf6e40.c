// Function: sub_DF6E40
// Address: 0xdf6e40
//
char __fastcall sub_DF6E40(__int64 a1, __int64 a2)
{
  char result; // al
  __int64 v3; // rdx
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // r8
  __int64 v7; // rdx
  __int64 v8; // rcx
  _BYTE *v9; // rdi

  if ( *(_BYTE *)a2 != 86 )
    return 0;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(a2 - 8);
  else
    v3 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  if ( **(_BYTE **)(v3 + 32) <= 0x15u && **(_BYTE **)(v3 + 64) <= 0x15u )
    return 0;
  v4 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  if ( !sub_BCAC40(v4, 1) )
    goto LABEL_15;
  if ( *(_BYTE *)a2 == 57 )
    return 0;
  v5 = *(_QWORD *)(a2 + 8);
  if ( *(_BYTE *)a2 == 86 && v5 == *(_QWORD *)(*(_QWORD *)(a2 - 96) + 8LL) && **(_BYTE **)(a2 - 32) <= 0x15u )
  {
    if ( !sub_AC30F0(*(_QWORD *)(a2 - 32)) )
    {
LABEL_15:
      v5 = *(_QWORD *)(a2 + 8);
      goto LABEL_16;
    }
    return 0;
  }
LABEL_16:
  if ( (unsigned int)*(unsigned __int8 *)(v5 + 8) - 17 <= 1 )
    v5 = **(_QWORD **)(v5 + 16);
  result = sub_BCAC40(v5, 1);
  if ( !result )
    return 1;
  if ( *(_BYTE *)a2 == 58 )
    return 0;
  if ( *(_BYTE *)a2 == 86 )
  {
    v7 = *(_QWORD *)(a2 - 96);
    v8 = *(_QWORD *)(a2 + 8);
    if ( *(_QWORD *)(v7 + 8) == v8 )
    {
      v9 = *(_BYTE **)(a2 - 64);
      if ( *v9 <= 0x15u )
        return !sub_AD7A80(v9, 1, v7, v8, v6);
    }
  }
  return result;
}
