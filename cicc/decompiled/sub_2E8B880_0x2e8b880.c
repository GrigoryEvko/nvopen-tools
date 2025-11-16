// Function: sub_2E8B880
// Address: 0x2e8b880
//
char __fastcall sub_2E8B880(__int64 a1)
{
  int v1; // eax
  char result; // al
  int v3; // eax
  char v4; // al

  if ( (unsigned int)*(unsigned __int16 *)(a1 + 68) - 1 <= 1 && (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 64LL) & 0x10) != 0 )
    return 1;
  v1 = *(_DWORD *)(a1 + 44);
  if ( (v1 & 4) == 0 && (v1 & 8) != 0 )
  {
    if ( !sub_2E88A90(a1, 0x100000, 1) )
      goto LABEL_9;
    return 1;
  }
  if ( (*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) & 0x100000LL) != 0 )
    return 1;
LABEL_9:
  v3 = *(_DWORD *)(a1 + 44);
  if ( (v3 & 4) != 0 || (v3 & 8) == 0 )
    v4 = (unsigned __int8)*(_QWORD *)(*(_QWORD *)(a1 + 16) + 24LL) >> 7;
  else
    v4 = sub_2E88A90(a1, 128, 1);
  if ( v4 )
    return 1;
  result = sub_2E8B090(a1);
  if ( result )
    return *(_WORD *)(a1 + 68) != 24;
  return result;
}
