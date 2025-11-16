// Function: sub_1BBA4E0
// Address: 0x1bba4e0
//
char __fastcall sub_1BBA4E0(__int64 a1)
{
  char v1; // dl
  char result; // al
  __int64 v3; // rdx
  __int64 v4; // rdi
  unsigned int v5; // ebx

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 54 || v1 == 55 )
    return !sub_15F32D0(a1) && !(*(_WORD *)(a1 + 18) & 1);
  result = 1;
  if ( v1 == 78 )
  {
    v3 = *(_QWORD *)(a1 - 24);
    if ( !*(_BYTE *)(v3 + 16)
      && (*(_BYTE *)(v3 + 33) & 0x20) != 0
      && (unsigned int)(*(_DWORD *)(v3 + 36) - 133) <= 4
      && ((1LL << (*(_BYTE *)(v3 + 36) + 123)) & 0x15) != 0 )
    {
      v4 = *(_QWORD *)(a1 + 24 * (3LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
      v5 = *(_DWORD *)(v4 + 32);
      if ( v5 <= 0x40 )
        return *(_QWORD *)(v4 + 24) == 0;
      else
        return v5 == (unsigned int)sub_16A57B0(v4 + 24);
    }
  }
  return result;
}
