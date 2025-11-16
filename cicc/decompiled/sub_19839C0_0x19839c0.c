// Function: sub_19839C0
// Address: 0x19839c0
//
bool __fastcall sub_19839C0(__int64 a1)
{
  char v1; // dl
  unsigned int v2; // edx
  bool result; // al
  __int64 v4; // rdx
  __int64 v5; // rdi
  unsigned int v6; // ebx

  v1 = *(_BYTE *)(a1 + 16);
  if ( v1 == 54 || v1 == 55 )
  {
    v2 = *(unsigned __int16 *)(a1 + 18);
    result = !(v2 & 1);
    if ( ((v2 >> 7) & 6) != 0 )
      return 0;
  }
  else
  {
    result = 0;
    if ( v1 == 78 )
    {
      v4 = *(_QWORD *)(a1 - 24);
      if ( !*(_BYTE *)(v4 + 16)
        && (*(_BYTE *)(v4 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v4 + 36) - 133) <= 4
        && ((1LL << (*(_BYTE *)(v4 + 36) + 123)) & 0x15) != 0 )
      {
        v5 = *(_QWORD *)(a1 + 24 * (3LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)));
        v6 = *(_DWORD *)(v5 + 32);
        if ( v6 <= 0x40 )
          return *(_QWORD *)(v5 + 24) == 0;
        else
          return v6 == (unsigned int)sub_16A57B0(v5 + 24);
      }
    }
  }
  return result;
}
