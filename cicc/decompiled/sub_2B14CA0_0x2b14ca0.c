// Function: sub_2B14CA0
// Address: 0x2b14ca0
//
char __fastcall sub_2B14CA0(__int64 a1)
{
  bool v1; // r8
  char result; // al
  __int64 v3; // rdx
  __int64 v4; // rdi
  unsigned int v5; // ebx

  if ( (unsigned __int8)(*(_BYTE *)a1 - 61) > 1u )
  {
    result = 1;
    if ( *(_BYTE *)a1 == 85 )
    {
      v3 = *(_QWORD *)(a1 - 32);
      if ( v3 )
      {
        if ( !*(_BYTE *)v3
          && *(_QWORD *)(v3 + 24) == *(_QWORD *)(a1 + 80)
          && (*(_BYTE *)(v3 + 33) & 0x20) != 0
          && (unsigned int)(*(_DWORD *)(v3 + 36) - 238) <= 7
          && ((1LL << (*(_BYTE *)(v3 + 36) + 18)) & 0xAD) != 0 )
        {
          v4 = *(_QWORD *)(a1 + 32 * (3LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
          v5 = *(_DWORD *)(v4 + 32);
          if ( v5 <= 0x40 )
            return *(_QWORD *)(v4 + 24) == 0;
          else
            return v5 == (unsigned int)sub_C444A0(v4 + 24);
        }
      }
    }
  }
  else
  {
    v1 = sub_B46500((unsigned __int8 *)a1);
    result = 0;
    if ( !v1 )
      return !(*(_WORD *)(a1 + 2) & 1);
  }
  return result;
}
