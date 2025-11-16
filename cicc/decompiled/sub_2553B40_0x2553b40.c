// Function: sub_2553B40
// Address: 0x2553b40
//
bool __fastcall sub_2553B40(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdx
  __int64 v3; // rdi
  unsigned int v4; // ebx

  result = 0;
  if ( *(_BYTE *)a1 == 85 )
  {
    v2 = *(_QWORD *)(a1 - 32);
    if ( v2 )
    {
      if ( !*(_BYTE *)v2
        && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80)
        && (*(_BYTE *)(v2 + 33) & 0x20) != 0
        && (unsigned int)(*(_DWORD *)(v2 + 36) - 238) <= 7
        && ((1LL << (*(_BYTE *)(v2 + 36) + 18)) & 0xAD) != 0 )
      {
        v3 = *(_QWORD *)(a1 + 32 * (3LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
        v4 = *(_DWORD *)(v3 + 32);
        if ( v4 <= 0x40 )
          return *(_QWORD *)(v3 + 24) == 0;
        else
          return v4 == (unsigned int)sub_C444A0(v3 + 24);
      }
    }
  }
  return result;
}
