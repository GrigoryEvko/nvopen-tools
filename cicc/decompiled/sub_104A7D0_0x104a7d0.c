// Function: sub_104A7D0
// Address: 0x104a7d0
//
char __fastcall sub_104A7D0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdi

  if ( (unsigned __int8)(*(_BYTE *)a1 - 63) <= 0x15u )
  {
    return (0x21FFF1uLL >> (*(_BYTE *)a1 - 63)) & 1;
  }
  else if ( *(_BYTE *)a1 == 42 )
  {
    if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
      v2 = *(_QWORD *)(a1 - 8);
    else
      v2 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    LOBYTE(v1) = **(_BYTE **)(v2 + 32) == 17;
  }
  else
  {
    LOBYTE(v1) = 0;
  }
  return v1;
}
