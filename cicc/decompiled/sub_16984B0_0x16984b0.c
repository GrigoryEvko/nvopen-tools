// Function: sub_16984B0
// Address: 0x16984b0
//
bool __fastcall sub_16984B0(__int64 a1)
{
  bool result; // al
  int v2; // r12d
  __int64 v3; // rax

  result = (*(_BYTE *)(a1 + 18) & 6) != 0 && (*(_BYTE *)(a1 + 18) & 7) != 3;
  if ( result )
  {
    result = 0;
    if ( *(_WORD *)(a1 + 16) == *(_WORD *)(*(_QWORD *)a1 + 2LL) )
    {
      v2 = *(_DWORD *)(*(_QWORD *)a1 + 4LL);
      v3 = sub_16984A0(a1);
      return (unsigned int)sub_16A70B0(v3, (unsigned int)(v2 - 1)) == 0;
    }
  }
  return result;
}
