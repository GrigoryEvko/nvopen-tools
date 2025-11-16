// Function: sub_1D18910
// Address: 0x1d18910
//
bool __fastcall sub_1D18910(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdi
  unsigned int v3; // ebx

  result = *(_WORD *)(a1 + 24) == 32 || *(_WORD *)(a1 + 24) == 10;
  if ( result )
  {
    v2 = *(_QWORD *)(a1 + 88);
    v3 = *(_DWORD *)(v2 + 32);
    if ( v3 <= 0x40 )
      return *(_QWORD *)(v2 + 24) == 1;
    else
      return v3 - 1 == (unsigned int)sub_16A57B0(v2 + 24);
  }
  return result;
}
