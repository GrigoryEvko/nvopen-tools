// Function: sub_14560B0
// Address: 0x14560b0
//
bool __fastcall sub_14560B0(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdi
  unsigned int v3; // ebx

  result = 0;
  if ( !*(_WORD *)(a1 + 24) )
  {
    v2 = *(_QWORD *)(a1 + 32);
    v3 = *(_DWORD *)(v2 + 32);
    if ( v3 <= 0x40 )
      return *(_QWORD *)(v2 + 24) == 0;
    else
      return v3 == (unsigned int)sub_16A57B0(v2 + 24);
  }
  return result;
}
