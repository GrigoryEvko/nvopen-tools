// Function: sub_13CFFB0
// Address: 0x13cffb0
//
bool __fastcall sub_13CFFB0(__int64 a1)
{
  unsigned int v1; // ebx
  bool result; // al
  int v3; // r13d

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 > 0x40 )
  {
    v3 = sub_16A58F0(a1);
    result = 0;
    if ( v3 )
      return (unsigned int)sub_16A57B0(a1) + v3 == v1;
  }
  else
  {
    result = 0;
    if ( *(_QWORD *)a1 )
      return (*(_QWORD *)a1 & (*(_QWORD *)a1 + 1LL)) == 0;
  }
  return result;
}
