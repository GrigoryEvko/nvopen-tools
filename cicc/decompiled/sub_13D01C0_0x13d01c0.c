// Function: sub_13D01C0
// Address: 0x13d01c0
//
bool __fastcall sub_13D01C0(__int64 a1)
{
  unsigned int v1; // ebx

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 <= 0x40 )
    return *(_QWORD *)a1 == 0;
  else
    return (unsigned int)sub_16A57B0(a1) == v1;
}
