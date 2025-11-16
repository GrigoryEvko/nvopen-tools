// Function: sub_15F8BF0
// Address: 0x15f8bf0
//
__int64 __fastcall sub_15F8BF0(__int64 a1)
{
  int v1; // eax
  __int64 v2; // rdi
  unsigned int v3; // ebx

  v2 = *(_QWORD *)(a1 - 24);
  if ( *(_BYTE *)(v2 + 16) != 13 )
    return 1;
  v3 = *(_DWORD *)(v2 + 32);
  if ( v3 <= 0x40 )
  {
    LOBYTE(v1) = *(_QWORD *)(v2 + 24) == 1;
  }
  else
  {
    v1 = sub_16A57B0(v2 + 24);
    LOBYTE(v1) = v3 - 1 == v1;
  }
  return v1 ^ 1u;
}
