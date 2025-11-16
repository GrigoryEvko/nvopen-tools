// Function: sub_1D13E90
// Address: 0x1d13e90
//
__int64 __fastcall sub_1D13E90(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  unsigned int v4; // ebx

  v2 = *a2;
  v3 = *(_QWORD *)(*a2 + 88);
  v4 = *(_DWORD *)(v3 + 32);
  if ( v4 <= 0x40 )
  {
    LOBYTE(v2) = *(_QWORD *)(v3 + 24) == 0;
  }
  else
  {
    LODWORD(v2) = sub_16A57B0(v3 + 24);
    LOBYTE(v2) = v4 == (_DWORD)v2;
  }
  return (unsigned int)v2 ^ 1;
}
