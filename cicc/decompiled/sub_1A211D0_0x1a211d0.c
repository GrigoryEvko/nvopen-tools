// Function: sub_1A211D0
// Address: 0x1a211d0
//
__int64 __fastcall sub_1A211D0(__int64 a1)
{
  __int64 v1; // rdx
  int v2; // eax
  __int64 v3; // rdi
  unsigned int v4; // ebx

  v1 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  v2 = 3 * (3 - v1);
  v3 = *(_QWORD *)(a1 + 24 * (3 - v1));
  v4 = *(_DWORD *)(v3 + 32);
  if ( v4 <= 0x40 )
  {
    LOBYTE(v2) = *(_QWORD *)(v3 + 24) == 0;
  }
  else
  {
    v2 = sub_16A57B0(v3 + 24);
    LOBYTE(v2) = v4 == v2;
  }
  return v2 ^ 1u;
}
