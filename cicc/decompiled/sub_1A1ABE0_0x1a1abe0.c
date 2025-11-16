// Function: sub_1A1ABE0
// Address: 0x1a1abe0
//
__int64 __fastcall sub_1A1ABE0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned int v3; // r12d
  bool v4; // al
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 - 72);
  if ( *(_BYTE *)(v2 + 16) == 13 )
  {
    v3 = *(_DWORD *)(v2 + 32);
    if ( v3 <= 0x40 )
      v4 = *(_QWORD *)(v2 + 24) == 0;
    else
      v4 = v3 == (unsigned int)sub_16A57B0(v2 + 24);
    return *(_QWORD *)(a1 + 24LL * ((v4 + 1) & 3) - 72);
  }
  else
  {
    result = *(_QWORD *)(a1 - 24);
    if ( *(_QWORD *)(a1 - 48) != result )
      return 0;
  }
  return result;
}
