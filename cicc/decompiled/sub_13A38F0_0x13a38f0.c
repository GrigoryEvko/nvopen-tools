// Function: sub_13A38F0
// Address: 0x13a38f0
//
bool __fastcall sub_13A38F0(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // r13d
  _QWORD *v3; // rax
  bool result; // al
  unsigned int v5; // r13d

  v2 = *(_DWORD *)(a1 + 8);
  if ( v2 <= 0x40 )
  {
    v3 = *(_QWORD **)a1;
    return a2 == v3;
  }
  v5 = v2 - sub_16A57B0(a1);
  result = 0;
  if ( v5 <= 0x40 )
  {
    v3 = **(_QWORD ***)a1;
    return a2 == v3;
  }
  return result;
}
