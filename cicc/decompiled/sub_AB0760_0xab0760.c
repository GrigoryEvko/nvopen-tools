// Function: sub_AB0760
// Address: 0xab0760
//
bool __fastcall sub_AB0760(__int64 a1)
{
  bool v1; // r8
  bool result; // al
  unsigned int v3; // esi
  __int64 v4; // rax

  v1 = sub_AB0120(a1);
  result = 0;
  if ( !v1 )
  {
    v3 = *(_DWORD *)(a1 + 8);
    v4 = *(_QWORD *)a1;
    if ( v3 > 0x40 )
      v4 = *(_QWORD *)(v4 + 8LL * ((v3 - 1) >> 6));
    return (v4 & (1LL << ((unsigned __int8)v3 - 1))) == 0;
  }
  return result;
}
