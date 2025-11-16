// Function: sub_987920
// Address: 0x987920
//
bool __fastcall sub_987920(__int64 a1)
{
  unsigned int v1; // r12d
  int v2; // r8d
  bool result; // al
  unsigned int v4; // r12d

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 <= 0x40 )
  {
    result = 0;
    if ( *(_QWORD *)a1 )
      return result;
  }
  else
  {
    v2 = sub_C444A0(a1);
    result = 0;
    if ( v1 != v2 )
      return result;
  }
  v4 = *(_DWORD *)(a1 + 24);
  if ( v4 <= 0x40 )
    return *(_QWORD *)(a1 + 16) == 0;
  else
    return v4 == (unsigned int)sub_C444A0(a1 + 16);
}
