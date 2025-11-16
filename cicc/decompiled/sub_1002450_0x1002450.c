// Function: sub_1002450
// Address: 0x1002450
//
bool __fastcall sub_1002450(__int64 a1)
{
  unsigned int v1; // ebx
  bool result; // al
  int v3; // r13d

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 > 0x40 )
  {
    v3 = sub_C445E0(a1);
    result = 0;
    if ( v3 )
      return (unsigned int)sub_C444A0(a1) + v3 == v1;
  }
  else
  {
    result = 0;
    if ( *(_QWORD *)a1 )
      return (*(_QWORD *)a1 & (*(_QWORD *)a1 + 1LL)) == 0;
  }
  return result;
}
