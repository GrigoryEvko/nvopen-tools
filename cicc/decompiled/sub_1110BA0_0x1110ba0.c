// Function: sub_1110BA0
// Address: 0x1110ba0
//
bool __fastcall sub_1110BA0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  _QWORD *v3; // rax
  bool result; // al
  unsigned int v5; // r12d

  v2 = *(_DWORD *)(a2 + 8);
  if ( v2 <= 0x40 )
  {
    v3 = *(_QWORD **)a2;
    return (unsigned __int64)v3 > 1;
  }
  v5 = v2 - sub_C444A0(a2);
  result = 1;
  if ( v5 <= 0x40 )
  {
    v3 = **(_QWORD ***)a2;
    return (unsigned __int64)v3 > 1;
  }
  return result;
}
