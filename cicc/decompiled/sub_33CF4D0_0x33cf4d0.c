// Function: sub_33CF4D0
// Address: 0x33cf4d0
//
bool __fastcall sub_33CF4D0(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdi
  unsigned int v3; // ebx

  result = *(_DWORD *)(a1 + 24) == 35 || *(_DWORD *)(a1 + 24) == 11;
  if ( result )
  {
    v2 = *(_QWORD *)(a1 + 96);
    v3 = *(_DWORD *)(v2 + 32);
    if ( v3 <= 0x40 )
      return *(_QWORD *)(v2 + 24) == 1;
    else
      return v3 - 1 == (unsigned int)sub_C444A0(v2 + 24);
  }
  return result;
}
