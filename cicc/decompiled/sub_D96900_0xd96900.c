// Function: sub_D96900
// Address: 0xd96900
//
bool __fastcall sub_D96900(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdi
  unsigned int v3; // ebx

  result = 0;
  if ( !*(_WORD *)(a1 + 24) )
  {
    v2 = *(_QWORD *)(a1 + 32);
    v3 = *(_DWORD *)(v2 + 32);
    if ( v3 <= 0x40 )
      return *(_QWORD *)(v2 + 24) == 1;
    else
      return v3 - 1 == (unsigned int)sub_C444A0(v2 + 24);
  }
  return result;
}
