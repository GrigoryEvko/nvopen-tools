// Function: sub_D96960
// Address: 0xd96960
//
bool __fastcall sub_D96960(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdi
  unsigned int v3; // ebx

  result = 0;
  if ( !*(_WORD *)(a1 + 24) )
  {
    result = 1;
    v2 = *(_QWORD *)(a1 + 32);
    v3 = *(_DWORD *)(v2 + 32);
    if ( v3 )
    {
      if ( v3 <= 0x40 )
        return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v3) == *(_QWORD *)(v2 + 24);
      else
        return v3 == (unsigned int)sub_C445E0(v2 + 24);
    }
  }
  return result;
}
