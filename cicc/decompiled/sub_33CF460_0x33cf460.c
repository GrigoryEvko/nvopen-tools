// Function: sub_33CF460
// Address: 0x33cf460
//
bool __fastcall sub_33CF460(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdi
  unsigned int v3; // ebx

  result = *(_DWORD *)(a1 + 24) == 35 || *(_DWORD *)(a1 + 24) == 11;
  if ( result )
  {
    v2 = *(_QWORD *)(a1 + 96);
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
