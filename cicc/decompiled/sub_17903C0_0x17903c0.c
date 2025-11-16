// Function: sub_17903C0
// Address: 0x17903c0
//
bool __fastcall sub_17903C0(__int64 a1, __int64 a2)
{
  unsigned __int8 v2; // dl
  bool result; // al
  __int64 v4; // rcx

  v2 = *(_BYTE *)(a1 + 16);
  result = 1;
  if ( v2 > 0x17u )
  {
    v4 = *(_QWORD *)(*(_QWORD *)(a2 - 72) + 40LL);
    if ( v2 != 77 || *(_QWORD *)(a1 + 40) != v4 )
    {
      result = 0;
      if ( *(_QWORD *)(a2 + 40) == v4 )
        return *(_QWORD *)(a1 + 40) != v4;
    }
  }
  return result;
}
