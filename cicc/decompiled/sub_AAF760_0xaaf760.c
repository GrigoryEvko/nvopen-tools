// Function: sub_AAF760
// Address: 0xaaf760
//
char __fastcall sub_AAF760(__int64 a1)
{
  unsigned int v1; // ebx
  char result; // al

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 <= 0x40 )
  {
    result = 0;
    if ( *(_QWORD *)a1 == *(_QWORD *)(a1 + 16) )
    {
      result = 1;
      if ( v1 )
        return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v1) == *(_QWORD *)a1;
    }
  }
  else
  {
    result = sub_C43C50(a1, a1 + 16);
    if ( result )
      return v1 == (unsigned int)sub_C445E0(a1);
  }
  return result;
}
