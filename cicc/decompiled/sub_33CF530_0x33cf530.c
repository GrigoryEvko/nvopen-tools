// Function: sub_33CF530
// Address: 0x33cf530
//
bool __fastcall sub_33CF530(__int64 a1)
{
  bool result; // al
  __int64 v2; // rdi
  unsigned int v3; // eax
  __int64 v4; // rsi
  unsigned int v5; // ebx

  result = *(_DWORD *)(a1 + 24) == 35 || *(_DWORD *)(a1 + 24) == 11;
  if ( result )
  {
    v2 = *(_QWORD *)(a1 + 96);
    v3 = *(_DWORD *)(v2 + 32);
    v4 = *(_QWORD *)(v2 + 24);
    v5 = v3 - 1;
    if ( v3 <= 0x40 )
    {
      return v4 == 1LL << v5;
    }
    else
    {
      result = 0;
      if ( (*(_QWORD *)(v4 + 8LL * (v5 >> 6)) & (1LL << v5)) != 0 )
        return (unsigned int)sub_C44590(v2 + 24) == v5;
    }
  }
  return result;
}
