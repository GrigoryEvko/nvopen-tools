// Function: sub_2E8E6C0
// Address: 0x2e8e6c0
//
bool __fastcall sub_2E8E6C0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdx
  bool result; // al
  __int64 v4; // rdx
  __int64 v5; // rdx
  int v6; // ecx
  unsigned int v7; // edx

  v2 = *(_QWORD *)(a1 + 32);
  result = 0;
  if ( !*(_BYTE *)(v2 + 40LL * a2) )
  {
    v4 = v2 + 40LL * (a2 - 1);
    if ( *(_BYTE *)v4 == 1 )
    {
      v5 = *(_QWORD *)(v4 + 24);
      v6 = v5 & 7;
      v7 = ((unsigned int)v5 >> 30) & 1;
      result = v6 == 1 || (unsigned __int8)(v6 - 2) <= 1u;
      if ( result )
        return v7;
    }
  }
  return result;
}
