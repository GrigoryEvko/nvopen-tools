// Function: sub_24E41D0
// Address: 0x24e41d0
//
bool __fastcall sub_24E41D0(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // rax
  bool result; // al

  while ( a1 != a2 )
  {
    if ( !a1 )
      BUG();
    v2 = *(_BYTE *)(a1 - 24);
    if ( v2 == 85 )
    {
      v3 = *(_QWORD *)(a1 - 56);
      if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a1 + 56) || (*(_BYTE *)(v3 + 33) & 0x20) == 0 )
        return 1;
    }
    else
    {
      result = v2 == 34 || v2 == 40;
      if ( result )
        return result;
    }
    a1 = *(_QWORD *)(a1 + 8);
  }
  return 0;
}
