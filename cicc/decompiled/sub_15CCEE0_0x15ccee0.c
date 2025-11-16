// Function: sub_15CCEE0
// Address: 0x15ccee0
//
char __fastcall sub_15CCEE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // r15
  char result; // al
  __int64 i; // rdx

  v4 = *(_QWORD *)(a3 + 40);
  if ( !sub_15CC510(a1, v4) )
    return 1;
  v5 = *(_QWORD *)(a2 + 40);
  result = (a3 == a2) | (sub_15CC510(a1, v5) == 0);
  if ( result )
    return 0;
  if ( *(_BYTE *)(a2 + 16) == 29 || *(_BYTE *)(a3 + 16) == 77 )
    return sub_15CCE20(a1, a2, v4);
  if ( v4 == v5 )
  {
    for ( i = *(_QWORD *)(v4 + 48); ; i = *(_QWORD *)(i + 8) )
    {
      if ( i )
      {
        if ( i - 24 == a2 )
          return 1;
        if ( i - 24 == a3 )
          return result;
      }
    }
  }
  return sub_15CC8F0(a1, v5, v4);
}
