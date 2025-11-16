// Function: sub_7E3130
// Address: 0x7e3130
//
__int64 __fastcall sub_7E3130(__int64 a1)
{
  unsigned int v2; // r13d
  __int64 i; // rax
  __int64 v5; // rbx

  while ( 1 )
  {
    if ( (unsigned int)sub_7E1F40(a1) )
      return 1;
    v2 = sub_8D3410(a1);
    if ( !v2 )
      break;
    a1 = sub_8D4050(a1);
  }
  if ( (unsigned int)sub_8D3A70(a1) )
  {
    for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v5 = *(_QWORD *)(i + 160);
    if ( v5 )
    {
      while ( !(unsigned int)sub_7E3130(*(_QWORD *)(v5 + 120)) )
      {
        if ( !(unsigned int)sub_8D3B10(a1) )
        {
          v5 = *(_QWORD *)(v5 + 112);
          if ( v5 )
            continue;
        }
        return v2;
      }
      return 1;
    }
  }
  return v2;
}
