// Function: sub_82EC50
// Address: 0x82ec50
//
__int64 __fastcall sub_82EC50(__int64 a1, __int64 a2, __int64 a3)
{
  char v5; // al
  __int64 v6; // rax

  while ( a1 )
  {
    while ( 1 )
    {
      if ( *(_QWORD *)(a1 + 16) )
        return 1;
      v5 = *(_BYTE *)(a1 + 8);
      if ( v5 )
      {
        if ( v5 == 1 )
        {
          if ( (unsigned int)sub_82EC50(*(_QWORD *)(a1 + 24)) )
            return 1;
        }
        else if ( v5 != 2 )
        {
          sub_721090();
        }
      }
      else if ( (unsigned int)sub_82ED00(*(_QWORD *)(a1 + 24) + 8LL, a2, a3) )
      {
        return 1;
      }
      v6 = *(_QWORD *)a1;
      if ( !*(_QWORD *)a1 )
        return 0;
      if ( *(_BYTE *)(v6 + 8) == 3 )
        break;
      a1 = *(_QWORD *)a1;
      if ( !v6 )
        return 0;
    }
    a1 = sub_6BBB10((_QWORD *)a1);
  }
  return 0;
}
