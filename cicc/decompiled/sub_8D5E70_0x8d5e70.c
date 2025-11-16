// Function: sub_8D5E70
// Address: 0x8d5e70
//
__int64 __fastcall sub_8D5E70(__int64 a1)
{
  __int64 result; // rax

  if ( (*(_BYTE *)(a1 + 96) & 2) != 0 )
  {
    result = (__int64)sub_72B780(a1)[1];
    if ( result )
    {
      while ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(result + 16) + 112LL) + 25LL) )
      {
        result = *(_QWORD *)result;
        if ( !result )
          return result;
      }
      return 1;
    }
    else
    {
      return 0;
    }
  }
  else
  {
    result = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 8LL);
    if ( result )
    {
      while ( !*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(result + 16) + 112LL) + 25LL) )
      {
        result = *(_QWORD *)result;
        if ( !result )
          return result;
      }
      return 1;
    }
    else
    {
      return 0;
    }
  }
}
