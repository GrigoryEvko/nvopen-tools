// Function: sub_38D7BC0
// Address: 0x38d7bc0
//
char __fastcall sub_38D7BC0(__int64 a1, __int64 a2, __int64 a3)
{
  char result; // al
  int v4; // eax

  result = 0;
  if ( !*(_QWORD *)(a1 + 176) )
  {
    if ( a3 == 5 )
    {
      if ( *(_DWORD *)a2 == 2019914798 && *(_BYTE *)(a2 + 4) == 116 )
      {
        return 1;
      }
      else
      {
        if ( *(_DWORD *)a2 != 1952539694 || (v4 = 0, *(_BYTE *)(a2 + 4) != 97) )
          v4 = 1;
        return v4 == 0;
      }
    }
    else if ( a3 == 4 )
    {
      return *(_DWORD *)a2 == 1936941614;
    }
  }
  return result;
}
