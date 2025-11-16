// Function: sub_E92970
// Address: 0xe92970
//
__int64 __fastcall sub_E92970(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax

  result = 0;
  if ( !*(_QWORD *)(a1 + 160) )
  {
    if ( a3 == 5 )
    {
      if ( *(_DWORD *)a2 == 2019914798 && *(_BYTE *)(a2 + 4) == 116
        || *(_DWORD *)a2 == 1952539694 && *(_BYTE *)(a2 + 4) == 97 )
      {
        return 1;
      }
    }
    else if ( a3 == 4 && *(_DWORD *)a2 == 1936941614 )
    {
      return 1;
    }
    return 0;
  }
  return result;
}
