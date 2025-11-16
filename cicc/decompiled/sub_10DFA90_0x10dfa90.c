// Function: sub_10DFA90
// Address: 0x10dfa90
//
__int64 __fastcall sub_10DFA90(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx

  result = 1;
  if ( *(_BYTE *)a1 != 60 )
  {
    result = 0;
    if ( *(_BYTE *)a1 == 85 )
    {
      v2 = *(_QWORD *)(a1 - 32);
      result = 3;
      if ( v2 )
      {
        if ( !*(_BYTE *)v2 && *(_QWORD *)(v2 + 24) == *(_QWORD *)(a1 + 80) && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
        {
          if ( *(_DWORD *)(v2 + 36) == 342 )
            return 2;
          else
            return (unsigned __int8)sub_B46970((unsigned __int8 *)a1) != 0 ? 3 : 0;
        }
      }
    }
  }
  return result;
}
